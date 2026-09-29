"""select -> fetch -> prepare -> push_cvat -> (annotate) -> pull_cvat, with a
tiny field DB, the copy transfer backend, and an in-memory stand-in for CVAT."""

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.labeling import fetch, prepare, pull_cvat, push_cvat, select
from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import encode_mask
from src.labeling.ledger import LabelLedger


class FakeCvat:
    """Just enough of CvatClient for push_cvat and pull_cvat."""

    def __init__(self):
        self.project = None
        self.tasks = {}
        self.job_state = {}

    def find_project(self, name):
        return self.project

    def create_project(self, name, labels):
        self.project = {"id": 1, "name": name, "labels": labels}
        return self.project

    def label_ids(self, project_id):
        return {label["name"]: 10 + n for n, label in enumerate(self.project["labels"])}

    def create_task(self, name, project_id, segment_size):
        task_id = 100 + len(self.tasks)
        self.tasks[task_id] = {"name": name, "segment_size": segment_size, "frames": [], "shapes": [], "tags": []}
        return {"id": task_id}

    def upload_images(self, task_id, paths, *, image_quality, max_request_bytes):
        # CVAT orders frames by name with sorting_method=natural.
        for path in sorted(paths, key=lambda p: p.name):
            h, w = cv2.imread(str(path)).shape[:2]
            self.tasks[task_id]["frames"].append({"name": path.name, "width": w, "height": h})

    def task_frames(self, task_id):
        return {"start_frame": 0, "frames": self.tasks[task_id]["frames"], "deleted_frames": []}

    def task_jobs(self, task_id):
        task = self.tasks[task_id]
        n, size = len(task["frames"]), task["segment_size"]
        return [
            {"id": task_id * 10 + k, "start_frame": start, "stop_frame": min(start + size, n) - 1,
             "state": self.job_state.get(task_id * 10 + k, "new"), "stage": "annotation", "type": "annotation"}
            for k, start in enumerate(range(0, n, size))
        ]

    def add_task_annotations(self, task_id, shapes, tags=None):
        self.tasks[task_id]["shapes"].extend(shapes)
        self.tasks[task_id]["tags"].extend(tags or [])

    def job_annotations(self, job_id):
        task_id = job_id // 10
        job = next(j for j in self.task_jobs(task_id) if j["id"] == job_id)
        in_job = lambda item: job["start_frame"] <= item["frame"] <= job["stop_frame"]
        task = self.tasks[task_id]
        return {"shapes": [s for s in task["shapes"] if in_job(s)], "tags": [t for t in task["tags"] if in_job(t)]}


def write_field_fixture(root: Path, db_path: Path, n_images: int = 3) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE file_status (base_name TEXT PRIMARY KEY, location_code TEXT, plant_type TEXT, species TEXT,
            growth_stage TEXT, crop_or_fallow TEXT, cover_crop_family TEXT, flower_fruit_or_seeds TEXT,
            processed_jpg_in_nfs BOOLEAN);
        CREATE TABLE file_locations (id INTEGER PRIMARY KEY, base_name TEXT, extension TEXT, artifact_kind TEXT,
            storage_location TEXT, path TEXT, batch_label TEXT);
        """
    )
    rng = np.random.default_rng(0)
    for n in range(n_images):
        base = f"IMG{n}"
        path = root / f"field-batches/B{n % 2}/developed-images/{base}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        conn.execute("INSERT INTO file_status VALUES (?, 'NC', 'COVERCROPS', 'Crimson clover', NULL, NULL, NULL, 'True', 1)", (base,))
        conn.execute(
            "INSERT INTO file_locations (base_name, extension, artifact_kind, storage_location, path, batch_label) "
            "VALUES (?, 'jpg', 'processed_jpg', 'nfs', ?, ?)", (base, str(path), f"B{n % 2}"),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def flow_cfg(tmp_path, make_cfg):
    root, db = tmp_path / "lts", tmp_path / "field_exploration.db"
    root.mkdir()
    write_field_fixture(root, db)
    cfg = make_cfg("label.round=r_test", "label.source=field", "transfer.backend=copy",
                   "label.prelabel.enable=false", "label.prepare.max_side=64", "cvat.segment_size=2")
    OmegaConf.update(cfg, "sources.field.db", str(db))
    OmegaConf.update(cfg, "sources.field.roots",
                     [{"name": "lts", "local_path": str(root), "endpoint": None, "endpoint_path": None}], merge=False)
    return cfg


def statuses(cfg):
    with LabelLedger(cfg.paths.labels_db) as ledger:
        return {i["image_id"]: i for i in ledger.items(round_name=cfg.label.round)}


def test_round_end_to_end(flow_cfg, monkeypatch):
    cfg = flow_cfg
    select.main(cfg)
    assert {i["status"] for i in statuses(cfg).values()} == {"selected"}
    assert len(statuses(cfg)) == 3
    select.main(cfg)  # round is full: no-op
    assert len(statuses(cfg)) == 3
    # select read a local snapshot, not the DB itself.
    assert (Path(cfg.paths.source_db_dir) / "field_exploration.db").exists()

    fetch.main(cfg)
    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"fetched"}
    assert Path(items["IMG0"]["native_path"]) == Path(cfg.paths.label_images_dir) / "field/IMG0.jpg"

    prepare.main(cfg)
    items = statuses(cfg)
    assert items["IMG0"]["native_width"] == 120 and items["IMG0"]["cvat_width"] == 64
    assert cv2.imread(items["IMG0"]["cvat_image_path"]).shape[:2] == (48, 64)

    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    # Pre-labels are normally written by prepare; add one to check it's uploaded.
    prelabel = np.zeros((48, 64), np.uint8)
    prelabel[10:20, 10:30] = 255
    prelabel_path = Path(cfg.paths.label_prelabels_dir) / "r_test/IMG0.png"
    prelabel_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(prelabel_path), prelabel)
    with LabelLedger(cfg.paths.labels_db) as ledger:
        ledger.update("field", "IMG0", prelabel_path=str(prelabel_path))

    push_cvat.main(cfg)
    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"in_cvat"}
    task = fake.tasks[items["IMG0"]["cvat_task_id"]]
    assert task["name"] == "r_test" and len(task["frames"]) == 3
    assert [items[f"IMG{n}"]["cvat_frame"] for n in range(3)] == [0, 1, 2]
    assert items["IMG0"]["cvat_prelabel_uploaded"] == 1
    assert len(task["shapes"]) == 1 and task["shapes"][0]["frame"] == 0

    # Nothing is pulled until a job is completed.
    pull_cvat.main(cfg)
    assert {i["status"] for i in statuses(cfg).values()} == {"in_cvat"}

    # Annotator: keeps IMG0's pre-label, adds a polygon on IMG1 (both job 1),
    # and tags IMG2 (job 2) as exclude but doesn't finish job 2 yet.
    labels = fake.label_ids(1)
    task["shapes"].append({"type": "polygon", "frame": 1, "label_id": labels["vegetation"],
                           "points": [0, 0, 31, 0, 31, 23, 0, 23]})
    task["tags"].append({"frame": 2, "label_id": labels["exclude"]})
    first_job, second_job = fake.task_jobs(100)
    fake.job_state[first_job["id"]] = "completed"

    pull_cvat.main(cfg)
    items = statuses(cfg)
    assert [items[f"IMG{n}"]["status"] for n in range(3)] == ["labeled", "labeled", "in_cvat"]
    native = cv2.imread(items["IMG0"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert native.shape == (90, 120) and set(np.unique(native)) <= {0, 1}
    assert items["IMG0"]["changed_frac"] == 0.0
    cvat_mask = cv2.imread(items["IMG1"]["cvat_mask_path"], cv2.IMREAD_GRAYSCALE)
    assert cvat_mask[:24, :32].all() and cvat_mask.sum() == 24 * 32

    fake.job_state[second_job["id"]] = "completed"
    pull_cvat.main(cfg)
    assert statuses(cfg)["IMG2"]["status"] == "excluded"


def test_prelabeler_tiles_and_loads_lightning_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    from src.labeling.prelabel import Prelabeler

    model = smp.create_model("Unet", encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, checkpoint)

    pcfg = OmegaConf.create({
        "checkpoint": str(checkpoint), "arch": "Unet", "encoder": "resnet18", "normalize": True,
        "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25], "input_scale": 1.0, "tile_size": 64,
        "overlap": 16, "batch_size": 3, "threshold": 0.5, "min_area_px": 4, "device": "cpu",
    })
    prelabeler = Prelabeler(pcfg)
    image = np.random.default_rng(0).integers(0, 255, (70, 150, 3), dtype=np.uint8)
    mask = prelabeler.predict(image, (75, 35))
    assert mask.shape == (35, 75) and mask.dtype == bool

    pcfg.encoder = "resnet34"  # wrong architecture: must fail loudly, not load partially
    with pytest.raises(RuntimeError, match="weights missing"):
        Prelabeler(pcfg)


def test_encode_mask_used_for_prelabels_roundtrips_through_pull():
    bitmap = np.zeros((48, 64), dtype=bool)
    bitmap[5:9, 40:60] = True
    from src.labeling.cvat_masks import rasterize
    mask, _ = rasterize([{"type": "mask", "points": encode_mask(bitmap)}], 48, 64)
    assert np.array_equal(mask, bitmap)
