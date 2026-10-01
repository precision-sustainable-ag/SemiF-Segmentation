"""select -> fetch -> prepare -> push_cvat -> (annotate) -> pull_cvat, with a
tiny field DB, the copy transfer backend, and an in-memory stand-in for CVAT."""

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from src import label
from src.labeling import fetch, prepare, pull_cvat, push_cvat, select
from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import encode_mask
from src.labeling.ledger import LabelLedger
from src.labeling.run_folder import RunFolder


class FakeCvat:
    """Just enough of CvatClient for push_cvat and pull_cvat."""

    base = "https://cvat.test"

    def __init__(self):
        self.project = None
        self.tasks = {}
        self.job_state = {}

    def find_project(self, name):
        return self.project if self.project and self.project["name"] == name else None

    def create_project(self, name, labels):
        self.project = {"id": 1, "name": name, "labels": labels}
        return self.project

    def label_ids(self, project_id):
        return {label["name"]: 10 + n for n, label in enumerate(self.project["labels"])}

    def create_task(self, name, project_id, segment_size):
        task_id = 100 + len(self.tasks)
        self.tasks[task_id] = {"name": name, "project_id": project_id, "segment_size": segment_size, "frames": [], "shapes": [], "tags": []}
        return {"id": task_id}

    def delete_task(self, task_id):
        del self.tasks[task_id]

    def get_task(self, task_id):
        return {"id": task_id, "name": self.tasks[task_id]["name"], "project_id": self.tasks[task_id]["project_id"]}

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
    run = RunFolder(cfg.paths.run_root)
    assert run.root == Path(cfg.paths.persistent_dir) / "outputs/runs/vegetation_001/r_test"
    select.main(cfg)
    assert {i["status"] for i in statuses(cfg).values()} == {"selected"}
    assert len(statuses(cfg)) == 3
    # select read a local snapshot, not the DB itself, and wrote agir-cv query's files.
    assert (Path(cfg.paths.source_db_dir) / "field_exploration.db").exists()
    query = pd.read_csv(run.query_csv)
    assert sorted(query["image_id"]) == ["IMG0", "IMG1", "IMG2"]
    assert query["image_path"].str.endswith(".jpg").all() and "meta" not in query
    spec = json.loads(run.query_spec.read_text())
    assert spec["query_metadata"]["run_id"] == "vegetation_001/r_test"
    assert spec["query_parameters"]["sample"]["parsed"]["strategy"] == "balanced"
    assert spec["execution"]["rows_returned"] == 3

    select.main(cfg)  # every match is already in the project: nothing new
    assert len(statuses(cfg)) == 3
    assert len(pd.read_csv(run.query_csv)) == 3
    assert json.loads(run.query_spec.read_text())["execution"]["rows_already_in_project"] == 3

    fetch.main(cfg)
    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"fetched"}
    # Fetched files keep their path below the source root, inside the run folder.
    assert Path(items["IMG0"]["native_path"]) == run.root / "field-batches/B0/developed-images/IMG0.jpg"
    assert json.loads(run.transfer_manifest.read_text())["num_items"] == 3

    prepare.main(cfg)
    items = statuses(cfg)
    assert items["IMG0"]["native_width"] == 120 and items["IMG0"]["cvat_width"] == 64
    assert Path(items["IMG0"]["cvat_image_path"]) == run.images / "IMG0.jpg"
    assert cv2.imread(items["IMG0"]["cvat_image_path"]).shape[:2] == (48, 64)

    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    # Pre-labels are normally written by prepare; add one to check it's uploaded.
    prelabel = np.zeros((48, 64), np.uint8)
    prelabel[10:20, 10:30] = 255
    prelabel_path = run.masks / "IMG0.png"
    prelabel_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(prelabel_path), prelabel)
    with LabelLedger(cfg.paths.labels_db) as ledger:
        ledger.update_tile("field", "IMG0", "IMG0", prelabel_path=str(prelabel_path))

    summary = push_cvat.main(cfg)
    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"in_cvat"}
    task = fake.tasks[items["IMG0"]["cvat_task_id"]]
    assert task["name"] == "vegetation_001_r_test" and len(task["frames"]) == 3
    assert summary["task_url"] == f"https://cvat.test/tasks/{items['IMG0']['cvat_task_id']}"
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
    downloads = run.cvat_downloads / "vegetation_001_r_test"
    assert Path(items["IMG0"]["mask_path"]) == downloads / "masks/IMG0.png"
    assert (downloads / f"annotations/job_{first_job['id']}.json").exists()
    mask = cv2.imread(items["IMG0"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (90, 120) and set(np.unique(mask)) <= {0, 1}  # full resolution
    assert items["IMG0"]["changed_frac"] == 0.0
    cvat_mask = cv2.imread(str(downloads / "cvat_masks/IMG1.png"), cv2.IMREAD_GRAYSCALE)
    assert cvat_mask.shape == (48, 64) and cvat_mask[:24, :32].all() and cvat_mask.sum() == 24 * 32
    mask = cv2.imread(items["IMG1"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (90, 120) and mask[:44, :59].all() and not mask[47:, :].any() and not mask[:, 62:].any()

    fake.job_state[second_job["id"]] = "completed"
    pull_cvat.main(cfg)
    assert statuses(cfg)["IMG2"]["status"] == "excluded"

    # A full `mode=label` run (nothing left to do) writes the rest of the run folder.
    label.main(cfg)
    saved = yaml.safe_load(run.cfg.read_text())
    assert saved["runtime"]["run_id"] == "vegetation_001/r_test" and saved["runtime"]["stage"] == "label"
    assert saved["paths"]["run_root"] == str(run.root) and saved["label"]["round"] == "r_test"
    assert list(run.logs.glob("*.log"))
    manifest = pd.read_csv(run.manifest)
    assert dict(zip(manifest["image_id"], manifest["status"])) == {"IMG0": "labeled", "IMG1": "labeled", "IMG2": "excluded"}
    metrics = json.loads(run.metrics.read_text())
    assert metrics["run_id"] == "vegetation_001/r_test"
    assert metrics["status_counts"] == {"labeled": 2, "excluded": 1}


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


def test_prepare_uses_existing_masks(flow_cfg):
    cfg = flow_cfg
    masks_root = Path(cfg.paths.persistent_dir) / "segs"
    OmegaConf.update(cfg, "label.prelabel.enable", True)
    OmegaConf.update(cfg, "label.prelabel.source", "masks")
    OmegaConf.update(cfg, "label.prelabel.masks_root", str(masks_root))
    OmegaConf.update(cfg, "label.prelabel.missing_mask", "empty")
    # Class-coded, native-resolution mask for IMG0 only (IMG0 is in batch B0).
    seg = np.zeros((90, 120), np.uint8)
    seg[0:30, 0:60] = 26
    seg[60:90, 90:120] = 3
    (masks_root / "B0/segmentations").mkdir(parents=True)
    cv2.imwrite(str(masks_root / "B0/segmentations/IMG0.png"), seg)

    select.main(cfg)
    fetch.main(cfg)
    counts = prepare.main(cfg)
    assert counts["from_masks"] == 1 and counts["no_mask"] == 2 and counts["errors"] == 0

    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"prepared"}
    assert items["IMG1"]["prelabel_path"] is None and items["IMG2"]["prelabel_path"] is None
    prelabel = cv2.imread(items["IMG0"]["prelabel_path"], cv2.IMREAD_GRAYSCALE)
    assert prelabel.shape == (48, 64)
    assert set(np.unique(prelabel)) == {0, 255}
    assert prelabel[5, 5] == 255 and prelabel[45, 60] == 255 and prelabel[40, 5] == 0


def test_push_cvat_splits_into_tasks_of_task_size(flow_cfg, monkeypatch):
    cfg = flow_cfg
    OmegaConf.update(cfg, "cvat.task_size", 2)
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)

    summary = push_cvat.main(cfg)
    assert summary["images"] == 3 and [t["images"] for t in summary["tasks"]] == [2, 1]
    assert [t["name"] for t in fake.tasks.values()] == ["vegetation_001_r_test", "vegetation_001_r_test-2"]
    items = statuses(cfg)
    assert {i["status"] for i in items.values()} == {"in_cvat"}
    assert len({i["cvat_task_id"] for i in items.values()}) == 2
    assert push_cvat.main(cfg) is None  # everything is already in CVAT


def test_push_cvat_deletes_task_when_upload_fails(flow_cfg, monkeypatch):
    cfg = flow_cfg
    fake = FakeCvat()

    def failing_upload(task_id, paths, **kwargs):
        raise RuntimeError("HTTP 415")

    fake.upload_images = failing_upload
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)

    with pytest.raises(RuntimeError, match="415"):
        push_cvat.main(cfg)
    assert fake.tasks == {}
    assert {i["status"] for i in statuses(cfg).values()} == {"prepared"}


def test_pull_cvat_uses_the_tasks_own_project(flow_cfg, monkeypatch):
    """Pulling with a different cvat.project_name than the push still reads the
    right labels, and doesn't create a project."""
    cfg = flow_cfg
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)
    push_cvat.main(cfg)

    labels = fake.label_ids(1)
    task = fake.tasks[100]
    task["shapes"].append({"type": "polygon", "frame": 0, "label_id": labels["vegetation"],
                           "points": [0, 0, 31, 0, 31, 23, 0, 23]})
    for job in fake.task_jobs(100):
        fake.job_state[job["id"]] = "completed"

    def no_new_projects(*args, **kwargs):
        raise AssertionError("pull_cvat must not create a CVAT project")

    fake.create_project = no_new_projects
    OmegaConf.update(cfg, "cvat.project_name", "some_other_project")
    pull_cvat.main(cfg)
    mask = cv2.imread(statuses(cfg)["IMG0"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (90, 120) and mask[:44, :59].all()


def test_tile_grid_covers_the_image_without_overlap():
    assert [t[4:] for t in prepare.tile_grid(9560, 6368, 6144)] == [(4780, 3184)] * 4
    grid = prepare.tile_grid(13368, 9520, 6144)
    assert [(r, c) for r, c, *_ in grid] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert max(w for *_, w, h in grid) <= 6144 and max(h for *_, w, h in grid) <= 6144
    covered = np.zeros((9520, 13368), np.uint8)
    for _, _, x, y, w, h in grid:
        covered[y:y + h, x:x + w] += 1
    assert (covered == 1).all()
    assert prepare.tile_grid(5000, 4000, 6144) == [(0, 0, 0, 0, 5000, 4000)]


@pytest.fixture
def split_cfg(flow_cfg):
    """Fixture images are 120x90: max_side=64 cuts each into 2x2 tiles of 60x45."""
    cfg = flow_cfg
    OmegaConf.update(cfg, "label.prepare.split", True)
    OmegaConf.update(cfg, "cvat.segment_size", 2)
    masks_root = Path(cfg.paths.persistent_dir) / "segs"
    OmegaConf.update(cfg, "label.prelabel.enable", True)
    OmegaConf.update(cfg, "label.prelabel.source", "masks")
    OmegaConf.update(cfg, "label.prelabel.masks_root", str(masks_root))
    OmegaConf.update(cfg, "label.prelabel.missing_mask", "empty")
    seg = np.zeros((90, 120), np.uint8)
    seg[50:70, 70:100] = 26  # inside IMG0's bottom-right tile
    (masks_root / "B0/segmentations").mkdir(parents=True)
    cv2.imwrite(str(masks_root / "B0/segmentations/IMG0.png"), seg)
    return cfg


def test_split_round_keeps_tiles_in_one_job_and_rebuilds_full_masks(split_cfg, monkeypatch):
    cfg = split_cfg
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    counts = prepare.main(cfg)
    assert counts["prepared"] == 3 and counts["frames"] == 12
    run = RunFolder(cfg.paths.run_root)
    with LabelLedger(cfg.paths.labels_db) as ledger:
        tiles = ledger.tiles("field", "IMG0")
    assert [t["name"] for t in tiles] == ["IMG0_r0c0", "IMG0_r0c1", "IMG0_r1c0", "IMG0_r1c1"]
    assert [(t["x"], t["y"], t["width"], t["height"]) for t in tiles] == [
        (0, 0, 60, 45), (60, 0, 60, 45), (0, 45, 60, 45), (60, 45, 60, 45)]
    assert cv2.imread(tiles[3]["cvat_image_path"]).shape[:2] == (45, 60)  # full resolution
    assert tiles[3]["prelabel_path"] and cv2.imread(tiles[3]["prelabel_path"], 0)[5:25, 10:40].all()

    summary = push_cvat.main(cfg)
    assert summary["images"] == 3 and summary["frames"] == 12 and summary["jobs"] == 2
    task = fake.tasks[100]
    assert task["segment_size"] == 8  # 2 images x 4 tiles
    items = statuses(cfg)
    assert [items[f"IMG{n}"]["cvat_job_id"] for n in range(3)] == [1000, 1000, 1001]
    with LabelLedger(cfg.paths.labels_db) as ledger:
        assert {t["cvat_job_id"] for t in ledger.tiles("field", "IMG2")} == {1001}
        assert all(t["cvat_prelabel_uploaded"] for t in ledger.tiles("field", "IMG0"))
    # IMG0's pre-label is on its bottom-right tile only.
    assert {s["frame"] for s in task["shapes"]} == {3}
    # CVAT has the frames; the local copies are gone, the pre-labels stay.
    assert not list(run.images.iterdir()) and tiles[3]["prelabel_path"] and Path(tiles[3]["prelabel_path"]).exists()

    # Annotator: fills IMG1's top-left tile, completes job 1 (IMG0 and IMG1) only.
    labels = fake.label_ids(1)
    task["shapes"].append({"type": "polygon", "frame": 4, "label_id": labels["vegetation"],
                           "points": [0, 0, 59, 0, 59, 44, 0, 44]})
    fake.job_state[1000] = "completed"
    counts = pull_cvat.main(cfg)
    assert counts == {"labeled": 2, "excluded": 0, "waiting": 1}
    items = statuses(cfg)
    downloads = run.cvat_downloads / "vegetation_001_r_test"
    img0 = cv2.imread(items["IMG0"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert img0.shape == (90, 120) and img0[50:70, 70:100].all() and img0.sum() == 20 * 30
    assert items["IMG0"]["changed_frac"] == 0.0
    img1 = cv2.imread(items["IMG1"]["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert img1[:45, :60].all() and img1.sum() == 45 * 60
    assert sorted(p.name for p in (downloads / "cvat_masks").iterdir()) == [
        f"IMG{n}_r{r}c{c}.png" for n in (0, 1) for r in (0, 1) for c in (0, 1)]


def test_push_cvat_refuses_tasks_that_split_an_image_across_jobs(split_cfg, monkeypatch):
    cfg = split_cfg
    fake = FakeCvat()
    real_jobs = fake.task_jobs
    fake.task_jobs = lambda task_id: [dict(j, stop_frame=j["stop_frame"] - 1 if j["start_frame"] == 0 else j["stop_frame"],
                                           start_frame=j["start_frame"] - 1 if j["start_frame"] else 0)
                                      for j in real_jobs(task_id)]  # boundary one frame early
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)
    with pytest.raises(Exception, match="landed in jobs"):
        push_cvat.main(cfg)
    assert fake.tasks == {}
    assert {i["status"] for i in statuses(cfg).values()} == {"prepared"}


def test_push_cvat_sends_images_with_missing_frames_back_to_prepare(split_cfg, monkeypatch):
    cfg = split_cfg
    OmegaConf.update(cfg, "cvat.delete_frames_after_push", False)
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)
    run = RunFolder(cfg.paths.run_root)
    (run.images / "IMG1_r0c1.jpg").unlink()

    summary = push_cvat.main(cfg)
    assert summary["images"] == 2 and len(list(run.images.iterdir())) == 11  # kept this time
    items = statuses(cfg)
    assert items["IMG1"]["status"] == "fetched" and items["IMG0"]["status"] == "in_cvat"
    assert prepare.main(cfg)["prepared"] == 1 and push_cvat.main(cfg)["images"] == 1
    assert {i["status"] for i in statuses(cfg).values()} == {"in_cvat"}


def test_push_cvat_uploads_each_tasks_prelabels_before_the_next_task(split_cfg, monkeypatch):
    cfg = split_cfg
    OmegaConf.update(cfg, "cvat.task_size", 1)
    fake = FakeCvat()
    events = []
    create, annotate = fake.create_task, fake.add_task_annotations
    fake.create_task = lambda *a, **k: events.append("create") or create(*a, **k)
    fake.add_task_annotations = lambda task_id, *a, **k: events.append(f"prelabels {task_id}") or annotate(task_id, *a, **k)
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)

    summary = push_cvat.main(cfg)
    # Only IMG0 has a pre-label; it's uploaded before the next task is created.
    assert events == ["create", "prelabels 100", "create", "create"]
    assert summary["prelabels_uploaded"] == 1 and [t["prelabels_uploaded"] for t in summary["tasks"]] == [1, 0, 0]


def test_push_cvat_logs_the_plan_before_creating_tasks(split_cfg, monkeypatch, caplog):
    cfg = split_cfg
    OmegaConf.update(cfg, "cvat.task_size", 2)
    OmegaConf.update(cfg, "cvat.segment_size", 1)
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)
    with caplog.at_level("INFO", logger="src.labeling.push_cvat"):
        summary = push_cvat.main(cfg)
    plan = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Push plan"))
    assert plan.splitlines()[0] == "Push plan: 3 images (12 frames) -> 2 tasks, 3 jobs"
    assert "4 frame(s) per image: 3 images -> 2 tasks (2, 1 images), 3 jobs of up to 1 images (4 frames)" in plan
    assert len(summary["tasks"]) == 2 and sum(t["jobs"] for t in summary["tasks"]) == 3


def test_push_cvat_holds_back_images_over_max_images(flow_cfg, monkeypatch):
    cfg = flow_cfg
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    select.main(cfg)
    fetch.main(cfg)
    prepare.main(cfg)
    OmegaConf.update(cfg, "label.push_cvat.image_ids", ["IMG2", "IMG0"])
    OmegaConf.update(cfg, "label.push_cvat.max_images", 1)
    assert push_cvat.main(cfg)["images"] == 1
    assert {k: i["status"] for k, i in statuses(cfg).items()} == {"IMG0": "in_cvat", "IMG1": "prepared", "IMG2": "prepared"}
    OmegaConf.update(cfg, "label.push_cvat", {"max_images": None, "image_ids": []})
    assert push_cvat.main(cfg)["images"] == 2
