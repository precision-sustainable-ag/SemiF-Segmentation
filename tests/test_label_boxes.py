"""Box rounds (label.prelabel.kind=sam3_boxes): select -> fetch -> prepare ->
push_cvat -> (annotate) -> pull_cvat with a fake SAM 3 generator and the
in-memory CVAT of test_label_flow."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src import label
from src.inferencing.tiled_instances import Instances
from src.labeling import fetch, prepare, prelabel_boxes, pull_cvat, push_cvat, select
from src.labeling.cvat_boxes import match_provenance
from src.labeling.cvat_client import CvatClient
from src.labeling.ledger import LabelLedger
from src.labeling.run_folder import RunFolder
from src.utils.instance_annotations import AnnotationSource
from test_label_flow import FakeCvat, flow_cfg, statuses  # noqa: F401 (flow_cfg is a fixture)

# Fixture images are 120x90; at proposals.full_image.scale=0.5 the generator sees 60x45.
SCALED_BOXES = [(5, 5, 25, 20), (30, 10, 50, 40)]


class FakeBoxGenerator:
    """Stands in for SAM 3's ProposalGenerator: the same plant and colorchecker in every image."""

    source = AnnotationSource.SAM3

    def __init__(self):
        self.calls = []

    def predict_tiled(self, image, semantic_mask=None, tile_size=1024, overlap=256, merge_iou=0.5, box_from="mask"):
        self.calls.append((image.shape[:2], tile_size, overlap))
        height, width = image.shape[:2]
        return Instances(height, width, boxes=np.array(SCALED_BOXES, np.int64), labels=np.array([1, 2]),
                         scores=np.array([0.9, 0.8], np.float32),
                         masks=[np.ones((y1 - y0, x1 - x0), bool) for x0, y0, x1, y1 in SCALED_BOXES])


@pytest.fixture
def box_cfg(flow_cfg, monkeypatch):
    cfg = flow_cfg
    OmegaConf.update(cfg, "label.prelabel.kind", "sam3_boxes")
    OmegaConf.update(cfg, "label.prelabel.enable", True)
    OmegaConf.update(cfg, "proposals.full_image.scale", 0.5)
    generator = FakeBoxGenerator()
    monkeypatch.setattr(prelabel_boxes, "build_proposal_generator", lambda name, config: generator)
    fake = FakeCvat()
    monkeypatch.setattr(CvatClient, "from_config", classmethod(lambda cls, ccfg: fake))
    return cfg, generator, fake


def test_box_round_end_to_end(box_cfg):
    cfg, generator, fake = box_cfg
    run = RunFolder(cfg.paths.run_root)
    select.main(cfg)
    fetch.main(cfg)
    counts = prepare.main(cfg)
    assert counts == {"prepared": 3, "frames": 3, "prelabeled": 3, "boxes": 6, "errors": 0}
    assert generator.calls[0] == ((45, 60), 1008, 256)  # downscaled by full_image.scale, SAM 3 tiles

    items = statuses(cfg)
    assert items["IMG0"]["cvat_width"] == 64 and Path(items["IMG0"]["cvat_image_path"]) == run.images / "IMG0.jpg"
    assert Path(items["IMG0"]["prelabel_path"]) == run.boxes / "IMG0.json"
    prelabels = json.loads((run.boxes / "IMG0.json").read_text())
    assert prelabels["image_size"] == [90, 120] and prelabels["cvat_size"] == [48, 64]
    first = prelabels["instances"][0]
    assert first["bbox"] == [10.0, 10.0, 50.0, 40.0] and first["label"] == "plant" and first["class_id"] == 1
    assert first["bbox_cvat"] == [5.33, 5.33, 26.67, 21.33] and first["source"] == "sam3"

    summary = push_cvat.main(cfg)
    assert summary["images"] == 3 and summary["prelabels_uploaded"] == 6
    # Detection labels go in their own project: one rectangle label per class, plus exclude.
    assert fake.project["name"] == "semif_plant_boxes"
    assert [(l["name"], l["type"]) for l in fake.project["labels"]] == [
        ("plant", "rectangle"), ("colorchecker", "rectangle"), ("exclude", "tag")]
    task = fake.tasks[100]
    labels = fake.label_ids(1)
    assert {s["type"] for s in task["shapes"]} == {"rectangle"}
    assert [s["label_id"] for s in task["shapes"] if s["frame"] == 0] == [labels["plant"], labels["colorchecker"]]
    assert prelabels["instances"][1]["label"] == "colorchecker" and prelabels["instances"][1]["class_id"] == 2
    assert [s["points"] for s in task["shapes"] if s["frame"] == 0][0] == first["bbox_cvat"]

    # Annotator: on IMG0 keeps box 1, moves box 2, adds one; on IMG1 deletes box 2; excludes IMG2.
    img0 = [s for s in task["shapes"] if s["frame"] == 0]
    img0[1]["points"] = [33.0, 11.0, 55.0, 44.0]
    task["shapes"].append({"type": "rectangle", "frame": 0, "label_id": labels["plant"], "rotation": 0.0,
                           "points": [40.0, 30.0, 60.0, 45.0]})
    task["shapes"].append({"type": "polygon", "frame": 0, "label_id": labels["plant"], "points": [0, 0, 5, 0, 5, 5]})
    task["shapes"].remove([s for s in task["shapes"] if s["frame"] == 1][1])
    task["tags"].append({"frame": 2, "label_id": labels["exclude"]})
    for job in fake.task_jobs(100):
        fake.job_state[job["id"]] = "completed"

    counts = pull_cvat.main(cfg)
    assert counts == {"labeled": 2, "excluded": 1, "waiting": 0,
                      "sam3": 2, "sam3_corrected": 1, "human": 1, "deleted": 1}
    items = statuses(cfg)
    assert [items[f"IMG{n}"]["status"] for n in range(3)] == ["labeled", "labeled", "excluded"]
    downloads = run.cvat_downloads / "vegetation_001_r_test"
    assert Path(items["IMG0"]["boxes_path"]) == downloads / "boxes/IMG0.json"
    pulled = json.loads((downloads / "boxes/IMG0.json").read_text())
    by_source = {b["source"]: b for b in pulled["instances"]}
    assert sorted(by_source) == ["human", "sam3", "sam3_corrected"] and len(pulled["instances"]) == 3
    assert by_source["sam3"]["bbox"] == pytest.approx([10, 10, 50, 40], abs=0.1)  # full resolution
    assert by_source["sam3_corrected"]["prelabel_id"] == 2
    assert by_source["sam3_corrected"]["bbox"] == pytest.approx([61.9, 20.6, 103.1, 82.5], abs=0.1)
    assert by_source["human"]["bbox"] == pytest.approx([75, 56.25, 112.5, 84.375], abs=0.1) and by_source["human"]["prelabel_id"] is None
    assert [(b["label"], b["class_id"]) for b in pulled["instances"]] == [("plant", 1), ("colorchecker", 2), ("plant", 1)]
    img1 = json.loads((downloads / "boxes/IMG1.json").read_text())
    assert img1["counts"] == {"sam3": 1, "sam3_corrected": 0, "human": 0, "deleted": 1}
    assert img1["deleted_prelabel_ids"] == [2]

    label.main(cfg)  # nothing left to do; writes manifest.csv with the box paths
    manifest = pd.read_csv(run.manifest)
    assert manifest.set_index("image_id").loc["IMG0", "boxes_path"] == str(downloads / "boxes/IMG0.json")


def test_box_round_without_prelabels_starts_empty(box_cfg):
    cfg, generator, fake = box_cfg
    OmegaConf.update(cfg, "label.prelabel.enable", False)
    select.main(cfg)
    fetch.main(cfg)
    assert prepare.main(cfg)["prelabeled"] == 0 and not generator.calls
    push_cvat.main(cfg)
    task = fake.tasks[100]
    assert task["shapes"] == []
    task["shapes"].append({"type": "rectangle", "frame": 0, "label_id": fake.label_ids(1)["plant"],
                           "points": [0.0, 0.0, 32.0, 24.0]})
    for job in fake.task_jobs(100):
        fake.job_state[job["id"]] = "completed"
    assert pull_cvat.main(cfg)["human"] == 1
    pulled = json.loads(Path(statuses(cfg)["IMG0"]["boxes_path"]).read_text())
    assert pulled["instances"][0]["bbox"] == [0.0, 0.0, 60.0, 45.0] and pulled["instances"][0]["source"] == "human"


def test_box_round_refuses_tiles(box_cfg):
    cfg, _, _ = box_cfg
    OmegaConf.update(cfg, "label.prepare.split", True)
    select.main(cfg)
    fetch.main(cfg)
    with pytest.raises(ValueError, match="split=false"):
        prepare.main(cfg)


def test_unknown_prelabel_kind(flow_cfg):
    OmegaConf.update(flow_cfg, "label.prelabel.kind", "sam3_masks")
    select.main(flow_cfg)
    fetch.main(flow_cfg)
    with pytest.raises(ValueError, match="label.prelabel.kind"):
        prepare.main(flow_cfg)


def test_match_provenance():
    uploaded = [{"id": 1, "label": "plant", "bbox_cvat": [0, 0, 10, 10]},
                {"id": 2, "label": "plant", "bbox_cvat": [20, 20, 30, 30]},
                {"id": 3, "label": "plant", "bbox_cvat": [50, 50, 60, 60]}]
    pulled = [{"label": "plant", "bbox_cvat": [21, 20, 31, 30]},      # moved box 2
              {"label": "plant", "bbox_cvat": [0.2, 0, 10, 10.3]},    # box 1, within tolerance
              {"label": "weed", "bbox_cvat": [50, 50, 60, 60]},       # box 3 relabeled
              {"label": "plant", "bbox_cvat": [80, 80, 90, 90]}]      # new
    sources, deleted = match_provenance(pulled, uploaded)
    assert sources == [("sam3_corrected", 2), ("sam3", 1), ("sam3_corrected", 3), ("human", None)]
    assert deleted == []
    claude = [dict(u, source="claude") for u in uploaded]
    sources, _ = match_provenance(pulled[:2], claude)
    assert sources == [("claude_corrected", 2), ("claude", 1)]
    sources, deleted = match_provenance([{"label": "plant", "bbox_cvat": [5, 5, 15, 15]}], uploaded)
    assert sources == [("human", None)] and deleted == [1, 2, 3]  # IoU 1/7 is below match_iou


def test_ledger_adds_boxes_path_to_old_databases(tmp_path):
    db = tmp_path / "labels.db"
    with LabelLedger(db) as ledger:
        ledger.conn.execute("ALTER TABLE items DROP COLUMN boxes_path")
    conn = sqlite3.connect(db)
    assert "boxes_path" not in {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    conn.close()
    with LabelLedger(db) as ledger:
        assert "boxes_path" in {r["name"] for r in ledger.conn.execute("PRAGMA table_info(items)")}
