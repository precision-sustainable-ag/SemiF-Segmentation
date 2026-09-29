import json

import pandas as pd
from omegaconf import OmegaConf

from src.labeling.run_folder import RunFolder, sanitize_name, write_metrics, write_query


def test_write_query_keeps_earlier_picks(tmp_path):
    run = RunFolder(tmp_path / "run")
    write_query(run, pd.DataFrame({"image_id": ["a", "b"], "image_path": ["x/a.jpg", "x/b.jpg"]}), {"n": 1})
    write_query(run, pd.DataFrame({"image_id": ["b", "c"], "image_path": ["x/b.jpg", "x/c.jpg"]}), {"n": 2})
    assert list(pd.read_csv(run.query_csv)["image_id"]) == ["a", "b", "c"]
    assert json.loads(run.query_spec.read_text()) == {"n": 2}  # the latest query


def test_write_metrics_keeps_summaries_of_tasks_with_nothing_to_do(tmp_path):
    run = RunFolder(tmp_path / "run")
    cfg = OmegaConf.create({"project": {"name": "proj", "subname": "r1"}})
    write_metrics(run, cfg, {"in_cvat": 3}, {"push_cvat": {"task_id": 7}, "pull_cvat": {"waiting": 3}})
    write_metrics(run, cfg, {"labeled": 3}, {"push_cvat": None, "pull_cvat": {"labeled": 3}})
    metrics = json.loads(run.metrics.read_text())
    assert metrics["run_id"] == "proj/r1" and metrics["status_counts"] == {"labeled": 3}
    assert metrics["tasks"]["push_cvat"]["task_id"] == 7
    assert metrics["tasks"]["pull_cvat"]["labeled"] == 3 and "waiting" not in metrics["tasks"]["pull_cvat"]


def test_sanitize_name_matches_cvat_download_folders():
    assert sanitize_name("vegetation_001_r001 Covercrops") == "vegetation_001_r001_covercrops"
    assert sanitize_name("a/b: c") == "a_b_c"
    assert sanitize_name("???") == "unnamed_task"
