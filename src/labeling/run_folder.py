"""A labeling round's run folder, in AgIR-CVToolkit's output format.

AgIR-CVToolkit (`agir-cv query`, `agir-cv scinet-transfer`) writes each run to
outputs/runs/<project.name>/<project.subname>/. A labeling round is one such
run, with project.subname = label.round:

    cfg.yaml                 config of the latest `mode=label` run, with runtime and paths blocks
    logs/<epoch>.log         one log file per run (Hydra's own files go to logs/hydra/)
    query/query.csv          select: the round's images, one row each, as the source DB describes them
    query/query_spec.json    select: database, filters, sampling, and who ran it, when, and where
    globus_batch.txt         fetch: "<source path> <destination path>" per file; one batch file per
                             source collection (globus_batch_<root>.txt) when a fetch spans several
    transfer_manifest.json   fetch: backend, endpoints, and one entry per transfer (Globus task ID)
    <path below its root>    fetch: each full-resolution image, laid out as on the source storage,
                             e.g. semifield-developed-images/<batch>/images/<image_id>.jpg
    images/<image_id>.jpg    prepare: the downscaled copy uploaded to CVAT
    masks/<image_id>.png     prepare: the model pre-label uploaded with it (0/255)
    cvat_downloads/<task>/   pull_cvat: annotations/job_<id>.json as CVAT returned them, and
                             masks/<image_id>.png, the human masks (0/1) at CVAT resolution
    manifest.csv             one row per image: status, paths, CVAT task/job/frame
    metrics.json             status counts and each task's latest summary

The project's ledger, labels.db, sits next to its rounds in
outputs/runs/<project.name>/, as do the preprocess/, train/ and inference/
folders of the other modes.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"  # AgIR-CVToolkit's

# manifest.csv column -> labels.db column
MANIFEST_COLUMNS = {
    "image_id": "image_id",
    "source": "source",
    "status": "status",
    "batch": "batch",
    "location": "location",
    "source_path": "path",
    "root": "root",
    "image_path": "native_path",
    "native_width": "native_width",
    "native_height": "native_height",
    "cvat_image_path": "cvat_image_path",
    "prelabel_path": "prelabel_path",
    "cvat_width": "cvat_width",
    "cvat_height": "cvat_height",
    "globus_task_id": "globus_task_id",
    "cvat_task_id": "cvat_task_id",
    "cvat_job_id": "cvat_job_id",
    "cvat_frame": "cvat_frame",
    "mask_path": "mask_path",
    "changed_frac": "changed_frac",
    "split": "split",
    "error": "error",
    "updated_at": "updated_at",
}


class RunFolder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cfg = self.root / "cfg.yaml"
        self.logs = self.root / "logs"
        self.query = self.root / "query"
        self.query_csv = self.query / "query.csv"
        self.query_spec = self.query / "query_spec.json"
        self.transfer_manifest = self.root / "transfer_manifest.json"
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.cvat_downloads = self.root / "cvat_downloads"
        self.manifest = self.root / "manifest.csv"
        self.metrics = self.root / "metrics.json"

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "RunFolder":
        return cls(cfg.paths.run_root)

    def batch_file(self, root_name: str | None = None) -> Path:
        """globus_batch.txt, or globus_batch_<root>.txt when a fetch spans several source roots."""
        return self.root / ("globus_batch.txt" if root_name is None else f"globus_batch_{root_name}.txt")


def run_id(cfg: DictConfig) -> str:
    """<project.name>/<project.subname>, as AgIR-CVToolkit names a run."""
    return f"{cfg.project.name}/{cfg.project.subname}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _cli_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig

        return list(HydraConfig.get().overrides.task)
    except Exception:  # not running under @hydra.main (e.g. tests)
        return []


def who() -> dict:
    return {"user": getpass.getuser(), "host": socket.gethostname(), "git_commit": _git_commit()}


def write_cfg(cfg: DictConfig, run: RunFolder) -> dict:
    """cfg.yaml, as AgIR-CVToolkit's finalize_cfg writes it: the resolved config
    (the parts a round uses), then runtime and paths blocks. Returns runtime."""
    source = cfg.label.source
    material = {
        "project": OmegaConf.to_container(cfg.project, resolve=True),
        "mode": cfg.mode,
        "label": OmegaConf.to_container(cfg.label, resolve=True),
        "sources": {source: OmegaConf.to_container(cfg.sources[source], resolve=True)},
        "transfer": OmegaConf.to_container(cfg.transfer, resolve=True),
        "cvat": OmegaConf.to_container(cfg.cvat, resolve=True),
    }
    runtime = {
        "stage": cfg.mode,
        "dataset": source,
        "created_local": datetime.now().isoformat(timespec="seconds"),
        **who(),
        "cli_overrides": _cli_overrides(),
        "hash": hashlib.md5(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:8],
        "run_id": run_id(cfg),
    }
    paths = {
        "out_root": cfg.paths.out_root,
        "run_root": str(run.root),
        "logs": str(run.logs),
        "query": str(run.query),
        "images": str(run.images),
        "masks": str(run.masks),
        "cvat_downloads": str(run.cvat_downloads),
        "cfg_path": str(run.cfg),
        "manifest_path": str(run.manifest),
        "metrics_path": str(run.metrics),
        "labels_db": cfg.paths.labels_db,
    }
    run.root.mkdir(parents=True, exist_ok=True)
    run.cfg.write_text(yaml.safe_dump({**material, "runtime": runtime, "paths": paths}, sort_keys=False))
    return runtime


def start_log(run: RunFolder) -> logging.Handler:
    """Send this run's log to logs/<epoch>.log, in AgIR-CVToolkit's format."""
    run.logs.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run.logs / f"{int(time.time())}.log")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return handler


def stop_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def write_query(run: RunFolder, rows: pd.DataFrame, spec: dict) -> None:
    """query/query.csv, which keeps the round's earlier picks when it's topped
    up, and query/query_spec.json for the latest query."""
    run.query.mkdir(parents=True, exist_ok=True)
    if run.query_csv.exists():
        rows = pd.concat([pd.read_csv(run.query_csv), rows], ignore_index=True)
        rows = rows.drop_duplicates("image_id", keep="first")
    rows.to_csv(run.query_csv, index=False)
    run.query_spec.write_text(json.dumps(spec, indent=2, default=str))


def write_manifest(run: RunFolder, items: list[dict]) -> None:
    """manifest.csv: one row per image in the round, from labels.db."""
    rows = [{column: item.get(field) for column, field in MANIFEST_COLUMNS.items()} for item in items]
    run.root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(run.manifest, index=False)


def write_metrics(run: RunFolder, cfg: DictConfig, status_counts: dict, summaries: dict) -> None:
    """metrics.json: the round's status counts and each task's latest summary.
    A task that had nothing to do (summary None) keeps its earlier summary."""
    tasks: dict = {}
    if run.metrics.exists():
        try:
            tasks = json.loads(run.metrics.read_text()).get("tasks", {})
        except ValueError:
            log.warning("Replacing unreadable %s", run.metrics)
    now = datetime.now().isoformat(timespec="seconds")
    for task, summary in summaries.items():
        if summary:
            tasks[task] = {"updated_local": now, **summary}
    metrics = {"run_id": run_id(cfg), "updated_local": now, "status_counts": status_counts, "tasks": tasks}
    run.root.mkdir(parents=True, exist_ok=True)
    run.metrics.write_text(json.dumps(metrics, indent=2, default=str))


def sanitize_name(name: str) -> str:
    """A CVAT task name as a folder name, as AgIR-CVToolkit's cvat_download made them."""
    name = re.sub(r"[^\w\-.]", "_", name.lower().replace(" ", "_"))
    name = re.sub(r"_+", "_", name).strip("_")[:200].rstrip("_")
    return name or "unnamed_task"
