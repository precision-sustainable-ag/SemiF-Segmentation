"""pull_cvat: turn finished CVAT jobs into masks.

Only jobs matching `cvat.pull_when` (by default state=completed) are read.
Like AgIR-CVToolkit's cvat_download, everything from one CVAT task lands in
the run folder's cvat_downloads/<task name>/: annotations/job_<id>.json as
CVAT returned them, and masks/<image_id>.png, each image's vegetation shapes
rasterized at the resolution it was annotated (0/1, which is what train
expects; tile scales masks to the image). Frames tagged with the exclude label,
or deleted in CVAT, become status "excluded".
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import rasterize
from src.labeling.ledger import LabelLedger
from src.labeling.push_cvat import ensure_project
from src.labeling.run_folder import RunFolder, sanitize_name

log = logging.getLogger(__name__)


def job_is_ready(job: dict, pull_when) -> bool:
    return all(
        want is None or job.get(key) == want
        for key, want in (("state", pull_when.state), ("stage", pull_when.stage))
    )


def main(cfg: DictConfig) -> dict | None:
    lcfg, ccfg = cfg.label, cfg.cvat
    source = lcfg.source
    statuses = ("in_cvat", "labeled", "excluded") if lcfg.pull_cvat.refresh else ("in_cvat",)
    run = RunFolder.from_cfg(cfg)

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = [i for i in ledger.items(round_name=lcfg.round, status=statuses) if i["cvat_task_id"]]
        if not items:
            log.info("No images in CVAT for round %s.", lcfg.round)
            return None

        client = CvatClient.from_config(ccfg)
        _, label_ids = ensure_project(client, ccfg)
        vegetation_id = label_ids[ccfg.vegetation_label.name]
        exclude_id = label_ids[ccfg.exclude_label.name]

        by_task = defaultdict(dict)
        for item in items:
            by_task[item["cvat_task_id"]][item["cvat_frame"]] = item

        counts = {"labeled": 0, "excluded": 0, "waiting": 0}
        for task_id, items_by_frame in by_task.items():
            task_dir = run.cvat_downloads / sanitize_name(client.get_task(task_id)["name"])
            meta = client.task_frames(task_id)
            start = meta.get("start_frame", 0)
            deleted = set(meta.get("deleted_frames") or [])
            for job in client.task_jobs(task_id):
                job_frames = [f for f in range(job["start_frame"], job["stop_frame"] + 1) if f in items_by_frame]
                if not job_frames:
                    continue
                if not job_is_ready(job, ccfg.pull_when):
                    counts["waiting"] += len(job_frames)
                    continue

                annotations = client.job_annotations(job["id"])
                (task_dir / "annotations").mkdir(parents=True, exist_ok=True)
                (task_dir / "annotations" / f"job_{job['id']}.json").write_text(json.dumps(annotations))
                excluded = {t["frame"] for t in annotations.get("tags", []) if t["label_id"] == exclude_id}
                shapes = defaultdict(list)
                for shape in annotations.get("shapes", []):
                    if shape["label_id"] == vegetation_id:
                        shapes[shape["frame"]].append(shape)

                for frame in job_frames:
                    item = items_by_frame[frame]
                    if frame in excluded or frame in deleted:
                        ledger.update(source, item["image_id"], status="excluded", cvat_job_id=job["id"], error=None)
                        counts["excluded"] += 1
                        continue
                    frame_meta = meta["frames"][frame - start]
                    _save_mask(ledger, source, item, job["id"], shapes[frame],
                               frame_meta["height"], frame_meta["width"], task_dir / "masks")
                    counts["labeled"] += 1

        log.info("Round %s: %d labeled, %d excluded, %d still waiting on unfinished jobs.",
                 lcfg.round, counts["labeled"], counts["excluded"], counts["waiting"])
        return counts


def _save_mask(ledger, source, item, job_id, shapes, height, width, masks_dir: Path) -> None:
    image_id = item["image_id"]
    mask, skipped = rasterize(shapes, height, width)
    if skipped:
        log.warning("%s: ignored shapes CVAT can't express as a mask here: %s", image_id, dict(skipped))

    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{image_id}.png"
    cv2.imwrite(str(mask_path), mask.astype(np.uint8))

    changed = None
    if item["prelabel_path"]:
        prelabel = cv2.imread(item["prelabel_path"], cv2.IMREAD_GRAYSCALE)
        if prelabel is not None:
            if prelabel.shape != mask.shape:
                prelabel = cv2.resize(prelabel, (width, height), interpolation=cv2.INTER_NEAREST)
            changed = float(np.mean((prelabel > 0) != mask))

    ledger.update(source, image_id, status="labeled", cvat_job_id=job_id,
                  mask_path=str(mask_path), changed_frac=changed, error=None)
