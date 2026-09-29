"""pull_cvat: turn finished CVAT jobs into masks.

Only jobs matching `cvat.pull_when` (by default state=completed) are read.
For each of their images the vegetation shapes are rasterized at the CVAT
resolution, saved, and scaled back to native resolution with nearest-neighbor
sampling. Frames tagged with the exclude label, or deleted in CVAT, become
status "excluded". Masks are 0/1 PNGs, which is what train expects.
"""

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

log = logging.getLogger(__name__)


def job_is_ready(job: dict, pull_when) -> bool:
    return all(
        want is None or job.get(key) == want
        for key, want in (("state", pull_when.state), ("stage", pull_when.stage))
    )


def main(cfg: DictConfig) -> None:
    lcfg, ccfg = cfg.label, cfg.cvat
    source = lcfg.source
    statuses = ("in_cvat", "labeled", "excluded") if lcfg.pull_cvat.refresh else ("in_cvat",)
    masks_dir = Path(cfg.paths.label_masks_dir) / source
    cvat_masks_dir = Path(cfg.paths.label_cvat_masks_dir) / source
    masks_dir.mkdir(parents=True, exist_ok=True)
    cvat_masks_dir.mkdir(parents=True, exist_ok=True)

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = [i for i in ledger.items(round_name=lcfg.round, status=statuses) if i["cvat_task_id"]]
        if not items:
            log.info("No images in CVAT for round %s.", lcfg.round)
            return

        client = CvatClient.from_config(ccfg)
        _, label_ids = ensure_project(client, ccfg)
        vegetation_id = label_ids[ccfg.vegetation_label.name]
        exclude_id = label_ids[ccfg.exclude_label.name]

        by_task = defaultdict(dict)
        for item in items:
            by_task[item["cvat_task_id"]][item["cvat_frame"]] = item

        counts = defaultdict(int)
        for task_id, items_by_frame in by_task.items():
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
                    _save_masks(ledger, source, item, job["id"], shapes[frame],
                                frame_meta["height"], frame_meta["width"], masks_dir, cvat_masks_dir)
                    counts["labeled"] += 1

        log.info("Round %s: %d labeled, %d excluded, %d still waiting on unfinished jobs.",
                 lcfg.round, counts["labeled"], counts["excluded"], counts["waiting"])


def _save_masks(ledger, source, item, job_id, shapes, height, width, masks_dir, cvat_masks_dir) -> None:
    image_id = item["image_id"]
    mask, skipped = rasterize(shapes, height, width)
    if skipped:
        log.warning("%s: ignored shapes CVAT can't express as a mask here: %s", image_id, dict(skipped))

    mask_u8 = mask.astype(np.uint8)
    cvat_mask_path = cvat_masks_dir / f"{image_id}.png"
    cv2.imwrite(str(cvat_mask_path), mask_u8)

    native_size = (item["native_width"], item["native_height"])
    native = cv2.resize(mask_u8, native_size, interpolation=cv2.INTER_NEAREST)
    mask_path = masks_dir / f"{image_id}.png"
    cv2.imwrite(str(mask_path), native)

    changed = None
    if item["prelabel_path"]:
        prelabel = cv2.imread(item["prelabel_path"], cv2.IMREAD_GRAYSCALE)
        if prelabel is not None:
            if prelabel.shape != mask.shape:
                prelabel = cv2.resize(prelabel, (width, height), interpolation=cv2.INTER_NEAREST)
            changed = float(np.mean((prelabel > 0) != mask))

    ledger.update(source, image_id, status="labeled", cvat_job_id=job_id,
                  cvat_mask_path=str(cvat_mask_path), mask_path=str(mask_path),
                  changed_frac=changed, error=None)
