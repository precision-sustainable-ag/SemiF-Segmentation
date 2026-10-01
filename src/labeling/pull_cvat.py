"""pull_cvat: turn finished CVAT jobs into masks.

Only jobs matching `cvat.pull_when` (by default state=completed) are read.
Like AgIR-CVToolkit's cvat_download, everything from one CVAT task lands in
the run folder's cvat_downloads/<task name>/: annotations/job_<id>.json as
CVAT returned them; cvat_masks/<frame name>.png, each frame's vegetation shapes
rasterized at the resolution it was annotated (one per tile when prepare split
the image); and masks/<image_id>.png, the full-resolution mask put back
together from its frames, which is what the dataset uses. Both are 0/1, which
is what train expects. An image is pulled once all its frames are in finished
jobs; if any of them is tagged with the exclude label, or deleted in CVAT, the
image becomes status "excluded".
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import rasterize, resize_mask
from src.labeling.ledger import LabelLedger
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

        by_task = defaultdict(list)
        for item in items:
            by_task[item["cvat_task_id"]].append((item, ledger.tiles(source, item["image_id"])))

        counts = {"labeled": 0, "excluded": 0, "waiting": 0}
        for task_id, task_items in by_task.items():
            task = client.get_task(task_id)
            # Labels come from the task's own project, whatever cvat.project_name says now.
            label_ids = client.label_ids(task["project_id"])
            missing = [n for n in (ccfg.vegetation_label.name, ccfg.exclude_label.name) if n not in label_ids]
            if missing:
                raise ValueError(f"CVAT project {task['project_id']} of task {task_id} has no {missing} label(s)")
            vegetation_id = label_ids[ccfg.vegetation_label.name]
            exclude_id = label_ids[ccfg.exclude_label.name]
            task_dir = run.cvat_downloads / sanitize_name(task["name"])
            meta = client.task_frames(task_id)
            start = meta.get("start_frame", 0)
            deleted = set(meta.get("deleted_frames") or [])
            wanted = {t["cvat_frame"] for _, tiles in task_items for t in tiles}

            # Shapes and exclude tags of every frame in a finished job.
            ready, excluded, shapes = set(), set(), defaultdict(list)
            for job in client.task_jobs(task_id):
                job_frames = set(range(job["start_frame"], job["stop_frame"] + 1))
                if not job_frames & wanted or not job_is_ready(job, ccfg.pull_when):
                    continue
                annotations = client.job_annotations(job["id"])
                (task_dir / "annotations").mkdir(parents=True, exist_ok=True)
                (task_dir / "annotations" / f"job_{job['id']}.json").write_text(json.dumps(annotations))
                ready |= job_frames
                excluded |= {t["frame"] for t in annotations.get("tags", []) if t["label_id"] == exclude_id}
                for shape in annotations.get("shapes", []):
                    if shape["label_id"] == vegetation_id:
                        shapes[shape["frame"]].append(shape)

            for item, tiles in task_items:
                frames = [t["cvat_frame"] for t in tiles]
                if not tiles or not set(frames) <= ready:
                    counts["waiting"] += 1
                    continue
                if set(frames) & (excluded | deleted):
                    ledger.update(source, item["image_id"], status="excluded", error=None)
                    counts["excluded"] += 1
                    continue
                frame_sizes = {f: (meta["frames"][f - start]["width"], meta["frames"][f - start]["height"])
                               for f in frames}
                _save_mask(ledger, source, item, tiles, shapes, frame_sizes, task_dir)
                counts["labeled"] += 1

        log.info("Round %s: %d labeled, %d excluded, %d still waiting on unfinished jobs.",
                 lcfg.round, counts["labeled"], counts["excluded"], counts["waiting"])
        return counts


def _save_mask(ledger, source, item, tiles, shapes, frame_sizes, task_dir: Path) -> None:
    """Rasterize each tile's shapes, keep them at CVAT resolution in
    cvat_masks/, and place them into the full-resolution mask in masks/."""
    image_id = item["image_id"]
    full = np.zeros((item["native_height"], item["native_width"]), dtype=bool)
    changed_px, total_px = 0, 0
    (task_dir / "cvat_masks").mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        width, height = frame_sizes[tile["cvat_frame"]]
        mask, skipped = rasterize(shapes[tile["cvat_frame"]], height, width)
        if skipped:
            log.warning("%s: ignored shapes CVAT can't express as a mask here: %s", tile["name"], dict(skipped))
        cv2.imwrite(str(task_dir / "cvat_masks" / f"{tile['name']}.png"), mask.astype(np.uint8))

        x, y = tile["x"], tile["y"]
        full[y:y + tile["height"], x:x + tile["width"]] = resize_mask(mask, (tile["width"], tile["height"]))

        prelabel = cv2.imread(tile["prelabel_path"], cv2.IMREAD_GRAYSCALE) if tile["prelabel_path"] else None
        if prelabel is not None:
            if prelabel.shape != mask.shape:
                prelabel = cv2.resize(prelabel, (width, height), interpolation=cv2.INTER_NEAREST)
            changed_px += int(np.count_nonzero((prelabel > 0) != mask))
            total_px += mask.size

    mask_path = task_dir / "masks" / f"{image_id}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), full.astype(np.uint8)):
        raise RuntimeError(f"could not write {mask_path}")
    ledger.update(source, image_id, status="labeled", mask_path=str(mask_path),
                  changed_frac=changed_px / total_px if total_px else None, error=None)
