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

Box rounds (`label.prelabel.kind=sam3_boxes`) write boxes/<image_id>.json
instead: each rectangle of a detection label, at full resolution (`bbox`) and
as annotated (`bbox_cvat`), with its label, class id, and where it came from
by comparison with the uploaded pre-labels (cvat_boxes.match_provenance):
sam3 (unchanged), sam3_corrected (edited), or human (added); the pre-labels
the annotator deleted are listed too.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from src.labeling.cvat_boxes import class_names, is_box_round, match_provenance
from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import rasterize, resize_mask
from src.labeling.ledger import LabelLedger
from src.labeling.push_cvat import required_labels
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
    boxes = is_box_round(lcfg)

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
            missing = [n for n in required_labels(ccfg, boxes) if n not in label_ids]
            if missing:
                raise ValueError(f"CVAT project {task['project_id']} of task {task_id} has no {missing} label(s)")
            # label id -> name of the labels whose shapes are kept
            if boxes:
                shape_labels = {label_ids[name]: name for name in class_names(ccfg).values()}
            else:
                shape_labels = {label_ids[ccfg.vegetation_label.name]: ccfg.vegetation_label.name}
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
                    if shape["label_id"] in shape_labels:
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
                if boxes:
                    found = _save_boxes(ledger, source, item, tiles[0], shapes[frames[0]], frame_sizes[frames[0]],
                                        task_dir, shape_labels, ccfg, float(lcfg.pull_cvat.match_iou))
                    for key, n in found.items():
                        counts[key] = counts.get(key, 0) + n
                else:
                    _save_mask(ledger, source, item, tiles, shapes, frame_sizes, task_dir)
                counts["labeled"] += 1

        log.info("Round %s: %d labeled, %d excluded, %d still waiting on unfinished jobs.",
                 lcfg.round, counts["labeled"], counts["excluded"], counts["waiting"])
        if boxes and counts["labeled"]:
            log.info("Boxes by provenance: %s", {k: v for k, v in counts.items()
                                                  if k not in ("labeled", "excluded", "waiting")})
        return counts


def _save_boxes(ledger, source, item, tile, shapes, frame_size, task_dir: Path, shape_labels: dict,
                ccfg, match_iou: float) -> dict:
    """The frame's rectangles as full-resolution boxes in boxes/<image_id>.json,
    with their provenance. Returns counts per provenance, and deleted."""
    image_id = item["image_id"]
    width, height = frame_size
    sx, sy = item["native_width"] / width, item["native_height"] / height
    class_of = {name: class_id for class_id, name in class_names(ccfg).items()}
    pulled, skipped = [], defaultdict(int)
    for shape in shapes:
        if shape["type"] != "rectangle" or shape.get("rotation"):
            skipped["rotated rectangle" if shape["type"] == "rectangle" else shape["type"]] += 1
            continue
        x0, y0, x1, y1 = (float(v) for v in shape["points"][:4])
        x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
        y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
        if x1 <= x0 or y1 <= y0:
            skipped["empty rectangle"] += 1
            continue
        label = shape_labels[shape["label_id"]]
        pulled.append({"bbox_cvat": [round(v, 2) for v in (x0, y0, x1, y1)], "label": label,
                       "class_id": class_of[label]})
    if skipped:
        log.warning("%s: ignored shapes that aren't boxes: %s", image_id, dict(skipped))

    uploaded = []
    if tile["prelabel_path"] and Path(tile["prelabel_path"]).exists():
        uploaded = json.loads(Path(tile["prelabel_path"]).read_text())["instances"]
    sources, deleted = match_provenance(pulled, uploaded, match_iou)

    # Top to bottom, left to right: the same order however CVAT returned them.
    order = sorted(range(len(pulled)), key=lambda i: (pulled[i]["bbox_cvat"][1], pulled[i]["bbox_cvat"][0]))
    instances = []
    for k, i in enumerate(order):
        box = pulled[i]
        x0, y0, x1, y1 = box["bbox_cvat"]
        full = [round(min(max(x0 * sx, 0), item["native_width"]), 1), round(min(max(y0 * sy, 0), item["native_height"]), 1),
                round(min(max(x1 * sx, 0), item["native_width"]), 1), round(min(max(y1 * sy, 0), item["native_height"]), 1)]
        instances.append({"id": k + 1, "bbox": full, "bbox_cvat": box["bbox_cvat"], "label": box["label"],
                          "class_id": box["class_id"], "source": sources[i][0], "prelabel_id": sources[i][1]})
    counts = {"sam3": 0, "sam3_corrected": 0, "human": 0}
    for inst in instances:
        counts[inst["source"]] = counts.get(inst["source"], 0) + 1
    counts["deleted"] = len(deleted)

    boxes_path = task_dir / "boxes" / f"{image_id}.json"
    boxes_path.parent.mkdir(parents=True, exist_ok=True)
    boxes_path.write_text(json.dumps({
        "image_id": image_id, "image_size": [item["native_height"], item["native_width"]],
        "cvat_size": [height, width], "cvat_task_id": tile["cvat_task_id"], "cvat_job_id": tile["cvat_job_id"],
        "cvat_frame": tile["cvat_frame"], "prelabel_path": tile["prelabel_path"],
        "counts": counts, "deleted_prelabel_ids": deleted, "instances": instances,
    }, indent=1))
    ledger.update(source, image_id, status="labeled", boxes_path=str(boxes_path), error=None)
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
