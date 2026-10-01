"""push_cvat: create CVAT tasks for the round's prepared images.

A task is named <project.name>_<project.subname>, as AgIR-CVToolkit's
cvat_upload named them, and goes into the configured project (created with the
vegetation and exclude labels if it doesn't exist). Each image is one frame, or
several when prepare split it into tiles. `cvat.task_size` and
`cvat.segment_size` count images: a task holds at most task_size images, and
its jobs segment_size images each, i.e. segment_size x tiles-per-image frames.
Images are grouped into tasks by their number of tiles so that every job
boundary falls between images, and each image's tiles are checked to have
landed in one job. The run folder's images/ are uploaded, and right after each
task its masks/ (the pre-labels) as editable masks, so a task is complete as
soon as it appears; that step is tracked per frame, so a crash after the
images are uploaded is recovered by re-running push_cvat. With
`cvat.delete_frames_after_push`, a task's frame files are deleted once it's
set up; an image whose files are gone goes back to `fetched` for prepare.

Box rounds (`label.prelabel.kind=sam3_boxes`) go into their own project,
`cvat.detection.project_name`, whose labels are rectangles (one per detection
class, `cvat.detection.labels`) plus the exclude tag; each image's boxes/ JSON
is uploaded as one rectangle per plant.
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import cv2
from omegaconf import DictConfig

from src.labeling.cvat_boxes import class_names, is_box_round, rectangle_shape
from src.labeling.cvat_client import CvatClient, CvatError
from src.labeling.cvat_masks import component_shapes, mask_shape
from src.labeling.ledger import LabelLedger

log = logging.getLogger(__name__)

# Keep each annotation upload request comfortably small.
_MAX_ANNOTATION_POINTS = 2_000_000


def project_labels(ccfg, boxes: bool = False) -> list[dict]:
    exclude = {"name": ccfg.exclude_label.name, "color": ccfg.exclude_label.color, "type": "tag", "attributes": []}
    if boxes:
        return [{"name": label.name, "color": label.color, "type": "rectangle", "attributes": []}
                for label in ccfg.detection.labels] + [exclude]
    return [
        {"name": ccfg.vegetation_label.name, "color": ccfg.vegetation_label.color, "type": "any", "attributes": []},
        exclude,
    ]


def required_labels(ccfg, boxes: bool = False) -> list[str]:
    shapes = list(class_names(ccfg).values()) if boxes else [ccfg.vegetation_label.name]
    return shapes + [ccfg.exclude_label.name]


def ensure_project(client: CvatClient, ccfg, boxes: bool = False) -> tuple[dict, dict[str, int]]:
    """The configured project (cvat.detection.project_name for box rounds) and
    its {label name: id}, creating the project if needed."""
    name = ccfg.detection.project_name if boxes else ccfg.project_name
    project = client.find_project(name)
    if project is None:
        project = client.create_project(name, project_labels(ccfg, boxes))
        log.info("Created CVAT project %s (id %s)", name, project["id"])
    label_ids = client.label_ids(project["id"])
    for label in required_labels(ccfg, boxes):
        if label not in label_ids:
            raise ValueError(f"CVAT project {name!r} has no {label!r} label; add it in CVAT")
    return project, label_ids


def main(cfg: DictConfig) -> dict | None:
    lcfg, ccfg = cfg.label, cfg.cvat
    source = lcfg.source

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items_for_task(lcfg.round, "prepared", "push_cvat", lcfg.retry_errors)
        items = _limit(items, lcfg.get("push_cvat") or {})
        if not items and not ledger.pending_prelabel_tiles(lcfg.round):
            log.info("Nothing to push to CVAT for round %s.", lcfg.round)
            return None

        boxes = is_box_round(lcfg)
        client = CvatClient.from_config(ccfg)
        project, label_ids = ensure_project(client, ccfg, boxes)

        summary = {"project_id": project["id"], "images": 0}
        task_name = f"{cfg.project.name}_{cfg.project.subname}"
        by_n_tiles = defaultdict(list)
        for item in items:
            tiles = ledger.tiles(source, item["image_id"])
            if not tiles:
                ledger.update(source, item["image_id"], status="error", error="push_cvat: no frames; re-run prepare")
                continue
            missing = [t["cvat_image_path"] for t in tiles if not Path(t["cvat_image_path"]).exists()]
            if missing:
                # Frame files are deleted after a push (cvat.delete_frames_after_push); prepare remakes them.
                log.warning("%s: %d frame file(s) missing (e.g. %s); back to fetched, re-run to prepare it again",
                            item["image_id"], len(missing), missing[0])
                ledger.update(source, item["image_id"], status="fetched", error=None)
                continue
            by_n_tiles[len(tiles)].append((item, tiles))
        _log_plan(by_n_tiles, ccfg)
        # Pre-labels left over from an earlier, interrupted push go first.
        prelabels = _upload_prelabels(client, ledger, source, ledger.pending_prelabel_tiles(lcfg.round),
                                      label_ids, ccfg, boxes)
        tasks = []
        for n_tiles, group in sorted(by_n_tiles.items()):
            task_size = int(ccfg.task_size or len(group))
            for start in range(0, len(group), task_size):
                task = _create_task(client, ccfg, ledger, lcfg.round, task_name, source, project,
                                    group[start:start + task_size], n_tiles)
                # Each task gets its pre-labels right away, so it's ready to annotate as soon as it appears.
                pending = [t for t in ledger.pending_prelabel_tiles(lcfg.round) if t["cvat_task_id"] == task["task_id"]]
                task["prelabels_uploaded"] = _upload_prelabels(client, ledger, source, pending, label_ids, ccfg, boxes)
                prelabels += task["prelabels_uploaded"]
                tasks.append(task)
        if len(tasks) == 1:
            summary.update(tasks[0])
        elif tasks:
            summary.update(images=sum(t["images"] for t in tasks), tasks=tasks)
        summary["prelabels_uploaded"] = prelabels
    return summary


def _limit(items: list[dict], pcfg) -> list[dict]:
    """label.push_cvat.image_ids (only these) and max_images (at most this many,
    by image id) hold back the rest of the prepared images for a later push."""
    image_ids = set(pcfg.get("image_ids") or [])
    if image_ids:
        items = [i for i in items if i["image_id"] in image_ids]
    max_images = pcfg.get("max_images")
    if max_images is not None and len(items) > int(max_images):
        log.info("Pushing %d of %d prepared images (label.push_cvat.max_images)", int(max_images), len(items))
        items = sorted(items, key=lambda i: i["image_id"])[:int(max_images)]
    return items


def _log_plan(by_n_tiles: dict, ccfg) -> None:
    """Before anything is created: how many tasks, jobs and frames the push will make."""
    images_per_job = int(ccfg.segment_size)
    total = {"images": 0, "frames": 0, "tasks": 0, "jobs": 0}
    lines = []
    for n_tiles, group in sorted(by_n_tiles.items()):
        task_size = int(ccfg.task_size or len(group))
        sizes = [min(task_size, len(group) - start) for start in range(0, len(group), task_size)]
        jobs = sum(math.ceil(n / images_per_job) for n in sizes)
        lines.append(f"  {n_tiles} frame(s) per image: {len(group)} images -> {len(sizes)} tasks "
                     f"({', '.join(map(str, sizes))} images), {jobs} jobs of up to {images_per_job} images "
                     f"({images_per_job * n_tiles} frames)")
        for key, value in (("images", len(group)), ("frames", len(group) * n_tiles),
                           ("tasks", len(sizes)), ("jobs", jobs)):
            total[key] += value
    if total["images"]:
        log.info("Push plan: %d images (%d frames) -> %d tasks, %d jobs\n%s", total["images"], total["frames"],
                 total["tasks"], total["jobs"], "\n".join(lines))


def _create_task(client, ccfg, ledger, round_name, task_name, source, project, group, n_tiles) -> dict:
    """One task for `group`, [(item, tiles), ...] with n_tiles tiles each."""
    previous_tasks = {i["cvat_task_id"] for i in ledger.items(round_name=round_name) if i["cvat_task_id"]}
    name = task_name if not previous_tasks else f"{task_name}-{len(previous_tasks) + 1}"
    images_per_job = int(ccfg.segment_size)
    task = client.create_task(name, project["id"], images_per_job * n_tiles)
    task_id = task["id"]
    log.info("Created CVAT task %s (id %s) for %d images (%d frames)", name, task_id, len(group), len(group) * n_tiles)

    try:
        client.upload_images(
            task_id, [Path(t["cvat_image_path"]) for _, tiles in group for t in tiles],
            image_quality=int(ccfg.image_quality), max_request_bytes=int(ccfg.upload_batch_mb * 2**20),
        )
        meta = client.task_frames(task_id)
        start = meta.get("start_frame", 0)
        frame_of = {Path(f["name"]).stem: start + idx for idx, f in enumerate(meta["frames"])}
        jobs = client.task_jobs(task_id)
        job_of = lambda frame: next((j["id"] for j in jobs if j["start_frame"] <= frame <= j["stop_frame"]), None)

        placed = []
        for item, tiles in group:
            frames = [frame_of.get(t["name"]) for t in tiles]
            if None in frames:
                raise CvatError(f"{item['image_id']}: not all of its frames are in task {task_id}")
            job_ids = {job_of(f) for f in frames}
            if len(job_ids) != 1:
                raise CvatError(f"{item['image_id']}: its {n_tiles} tiles landed in jobs {sorted(job_ids)}")
            placed.append((item, tiles, frames, job_ids.pop()))
    except Exception:
        # The images stay `prepared`, so the next run makes a new task; don't leave this one behind.
        log.error("Setting up task %s failed; deleting it", task_id)
        try:
            client.delete_task(task_id)
        except Exception:
            log.exception("Could not delete CVAT task %s; delete it by hand", task_id)
        raise

    for item, tiles, frames, job_id in placed:
        for tile, frame in zip(tiles, frames):
            ledger.update_tile(source, item["image_id"], tile["name"], cvat_task_id=task_id,
                               cvat_frame=frame, cvat_job_id=job_id, cvat_prelabel_uploaded=0)
        ledger.update(source, item["image_id"], status="in_cvat", cvat_task_id=task_id,
                      cvat_frame=min(frames), cvat_job_id=job_id, cvat_prelabel_uploaded=0, error=None)
    log.info("Task %s: %d jobs of up to %d images (%d frames)", task_id, len(jobs), images_per_job,
             images_per_job * n_tiles)

    if ccfg.delete_frames_after_push:
        # CVAT has them now; pull_cvat and training don't read them.
        for _, tiles, _, _ in placed:
            for tile in tiles:
                Path(tile["cvat_image_path"]).unlink(missing_ok=True)

    if ccfg.assignees:
        _assign_jobs(client, jobs, list(ccfg.assignees))
    return {"task_id": task_id, "task_name": name, "task_url": f"{client.base}/tasks/{task_id}",
            "images": len(group), "frames": len(group) * n_tiles, "jobs": len(jobs)}


def _assign_jobs(client, jobs, usernames) -> None:
    user_ids = []
    for username in usernames:
        user_id = client.find_user_id(username)
        if user_id is None:
            log.warning("CVAT user %s not found; not assigning jobs to them", username)
        else:
            user_ids.append((username, user_id))
    for n, job in enumerate(jobs):
        if user_ids:
            username, user_id = user_ids[n % len(user_ids)]
            client.assign_job(job["id"], user_id)
            log.info("Assigned job %s to %s", job["id"], username)


def _upload_prelabels(client, ledger, source, tiles, label_ids, ccfg, boxes=False) -> int:
    """Upload the pre-labels of `tiles` (ledger tiles rows, sorted by task):
    vegetation masks, or with `boxes` one rectangle per box in its JSON."""
    label_id = label_ids[ccfg.vegetation_label.name] if not boxes else None
    split = ccfg.prelabel_shapes == "components"
    if ccfg.prelabel_shapes not in ("components", "single"):
        raise ValueError(f"cvat.prelabel_shapes must be components or single, not {ccfg.prelabel_shapes!r}")
    min_area_px = int(ccfg.prelabel_min_area_px or 0)
    by_task = defaultdict(list)
    for tile in tiles:
        by_task[tile["cvat_task_id"]].append(tile)

    uploaded = 0
    for task_id, task_tiles in by_task.items():
        meta = client.task_frames(task_id)
        start = meta.get("start_frame", 0)
        shapes, batch, n_points = [], [], 0
        for tile in task_tiles:
            frame_meta = meta["frames"][tile["cvat_frame"] - start]
            mask = None if boxes else cv2.imread(tile["prelabel_path"], cv2.IMREAD_GRAYSCALE)
            if boxes:
                new = _box_shapes(tile, label_ids)
                shapes.extend(new)
                n_points += sum(len(s["points"]) for s in new)
            elif mask is None:
                log.warning("Missing pre-label %s; %s starts empty in CVAT", tile["prelabel_path"], tile["name"])
            else:
                size = (frame_meta["width"], frame_meta["height"])
                if (mask.shape[1], mask.shape[0]) != size:
                    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                if split:
                    new = component_shapes(mask > 0, tile["cvat_frame"], label_id, min_area_px)
                else:
                    new = [s for s in [mask_shape(mask > 0, tile["cvat_frame"], label_id)] if s is not None]
                shapes.extend(new)
                n_points += sum(len(s["points"]) for s in new)
            batch.append(tile)
            if n_points >= _MAX_ANNOTATION_POINTS:
                uploaded += _send(client, ledger, source, task_id, shapes, batch)
                shapes, batch, n_points = [], [], 0
        if batch:
            uploaded += _send(client, ledger, source, task_id, shapes, batch)
    return uploaded


def _box_shapes(tile, label_ids) -> list[dict]:
    """CVAT rectangles of a tile's boxes/ JSON (each box's bbox_cvat)."""
    path = Path(tile["prelabel_path"])
    if not path.exists():
        log.warning("Missing pre-label %s; %s starts empty in CVAT", path, tile["name"])
        return []
    shapes = []
    for box in json.loads(path.read_text())["instances"]:
        if box["label"] not in label_ids:
            raise ValueError(f"{path.name}: label {box['label']!r} is not in the CVAT project")
        shapes.append(rectangle_shape(box["bbox_cvat"], tile["cvat_frame"], label_ids[box["label"]]))
    return shapes


def _send(client, ledger, source, task_id, shapes, tiles) -> int:
    if shapes:
        client.add_task_annotations(task_id, shapes)
    for tile in tiles:
        ledger.update_tile(source, tile["image_id"], tile["name"], cvat_prelabel_uploaded=1)
    for image_id in {t["image_id"] for t in tiles}:
        ledger.update(source, image_id, cvat_prelabel_uploaded=1)
    log.info("Uploaded %d pre-label shapes to task %s (%.1f MB of points)",
             len(shapes), task_id, len(json.dumps(shapes)) / 1e6 if shapes else 0.0)
    return len(shapes)
