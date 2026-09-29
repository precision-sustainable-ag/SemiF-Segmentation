"""push_cvat: create a CVAT task for the round's prepared images.

The task goes into the configured project (created with the vegetation and
exclude labels if it doesn't exist) and is split into jobs of
`cvat.segment_size` images. Pre-labels are uploaded as editable masks; that
step is tracked per image, so a crash after the images are uploaded is
recovered by simply re-running push_cvat.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
from omegaconf import DictConfig

from src.labeling.cvat_client import CvatClient
from src.labeling.cvat_masks import mask_shape
from src.labeling.ledger import LabelLedger

log = logging.getLogger(__name__)

# Keep each annotation upload request comfortably small.
_MAX_ANNOTATION_POINTS = 2_000_000


def project_labels(ccfg) -> list[dict]:
    return [
        {"name": ccfg.vegetation_label.name, "color": ccfg.vegetation_label.color, "type": "any", "attributes": []},
        {"name": ccfg.exclude_label.name, "color": ccfg.exclude_label.color, "type": "tag", "attributes": []},
    ]


def ensure_project(client: CvatClient, ccfg) -> tuple[dict, dict[str, int]]:
    """The configured project and its {label name: id}, creating the project if needed."""
    project = client.find_project(ccfg.project_name)
    if project is None:
        project = client.create_project(ccfg.project_name, project_labels(ccfg))
        log.info("Created CVAT project %s (id %s)", ccfg.project_name, project["id"])
    label_ids = client.label_ids(project["id"])
    for label in (ccfg.vegetation_label.name, ccfg.exclude_label.name):
        if label not in label_ids:
            raise ValueError(f"CVAT project {ccfg.project_name!r} has no {label!r} label; add it in CVAT")
    return project, label_ids


def main(cfg: DictConfig) -> None:
    lcfg, ccfg = cfg.label, cfg.cvat
    source = lcfg.source

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items_for_task(lcfg.round, "prepared", "push_cvat", lcfg.retry_errors)
        pending_prelabels = [
            i for i in ledger.items(round_name=lcfg.round, status="in_cvat")
            if i["prelabel_path"] and not i["cvat_prelabel_uploaded"]
        ]
        if not items and not pending_prelabels:
            log.info("Nothing to push to CVAT for round %s.", lcfg.round)
            return

        client = CvatClient.from_config(ccfg)
        project, label_ids = ensure_project(client, ccfg)

        if items:
            _create_task(client, ccfg, ledger, lcfg.round, source, project, items)

        pending = [
            i for i in ledger.items(round_name=lcfg.round, status="in_cvat")
            if i["prelabel_path"] and not i["cvat_prelabel_uploaded"]
        ]
        if pending:
            _upload_prelabels(client, ledger, source, pending, label_ids[ccfg.vegetation_label.name])


def _create_task(client, ccfg, ledger, round_name, source, project, items) -> None:
    previous_tasks = {i["cvat_task_id"] for i in ledger.items(round_name=round_name) if i["cvat_task_id"]}
    name = round_name if not previous_tasks else f"{round_name}-{len(previous_tasks) + 1}"
    task = client.create_task(name, project["id"], int(ccfg.segment_size))
    task_id = task["id"]
    log.info("Created CVAT task %s (id %s) for %d images", name, task_id, len(items))

    client.upload_images(
        task_id, [Path(i["cvat_image_path"]) for i in items],
        image_quality=int(ccfg.image_quality), max_request_bytes=int(ccfg.upload_batch_mb * 2**20),
    )

    meta = client.task_frames(task_id)
    start = meta.get("start_frame", 0)
    frame_of = {Path(f["name"]).stem: start + idx for idx, f in enumerate(meta["frames"])}
    jobs = client.task_jobs(task_id)

    for item in items:
        frame = frame_of.get(item["image_id"])
        if frame is None:
            ledger.update(source, item["image_id"], status="error",
                          error=f"push_cvat: not among task {task_id}'s frames")
            continue
        job_id = next((j["id"] for j in jobs if j["start_frame"] <= frame <= j["stop_frame"]), None)
        ledger.update(source, item["image_id"], status="in_cvat", cvat_task_id=task_id,
                      cvat_frame=frame, cvat_job_id=job_id, cvat_prelabel_uploaded=0, error=None)
    log.info("Task %s: %d jobs of up to %d images", task_id, len(jobs), ccfg.segment_size)

    if ccfg.assignees:
        _assign_jobs(client, jobs, list(ccfg.assignees))


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


def _upload_prelabels(client, ledger, source, items, label_id) -> None:
    by_task = defaultdict(list)
    for item in items:
        by_task[item["cvat_task_id"]].append(item)

    for task_id, task_items in by_task.items():
        meta = client.task_frames(task_id)
        start = meta.get("start_frame", 0)
        shapes, batch_items, n_points = [], [], 0
        for item in task_items:
            frame_meta = meta["frames"][item["cvat_frame"] - start]
            mask = cv2.imread(item["prelabel_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                log.warning("Missing pre-label %s; %s starts empty in CVAT", item["prelabel_path"], item["image_id"])
            else:
                size = (frame_meta["width"], frame_meta["height"])
                if (mask.shape[1], mask.shape[0]) != size:
                    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                shape = mask_shape(mask > 0, item["cvat_frame"], label_id)
                if shape is not None:
                    shapes.append(shape)
                    n_points += len(shape["points"])
            batch_items.append(item)
            if n_points >= _MAX_ANNOTATION_POINTS:
                _send(client, ledger, source, task_id, shapes, batch_items)
                shapes, batch_items, n_points = [], [], 0
        if batch_items:
            _send(client, ledger, source, task_id, shapes, batch_items)


def _send(client, ledger, source, task_id, shapes, items) -> None:
    if shapes:
        client.add_task_annotations(task_id, shapes)
    for item in items:
        ledger.update(source, item["image_id"], cvat_prelabel_uploaded=1)
    log.info("Uploaded %d pre-label masks to task %s (%.1f MB of points)",
             len(shapes), task_id, len(json.dumps(shapes)) / 1e6 if shapes else 0.0)
