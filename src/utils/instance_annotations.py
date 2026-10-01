"""Instance annotations stored next to the semantic tiles, with provenance.

One annotation per tile, in paths.split_dir/<split>/<dirname>/:
  <stem>.png   uint16 instance-id map: 0 = no instance, k = instance k
  <stem>.json  {"image_size": [H, W], "instances": [{"id": k, "label": class,
               "score": float|null, "annotation_source": ..., ...}, ...], ...}

An id map holds non-overlapping instances only, which is what a plant
annotation is (a pixel belongs to one plant), and it opens in any image
viewer or CVAT for correction. Whoever corrects an instance sets its
annotation_source to "sam3_corrected", so training data can be split by
where its labels came from.

Nothing here depends on how the instances were made (SAM 3
pseudo-labels, human annotation, ...): src.utils.datasets.InstanceDataset reads these files.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch


class AnnotationSource(str, Enum):
    """Where an instance annotation came from."""
    HUMAN = "human"                    # drawn by an annotator
    SAM3 = "sam3"                      # SAM 3 pseudo-instance, unreviewed
    SAM3_CORRECTED = "sam3_corrected"  # SAM 3 pseudo-instance edited by an annotator
    CONNECTED_COMPONENT = "connected_component"  # a blob of the semantic mask (src.utils.datasets.masks_to_instances)


def mask_box(mask: np.ndarray | torch.Tensor) -> tuple[int, int, int, int] | None:
    """Pixel-edge box (x0, y0, x1, y1) of a (H, W) mask -- columns x0 .. x1 - 1,
    as src.utils.datasets.masks_to_instances -- or None if the mask is empty."""
    if isinstance(mask, torch.Tensor):
        rows, cols = mask.any(dim=1), mask.any(dim=0)
        if not bool(rows.any()):
            return None
        y, x = torch.where(rows)[0], torch.where(cols)[0]
        return int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    if not rows.any():
        return None
    y, x = np.where(rows)[0], np.where(cols)[0]
    return int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1


def empty_target(height: int, width: int) -> dict:
    """A torchvision detection target with no instances."""
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "masks": torch.zeros((0, height, width), dtype=torch.uint8),
        "scores": torch.zeros(0, dtype=torch.float32),
        "annotation_source": [],
    }


def save_instance_annotation(out_dir: Path, stem: str, target: dict, extra: dict | None = None) -> None:
    """Write target ({"masks" (N, H, W), "labels", optional "scores",
    "annotation_source", "prompts"}) as <stem>.png + <stem>.json in out_dir.
    Masks must not overlap."""
    masks = torch.as_tensor(target["masks"]).bool()
    n, height, width = masks.shape
    if n >= 2**16:
        raise ValueError(f"{n} instances do not fit a uint16 id map")
    if n and int(masks.sum(0).max()) > 1:
        raise ValueError("instance masks overlap; an id map needs exclusive masks (pseudo_labels.exclusive_masks)")
    id_map = np.zeros((height, width), np.uint16)
    for k in range(n):
        id_map[masks[k].cpu().numpy()] = k + 1

    scores = target.get("scores")
    sources = target.get("annotation_source") or [AnnotationSource.HUMAN.value] * n
    prompts = target.get("prompts") or [None] * n
    instances = []
    for k in range(n):
        instances.append({
            "id": k + 1,
            "label": int(target["labels"][k]),
            "score": None if scores is None else round(float(scores[k]), 4),
            "annotation_source": str(AnnotationSource(sources[k]).value),
            "prompt": prompts[k],
            "bbox": [int(v) for v in target["boxes"][k].tolist()],
        })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_dir / f"{stem}.png"), id_map):
        raise OSError(f"could not write {out_dir / f'{stem}.png'}")
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"image_size": [height, width], "instances": instances, **(extra or {})}, f, indent=1)


def load_instance_annotation(ann_dir: Path, stem: str) -> tuple[np.ndarray, dict]:
    """(id map (H, W) uint16, metadata) written by save_instance_annotation."""
    id_map = cv2.imread(str(Path(ann_dir) / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if id_map is None:
        raise FileNotFoundError(Path(ann_dir) / f"{stem}.png")
    with open(Path(ann_dir) / f"{stem}.json") as f:
        meta = json.load(f)
    return id_map, meta


def id_map_to_instances(id_map: np.ndarray, meta: dict, min_area: int = 0,
                        max_instances: int | None = None) -> dict:
    """Instance targets (torchvision detection format, as
    src.utils.datasets.masks_to_instances) from an id map, e.g. after
    augmentation: boxes are recomputed from what is left of each instance.

    Returns {"boxes" (N, 4) float xyxy pixel edges, "labels" (N,) int64,
    "masks" (N, H, W) uint8, "scores" (N,) float (1 where unknown),
    "annotation_source" list of N str}, largest first.
    """
    height, width = id_map.shape
    by_id = {int(inst["id"]): inst for inst in meta.get("instances", [])}
    ids, counts = np.unique(id_map, return_counts=True)
    found = sorted(((int(c), int(i)) for i, c in zip(ids, counts) if i != 0 and c >= min_area and int(i) in by_id),
                   reverse=True)
    if max_instances is not None:
        found = found[:max_instances]
    if not found:
        return empty_target(height, width)

    masks = np.zeros((len(found), height, width), np.uint8)
    boxes = np.zeros((len(found), 4), np.float32)
    for k, (_, i) in enumerate(found):
        masks[k] = id_map == i
        boxes[k] = mask_box(masks[k])
    insts = [by_id[i] for _, i in found]
    return {
        "boxes": torch.from_numpy(boxes),
        "labels": torch.tensor([int(inst["label"]) for inst in insts], dtype=torch.int64),
        "masks": torch.from_numpy(masks),
        "scores": torch.tensor([1.0 if inst.get("score") is None else float(inst["score"]) for inst in insts],
                               dtype=torch.float32),
        "annotation_source": [inst.get("annotation_source", AnnotationSource.HUMAN.value) for inst in insts],
    }
