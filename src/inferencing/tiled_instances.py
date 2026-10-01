"""Full-image instance segmentation by overlapping tiles.

Images (~9500 x 13400 px) are far larger than the tiles the detectors train
on, and running a detector on the whole image at training scale neither
fits torchvision's detection limits (100 detections, 512 px anchors) nor
memory for training. So the image is cut into overlapping tiles, each tile is
predicted, and the detections are stitched back into full-image instances:

- every tile's detections are moved to full-image coordinates;
- detections from two tiles whose masks agree inside the region both tiles
  saw (IoU >= merge_iou there) are one instance and are merged -- this
  removes the duplicates of plants inside an overlap and joins the pieces of
  plants cut by a tile edge (chains of merges join plants spanning several
  tiles). Only confident detections (score >= merge_min_score) merge: chains
  through low-score fragments would otherwise glue a whole field into one
  high-scoring instance.

Instance masks are kept as crops of their box (a full-image mask is ~127 MB).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou


@dataclass
class Instances:
    """Full-image instances; masks[i] is a bool crop covering boxes[i]."""
    height: int
    width: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.int64))  # x0, y0, x1, y1 (exclusive)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    masks: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.masks)

    def areas(self) -> np.ndarray:
        return np.array([int(m.sum()) for m in self.masks], np.int64)


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Tile origins covering [0, length) with at least `overlap` px shared by
    neighbours; the last tile ends at the edge."""
    if length <= tile:
        return [0]
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


def _crop_mask(mask: torch.Tensor):
    """(y0, x0, crop) of a (H, W) bool tensor's tight box, or None if empty."""
    rows, cols = mask.any(dim=1), mask.any(dim=0)
    if not rows.any():
        return None
    y = torch.where(rows)[0]
    x = torch.where(cols)[0]
    y0, y1, x0, x1 = int(y[0]), int(y[-1]) + 1, int(x[0]), int(x[-1]) + 1
    return y0, x0, mask[y0:y1, x0:x1].cpu().numpy()


def _window(inst: Instances, i: int, region) -> np.ndarray:
    """Instance i's mask inside region (x0, y0, x1, y1), as a region-sized bool array."""
    rx0, ry0, rx1, ry1 = region
    out = np.zeros((ry1 - ry0, rx1 - rx0), bool)
    x0, y0, x1, y1 = inst.boxes[i]
    ix0, iy0, ix1, iy1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
    if ix0 < ix1 and iy0 < iy1:
        out[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0] = inst.masks[i][iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    return out


def _intersect(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def merge_across_tiles(inst: Instances, tile_ids: np.ndarray, tile_rects: list, merge_iou: float = 0.5,
                       min_pixels: int = 16, min_score: float = 0.5) -> Instances:
    """Merge detections of different tiles that are the same instance: both
    scored >= min_score, same label, and masks with IoU >= merge_iou inside
    the two tiles' shared region (where both tiles saw the same pixels).
    Merged: union mask, highest score and its label. Detections below
    min_score are kept as they are."""
    n = len(inst)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        boxes = torch.as_tensor(inst.boxes, dtype=torch.float32)
        touching = box_iou(boxes, boxes) > 0
        pairs = torch.nonzero(torch.triu(touching, diagonal=1)).tolist()
        for i, j in pairs:
            if tile_ids[i] == tile_ids[j] or inst.labels[i] != inst.labels[j]:
                continue
            if min(inst.scores[i], inst.scores[j]) < min_score:
                continue
            shared = _intersect(tile_rects[tile_ids[i]], tile_rects[tile_ids[j]])
            if shared is None:
                continue
            union_box = (*np.minimum(inst.boxes[i][:2], inst.boxes[j][:2]), *np.maximum(inst.boxes[i][2:], inst.boxes[j][2:]))
            region = _intersect(shared, union_box)
            if region is None:
                continue
            a, b = _window(inst, i, region), _window(inst, j, region)
            if min(a.sum(), b.sum()) < min_pixels:
                continue
            if (a & b).sum() / (a | b).sum() >= merge_iou:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = Instances(inst.height, inst.width)
    boxes, labels, scores = [], [], []
    for members in groups.values():
        best = max(members, key=lambda k: inst.scores[k])
        x0, y0 = inst.boxes[members, 0].min(), inst.boxes[members, 1].min()
        x1, y1 = inst.boxes[members, 2].max(), inst.boxes[members, 3].max()
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        for k in members:
            bx0, by0, bx1, by1 = inst.boxes[k]
            mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] |= inst.masks[k]
        boxes.append((x0, y0, x1, y1))
        labels.append(inst.labels[best])
        scores.append(inst.scores[best])
        out.masks.append(mask)
    if out.masks:
        out.boxes, out.labels, out.scores = np.array(boxes, np.int64), np.array(labels, np.int64), np.array(scores, np.float32)
    return out


@torch.no_grad()
def predict_tiled_instances(model: torch.nn.Module, image: torch.Tensor, tile_size: int = 2048, overlap: int = 512,
                            merge_iou: float = 0.5, merge_min_score: float = 0.5, mask_threshold: float = 0.5,
                            batch_size: int = 4) -> Instances:
    """Full-image instances from a torchvision-style detector (eval mode) run on
    overlapping tiles of `image` (C, H, W, preprocessed as for training)."""
    device = next(model.parameters()).device
    _, H, W = image.shape
    rects = [(x, y, min(x + tile_size, W), min(y + tile_size, H))
             for y in tile_starts(H, tile_size, overlap) for x in tile_starts(W, tile_size, overlap)]
    raw = Instances(H, W)
    boxes, labels, scores, tile_ids = [], [], [], []
    for start in range(0, len(rects), batch_size):
        batch = rects[start:start + batch_size]
        crops = [image[:, y0:y1, x0:x1].to(device) for x0, y0, x1, y1 in batch]
        for t, (rect, out) in enumerate(zip(batch, model(crops))):
            for k in range(len(out["scores"])):
                crop = _crop_mask(out["masks"][k, 0] > mask_threshold)
                if crop is None:
                    continue
                y0, x0, m = crop
                gx0, gy0 = rect[0] + x0, rect[1] + y0
                boxes.append((gx0, gy0, gx0 + m.shape[1], gy0 + m.shape[0]))
                labels.append(int(out["labels"][k]))
                scores.append(float(out["scores"][k]))
                tile_ids.append(start + t)
                raw.masks.append(m)
    if not raw.masks:
        return raw
    raw.boxes, raw.labels, raw.scores = np.array(boxes, np.int64), np.array(labels, np.int64), np.array(scores, np.float32)
    return merge_across_tiles(raw, np.array(tile_ids), rects, merge_iou, min_score=merge_min_score)


def instances_from_mask(mask: np.ndarray, min_area: int = 0) -> Instances:
    """Ground-truth instances: one per 8-connected blob of each class value > 0
    (as src.utils.datasets.masks_to_instances, on the full image)."""
    H, W = mask.shape
    inst = Instances(H, W)
    boxes, labels = [], []
    for value in np.unique(mask):
        if value == 0:
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats((mask == value).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            boxes.append((x, y, x + w, y + h))
            labels.append(int(value))
            inst.masks.append(comp[y:y + h, x:x + w] == i)
    if inst.masks:
        inst.boxes, inst.labels = np.array(boxes, np.int64), np.array(labels, np.int64)
        inst.scores = np.ones(len(boxes), np.float32)
    return inst


def paint_semantic(inst: Instances, score_threshold: float = 0.5) -> np.ndarray:
    """(H, W) class indices: instances with score >= score_threshold, higher scores on top."""
    semantic = np.zeros((inst.height, inst.width), np.uint8)
    for i in np.argsort(inst.scores):
        if inst.scores[i] < score_threshold:
            continue
        x0, y0, x1, y1 = inst.boxes[i]
        semantic[y0:y1, x0:x1][inst.masks[i]] = inst.labels[i]
    return semantic


def crop_to_rle(crop: np.ndarray, box, height: int, width: int) -> dict:
    """COCO RLE of a full-image mask given as a crop, without building the full mask.
    COCO RLE runs over the image in column-major order, alternating background/foreground."""
    x0, y0 = int(box[0]), int(box[1])
    cols, rows = np.nonzero(crop.T)  # column-major order
    pos = (x0 + cols).astype(np.int64) * height + (y0 + rows)
    if pos.size == 0:
        return mask_utils.frPyObjects({"size": [height, width], "counts": [height * width]}, height, width)
    breaks = np.where(np.diff(pos) != 1)[0]
    run_starts = np.concatenate([[pos[0]], pos[breaks + 1]])
    run_ends = np.concatenate([pos[breaks] + 1, [pos[-1] + 1]])
    counts = np.empty(2 * len(run_starts) + 1, np.int64)
    counts[0] = run_starts[0]
    counts[1:-1:2] = run_ends - run_starts
    counts[2:-1:2] = run_starts[1:] - run_ends[:-1]
    counts[-1] = height * width - run_ends[-1]
    return mask_utils.frPyObjects({"size": [height, width], "counts": counts.tolist()}, height, width)


def coco_ap(gts: list[Instances], dts: list[Instances], max_dets: int = 1000) -> dict[str, float]:
    """COCO box and mask AP / AP50 / AP75 and AR over images (gts[i] <-> dts[i]).
    max_dets replaces COCO's 100 per image: full images hold hundreds of plants."""
    images, gt_anns, dt_anns = [], [], []
    for img_id, (gt, dt) in enumerate(zip(gts, dts)):
        images.append({"id": img_id, "height": gt.height, "width": gt.width})
        for i, area in enumerate(gt.areas()):
            x0, y0, x1, y1 = (int(v) for v in gt.boxes[i])
            gt_anns.append({"id": len(gt_anns) + 1, "image_id": img_id, "category_id": int(gt.labels[i]),
                            "bbox": [x0, y0, x1 - x0, y1 - y0], "area": int(area), "iscrowd": 0,
                            "segmentation": crop_to_rle(gt.masks[i], gt.boxes[i], gt.height, gt.width)})
        for i in range(len(dt)):
            x0, y0, x1, y1 = (int(v) for v in dt.boxes[i])
            dt_anns.append({"image_id": img_id, "category_id": int(dt.labels[i]), "score": float(dt.scores[i]),
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "segmentation": crop_to_rle(dt.masks[i], dt.boxes[i], dt.height, dt.width)})
    categories = sorted({a["category_id"] for a in gt_anns} | {a["category_id"] for a in dt_anns}) or [1]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gt_anns, "categories": [{"id": c} for c in categories]}
    coco_gt.createIndex()
    results = {}
    for iou_type in ("bbox", "segm"):
        if not dt_anns:
            results.update({f"{iou_type}_{k}": 0.0 for k in ("map", "map_50", "map_75", "mar")})
            continue
        anns = [dict(a) for a in dt_anns]
        if iou_type == "bbox":
            for a in anns:
                a.pop("segmentation")
        coco_dt = coco_gt.loadRes(anns)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.maxDets = [1, 100, max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        results.update({f"{iou_type}_map": ev.stats[0], f"{iou_type}_map_50": ev.stats[1],
                        f"{iou_type}_map_75": ev.stats[2], f"{iou_type}_mar": ev.stats[8]})
    return {k: float(v) for k, v in results.items()}
