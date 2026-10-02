"""sam3_finetune task `dataset`: tiled COCO train/valid sets from the project's
labeled images, with either kind of label (sam3_finetune.labels):

  boxes  box rounds (label.prelabel.kind=sam3_boxes): each image's pulled boxes
         (cvat_downloads/<task>/boxes/<image_id>.json, labels.db boxes_path),
         every one, whatever its provenance.
  masks  semantic rounds: each image's pulled vegetation mask (labels.db
         mask_path), one `plant` box per blob: pixels within merge_px of each
         other are one plant (stray leaf fragments join theirs), as is a blob
         whose box lies absorb_contained inside a bigger blob's box; blobs
         under min_area_px are dropped. Plants that touch become one box. Only the
         plant category is trained, since nothing else is labeled there.

Each image is one training input, downscaled so its longest side is max_side
(SAM 3's 1008: the whole field of view, as context). With max_side null,
images are instead seen as tiled pre-labeling sees them: downscaled by
`scale`, then cut into overlapping tile_size tiles; a box cut by a tile edge
is kept, clipped, if at least min_visible of it is inside, otherwise dropped. Tiles without boxes stay, as
negatives (sam3's loader asks each tile for every category).

Category names are the SAM 3 prompts the CVAT labels are trained under
(sam3_finetune.prompts), since SAM 3 detects by text prompt.

Images are split into train/valid once, per image, and the split is kept in
split.json, so adding rounds never moves an image between the sets. Writes,
in <run>/dataset/:
  {train,valid}/<image_id>_y<y>_x<x>.jpg
  {train,valid}/_annotations.coco.json   (sam3.train's Roboflow layout)
  split.json, summary.json
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.inferencing.tiled_instances import tile_starts
from src.labeling.ledger import LabelLedger
from src.preprocessing.full_image_boxes import working_scale

log = logging.getLogger(__name__)

SPLITS = ("train", "valid")


LABEL_KINDS = {"boxes": "boxes_path", "masks": "mask_path"}


def labeled_images(labels_db: str | Path, rounds, kind: str = "boxes") -> list[dict]:
    """{image_id, round, native_path, boxes_path | mask_path} of every labeled
    image with pulled labels of `kind` (boxes | masks)."""
    if kind not in LABEL_KINDS:
        raise ValueError(f"sam3_finetune.labels must be one of {sorted(LABEL_KINDS)}, not {kind!r}")
    column = LABEL_KINDS[kind]
    rounds = set(rounds or [])
    with LabelLedger(labels_db) as ledger:
        items = ledger.items(status="labeled")
    rows = [{k: i[k] for k in ("image_id", "round", "native_path", column)}
            for i in items if i.get(column) and (not rounds or i["round"] in rounds)]
    missing = rounds - {r["round"] for r in rows}
    if missing:
        log.warning("No pulled %s in round(s) %s", kind, sorted(missing))
    return rows


def mask_instances(mask_path: str | Path, size: tuple[int, int], mcfg: dict) -> list[dict]:
    """Plant boxes (xyxy at `size` = (width, height)) of a pulled vegetation mask.
    merge_px and min_area_px are full-resolution pixels."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"can't read {mask_path}")
    scale = size[0] / mask.shape[1]
    # Plant coverage per working-scale pixel; >= 25% keeps thin grass leaves that a majority vote would erase.
    mask = (cv2.resize((mask > 0).astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) >= 64).astype(np.uint8)
    merge = int(round(float(mcfg["merge_px"]) * scale))
    joined = cv2.dilate(mask, np.ones((2 * merge + 1, 2 * merge + 1), np.uint8)) if merge > 0 else mask
    n, comp = cv2.connectedComponents(joined, connectivity=8)
    comp[mask == 0] = 0  # boxes cover the mask's own pixels, not the dilation
    blobs = []  # [x0, y0, x1, y1, pixels]
    for i in range(1, n):
        ys, xs = np.nonzero(comp == i)
        if len(xs):
            blobs.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, len(xs)])
    # A blob whose box lies (mostly) inside a bigger blob's box is a piece of that plant.
    blobs.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    absorb = float(mcfg.get("absorb_contained", 0.9))
    kept = []
    for b in blobs:
        area = (b[2] - b[0]) * (b[3] - b[1])
        host = next((k for k in kept if area and _overlap(b, k) >= absorb * area), None) if absorb > 0 else None
        if host is None:
            kept.append(b)
        else:
            host[:4] = [min(host[0], b[0]), min(host[1], b[1]), max(host[2], b[2]), max(host[3], b[3])]
            host[4] += b[4]
    return [{"bbox": [float(v) for v in b[:4]], "label": mcfg["label"], "source": "mask_blob"}
            for b in sorted(kept, key=lambda b: (b[1], b[0])) if b[4] >= float(mcfg["min_area_px"]) * scale ** 2]


def _overlap(a, b) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def assign_splits(image_ids, split_path: Path, val_fraction: float, seed: int) -> dict[str, str]:
    """image_id -> train | valid. Ids already in split_path keep theirs; new ones
    are shuffled (seed) and topped up so valid holds about val_fraction of all."""
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    new = sorted(set(image_ids) - set(split))
    random.Random(seed).shuffle(new)
    total = len(split) + len(new)
    n_valid = max(0, round(total * val_fraction) - sum(v == "valid" for v in split.values()))
    for k, image_id in enumerate(new):
        split[image_id] = "valid" if k < n_valid else "train"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(dict(sorted(split.items())), indent=1))
    return split


def tile_boxes(boxes: np.ndarray, rect, min_visible: float, min_box_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(kept boxes in tile coordinates, their indices) for xyxy `boxes` at the
    working scale and the tile rect (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    if not len(boxes):
        return np.zeros((0, 4)), np.zeros(0, np.int64)
    clipped = np.stack([boxes[:, 0].clip(x0, x1), boxes[:, 1].clip(y0, y1),
                        boxes[:, 2].clip(x0, x1), boxes[:, 3].clip(y0, y1)], axis=1)
    w, h = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (w >= min_box_px) & (h >= min_box_px) & (w * h >= min_visible * np.maximum(area, 1e-9))
    return clipped[keep] - np.array([x0, y0, x0, y0]), np.nonzero(keep)[0]


def build(images: list[dict], out_dir: str | Path, prompts: dict, dcfg: dict, mcfg: dict | None = None) -> dict:
    """Write the dataset for `images` (labeled_images rows: boxes_path or mask_path,
    with mcfg = sam3_finetune.masks for the latter); returns summary.json's content."""
    out_dir = Path(out_dir)
    from_masks = bool(images) and "mask_path" in images[0]
    if from_masks:
        prompts = {mcfg["label"]: prompts[mcfg["label"]]}  # the only category a vegetation mask labels
    categories = [{"id": k + 1, "name": prompt} for k, prompt in enumerate(dict.fromkeys(prompts.values()))]
    cat_of = {label: next(c["id"] for c in categories if c["name"] == prompt) for label, prompt in prompts.items()}
    split = assign_splits([i["image_id"] for i in images], out_dir / "split.json",
                          float(dcfg["val_fraction"]), int(dcfg["seed"]))
    coco = {s: {"images": [], "annotations": [], "categories": categories} for s in SPLITS}
    stats = {s: Counter() for s in SPLITS}
    whole = bool(dcfg.get("max_side"))
    overlap = int(dcfg["overlap"])
    for s in SPLITS:
        (out_dir / s).mkdir(parents=True, exist_ok=True)
        for old in (out_dir / s).glob("*.jpg"):  # rebuilt from scratch: the boxes may have been re-pulled
            old.unlink()

    for item in images:
        s = split[item["image_id"]]
        image = cv2.imread(item["native_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{item['image_id']}: can't read {item['native_path']} (fetch it again)")
        full_h, full_w = image.shape[:2]
        scale = working_scale((full_h, full_w), dcfg)
        size = (max(1, round(full_w * scale)), max(1, round(full_h * scale)))
        tile = max(size) if whole else int(dcfg["tile_size"])  # whole: one "tile", the image
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size != (full_w, full_h) else image
        sx, sy = size[0] / full_w, size[1] / full_h
        if from_masks:
            instances = mask_instances(item["mask_path"], size, mcfg)  # already at the working scale
            boxes = np.array([b["bbox"] for b in instances], np.float64).reshape(-1, 4)
        else:
            instances = json.loads(Path(item["boxes_path"]).read_text())["instances"]
            unknown = {b["label"] for b in instances} - set(prompts)
            if unknown:
                raise ValueError(f"{item['boxes_path']}: labels {sorted(unknown)} have no sam3_finetune.prompts entry")
            boxes = np.array([b["bbox"] for b in instances], np.float64).reshape(-1, 4) * [sx, sy, sx, sy]
        labels = [b["label"] for b in instances]
        stats[s]["images"] += 1
        stats[s].update(f"source:{b.get('source', 'unknown')}" for b in instances)

        height, width = image.shape[:2]
        for ty in tile_starts(height, tile, overlap):
            for tx in tile_starts(width, tile, overlap):
                rect = (tx, ty, min(tx + tile, width), min(ty + tile, height))
                name = f"{item['image_id']}_y{ty}_x{tx}.jpg"
                crop = image[rect[1]:rect[3], rect[0]:rect[2]]
                if not cv2.imwrite(str(out_dir / s / name), crop, [cv2.IMWRITE_JPEG_QUALITY, int(dcfg["jpeg_quality"])]):
                    raise RuntimeError(f"could not write {out_dir / s / name}")
                image_entry = {"id": len(coco[s]["images"]) + 1, "file_name": name, "width": crop.shape[1],
                               "height": crop.shape[0], "source_image": item["image_id"], "tile_xy": [tx, ty],
                               "scale": [sx, sy]}
                coco[s]["images"].append(image_entry)
                kept, idx = tile_boxes(boxes, rect, float(dcfg["min_visible"]), float(dcfg["min_box_px"]))
                stats[s]["tiles"] += 1
                stats[s]["empty_tiles"] += int(not len(kept))
                for (bx0, by0, bx1, by1), i in zip(kept, idx):
                    w, h = bx1 - bx0, by1 - by0
                    coco[s]["annotations"].append({
                        "id": len(coco[s]["annotations"]) + 1, "image_id": image_entry["id"],
                        "category_id": cat_of[labels[i]], "bbox": [round(float(v), 2) for v in (bx0, by0, w, h)],
                        "area": round(float(w * h), 2), "iscrowd": 0})
                    stats[s][f"boxes:{labels[i]}"] += 1

    for s in SPLITS:
        (out_dir / s / "_annotations.coco.json").write_text(json.dumps(coco[s]))
    summary = {"categories": categories, "prompts": dict(prompts), "settings": dict(dcfg),
               **{s: dict(sorted(stats[s].items())) for s in SPLITS}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("SAM 3 box dataset in %s: %s", out_dir, {s: dict(stats[s]) for s in SPLITS})
    if not coco["train"]["annotations"]:
        raise ValueError("The training set has no boxes; pull some finished jobs first")
    return summary
