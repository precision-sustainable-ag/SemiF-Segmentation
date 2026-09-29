"""Conversions between binary masks and CVAT annotation shapes.

CVAT stores a "mask" shape's `points` as run lengths over its bounding box
(row-major, starting with a possibly-empty background run), followed by the
inclusive box [left, top, right, bottom] -- the same encoding as
cvat_sdk.masks.encode_mask.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def encode_mask(bitmap: np.ndarray) -> list[int] | None:
    """CVAT mask `points` for the tight box around `bitmap`; None if it's empty."""
    bitmap = np.asarray(bitmap, dtype=bool)
    ys, xs = np.nonzero(bitmap)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    flat = bitmap[y1:y2, x1:x2].ravel()
    (run_indices,) = np.diff(flat, prepend=[not flat[0]], append=[not flat[-1]]).nonzero()
    run_lengths = np.diff(run_indices, prepend=[0]) if flat[0] else np.diff(run_indices)
    return run_lengths.tolist() + [x1, y1, x2 - 1, y2 - 1]


def decode_mask(points, height: int, width: int) -> np.ndarray:
    """Boolean (height, width) mask from a CVAT mask shape's `points`."""
    if len(points) < 5:
        raise ValueError(f"mask points need run lengths plus a 4-value box, got {len(points)} values")
    left, top, right, bottom = (int(round(v)) for v in points[-4:])
    box_w, box_h = right - left + 1, bottom - top + 1
    runs = np.asarray(points[:-4], dtype=np.int64)
    values = np.zeros(len(runs), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, runs)
    region = np.zeros(box_w * box_h, dtype=bool)
    n = min(flat.size, region.size)
    region[:n] = flat[:n]
    region = region.reshape(box_h, box_w)

    out = np.zeros((height, width), dtype=bool)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(bottom + 1, height), min(right + 1, width)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = region[y0 - top:y1 - top, x0 - left:x1 - left]
    return out


def mask_shape(bitmap: np.ndarray, frame: int, label_id: int) -> dict | None:
    """A CVAT "mask" shape for `bitmap`, or None if it's empty."""
    points = encode_mask(bitmap)
    if points is None:
        return None
    return {
        "type": "mask", "frame": frame, "label_id": label_id, "points": points,
        "occluded": False, "z_order": 0, "group": 0, "attributes": [], "source": "auto",
    }


def rasterize(shapes: list[dict], height: int, width: int) -> tuple[np.ndarray, Counter]:
    """Union of mask, polygon, and rectangle shapes as a boolean mask.

    Returns (mask, counts of shape types that were skipped).
    """
    mask = np.zeros((height, width), dtype=bool)
    canvas = np.zeros((height, width), dtype=np.uint8)
    skipped: Counter = Counter()
    for shape in shapes:
        kind, points = shape["type"], shape["points"]
        if kind == "mask":
            mask |= decode_mask(points, height, width)
        elif kind == "polygon":
            polygon = np.round(np.asarray(points, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(canvas, [polygon], 1)
        elif kind == "rectangle" and not shape.get("rotation"):
            x1, y1, x2, y2 = (int(round(v)) for v in points[:4])
            canvas[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
        else:
            skipped[kind] += 1
    return mask | canvas.astype(bool), skipped
