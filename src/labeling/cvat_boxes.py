"""Plant boxes in a labeling round (label.prelabel.kind=sam3_boxes): CVAT
rectangles, the detection labels, and where each corrected box came from.

Boxes are xyxy pixel edges (x1, y1 exclusive), like full_image_boxes writes
them: `bbox` at full resolution, `bbox_cvat` in the downscaled frame uploaded to
CVAT, which is also what a CVAT rectangle's points are.
"""

from __future__ import annotations

from src.utils.instance_annotations import AnnotationSource

KINDS = ("unet_masks", "sam3_boxes")


def is_box_round(lcfg) -> bool:
    """True for label.prelabel.kind=sam3_boxes (boxes), False for unet_masks (masks)."""
    kind = lcfg.prelabel.get("kind", "unet_masks")
    if kind not in KINDS:
        raise ValueError(f"label.prelabel.kind must be one of {KINDS}, not {kind!r}")
    return kind == "sam3_boxes"


def class_names(ccfg) -> dict[int, str]:
    """Detection class id -> CVAT label name (cvat.detection.labels)."""
    return {int(label.id): str(label.name) for label in ccfg.detection.labels}


def scale_box(box, sx: float, sy: float) -> list[float]:
    x0, y0, x1, y1 = box
    return [round(x0 * sx, 2), round(y0 * sy, 2), round(x1 * sx, 2), round(y1 * sy, 2)]


def rectangle_shape(box, frame: int, label_id: int) -> dict:
    return {
        "type": "rectangle", "frame": frame, "label_id": label_id, "points": [float(v) for v in box],
        "occluded": False, "z_order": 0, "group": 0, "rotation": 0.0, "attributes": [], "source": "auto",
    }


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_provenance(pulled: list[dict], uploaded: list[dict], match_iou: float = 0.5,
                     tolerance_px: float = 0.5) -> tuple[list[tuple[str, int | None]], list[int]]:
    """Where each pulled box came from, by comparing `bbox_cvat` and `label`
    with the uploaded pre-labels: the pre-label's own `source` (sam3 by
    default) if unchanged (within tolerance_px), <source>_corrected if moved,
    resized or relabeled (the best remaining IoU match, at least match_iou),
    or human (added). Returns ([(source, uploaded id or None)] per pulled box,
    ids of the uploaded boxes the annotator deleted)."""
    result: list[tuple[str, int | None]] = [(AnnotationSource.HUMAN.value, None)] * len(pulled)
    free = {u["id"]: u for u in uploaded}
    for i, box in enumerate(pulled):
        for uid, up in free.items():
            if up["label"] == box["label"] and max(abs(a - b) for a, b in zip(up["bbox_cvat"], box["bbox_cvat"])) <= tolerance_px:
                result[i] = (up.get("source", AnnotationSource.SAM3.value), uid)
                del free[uid]
                break
    pairs = sorted(((iou(box["bbox_cvat"], up["bbox_cvat"]), i, uid)
                    for i, box in enumerate(pulled) if result[i][1] is None
                    for uid, up in free.items()), reverse=True)
    for overlap, i, uid in pairs:
        if overlap < match_iou:
            break
        if result[i][1] is None and uid in free:
            result[i] = (f"{free[uid].get('source', AnnotationSource.SAM3.value)}_corrected", uid)
            del free[uid]
    return result, sorted(free)
