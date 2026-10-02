"""sam3_finetune task `evaluate`: zero-shot vs fine-tuned SAM 3 on the held-out
tiles (dataset/valid), as box detectors.

Each checkpoint runs through the proposal generator the box rounds pre-label
with (conf/proposals: prompts, filtering), on every valid tile, one prompt per
category, using the predicted boxes. Reported per checkpoint:
  AP, AP50, AP75 (COCO, every class) and AP per class, from detections down to
    evaluate.score_threshold;
  precision / recall / F1 at IoU 0.5 for detections at or above
    proposals.score_threshold: what annotators would get as pre-labels.
Writes <run>/eval.json, every detection in <run>/detections/<checkpoint>.json
(COCO results), and for the first evaluate.previews valid images
<run>/previews/<image>.jpg: ground truth | each checkpoint's detections at
the operating threshold, side by side.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def predict_tiles(generator, valid_dir: Path, coco: dict) -> list[dict]:
    """COCO-style detections of `generator` on every tile of coco["images"]."""
    cat_of = {c["name"]: c["id"] for c in coco["categories"]}
    generator.prompts = list(cat_of)
    detections = []
    for entry in coco["images"]:
        image = cv2.cvtColor(cv2.imread(str(valid_dir / entry["file_name"])), cv2.COLOR_BGR2RGB)
        for p in generator.predict(image).proposals:
            x0, y0, x1, y1 = (float(v) for v in p.bbox)
            detections.append({"image_id": entry["id"], "category_id": cat_of[p.prompt],
                               "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(p.score)})
    return detections


def coco_ap(coco: dict, detections: list[dict]) -> dict:
    """AP, AP50, AP75 overall and AP per category name."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "per_class": {c["name"]: 0.0 for c in coco["categories"]}}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco
        gt.createIndex()
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    out = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2]), "per_class": {}}
    precision = ev.eval["precision"]  # (iou, recall, class, area, maxdet)
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, -1]
        out["per_class"][gt.cats[cat_id]["name"]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def precision_recall(coco: dict, detections: list[dict], score_threshold: float, iou: float = 0.5) -> dict:
    """Greedy (by score) one-to-one matching per tile and category, at `iou`."""
    gts: dict = {}
    for a in coco["annotations"]:
        gts.setdefault((a["image_id"], a["category_id"]), []).append(a["bbox"])
    tp = fp = 0
    used = {k: [False] * len(v) for k, v in gts.items()}
    for d in sorted((d for d in detections if d["score"] >= score_threshold), key=lambda d: -d["score"]):
        key = (d["image_id"], d["category_id"])
        best, best_iou = None, iou
        for j, g in enumerate(gts.get(key, [])):
            overlap = _iou(d["bbox"], g)
            if not used[key][j] and overlap >= best_iou:
                best, best_iou = j, overlap
        if best is None:
            fp += 1
        else:
            used[key][best] = True
            tp += 1
    n_gt = len(coco["annotations"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"score_threshold": score_threshold, "iou": iou, "tp": tp, "fp": fp, "fn": n_gt - tp,
            "precision": precision, "recall": recall, "f1": f1}


COLORS = [(255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0)]  # BGR per category: magenta, cyan, ...


def draw_panel(image: np.ndarray, boxes, title: str, width: int) -> np.ndarray:
    """`image` (BGR) resized to `width`, with (bbox xywh, category_id, score|None) boxes and a title bar."""
    f = width / image.shape[1]
    view = cv2.resize(image, (width, max(1, round(image.shape[0] * f))), interpolation=cv2.INTER_AREA)
    thickness = max(1, round(width / 400))
    for (x, y, w, h), cat, score in boxes:
        color = COLORS[(cat - 1) % len(COLORS)]
        p0, p1 = (round(x * f), round(y * f)), (round((x + w) * f), round((y + h) * f))
        cv2.rectangle(view, p0, p1, color, thickness)
        if score is not None:
            cv2.putText(view, f"{score:.2f}", (p0[0] + 2, max(p0[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * thickness, color, 1, cv2.LINE_AA)
    bar = np.full((28, width, 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, view])


def save_previews(out_dir: Path, valid_dir: Path, coco: dict, detections: dict[str, list[dict]],
                  score_threshold: float, n: int, panel_width: int = 900) -> int:
    """previews/<image>.jpg for the first n valid images (most boxes first):
    ground truth | each checkpoint's detections at >= score_threshold."""
    gts: dict = {}
    for a in coco["annotations"]:
        gts.setdefault(a["image_id"], []).append((a["bbox"], a["category_id"], None))
    entries = sorted(coco["images"], key=lambda e: (-len(gts.get(e["id"], [])), e["file_name"]))[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    legend = ", ".join(f"{names[c]}: {'magenta cyan orange green'.split()[(c - 1) % 4]}" for c in sorted(names))
    for entry in entries:
        image = cv2.imread(str(valid_dir / entry["file_name"]))
        truth = gts.get(entry["id"], [])
        panels = [draw_panel(image, truth, f"ground truth: {len(truth)} ({legend})", panel_width)]
        for name, dets in detections.items():
            shown = [(d["bbox"], d["category_id"], d["score"]) for d in dets
                     if d["image_id"] == entry["id"] and d["score"] >= score_threshold]
            panels.append(draw_panel(image, shown, f"{name}: {len(shown)} at score >= {score_threshold:.2f}",
                                     panel_width))
        cv2.imwrite(str(out_dir / Path(entry["file_name"]).with_suffix(".jpg").name), np.hstack(panels),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(entries)


def evaluate(dataset_dir: str | Path, checkpoints: dict[str, str], scfg: dict, ecfg: dict, build_generator,
             out_dir: str | Path | None = None) -> dict:
    """{name: metrics} for each checkpoint in `checkpoints` ({name: proposals.checkpoint}).
    build_generator(scfg) returns a proposal generator for proposals config scfg. With
    out_dir, also writes detections/ and previews/ there."""
    valid_dir = Path(dataset_dir) / "valid"
    coco = json.loads((valid_dir / "_annotations.coco.json").read_text())
    if not coco["images"]:
        raise ValueError(f"{valid_dir} has no tiles; raise sam3_finetune.dataset.val_fraction")
    operating = float(scfg.get("score_threshold", 0.5))
    results = {"valid_tiles": len(coco["images"]), "valid_boxes": len(coco["annotations"]), "checkpoints": {}}
    all_detections = {}
    for name, checkpoint in checkpoints.items():
        # Low threshold for AP; the generator's filters (area, duplicates) as in the rounds.
        generator = build_generator({**scfg, "checkpoint": checkpoint,
                                     "score_threshold": min(float(ecfg["score_threshold"]), operating)})
        detections = predict_tiles(generator, valid_dir, coco)
        all_detections[name] = detections
        del generator
        metrics = {"checkpoint": str(checkpoint), "detections": len(detections), **coco_ap(coco, detections),
                   "at_operating_threshold": precision_recall(coco, detections, operating)}
        results["checkpoints"][name] = metrics
        pr = metrics["at_operating_threshold"]
        log.info("%s: AP %.3f, AP50 %.3f, per class %s; at score >= %.2f: precision %.3f, recall %.3f, F1 %.3f",
                 name, metrics["AP"], metrics["AP50"], {k: round(v, 3) for k, v in metrics["per_class"].items()},
                 operating, pr["precision"], pr["recall"], pr["f1"])
        _free_gpu()
    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / "detections").mkdir(parents=True, exist_ok=True)
        for name, detections in all_detections.items():
            (out_dir / "detections" / f"{name}.json").write_text(json.dumps(detections))
        shown = all_detections
        if len(all_detections) > 3:  # zero-shot vs the best fine-tuned one, not every snapshot
            key = ecfg.get("select_by", "AP")
            tuned = {k: v for k, v in results["checkpoints"].items() if k != "zero_shot"}
            best = max(tuned, key=lambda k: tuned[k][key])
            shown = {k: all_detections[k] for k in ("zero_shot", best) if k in all_detections}
        n = save_previews(out_dir / "previews", valid_dir, coco, shown, operating, int(ecfg.get("previews", 8)))
        log.info("Wrote %d previews to %s", n, out_dir / "previews")
    return results


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
