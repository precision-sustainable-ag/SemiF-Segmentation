"""Evaluate an instance model on whole images rather than training tiles.

For the full images of one manifest split (inference.full_image_eval.split),
at the training scale: predict with overlapping tiles stitched into
full-image instances (src.inferencing.tiled_instances), and compare with the
full-image ground-truth instances (connected blobs of the label mask).
Plants cut by tile edges are therefore scored whole.

Writes to paths.project_inference_dir/full_image_eval/version_<v>/<split>/:
  summary.json    COCO box/mask AP (AP, AP50, AP75, AR), instance counts,
                  detail metrics (averaged over images), settings
  per_image.csv   the same per image, plus timings
  plots/          <image>_full.png (downscaled) and <image>_zoom.png (a
                  full-resolution window): image | ground truth | prediction
"""

import json
import logging
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.inferencing.tiled_instances import (
    Instances,
    _window,
    coco_ap,
    instances_from_mask,
    paint_semantic,
    predict_tiled_instances,
)
from src.utils.instance_model import InstanceSegmentationModule
from src.utils.model import detail_metrics_config
from src.utils.seg_metrics import detail_metrics
from src.utils.viz_utils import draw_instances

log = logging.getLogger(__name__)


def load_full_image(image_path, mask_path, scale: float):
    """RGB image and label mask at the training scale, resized as src/preprocessing/tile.py does."""
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != (width, height):
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def to_model_input(image: np.ndarray, normalize: bool, mean, std) -> torch.Tensor:
    """(C, H, W) float tensor, preprocessed as src.utils.datasets.Dataset does."""
    x = image.astype(np.float32) / 255.0
    if normalize and mean is not None:
        x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def _dense_masks(inst: Instances, keep, region, scale: float = 1.0):
    """Those of instances `keep` that reach into region (x0, y0, x1, y1): dense bool masks
    over it (resized by scale), their boxes there, and their indices."""
    rx0, ry0, rx1, ry1 = region
    size = (max(1, round((rx1 - rx0) * scale)), max(1, round((ry1 - ry0) * scale)))
    masks, boxes, used = [], [], []
    for i in keep:
        m = _window(inst, i, region)
        if not m.any():
            continue
        if scale != 1:
            m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        x0, y0, x1, y1 = inst.boxes[i]
        boxes.append([(max(x0, rx0) - rx0) * scale, (max(y0, ry0) - ry0) * scale,
                      (min(x1, rx1) - rx0) * scale, (min(y1, ry1) - ry0) * scale])
        masks.append(m)
        used.append(i)
    shape = (size[1], size[0])
    return (np.stack(masks) if masks else np.zeros((0, *shape), bool)), np.array(boxes).reshape(-1, 4), used


def plot_instances(image, gt: Instances, dt: Instances, region, scale, score_threshold, title, output_path):
    """Image | ground-truth instances | predicted instances (score >= threshold) over region."""
    rx0, ry0, rx1, ry1 = region
    view = image[ry0:ry1, rx0:rx1]
    if scale != 1:
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gt_masks, gt_boxes, _ = _dense_masks(gt, range(len(gt)), region, scale)
    keep = [i for i in np.argsort(-dt.scores) if dt.scores[i] >= score_threshold]
    dt_masks, dt_boxes, used = _dense_masks(dt, keep, region, scale)
    texts = [f"{dt.scores[i]:.2f}" for i in used]
    panels = [(view, title),
              (draw_instances(view, gt_masks, gt_boxes), f"Ground truth: {len(gt_masks)} instances"),
              (draw_instances(view, dt_masks, dt_boxes, texts), f"Prediction: {len(dt_masks)} (score >= {score_threshold})")]
    fig, axs = plt.subplots(1, 3, figsize=(21, 21 * view.shape[0] / view.shape[1] / 3 + 0.8))
    for ax, (panel, name) in zip(axs, panels):
        ax.imshow(panel)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def densest_window(mask: np.ndarray, size: int):
    """(x0, y0, x1, y1) of the size x size window (on a size/2 grid) with the most foreground."""
    H, W = mask.shape
    best, best_count = (0, 0, min(size, W), min(size, H)), -1
    for y in range(0, max(H - size, 0) + 1, size // 2):
        for x in range(0, max(W - size, 0) + 1, size // 2):
            count = int(np.count_nonzero(mask[y:y + size, x:x + size]))
            if count > best_count:
                best, best_count = (x, y, min(x + size, W), min(y + size, H)), count
    return best


def main(cfg: DictConfig):
    ecfg, tcfg = cfg.inference.full_image_eval, cfg.inference.instance_tiling
    if cfg.model.get("task", "semantic") != "instance":
        raise ValueError(f"full_image_eval needs an instance model (model=maskrcnn or maskrcnn_pointrend), not {cfg.model.arch_name}")
    version = cfg.inference.inference.version
    ckpt = Path(cfg.paths.inference.checkpoints) / ecfg.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = InstanceSegmentationModule.load_from_checkpoint(ckpt, cfg=cfg, weights_only=False).to(device).eval()
    log.info("Loaded %s", ckpt)

    stats_file = Path(cfg.paths.project_preprocess_dir) / "data/data_stats/rgb_mean_std.json"
    mean = std = None
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        mean, std = stats["mean"], stats["std"]
    manifest = pd.read_csv(cfg.paths.dataset_manifest)
    rows = manifest[manifest["split"] == ecfg.split]
    if ecfg.max_images:
        rows = rows.head(int(ecfg.max_images))
    out_dir = Path(cfg.paths.project_inference_dir) / "full_image_eval" / f"version_{version}" / ecfg.split
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    scale = cfg.preprocess.tile.scale
    score_threshold = module.score_threshold
    detail_cfg = detail_metrics_config(cfg)
    gts, dts, records = [], [], []
    for _, row in rows.iterrows():
        image, mask = load_full_image(row["image_path"], row["mask_path"], scale)
        gt = instances_from_mask(mask, cfg.model.instances.min_area)
        start = time.time()
        dt = predict_tiled_instances(
            module.model, to_model_input(image, cfg.train.dataset.normalize, mean, std),
            tile_size=tcfg.tile_size, overlap=tcfg.overlap, merge_iou=tcfg.merge_iou,
            merge_min_score=tcfg.merge_min_score,
            mask_threshold=module.mask_threshold, batch_size=tcfg.batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        predict_s = time.time() - start

        semantic = paint_semantic(dt, score_threshold)
        detail = detail_metrics(torch.from_numpy(semantic)[None].to(device), torch.from_numpy(mask)[None].to(device),
                                detail_cfg)
        n_pred = int((dt.scores >= score_threshold).sum())
        records.append({"image_id": row["image_id"], "height": mask.shape[0], "width": mask.shape[1],
                        "gt_instances": len(gt), "pred_instances": n_pred, "raw_detections": len(dt),
                        "predict_s": round(predict_s, 2), **{k: float(v[0]) for k, v in detail.items()},
                        **{f"img_{k}": v for k, v in coco_ap([gt], [dt], ecfg.max_dets).items()}})
        gts.append(gt)
        dts.append(dt)
        log.info("%s: %d gt / %d predicted instances, IoU %.3f, boundary F1 %.3f, %.1f s",
                 row["image_id"], len(gt), n_pred, records[-1]["iou"], records[-1]["boundary_f1"], predict_s)

        if ecfg.plot:
            H, W = mask.shape
            plot_instances(image, gt, dt, (0, 0, W, H), min(1.0, 2400 / max(H, W)), score_threshold,
                           f"{row['image_id']} ({W} x {H})", out_dir / "plots" / f"{row['image_id']}_full.png")
            window = densest_window(mask, ecfg.zoom_size)
            plot_instances(image, gt, dt, window, 1.0, score_threshold,
                           f"{row['image_id']} x={window[0]} y={window[1]}", out_dir / "plots" / f"{row['image_id']}_zoom.png")

    per_image = pd.DataFrame(records)
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    metric_names = ["dice", "iou", "boundary_iou", "boundary_f1", "thin_recall"]
    summary = {
        "checkpoint": str(ckpt),
        "model": cfg.model.name,
        "split": ecfg.split,
        "images": len(per_image),
        **coco_ap(gts, dts, ecfg.max_dets),
        "gt_instances": int(per_image["gt_instances"].sum()),
        "pred_instances": int(per_image["pred_instances"].sum()),
        "count_mae": float((per_image["pred_instances"] - per_image["gt_instances"]).abs().mean()),
        **{k: float(np.nanmean(per_image[k])) for k in metric_names},
        "predict_s_per_image": float(per_image["predict_s"].mean()),
        "settings": {"scale": scale, "score_threshold": score_threshold,
                     **OmegaConf.to_container(tcfg, resolve=True)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Full-image evaluation of %d images saved to %s: segm AP %.3f, bbox AP %.3f",
             len(per_image), out_dir, summary["segm_map"], summary["bbox_map"])
    return summary
