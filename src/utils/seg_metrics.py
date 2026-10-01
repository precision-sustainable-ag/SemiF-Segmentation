"""Per-image mask-quality metrics that are sensitive to fine detail, for
binary (foreground vs background) masks on the GPU.

- dice, iou: region overlap.
- boundary_iou (Cheng et al., CVPR 2021): IoU of the inner bands of width
  `boundary_width` px along each mask's contour; unlike mask IoU it isn't
  dominated by the interior of large plants.
- boundary_f1: F1 of contour pixels matched within `boundary_tolerance` px
  (the BSDS / DAVIS F-measure).
- thin_recall: fraction of thin foreground -- gt pixels removed by a
  morphological opening of width `thin_width` px, i.e. stems, petioles, grass
  blades narrower than that -- that the prediction covers.

Morphology uses square structuring elements via max-pooling. Image edges are
not treated as contours: on tiles they are cut lines, not plant boundaries.

Each metric is NaN for an image where it's undefined (e.g. thin_recall with
no thin gt structure); average with nanmean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

METRIC_NAMES = ("dice", "iou", "boundary_iou", "boundary_f1", "thin_recall")


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)  # -inf padding: edges don't erode


def inner_boundary(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Foreground pixels within `width` px of the background (B, 1, H, W float {0, 1})."""
    return mask * (1 - _erode(mask, width))


def _ratio(num: torch.Tensor, den: torch.Tensor, both_empty: float = 1.0) -> torch.Tensor:
    """num / den per image; `both_empty` where num and den are both 0."""
    out = num / den.clamp(min=1)
    return torch.where(den > 0, out, torch.full_like(out, both_empty))


@dataclass
class DetailMetricsConfig:
    boundary_width: int = 3
    boundary_tolerance: int = 2
    thin_width: int = 7


@torch.no_grad()
def detail_metrics(pred: torch.Tensor, gt: torch.Tensor, cfg: DetailMetricsConfig | None = None) -> dict[str, torch.Tensor]:
    """Per-image metrics for (B, H, W) or (B, 1, H, W) masks (nonzero = foreground).

    Returns {name: (B,) float tensor} for each name in METRIC_NAMES.
    """
    cfg = cfg or DetailMetricsConfig()
    if pred.ndim == 3:
        pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
    p = (pred > 0).float()
    g = (gt > 0).float()
    total = lambda x: x.sum(dim=(1, 2, 3))  # noqa: E731

    inter, p_area, g_area = total(p * g), total(p), total(g)
    dice = _ratio(2 * inter, p_area + g_area)
    iou = _ratio(inter, p_area + g_area - inter)

    pb, gb = inner_boundary(p, cfg.boundary_width), inner_boundary(g, cfg.boundary_width)
    b_inter = total(pb * gb)
    boundary_iou = _ratio(b_inter, total(pb) + total(gb) - b_inter)

    pc, gc = inner_boundary(p, 1), inner_boundary(g, 1)  # 1-px contours
    precision = _ratio(total(pc * _dilate(gc, cfg.boundary_tolerance)), total(pc), both_empty=float("nan"))
    recall = _ratio(total(gc * _dilate(pc, cfg.boundary_tolerance)), total(gc), both_empty=float("nan"))
    boundary_f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    boundary_f1 = torch.where(total(pc) + total(gc) == 0, torch.ones_like(boundary_f1), boundary_f1)
    boundary_f1 = torch.where(  # contours on one side only: F1 = 0
        (total(pc) == 0) ^ (total(gc) == 0), torch.zeros_like(boundary_f1), boundary_f1
    )

    r = cfg.thin_width // 2
    thin = g * (1 - _dilate(_erode(g, r), r))
    thin_recall = _ratio(total(thin * p), total(thin), both_empty=float("nan"))

    return {"dice": dice, "iou": iou, "boundary_iou": boundary_iou,
            "boundary_f1": boundary_f1, "thin_recall": thin_recall}
