"""PointRend training losses: the coarse mask loss (standard Mask R-CNN mask
loss at the coarse resolution) and the point-wise loss at sampled points."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import project_masks_on_boxes

from src.models.pointrend.sampling import box_coords_to_image_coords, sample_masks_at_points
from src.models.pointrend.uncertainty import select_class_logits


def coarse_mask_loss(
    coarse_logits: torch.Tensor,
    proposals: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    matched_idxs: list[torch.Tensor],
    mask_classes: torch.Tensor | None,
) -> torch.Tensor:
    """BCE between each box's coarse (R, C, S, S) logits and its matched
    ground-truth mask ROIAligned to S x S (torchvision's project_masks_on_boxes)."""
    if coarse_logits.shape[0] == 0:
        return coarse_logits.sum() * 0
    S = coarse_logits.shape[-1]
    targets = torch.cat(
        [project_masks_on_boxes(m, p, i, S) for m, p, i in zip(gt_masks, proposals, matched_idxs)]
    )
    logits = select_class_logits(coarse_logits, mask_classes)[:, 0]
    return F.binary_cross_entropy_with_logits(logits, targets)


def point_targets(
    gt_masks: list[torch.Tensor],
    proposals: list[torch.Tensor],
    matched_idxs: list[torch.Tensor],
    point_coords: torch.Tensor,
) -> torch.Tensor:
    """(R, P) ground-truth mask values at box-relative `point_coords`, read
    from the full-resolution instance masks (not the ROIAligned ones)."""
    targets = []
    start = 0
    for masks, boxes, idx in zip(gt_masks, proposals, matched_idxs):
        n = boxes.shape[0]
        abs_coords = box_coords_to_image_coords(boxes, point_coords[start:start + n])
        targets.append(sample_masks_at_points(masks, idx, abs_coords))
        start += n
    return torch.cat(targets)


def point_mask_loss(
    point_logits: torch.Tensor, mask_classes: torch.Tensor | None, targets: torch.Tensor
) -> torch.Tensor:
    """BCE between (R, C, P) point logits (each box's mask channel) and (R, P) targets."""
    if point_logits.shape[0] == 0:
        return point_logits.sum() * 0
    logits = select_class_logits(point_logits, mask_classes)[:, 0]
    return F.binary_cross_entropy_with_logits(logits, targets)
