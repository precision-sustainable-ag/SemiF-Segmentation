"""PointRend mask branch: ROIAlign -> coarse head -> point head.

Training (PointRend sec. 3.1): the coarse logits get the standard mask loss;
then, per box, `num_points` points are picked -- mostly the most uncertain of
an oversampled random set, the rest uniformly random -- and the point head is
trained on the full-resolution ground truth at those points.

Inference (sec. 3.1, adaptive subdivision): starting from the coarse logits,
repeatedly upsample 2x and re-predict the `subdivision_num_points` most
uncertain cells with the point head, `subdivision_steps` times.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MultiScaleRoIAlign

from src.models.pointrend.coarse_head import CoarseMaskHead
from src.models.pointrend.losses import coarse_mask_loss, point_mask_loss, point_targets
from src.models.pointrend.point_head import PointHead
from src.models.pointrend.sampling import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sample_fine_grained_features,
)
from src.models.pointrend.uncertainty import calculate_uncertainty, select_class_logits


class PointRendMaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_mask_classes: int = 1,
        coarse_feature_names: tuple[str, ...] = ("0",),
        fine_feature_names: tuple[str, ...] = ("0",),
        coarse_in_resolution: int = 14,
        coarse_out_resolution: int = 7,
        coarse_fc_dim: int = 1024,
        coarse_num_fc: int = 2,
        point_fc_dim: int = 256,
        point_num_fc: int = 3,
        num_points: int = 196,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        subdivision_steps: int = 5,
        subdivision_num_points: int = 784,
    ):
        """
        Args:
            in_channels: channels of each FPN level (256).
            num_mask_classes: 1 for a class-agnostic mask, else the detector's
                num_classes (background included, as in torchvision).
            coarse_feature_names / fine_feature_names: FPN levels ("0" is p2,
                stride 4) the coarse head pools from / the point head samples.
        """
        super().__init__()
        self.num_mask_classes = num_mask_classes
        self.fine_feature_names = list(fine_feature_names)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = subdivision_num_points

        self.mask_roi_pool = MultiScaleRoIAlign(list(coarse_feature_names), coarse_in_resolution, sampling_ratio=2)
        self.coarse_head = CoarseMaskHead(
            in_channels, coarse_in_resolution, num_mask_classes,
            conv_dim=in_channels, fc_dim=coarse_fc_dim, num_fc=coarse_num_fc, out_resolution=coarse_out_resolution,
        )
        self.point_head = PointHead(
            in_channels * len(self.fine_feature_names), num_mask_classes, fc_dim=point_fc_dim, num_fc=point_num_fc,
        )

    @property
    def class_agnostic(self) -> bool:
        return self.num_mask_classes == 1

    @property
    def output_resolution(self) -> int:
        return self.coarse_head.out_resolution * 2**self.subdivision_steps

    def _coarse_logits(self, features, boxes, image_shapes) -> torch.Tensor:
        return self.coarse_head(self.mask_roi_pool(features, boxes, image_shapes))

    def _point_logits(self, features, boxes, image_shapes, coarse_logits, point_coords) -> torch.Tensor:
        fine = sample_fine_grained_features(
            [features[k] for k in self.fine_feature_names], image_shapes, boxes, point_coords
        )
        coarse = point_sample(coarse_logits, point_coords)
        return self.point_head(fine, coarse)

    def losses(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        gt_masks: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        mask_classes: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Losses for the positive proposals of each image.

        Args:
            proposals: per-image (R_i, 4) positive proposals.
            gt_masks: per-image (G_i, H, W) instance masks.
            matched_idxs: per-image (R_i,) index of each proposal's gt instance.
            mask_classes: (sum R_i,) mask channel per proposal; None if class-agnostic.
        """
        coarse_logits = self._coarse_logits(features, proposals, image_shapes)
        loss_mask = coarse_mask_loss(coarse_logits, proposals, gt_masks, matched_idxs, mask_classes)

        num_boxes = coarse_logits.shape[0]
        if num_boxes == 0:
            # Still run the point head so every parameter is in the graph (DDP).
            fine_channels = self.point_head.fcs[0].in_channels - self.num_mask_classes
            empty = coarse_logits.new_zeros(0, fine_channels, self.num_points)
            point_logits = self.point_head(empty, coarse_logits.new_zeros(0, self.num_mask_classes, self.num_points))
            return {"loss_mask": loss_mask, "loss_mask_point": point_logits.sum() * 0}

        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits,
            lambda logits: calculate_uncertainty(logits, mask_classes),
            self.num_points,
            self.oversample_ratio,
            self.importance_sample_ratio,
        )
        point_logits = self._point_logits(features, proposals, image_shapes, coarse_logits, point_coords)
        with torch.no_grad():
            targets = point_targets(gt_masks, proposals, matched_idxs, point_coords)
        return {"loss_mask": loss_mask, "loss_mask_point": point_mask_loss(point_logits, mask_classes, targets)}

    @torch.no_grad()
    def inference(
        self,
        features: dict[str, torch.Tensor],
        boxes: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        mask_classes: torch.Tensor | None,
    ) -> torch.Tensor:
        """(sum R_i, 1, M, M) mask logits of each box's class, M = output_resolution."""
        num_boxes = sum(b.shape[0] for b in boxes)
        if num_boxes == 0:
            M = self.output_resolution
            return features[self.fine_feature_names[0]].new_zeros(0, 1, M, M)

        coarse_logits = self._coarse_logits(features, boxes, image_shapes)
        mask_logits = coarse_logits.clone()
        for step in range(self.subdivision_steps):
            mask_logits = F.interpolate(mask_logits, scale_factor=2, mode="bilinear", align_corners=False)
            R, C, H, W = mask_logits.shape
            # If the next step will refine every cell anyway, refining now is wasted work.
            if self.subdivision_num_points >= 4 * H * W and step < self.subdivision_steps - 1:
                continue
            uncertainty_map = calculate_uncertainty(mask_logits, mask_classes)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, self.subdivision_num_points
            )
            point_logits = self._point_logits(features, boxes, image_shapes, coarse_logits, point_coords)
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            mask_logits = (
                mask_logits.reshape(R, C, H * W).scatter_(2, point_indices, point_logits).view(R, C, H, W)
            )
        return select_class_logits(mask_logits, mask_classes)
