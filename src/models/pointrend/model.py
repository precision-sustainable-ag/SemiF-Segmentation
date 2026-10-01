"""Mask R-CNN + PointRend: the torchvision Mask R-CNN from src.models.maskrcnn
with its mask branch swapped for PointRend. Backbone, RPN and box branch are
identical to the baseline (and load the same COCO weights); the PointRend
heads start from scratch."""

from __future__ import annotations

from torchvision.models.detection import MaskRCNN

from src.models.maskrcnn import maskrcnn
from src.models.pointrend.mask_head import PointRendMaskHead
from src.models.pointrend.roi_heads import PointRendRoIHeads


def maskrcnn_pointrend(
    num_classes: int,
    pretrained: bool | str | None = True,
    class_agnostic_mask: bool = True,
    num_points: int = 196,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
    subdivision_steps: int = 5,
    subdivision_num_points: int = 784,
    coarse_in_resolution: int = 14,
    coarse_out_resolution: int = 7,
    coarse_feature_names: tuple[str, ...] = ("0",),
    fine_feature_names: tuple[str, ...] = ("0",),
    coarse_fc_dim: int = 1024,
    coarse_num_fc: int = 2,
    point_fc_dim: int = 256,
    point_num_fc: int = 3,
    **maskrcnn_kwargs,
) -> MaskRCNN:
    """
    Args:
        num_classes: detection classes including background.
        class_agnostic_mask: one foreground mask channel shared by all classes
            (the box head stays multi-class either way).
        num_points, oversample_ratio, importance_sample_ratio: training point
            sampling (Detectron2 defaults: 14*14, 3, 0.75).
        subdivision_steps, subdivision_num_points: inference refinement; masks
            come out at coarse_out_resolution * 2**subdivision_steps per box
            (Detectron2 defaults: 5 steps, 28*28 points -> 224 x 224).
        coarse_feature_names / fine_feature_names: FPN levels ("0" = p2,
            stride 4) for the coarse head's ROIAlign / the point features.
        maskrcnn_kwargs: passed to src.models.maskrcnn.maskrcnn (backbone,
            pretrained layers, resize, detector settings).
    """
    model = maskrcnn(num_classes=num_classes, pretrained=pretrained, **maskrcnn_kwargs)
    point_mask_head = PointRendMaskHead(
        in_channels=model.backbone.out_channels,
        num_mask_classes=1 if class_agnostic_mask else num_classes,
        coarse_feature_names=tuple(coarse_feature_names),
        fine_feature_names=tuple(fine_feature_names),
        coarse_in_resolution=coarse_in_resolution,
        coarse_out_resolution=coarse_out_resolution,
        coarse_fc_dim=coarse_fc_dim,
        coarse_num_fc=coarse_num_fc,
        point_fc_dim=point_fc_dim,
        point_num_fc=point_num_fc,
        num_points=num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
        subdivision_steps=subdivision_steps,
        subdivision_num_points=subdivision_num_points,
    )
    model.roi_heads = PointRendRoIHeads.from_roi_heads(model.roi_heads, point_mask_head)
    return model
