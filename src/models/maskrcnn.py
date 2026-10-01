"""Plain torchvision Mask R-CNN (R50-FPN-V2): the baseline PointRend is
compared against, and the detector PointRend is built on.

Two departures from torchvision's defaults, both to fit this repo's data path:
- Images are not resized (resize=False): torchvision would scale every image
  to min_size=800, losing detail on 1024-2048 px tiles. The repo already sets
  the working resolution (preprocess.tile.scale / inference.rescale_factor).
- Images are not normalized (mean 0, std 1): the repo's Dataset and inference
  preprocessing already normalize with the dataset's RGB mean/std.
"""

from __future__ import annotations

import logging

from torch import nn
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from src.models.backbones import build_backbone

log = logging.getLogger(__name__)

PRETRAINED_CHOICES = (True, False, None, "coco", "imagenet")


class NoResizeTransform(GeneralizedRCNNTransform):
    """GeneralizedRCNNTransform that keeps images (and targets) at their input size."""

    def resize(self, image, target=None):
        return image, target


def _load_matching_weights(model: nn.Module, state_dict: dict) -> None:
    """Load the entries of `state_dict` whose name and shape match `model`;
    heads sized for another number of classes (COCO's 91) stay freshly initialized."""
    own = model.state_dict()
    matching = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matching, strict=False)
    skipped = sorted(set(state_dict) - set(matching))
    fresh = sorted(set(own) - set(matching))
    log.info("Loaded %d pretrained tensors; skipped %d (%s); %d initialized fresh",
             len(matching), len(skipped), ", ".join(skipped) or "-", len(fresh))


def maskrcnn(
    num_classes: int,
    pretrained: bool | str | None = True,
    backbone: str = "resnet50_fpn_v2",
    trainable_backbone_layers: int = 3,
    resize: bool = False,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    **detector_kwargs,
) -> MaskRCNN:
    """torchvision Mask R-CNN R50-FPN-V2 (same modules as maskrcnn_resnet50_fpn_v2).

    Args:
        num_classes: detection classes including background (torchvision convention).
        pretrained: True/"coco" loads the COCO detector (all but the
            class-sized predictors), "imagenet" only the ResNet, False/None nothing.
        trainable_backbone_layers: ResNet stages left trainable (5 = all);
            ignored without pretrained weights, where everything trains.
        resize: resize images to min_size/max_size (torchvision's default
            behaviour) instead of keeping them as they are.
        image_mean, image_std: normalization inside the model; default
            identity, as inputs are already normalized.
        detector_kwargs: passed to torchvision's MaskRCNN (min_size, max_size,
            box_detections_per_img, box_score_thresh, rpn_* ...).
    """
    if pretrained not in PRETRAINED_CHOICES:
        raise ValueError(f"pretrained must be one of {PRETRAINED_CHOICES}, got {pretrained!r}")
    coco = pretrained in (True, "coco")
    if backbone != "resnet50_fpn_v2" and coco:
        raise ValueError("COCO detector weights exist only for backbone=resnet50_fpn_v2")

    body = build_backbone(
        backbone,
        imagenet_weights=pretrained == "imagenet",
        trainable_layers=trainable_backbone_layers if pretrained else 5,
    )
    anchor_generator = AnchorGenerator(((32,), (64,), (128,), (256,), (512,)), ((0.5, 1.0, 2.0),) * 5)
    model = MaskRCNN(
        body,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        rpn_head=RPNHead(body.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2),
        box_head=FastRCNNConvFCHead((body.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d),
        mask_head=MaskRCNNHeads(body.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d),
        image_mean=image_mean or [0.0, 0.0, 0.0],
        image_std=image_std or [1.0, 1.0, 1.0],
        **detector_kwargs,
    )
    if not resize:
        t = model.transform
        model.transform = NoResizeTransform(
            t.min_size, t.max_size, t.image_mean, t.image_std,
            size_divisible=t.size_divisible, fixed_size=t.fixed_size,
        )
    if coco:
        state_dict = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(progress=False)
        _load_matching_weights(model, state_dict)
    return model

