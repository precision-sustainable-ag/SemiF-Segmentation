"""ResNet-FPN backbones for the torchvision detectors.

resnet50_fpn_v2 is the backbone of torchvision's maskrcnn_resnet50_fpn_v2
(Li et al., "Benchmarking Detection Transfer Learning with Vision
Transformers"): trainable BatchNorm instead of frozen BN. Its module names
match that builder, so the COCO detector weights load into it.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.backbone_utils import BackboneWithFPN, _resnet_fpn_extractor


def resnet50_fpn_v2(imagenet_weights: bool = False, trainable_layers: int = 3) -> BackboneWithFPN:
    """
    Args:
        imagenet_weights: start the ResNet from ImageNet weights.
        trainable_layers: ResNet stages (from the last) left trainable, 0-5;
            earlier ones are frozen.
    """
    if not 0 <= trainable_layers <= 5:
        raise ValueError(f"trainable_layers must be in [0, 5], got {trainable_layers}")
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if imagenet_weights else None)
    return _resnet_fpn_extractor(backbone, trainable_layers, norm_layer=nn.BatchNorm2d)


BACKBONES = {
    "resnet50_fpn_v2": resnet50_fpn_v2,
}


def build_backbone(name: str, **kwargs) -> BackboneWithFPN:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)
