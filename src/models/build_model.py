"""Instance-segmentation model factory.

    model = build_model("maskrcnn_pointrend", num_classes=2, pretrained=True)

Models are plain torch.nn.Modules with torchvision's detection API: in train
mode model(images, targets) returns a loss dict, in eval mode model(images)
returns per-image {"boxes", "labels", "scores", "masks"}. (The repo's
semantic models are built by segmentation_models_pytorch in
src.utils.model.SegmentationModule instead.)

Proposal generators (optional teachers for the instance models, not models
trained here) are built separately:

    generator = build_proposal_generator(cfg.proposals.name, cfg.proposals)   # "sam3"
"""

from __future__ import annotations

from torch import nn

from src.models.maskrcnn import maskrcnn
from src.models.pointrend import maskrcnn_pointrend

MODEL_REGISTRY = {
    "maskrcnn": maskrcnn,
    "maskrcnn_pointrend": maskrcnn_pointrend,
}


def build_model(model_name: str, num_classes: int, pretrained: bool | str | None = True, **kwargs) -> nn.Module:
    """
    Args:
        model_name: a key of MODEL_REGISTRY.
        num_classes: detection classes including background (torchvision convention).
        pretrained: True/"coco", "imagenet", or False/None.
        kwargs: model options (see each builder), e.g. the PointRend settings.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def _sam3_generator(config):
    from src.models.proposals import SAM3ProposalGenerator  # imports the sam3 package only when loading
    return SAM3ProposalGenerator.from_config(config)


PROPOSAL_GENERATOR_REGISTRY = {
    "sam3": _sam3_generator,
}


def build_proposal_generator(name: str, config):
    """
    Args:
        name: a key of PROPOSAL_GENERATOR_REGISTRY.
        config: the generator's config (cfg.proposals: conf/proposals/<name>.yaml).
    """
    if name not in PROPOSAL_GENERATOR_REGISTRY:
        raise ValueError(f"Unknown proposal generator {name!r}; choose from {sorted(PROPOSAL_GENERATOR_REGISTRY)}")
    return PROPOSAL_GENERATOR_REGISTRY[name](config)
