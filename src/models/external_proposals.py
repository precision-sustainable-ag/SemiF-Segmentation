"""Run a torchvision Mask R-CNN / PointRend model on externally supplied
proposal boxes (e.g. SAM 3's) instead of, or besides, its own RPN.

The model is unchanged and needs no external proposals at other times: this
only re-wires its forward pass. The box head still classifies, re-scores and
refines each proposal (dropping those it calls background) and the mask
branch (PointRend) predicts the masks, so the output is the model's usual
per-image {"boxes", "labels", "scores", "masks"}.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.transform import resize_boxes

PROPOSAL_MODES = ("replace", "union")


@torch.no_grad()
def predict_with_proposals(model: MaskRCNN, images: list[torch.Tensor], proposals: list[torch.Tensor],
                           mode: str = "replace") -> list[dict[str, torch.Tensor]]:
    """
    Args:
        model: a torchvision-style detector (src.models.build_model), in eval mode.
        images: (C, H, W) tensors, preprocessed as for the model.
        proposals: per image, (N, 4) float xyxy boxes in that image's pixels
            (scale them first if the proposals came from another resolution).
        mode: "replace" uses only the external boxes; "union" adds them to
            the RPN's proposals.
    """
    if mode not in PROPOSAL_MODES:
        raise ValueError(f"mode must be one of {PROPOSAL_MODES}, got {mode!r}")
    if model.training:
        raise RuntimeError("predict_with_proposals needs the model in eval mode")
    if len(images) != len(proposals):
        raise ValueError(f"{len(images)} images but {len(proposals)} proposal sets")
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    image_list, _ = model.transform(list(images))
    features = model.backbone(image_list.tensors)
    device = image_list.tensors.device
    boxes = [resize_boxes(p.to(device=device, dtype=torch.float32).reshape(-1, 4), orig, new)
             for p, orig, new in zip(proposals, original_sizes, image_list.image_sizes)]
    if mode == "union":
        rpn_boxes, _ = model.rpn(image_list, features)
        boxes = [torch.cat([b, r]) for b, r in zip(boxes, rpn_boxes)]
    detections, _ = model.roi_heads(features, boxes, image_list.image_sizes)
    return model.transform.postprocess(detections, image_list.image_sizes, original_sizes)
