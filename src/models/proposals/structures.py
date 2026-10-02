"""Repository-native proposal types, and model outputs converted to them.

Downstream code (filtering, pseudo-labels, Mask R-CNN / PointRend) sees only
InstanceProposal / ProposalResult, never SAM 3's API. Conventions:
- image_size is (H, W);
- bbox is (4,) float xyxy in pixels of the image the proposal was made on;
- mask is (H, W) bool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class InstanceProposal:
    bbox: torch.Tensor                  # (4,) float xyxy
    mask: torch.Tensor                  # (H, W) bool
    score: float
    prompt: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)  # filled by src.models.proposals.filtering

    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class ProposalResult:
    proposals: list[InstanceProposal]
    image_size: tuple[int, int]         # (H, W)
    # proposals dropped by the filters, with the reason, for inspection
    rejected: list[tuple[InstanceProposal, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.proposals)

    @property
    def boxes(self) -> torch.Tensor:
        """(N, 4) float xyxy."""
        if not self.proposals:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.stack([p.bbox for p in self.proposals])

    @property
    def masks(self) -> torch.Tensor:
        """(N, H, W) bool."""
        if not self.proposals:
            return torch.zeros((0, *self.image_size), dtype=torch.bool)
        return torch.stack([p.mask for p in self.proposals])

    @property
    def scores(self) -> torch.Tensor:
        """(N,) float."""
        return torch.tensor([p.score for p in self.proposals], dtype=torch.float32)

    def to(self, device: torch.device | str) -> "ProposalResult":
        """A copy with every box and mask on `device` (metrics and rejected kept)."""
        def move(p: InstanceProposal) -> InstanceProposal:
            return InstanceProposal(p.bbox.to(device), p.mask.to(device), p.score, p.prompt, dict(p.metrics))
        return ProposalResult([move(p) for p in self.proposals], self.image_size,
                              [(move(p), reason) for p, reason in self.rejected])

    def extend(self, other: "ProposalResult") -> None:
        """Add another result's proposals for the same image (e.g. another prompt)."""
        if tuple(other.image_size) != tuple(self.image_size):
            raise ValueError(f"image sizes differ: {self.image_size} vs {other.image_size}")
        self.proposals.extend(other.proposals)
        self.rejected.extend(other.rejected)


def normalize_output(output: dict, image_size: tuple[int, int], prompt: str | None = None,
                          mask_threshold: float = 0.5, device: torch.device | str | None = "cpu") -> ProposalResult:
    """ProposalResult from a generator's raw masks / boxes / scores (e.g. what
    Sam3Processor.set_text_prompt returns).

    Args:
        output: {"masks": (N, 1, H, W) or (N, H, W) bool or probabilities,
                 "boxes": (N, 4) xyxy in image pixels, "scores": (N,)}.
        image_size: (H, W) of the image given to the model; masks of another size
            are resized to it (nearest).
        prompt: the text prompt the proposals answer.
        mask_threshold: binarizes probability masks.
        device: where the proposals are kept (None = leave them where they are);
            "cpu" frees GPU memory between tiles.
    Returns:
        proposals sorted by descending score, boxes clipped to the image.
    """
    height, width = image_size
    masks = torch.as_tensor(output["masks"])
    boxes = torch.as_tensor(output["boxes"]).float().reshape(-1, 4)
    scores = torch.as_tensor(output["scores"]).float().reshape(-1)
    if masks.ndim == 4:
        if masks.shape[1] != 1:
            raise ValueError(f"expected one mask per proposal, got masks of shape {tuple(masks.shape)}")
        masks = masks[:, 0]
    if masks.ndim != 3 or not len(masks) == len(boxes) == len(scores):
        raise ValueError(f"inconsistent proposal output: masks {tuple(masks.shape)}, "
                         f"boxes {tuple(boxes.shape)}, scores {tuple(scores.shape)}")
    if len(masks) == 0:
        return ProposalResult([], (height, width))

    if masks.dtype != torch.bool:
        masks = masks.float() > mask_threshold
    if tuple(masks.shape[-2:]) != (height, width):
        masks = F.interpolate(masks[:, None].float(), size=(height, width), mode="nearest")[:, 0].bool()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    if device is not None:
        masks, boxes, scores = masks.to(device), boxes.to(device), scores.to(device)

    order = scores.argsort(descending=True)
    proposals = [InstanceProposal(boxes[i], masks[i], float(scores[i]), prompt) for i in order.tolist()]
    return ProposalResult(proposals, (height, width))
