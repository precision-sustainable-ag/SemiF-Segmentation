"""Mask-logit uncertainty for PointRend point selection."""

from __future__ import annotations

import torch


def select_class_logits(logits: torch.Tensor, classes: torch.Tensor | None) -> torch.Tensor:
    """The mask channel of each box: `logits` is (R, C, ...), `classes` (R,).
    With a single (class-agnostic) channel, or classes=None, that channel is used."""
    if logits.shape[1] == 1 or classes is None:
        return logits[:, :1]
    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, classes.long()].unsqueeze(1)


def calculate_uncertainty(logits: torch.Tensor, classes: torch.Tensor | None = None) -> torch.Tensor:
    """-|logit| of each box's mask channel: highest (0) on the decision boundary.

    Args:
        logits: (R, C, ...) mask logits (a grid or points).
        classes: (R,) channel per box; ignored for class-agnostic logits.
    Returns:
        (R, 1, ...)
    """
    return -select_class_logits(logits, classes).abs()
