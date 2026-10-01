"""Filtered proposals -> Mask R-CNN / PointRend training targets.

SAM 3 supplies what the semantic labels lack -- which plant pixels belong
together -- while the semantic mask stays the authority on where plants are
and what class they are:
- clip_to_semantic: each instance keeps only its pixels on semantic foreground,
  so instance boundaries follow the (human) semantic labels;
- exclusive_masks: overlapping pixels go to the higher-scored instance, as a
  pixel belongs to one plant (and the id-map file format needs it);
- the class is the majority semantic class under the instance, falling back
  to the prompt's class (prompt_labels) when there is no semantic mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import torch

from src.models.proposals.structures import ProposalResult
from src.utils.instance_annotations import AnnotationSource, empty_target, mask_box


@dataclass
class PseudoLabelConfig:
    """conf/proposals pseudo_labels (only the options that shape targets)."""
    clip_to_semantic: bool = True
    exclusive_masks: bool = True
    min_area: int = 0                       # px an instance must keep after clipping / exclusivity
    default_label: int = 1                  # class of prompts not in prompt_labels, without a semantic mask
    prompt_labels: dict[str, int] = field(default_factory=dict)
    ignore_value: int | None = None         # semantic value that is neither plant nor background

    @classmethod
    def from_dict(cls, d: dict | None) -> "PseudoLabelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})


def majority_class(mask: torch.Tensor, semantic: torch.Tensor, ignore_value: int | None = None) -> int | None:
    """Most frequent semantic class > 0 (other than ignore_value) under mask, or None."""
    values = semantic[mask]
    values = values[values > 0]
    if ignore_value is not None:
        values = values[values != ignore_value]
    if values.numel() == 0:
        return None
    return int(torch.bincount(values.long()).argmax())


def create_pseudo_target(result: ProposalResult, semantic_mask=None, cfg: PseudoLabelConfig | None = None,
                         source: AnnotationSource = AnnotationSource.SAM3) -> dict:
    """Torchvision instance target from (already filtered) proposals.

    Returns {"boxes" (N, 4) float32 xyxy pixel edges, "labels" (N,) int64,
    "masks" (N, H, W) uint8, "scores" (N,) float32 (generator score),
    "annotation_source" list of N str, "prompts" list of N str|None,
    "class_source" list of N "semantic"|"prompt"}, in descending score order.
    Model code uses boxes/labels/masks; the rest is provenance.
    """
    cfg = cfg or PseudoLabelConfig()
    height, width = result.image_size
    semantic = None
    if semantic_mask is not None:
        semantic = semantic_mask if isinstance(semantic_mask, torch.Tensor) else torch.from_numpy(np.asarray(semantic_mask))
        semantic = semantic.cpu().long()
        if tuple(semantic.shape) != (height, width):
            raise ValueError(f"semantic mask {tuple(semantic.shape)} does not match the image {(height, width)}")
        foreground = semantic > 0
        if cfg.ignore_value is not None:
            foreground &= semantic != cfg.ignore_value

    claimed = torch.zeros((height, width), dtype=torch.bool)
    masks, boxes, labels, scores, prompts, class_sources = [], [], [], [], [], []
    for p in sorted(result.proposals, key=lambda p: -p.score):
        mask = p.mask.cpu().bool().clone()
        if semantic is not None and cfg.clip_to_semantic:
            mask &= foreground
        if cfg.exclusive_masks:
            mask &= ~claimed
        box = mask_box(mask)
        if box is None or int(mask.sum()) < max(cfg.min_area, 1):
            continue
        label = majority_class(mask, semantic, cfg.ignore_value) if semantic is not None else None
        class_sources.append("prompt" if label is None else "semantic")
        if label is None:
            label = int(cfg.prompt_labels.get(p.prompt, cfg.default_label))
        claimed |= mask
        masks.append(mask)
        boxes.append(box)
        labels.append(label)
        scores.append(p.score)
        prompts.append(p.prompt)

    if not masks:
        return {**empty_target(height, width), "prompts": [], "class_source": []}
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.stack(masks).to(torch.uint8),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "annotation_source": [AnnotationSource(source).value] * len(masks),
        "prompts": prompts,
        "class_source": class_sources,
    }


def target_stats(target: dict, semantic_mask=None, ignore_value: int | None = None) -> dict[str, float]:
    """Quality summary of one pseudo target: instance count, and -- given the
    semantic mask -- the share of plant pixels no instance claims (plants the
    generator missed or the filters rejected)."""
    stats = {"instances": int(len(target["boxes"]))}
    if semantic_mask is not None:
        semantic = torch.as_tensor(np.asarray(semantic_mask)).long()
        foreground = semantic > 0
        if ignore_value is not None:
            foreground &= semantic != ignore_value
        fg = int(foreground.sum())
        covered = target["masks"].bool().any(0) & foreground if len(target["masks"]) else torch.zeros_like(foreground)
        stats["foreground_px"] = fg
        stats["unassigned_foreground_fraction"] = (fg - int(covered.sum())) / fg if fg else 0.0
    return stats


def clip_instances_to_semantic(inst, semantic_mask, ignore_value: int | None = None, min_area: int = 0):
    """Full-image Instances (src.inferencing.tiled_instances, e.g. from
    predict_tiled) clipped to the semantic foreground as create_pseudo_target
    does: each mask keeps its plant pixels, its box shrinks to them, its label
    becomes their majority class; instances left with < min_area px are dropped."""
    from src.inferencing.tiled_instances import Instances

    semantic = np.asarray(semantic_mask)
    out = Instances(inst.height, inst.width)
    boxes, labels, scores = [], [], []
    for i in range(len(inst)):
        x0, y0, x1, y1 = (int(v) for v in inst.boxes[i])
        sem = semantic[y0:y1, x0:x1]
        foreground = sem > 0 if ignore_value is None else (sem > 0) & (sem != ignore_value)
        mask = inst.masks[i] & foreground
        box = mask_box(mask)
        if box is None or int(mask.sum()) < max(min_area, 1):
            continue
        bx0, by0, bx1, by1 = box
        values = sem[mask]
        values = values[values > 0] if ignore_value is None else values[(values > 0) & (values != ignore_value)]
        out.masks.append(mask[by0:by1, bx0:bx1])
        boxes.append((x0 + bx0, y0 + by0, x0 + bx1, y0 + by1))
        labels.append(int(np.bincount(values).argmax()))
        scores.append(float(inst.scores[i]))
    if out.masks:
        out.boxes = np.array(boxes, np.int64)
        out.labels = np.array(labels, np.int64)
        out.scores = np.array(scores, np.float32)
    return out
