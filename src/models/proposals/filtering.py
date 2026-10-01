"""Proposal filtering: score and area, agreement with a semantic mask, and
mask-level duplicate / containment resolution.

Every filter takes and returns a ProposalResult; dropped proposals go to
result.rejected with the reason ("score", "min_area", "semantic_overlap",
"duplicate_of:<k>", ...), so a tile's decisions can be inspected afterwards.

Semantic metrics (per proposal mask m, foreground F = semantic > 0 minus
ignore_value, background B = semantic == 0; "box" is m's tight box):
  area                 |m|
  score                generator confidence
  semantic_overlap     |m & F| / |m|   foreground purity: share of the proposal on labeled plant
  background_fraction  |m & B| / |m|   share on labeled background (= 1 - semantic_overlap
                                       unless pixels carry ignore_value)
  foreground_iou       IoU of m with F inside the box -- low when the proposal
                       misses a plant it overlaps or spills beyond it
  foreground_coverage  |m & F| / |F in box|  containment: share of the box's plant pixels the proposal covers
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from src.models.proposals.structures import InstanceProposal, ProposalResult
from src.utils.instance_annotations import mask_box

CONTAINMENT_POLICIES = ("keep_higher_score", "keep_larger", "keep_smaller", "keep_both")


@dataclass
class FilterConfig:
    """conf/proposals filtering (null disables a limit)."""
    min_area: int = 0
    max_area: int | None = None
    min_semantic_overlap: float | None = None
    max_background_fraction: float | None = None
    min_foreground_iou: float | None = None
    min_foreground_coverage: float | None = None
    ignore_value: int | None = None           # semantic value that is neither plant nor background
    mask_nms_threshold: float | None = 0.7    # mask IoU at which the lower-scored proposal is a duplicate
    containment_threshold: float | None = 0.9  # share of a mask inside another at which one contains the other
    containment_policy: str = "keep_higher_score"

    @classmethod
    def from_dict(cls, d: dict | None) -> "FilterConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown filtering options {sorted(unknown)}; expected {sorted(known)}")
        cfg = cls(**d)
        if cfg.containment_policy not in CONTAINMENT_POLICIES:
            raise ValueError(f"containment_policy must be one of {CONTAINMENT_POLICIES}, got {cfg.containment_policy!r}")
        return cfg


def _split(result: ProposalResult, reasons: list[str | None]) -> ProposalResult:
    """Keep proposals whose reason is None; reject the rest with their reason."""
    kept = [p for p, r in zip(result.proposals, reasons) if r is None]
    rejected = result.rejected + [(p, r) for p, r in zip(result.proposals, reasons) if r is not None]
    return ProposalResult(kept, result.image_size, rejected)


def filter_by_score_area(result: ProposalResult, score_threshold: float | None = None, min_area: int = 0,
                         max_area: int | None = None) -> ProposalResult:
    """Drop proposals below score_threshold, with empty masks, or outside [min_area, max_area] px."""
    reasons = []
    for p in result.proposals:
        area = p.area()
        p.metrics.update(score=p.score, area=area)
        if score_threshold is not None and p.score < score_threshold:
            reasons.append("score")
        elif area == 0:
            reasons.append("empty_mask")
        elif area < min_area:
            reasons.append("min_area")
        elif max_area is not None and area > max_area:
            reasons.append("max_area")
        else:
            reasons.append(None)
    return _split(result, reasons)


def semantic_metrics(mask: torch.Tensor, semantic: torch.Tensor, ignore_value: int | None = None) -> dict[str, float]:
    """The metrics of the module docstring for one (H, W) bool mask against a (H, W) semantic mask."""
    box = mask_box(mask)
    if box is None:
        return {"area": 0, "semantic_overlap": 0.0, "background_fraction": 0.0,
                "foreground_iou": 0.0, "foreground_coverage": 0.0}
    x0, y0, x1, y1 = box
    m = mask[y0:y1, x0:x1]
    s = semantic[y0:y1, x0:x1]
    background = s == 0
    foreground = ~background if ignore_value is None else ~background & (s != ignore_value)
    area = int(m.sum())
    inter = int((m & foreground).sum())
    fg_in_box = int(foreground.sum())
    return {
        "area": area,
        "semantic_overlap": inter / area,
        "background_fraction": int((m & background).sum()) / area,
        "foreground_iou": inter / (area + fg_in_box - inter),
        "foreground_coverage": inter / fg_in_box if fg_in_box else 0.0,
    }


def filter_by_semantics(result: ProposalResult, semantic_mask, cfg: FilterConfig) -> ProposalResult:
    """Record semantic metrics on each proposal and drop those that disagree with the semantic mask."""
    semantic = torch.as_tensor(np.asarray(semantic_mask) if not isinstance(semantic_mask, torch.Tensor) else semantic_mask)
    if tuple(semantic.shape) != tuple(result.image_size):
        raise ValueError(f"semantic mask {tuple(semantic.shape)} does not match the image {tuple(result.image_size)}")
    limits = (  # metric, threshold, keep if metric >= threshold (else <=)
        ("semantic_overlap", cfg.min_semantic_overlap, True),
        ("background_fraction", cfg.max_background_fraction, False),
        ("foreground_iou", cfg.min_foreground_iou, True),
        ("foreground_coverage", cfg.min_foreground_coverage, True),
    )
    if result.proposals:
        semantic = semantic.to(result.proposals[0].mask.device)
    reasons = []
    for p in result.proposals:
        p.metrics.update(semantic_metrics(p.mask, semantic, cfg.ignore_value))
        reason = None
        for name, threshold, at_least in limits:
            if threshold is None:
                continue
            if (p.metrics[name] < threshold) if at_least else (p.metrics[name] > threshold):
                reason = name
                break
        reasons.append(reason)
    return _split(result, reasons)


def pairwise_intersections(masks: list[torch.Tensor], chunk_elements: int = 2**26) -> torch.Tensor:
    """(N, N) pixel intersections of (H, W) bool masks, as a matrix product of
    the flattened masks, a chunk of pixels at a time (on the masks' device).
    Exact: products of 0/1 values, summed in float32 over < 2**24 pixels per chunk."""
    n = len(masks)
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.long)
    flat = [m.reshape(-1) for m in masks]
    step = max(1, min(chunk_elements // n, 2**24))
    inter = torch.zeros((n, n), dtype=torch.float64, device=flat[0].device)
    for start in range(0, flat[0].numel(), step):
        chunk = torch.stack([f[start:start + step] for f in flat]).float()
        inter += (chunk @ chunk.T).double()
    return inter.round().long().cpu()


def mask_iou(masks: list[torch.Tensor]) -> torch.Tensor:
    """(N, N) mask IoU."""
    inter = pairwise_intersections(masks).float()
    area = inter.diagonal()
    return inter / (area[:, None] + area[None, :] - inter).clamp(min=1)


def resolve_overlaps(result: ProposalResult, nms_threshold: float | None = 0.7,
                     containment_threshold: float | None = 0.9,
                     containment_policy: str = "keep_higher_score") -> ProposalResult:
    """Greedy mask NMS with containment handling, in descending score order.

    A proposal is a duplicate of a kept one if their mask IoU >= nms_threshold
    (the kept, higher-scored one wins). Otherwise, if either mask lies
    >= containment_threshold inside the other (e.g. a leaf inside its plant,
    or one proposal covering two touching plants), containment_policy decides:
      keep_higher_score  the kept (higher-scored) proposal stays
      keep_larger        the outer one stays (whole plants over parts)
      keep_smaller       the inner ones stay (separate plants over a merged proposal)
      keep_both          both stay
    Proposals are compared by mask pixels, not boxes.
    """
    if containment_policy not in CONTAINMENT_POLICIES:
        raise ValueError(f"containment_policy must be one of {CONTAINMENT_POLICIES}, got {containment_policy!r}")
    props = sorted(result.proposals, key=lambda p: -p.score)
    if len(props) < 2:
        return ProposalResult(props, result.image_size, list(result.rejected))
    inter = pairwise_intersections([p.mask for p in props]).float()
    area = inter.diagonal().clamp(min=1)
    iou = inter / (area[:, None] + area[None, :] - inter)
    inside = inter / area[:, None]  # inside[i, j]: share of i within j

    kept: list[int] = []
    reasons: dict[int, str] = {}
    for c in range(len(props)):
        dup = next((k for k in kept if nms_threshold is not None and iou[c, k] >= nms_threshold), None)
        if dup is not None:
            reasons[c] = f"duplicate_of:{dup}"
            continue
        contained = [] if containment_threshold is None or containment_policy == "keep_both" else [
            k for k in kept if max(inside[c, k], inside[k, c]) >= containment_threshold]
        if not contained:
            kept.append(c)
            continue
        if containment_policy == "keep_higher_score":
            reasons[c] = f"contained_with:{contained[0]}"
            continue
        wins = (lambda k: area[c] > area[k]) if containment_policy == "keep_larger" else (lambda k: area[c] < area[k])
        if all(wins(k) for k in contained):
            for k in contained:
                kept.remove(k)
                reasons[k] = f"contained_with:{c}"
            kept.append(c)
        else:
            reasons[c] = f"contained_with:{next(k for k in contained if not wins(k))}"
    for i, p in enumerate(props):
        p.metrics["rank"] = i  # the index the reasons refer to
    keep = set(kept)
    return ProposalResult([props[i] for i in sorted(keep)], result.image_size,
                          list(result.rejected) + [(props[i], reasons[i]) for i in range(len(props)) if i not in keep])


def apply_filters(result: ProposalResult, cfg: FilterConfig, score_threshold: float | None = None,
                  semantic_mask=None) -> ProposalResult:
    """Score/area, then semantic agreement (if a semantic mask is given), then
    overlap resolution -- semantic filtering first, so a high-scoring proposal
    over soil can't suppress a good one."""
    result = filter_by_score_area(result, score_threshold, cfg.min_area, cfg.max_area)
    if semantic_mask is not None:
        result = filter_by_semantics(result, semantic_mask, cfg)
    return resolve_overlaps(result, cfg.mask_nms_threshold, cfg.containment_threshold, cfg.containment_policy)
