"""What every proposal generator (SAM 3, ...) shares: filtering,
pseudo-targets, and tiled prediction on top of one model call, predict_raw.

    generator = build_proposal_generator(cfg.proposals.name, cfg.proposals)   # src.models
    result = generator.predict(image)                                           # ProposalResult
    target = generator.create_pseudo_target(image, semantic_mask=mask)

Generators are teachers for the Mask R-CNN / PointRend models, not part of
them: each imports its model's package lazily, and returns only
src.models.proposals.structures types.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from PIL import Image

from src.models.proposals.filtering import FilterConfig, apply_filters
from src.models.proposals.pseudo_labels import PseudoLabelConfig, create_pseudo_target
from src.models.proposals.structures import ProposalResult
from src.utils.instance_annotations import AnnotationSource, mask_box


def to_pil(image) -> Image.Image:
    """RGB PIL image from a PIL image, (H, W, 3) uint8 array, or (3, H, W)
    tensor (uint8, or float in [0, 1] -- not a mean/std-normalized one)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"expected a (3, H, W) tensor, got {tuple(image.shape)}")
        if image.is_floating_point():
            if image.min() < 0 or image.max() > 1:
                raise ValueError("float tensors must be in [0, 1]; the generators do their own normalization")
            image = (image * 255).round()
        image = image.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {image.shape} {image.dtype}")
    return Image.fromarray(image)


def model_box(bbox, height: int, width: int) -> tuple[int, int, int, int] | None:
    """A predicted xyxy box as whole pixel edges (outward), clipped to the image; None if empty."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def to_dict(cfg) -> dict:
    """A plain dict from a DictConfig, dict or None."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)


def common_kwargs(cfg: dict) -> dict:
    """ProposalGenerator.__init__ arguments from conf/proposals/common.yaml keys."""
    filtering = dict(cfg.get("filtering") or {})
    pseudo = {**(cfg.get("pseudo_labels") or {}), "min_area": filtering.get("min_area", 0),
              "prompt_labels": cfg.get("prompt_labels") or {}, "ignore_value": filtering.get("ignore_value")}
    return {
        "score_threshold": cfg.get("score_threshold", 0.5),
        "device": cfg.get("device", "cuda"),
        "mask_threshold": cfg.get("mask_threshold", 0.5),
        "filtering": filtering,
        "pseudo_labels": pseudo,
        "autocast_dtype": cfg.get("autocast_dtype", "bfloat16"),
        "output_device": cfg.get("output_device", "cpu"),
    }


class ProposalGenerator:
    """Subclasses implement predict_raw (and set name / source)."""

    name = "base"
    source = AnnotationSource.HUMAN

    def __init__(
        self,
        score_threshold: float = 0.5,
        device: str = "cuda",
        mask_threshold: float = 0.5,
        filtering: FilterConfig | dict | None = None,
        pseudo_labels: PseudoLabelConfig | dict | None = None,
        autocast_dtype: torch.dtype | str | None = "bfloat16",
        output_device: str = "cpu",
    ):
        """
        Args:
            score_threshold: model confidence below which proposals are dropped.
            filtering: FilterConfig (or its dict) for predict / create_pseudo_target.
            pseudo_labels: PseudoLabelConfig (or its dict) for create_pseudo_target.
            autocast_dtype: CUDA autocast dtype for the model, None = off.
            output_device: where proposal masks are kept ("cpu" frees GPU memory).
        """
        self.score_threshold = float(score_threshold)
        self.device = device
        self.mask_threshold = mask_threshold
        self.filtering = filtering if isinstance(filtering, FilterConfig) else FilterConfig.from_dict(filtering)
        self.pseudo_labels = (pseudo_labels if isinstance(pseudo_labels, PseudoLabelConfig)
                              else PseudoLabelConfig.from_dict(pseudo_labels))
        if isinstance(autocast_dtype, str):
            autocast_dtype = getattr(torch, autocast_dtype)
        self.autocast_dtype = autocast_dtype
        self.output_device = output_device

    def _autocast(self):
        if self.autocast_dtype is None or not str(self.device).startswith("cuda"):
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def predict_raw(self, image, semantic_mask=None, **kwargs) -> ProposalResult:
        """The model's proposals above score_threshold, unfiltered otherwise.
        semantic_mask is a hint a generator may use (SAM 3 ignores it)."""
        raise NotImplementedError

    def predict(self, image, semantic_mask=None, apply_filtering: bool = True, **kwargs) -> ProposalResult:
        """Proposals for one image (PIL, (H, W, 3) uint8 RGB, or (3, H, W) tensor).

        With apply_filtering, the configured filters run (score/area, semantic
        agreement when semantic_mask -- (H, W) class indices -- is given, mask
        NMS / containment); result.rejected lists what they dropped and why.
        """
        result = self.predict_raw(image, semantic_mask, **kwargs)
        if not apply_filtering:
            return result
        return apply_filters(result, self.filtering, self.score_threshold, semantic_mask)

    def create_pseudo_target(self, image, semantic_mask=None, return_result: bool = False):
        """Torchvision instance target (src.models.proposals.pseudo_labels.create_pseudo_target)
        for one image, annotation_source = this generator; with return_result also the
        filtered ProposalResult."""
        result = self.predict(image, semantic_mask)
        target = create_pseudo_target(result, semantic_mask, self.pseudo_labels, self.source)
        return (target, result) if return_result else target

    def predict_tiled(self, image, semantic_mask=None, tile_size: int = 1024, overlap: int = 256,
                      merge_iou: float = 0.5, box_from: str = "mask"):
        """Full-image proposals for images too large (or dense) for one pass:
        filtered proposals of overlapping tiles, moved to image coordinates and
        merged across tiles as src.inferencing.tiled_instances does for the
        detectors. Returns its Instances (masks as crops of their boxes, labels
        from prompt_labels), usable with paint_semantic, coco_ap, etc.

        box_from: "mask", each box is its mask's extent; "model", the box the
        model predicted (what detection fine-tuning trains, src/finetune)."""
        if box_from not in ("mask", "model"):
            raise ValueError(f"box_from must be mask or model, not {box_from!r}")
        from src.inferencing.tiled_instances import Instances, merge_across_tiles, tile_starts

        array = np.asarray(to_pil(image))
        semantic = None if semantic_mask is None else np.asarray(semantic_mask)
        height, width = array.shape[:2]
        rects = [(x, y, min(x + tile_size, width), min(y + tile_size, height))
                 for y in tile_starts(height, tile_size, overlap) for x in tile_starts(width, tile_size, overlap)]
        raw = Instances(height, width)
        boxes, labels, scores, tile_ids = [], [], [], []
        for t, (x0, y0, x1, y1) in enumerate(rects):
            crop_sem = None if semantic is None else semantic[y0:y1, x0:x1]
            for p in self.predict(array[y0:y1, x0:x1], crop_sem).proposals:
                box = mask_box(p.mask) if box_from == "mask" else model_box(p.bbox, y1 - y0, x1 - x0)
                if box is None:
                    continue
                bx0, by0, bx1, by1 = box
                raw.masks.append(p.mask[by0:by1, bx0:bx1].cpu().numpy())
                boxes.append((x0 + bx0, y0 + by0, x0 + bx1, y0 + by1))
                labels.append(int(self.pseudo_labels.prompt_labels.get(p.prompt, self.pseudo_labels.default_label)))
                scores.append(p.score)
                tile_ids.append(t)
        if not raw.masks:
            return raw
        raw.boxes = np.array(boxes, np.int64)
        raw.labels = np.array(labels, np.int64)
        raw.scores = np.array(scores, np.float32)
        return merge_across_tiles(raw, np.array(tile_ids), rects, merge_iou)
