"""SAM 3 (facebookresearch/sam3): text-prompted instance proposals.

Only this module imports the sam3 package (lazily, in load_sam3), and only
through Sam3Processor's set_image / set_text_prompt.

Sam3Processor resizes every image to 1008 x 1008 and returns at most 200
detections per prompt, so large or dense images belong in predict_tiled
(tiles near 1008 px), not predict.
"""

from __future__ import annotations

import logging

import torch

from src.models.proposals.base import ProposalGenerator, common_kwargs, to_dict, to_pil
from src.models.proposals.structures import ProposalResult, normalize_output
from src.utils.instance_annotations import AnnotationSource

log = logging.getLogger(__name__)

SAM3_INSTALL_HINT = ("SAM 3 is an optional dependency. Install it with `uv sync --extra sam3` "
                     "(see README, 'SAM 3 pseudo-instances').")


def load_sam3(checkpoint: str | None = "sam3", device: str = "cuda", resolution: int = 1008,
              confidence_threshold: float = 0.5, compile: bool = False):
    """(model, Sam3Processor) from the sam3 package.

    Args:
        checkpoint: "sam3" downloads facebook/sam3 from Hugging Face (gated:
            request access, then `hf auth login`); a path loads that sam3.pt, or
            a mode=sam3_finetune export (its base, then the fine-tuned detector);
            "random" builds the architecture untrained (plumbing tests only).
    """
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        raise ImportError(SAM3_INSTALL_HINT) from e
    if checkpoint in (None, "sam3"):
        model = build_sam3_image_model(device=device, compile=compile)
    elif checkpoint == "random":
        log.warning("SAM 3 built with random weights (checkpoint=random): proposals are meaningless")
        model = build_sam3_image_model(device=device, load_from_HF=False, compile=compile)
    else:
        from src.finetune.sam3_model import apply_delta, read_delta

        delta = read_delta(checkpoint)
        if delta is not None:  # mode=sam3_finetune's export: its base, plus the fine-tuned detector
            log.info("SAM 3 fine-tuned checkpoint %s (base %s)", checkpoint, delta["base"])
            model, _ = load_sam3(delta["base"], "cpu", resolution, confidence_threshold, compile=False)
            apply_delta(model, delta)
            model = model.to(device)
        else:
            model = build_sam3_image_model(device=device, checkpoint_path=str(checkpoint), load_from_HF=False,
                                           compile=compile)
    processor = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence_threshold)
    return model, processor


class SAM3ProposalGenerator(ProposalGenerator):
    name = "sam3"
    source = AnnotationSource.SAM3

    def __init__(self, model, processor, prompts, score_threshold: float = 0.5, device: str = "cuda", **kwargs):
        """
        Args:
            model, processor: from load_sam3 (processor may be any object with
                Sam3Processor's set_image / set_text_prompt, e.g. a test fake).
            prompts: text prompts, each run on the image ("plant", "weed", ...).
            kwargs: see ProposalGenerator.
        """
        super().__init__(score_threshold=score_threshold, device=device, **kwargs)
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("at least one prompt is needed")
        self.model = model
        self.processor = processor
        self.prompts = list(prompts)
        if hasattr(processor, "confidence_threshold"):
            processor.confidence_threshold = self.score_threshold  # drop low scores before masks are upsampled

    @classmethod
    def from_config(cls, cfg) -> "SAM3ProposalGenerator":
        """From conf/proposals/sam3.yaml (a DictConfig or dict); loads SAM 3."""
        cfg = to_dict(cfg)
        model, processor = load_sam3(cfg.get("checkpoint", "sam3"), cfg.get("device", "cuda"),
                                     cfg.get("resolution", 1008), cfg.get("score_threshold", 0.5),
                                     cfg.get("compile", False))
        return cls.from_components(model, processor, cfg)

    @classmethod
    def from_components(cls, model, processor, cfg) -> "SAM3ProposalGenerator":
        """From an already loaded (or fake) model/processor and the config."""
        cfg = to_dict(cfg)
        return cls(model, processor, cfg.get("prompts", ["plant"]), **common_kwargs(cfg))

    @torch.inference_mode()
    def predict_raw(self, image, semantic_mask=None, prompts: list[str] | None = None) -> ProposalResult:
        """Every prompt's proposals above score_threshold (semantic_mask unused)."""
        pil = to_pil(image)
        image_size = (pil.height, pil.width)
        result = ProposalResult([], image_size)
        with self._autocast():
            state = self.processor.set_image(pil)
            for prompt in prompts or self.prompts:
                output = self.processor.set_text_prompt(prompt=prompt, state=state)
                result.extend(normalize_output(output, image_size, prompt, self.mask_threshold, self.output_device))
        del state
        return result
