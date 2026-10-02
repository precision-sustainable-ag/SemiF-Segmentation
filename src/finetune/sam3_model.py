"""SAM 3 for detection fine-tuning: the trainable model, and the checkpoint the
proposal generator loads afterwards.

Fine-tuning trains the detector only (sam3.train's box-only setup: no
segmentation head, box + classification losses). Parameters matching
sam3_finetune.train.freeze (unix patterns, e.g. "backbone.*") keep their SAM 3
values. The trainer's checkpoint then holds the detector without its mask head,
under keys sam3's own loader skips (it only reads "detector.*"), so `export`
merges it into the base checkpoint, mask head included, in sam3.pt's layout:
proposals.checkpoint=<that file> uses it like facebook/sam3.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def base_checkpoint_path(checkpoint: str | None) -> str:
    """The sam3.pt to start from: proposals.checkpoint, or facebook/sam3 (Hugging Face cache) for "sam3"."""
    if checkpoint in (None, "sam3"):
        from sam3.model_builder import download_ckpt_from_hf

        return download_ckpt_from_hf(version="sam3")
    return str(checkpoint)


def frozen_names(names, patterns) -> set[str]:
    return {n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns)}


def build_trainable_model(checkpoint_path: str, freeze: list[str], enable_segmentation: bool = False, **kwargs):
    """sam3's image model from checkpoint_path (a sam3.pt), on the CPU in train
    mode (the trainer moves it), with the `freeze` parameters not trained.
    Named in the trainer config as its model `_target_`."""
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device="cpu", eval_mode=False, checkpoint_path=checkpoint_path,
                                   load_from_HF=False, enable_segmentation=enable_segmentation, **kwargs)
    names = [n for n, _ in model.named_parameters()]
    frozen = frozen_names(names, freeze)
    for name, param in model.named_parameters():
        param.requires_grad_(name not in frozen)
    for part in ("vision_backbone", "language_backbone"):
        module = getattr(model.backbone, part)
        if not any(p.requires_grad for p in module.parameters()):
            # Frozen: no activations kept for backprop. Also required for the vision
            # trunk, whose MLPs use a fused kernel that refuses to run with grad enabled.
            module.forward = torch.no_grad()(module.forward)
        elif part == "vision_backbone":
            _allow_vit_grad()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("SAM 3: training %.1fM of %.1fM parameters (frozen: %s)", trainable / 1e6, total / 1e6, list(freeze))
    if not trainable:
        raise ValueError(f"sam3_finetune.train.freeze={list(freeze)} freezes every parameter")
    return model


def _allow_vit_grad() -> None:
    """Let SAM 3's ViT MLPs run with grad enabled (training the vision trunk): the
    standard fc1 + activation when grad is on, sam3's fused kernel otherwise."""
    from sam3.model import vitdet

    if getattr(vitdet.Mlp.forward, "_allows_grad", False):
        return
    fused = vitdet.Mlp.forward

    def forward(self, x):
        if not torch.is_grad_enabled():
            return fused(self, x)
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))

    forward._allows_grad = True
    vitdet.Mlp.forward = forward


DELTA_FORMAT = "sam3_detector_delta"


def export_checkpoint(base: str | None, trained_path: str | Path, out_path: str | Path) -> dict:
    """The trained detector weights (as the trainer saved them: frozen ones
    excluded), with the base checkpoint they apply to (proposals.checkpoint:
    "sam3" or a sam3.pt path), so the file stays ~0.4 GB instead of a full
    3.4 GB sam3.pt. load_sam3 rebuilds the base and puts them in."""
    trained = torch.load(str(trained_path), map_location="cpu", weights_only=False)
    trained = trained.get("model", trained)
    if not trained:
        raise ValueError(f"{trained_path} holds no weights")
    base = "sam3" if base in (None, "sam3") else str(Path(base).resolve())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": DELTA_FORMAT, "base": base, "detector": trained}, out_path)
    log.info("Exported %s: %d trained detector tensors on top of %s", out_path, len(trained), base)
    return {"tensors": len(trained), "base": base}


def read_delta(path: str | Path) -> dict | None:
    """The export_checkpoint dict at `path`, or None if it's a full sam3.pt
    (memory-mapped, so a full checkpoint isn't read just to find out)."""
    obj = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    return obj if isinstance(obj, dict) and obj.get("format") == DELTA_FORMAT else None


def apply_delta(model, delta: dict) -> None:
    """Load delta's trained weights into a built SAM 3 image model, all or nothing."""
    state = model.state_dict()
    unknown = [k for k in delta["detector"] if k not in state]
    if unknown:
        raise ValueError(f"{len(unknown)} fine-tuned weights aren't in the model, e.g. {unknown[:5]}")
    for name, value in delta["detector"].items():
        if state[name].shape != value.shape:
            raise ValueError(f"{name}: fine-tuned shape {tuple(value.shape)} != model {tuple(state[name].shape)}")
    model.load_state_dict(delta["detector"], strict=False)
