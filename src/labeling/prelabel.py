"""Model-assisted pre-labels: a binary vegetation mask per image, uploaded to
CVAT as an editable mask so annotators correct instead of painting from scratch.

Works with any binary SMP checkpoint: a Lightning `.ckpt` or a `.pth` whose
state_dict keys are prefixed "model." (both what this repo's train mode writes
and what Field-SegmentationTraining writes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


def load_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")
    stripped = {}
    for key, value in state.items():
        for prefix in ("model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        stripped[key] = value
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        # A partially loaded model would silently produce garbage pre-labels.
        raise RuntimeError(
            f"{len(missing)} weights missing loading {checkpoint} (check label.prelabel.arch/encoder): {missing[:5]}"
        )
    if unexpected:
        log.warning("Ignoring %d unexpected keys in %s: %s", len(unexpected), checkpoint, unexpected[:5])


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering [0, length); the last tile is aligned to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


class Prelabeler:
    def __init__(self, pcfg):
        import segmentation_models_pytorch as smp

        if pcfg.tile_size % 32:
            raise ValueError("label.prelabel.tile_size must be a multiple of 32")
        if not 0 <= pcfg.overlap < pcfg.tile_size:
            raise ValueError("label.prelabel.overlap must be in [0, tile_size)")

        wants_cuda = str(pcfg.device).startswith("cuda")
        self.device = torch.device(pcfg.device if (not wants_cuda or torch.cuda.is_available()) else "cpu")
        if wants_cuda and self.device.type == "cpu":
            log.warning("CUDA requested for pre-labels but unavailable; using CPU (slow).")

        self.model = smp.create_model(
            pcfg.arch, encoder_name=pcfg.encoder, encoder_weights=None, in_channels=3, classes=1
        )
        load_weights(self.model, pcfg.checkpoint)
        self.model.to(self.device).eval()
        log.info("Pre-label model: %s/%s from %s on %s", pcfg.arch, pcfg.encoder, pcfg.checkpoint, self.device)

        self.normalize = bool(pcfg.normalize)
        self.mean = np.asarray(pcfg.mean, dtype=np.float32)
        self.std = np.asarray(pcfg.std, dtype=np.float32)
        self.input_scale = float(pcfg.input_scale)
        self.tile = int(pcfg.tile_size)
        self.stride = self.tile - int(pcfg.overlap)
        self.batch_size = int(pcfg.batch_size)
        self.threshold = float(pcfg.threshold)
        self.min_area_px = int(pcfg.min_area_px)
        hann = np.hanning(self.tile + 2)[1:-1]  # strictly positive, so edge tiles still count
        self.window = np.outer(hann, hann).astype(np.float32)

    def predict(self, image_bgr: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        """Boolean vegetation mask of size out_size (width, height)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.input_scale != 1.0:
            h, w = rgb.shape[:2]
            size = (max(1, round(w * self.input_scale)), max(1, round(h * self.input_scale)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

        prob = self._probabilities(rgb)
        if (prob.shape[1], prob.shape[0]) != tuple(out_size):
            prob = cv2.resize(prob, tuple(out_size), interpolation=cv2.INTER_LINEAR)
        return self._drop_small_blobs(prob >= self.threshold)

    def _probabilities(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        # Images smaller than a tile are reflect-padded up to one tile.
        pad_h, pad_w = max(0, self.tile - h), max(0, self.tile - w)
        if pad_h or pad_w:
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        H, W = rgb.shape[:2]

        acc = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        coords = [(y, x) for y in tile_starts(H, self.tile, self.stride) for x in tile_starts(W, self.tile, self.stride)]

        for i in range(0, len(coords), self.batch_size):
            batch_coords = coords[i:i + self.batch_size]
            tiles = np.stack([rgb[y:y + self.tile, x:x + self.tile] for y, x in batch_coords]).astype(np.float32) / 255.0
            if self.normalize:
                tiles = (tiles - self.mean) / self.std
            x_in = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                probs = torch.sigmoid(self.model(x_in).float())[:, 0].cpu().numpy()
            for (y, x), p in zip(batch_coords, probs):
                acc[y:y + self.tile, x:x + self.tile] += p * self.window
                weight[y:y + self.tile, x:x + self.tile] += self.window

        return (acc / np.maximum(weight, 1e-6))[:h, :w]

    def _drop_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        if self.min_area_px <= 0 or not mask.any():
            return mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_area_px
        return keep[labels]
