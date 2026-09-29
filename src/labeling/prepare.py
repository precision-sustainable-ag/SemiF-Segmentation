"""prepare: make the CVAT copy of each fetched image, and its pre-label.

Full frames (up to 13368x9520) are too big to annotate comfortably in the
browser, so each image is downscaled until its longest side is at most
`label.prepare.max_side`, into the run folder's images/. The pre-label mask
is written at that same size into masks/ (0/255), as AgIR-CVToolkit's
seg-infer wrote images/ and masks/ for its CVAT upload. With
`label.prelabel.source=masks` it's taken from an existing mask instead of the
model (see `existing_mask_path`).
"""

import logging
from pathlib import Path

import cv2
from omegaconf import DictConfig
from tqdm import tqdm

from src.labeling.ledger import LabelLedger
from src.labeling.run_folder import RunFolder

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> dict | None:
    lcfg = cfg.label
    source = lcfg.source
    run = RunFolder.from_cfg(cfg)
    run.images.mkdir(parents=True, exist_ok=True)

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items_for_task(lcfg.round, "fetched", "prepare", lcfg.retry_errors)
        if not items:
            log.info("No fetched images waiting to be prepared in round %s.", lcfg.round)
            return None

        pcfg = lcfg.prelabel
        use_masks = pcfg.enable and pcfg.source == "masks"
        if pcfg.enable and pcfg.source not in ("model", "masks"):
            raise ValueError(f"label.prelabel.source must be model or masks, not {pcfg.source!r}")
        if use_masks and not pcfg.masks_root:
            raise ValueError("label.prelabel.source=masks needs label.prelabel.masks_root")
        if use_masks and pcfg.missing_mask not in ("model", "empty"):
            raise ValueError(f"label.prelabel.missing_mask must be model or empty, not {pcfg.missing_mask!r}")
        if pcfg.enable:
            run.masks.mkdir(parents=True, exist_ok=True)
        prelabeler = None

        def get_prelabeler():
            nonlocal prelabeler
            if prelabeler is None:
                from src.labeling.prelabel import Prelabeler  # torch/smp only when needed

                prelabeler = Prelabeler(pcfg)
            return prelabeler

        max_side = int(lcfg.prepare.max_side)
        quality = int(lcfg.prepare.jpeg_quality)
        counts = {"prepared": 0, "prelabeled": 0, "from_masks": 0, "no_mask": 0, "errors": 0}
        for item in tqdm(items, desc="prepare"):
            image_id = item["image_id"]
            try:
                image = cv2.imread(item["native_path"], cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"could not read {item['native_path']}")
                height, width = image.shape[:2]
                scale = min(1.0, max_side / max(height, width))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else image

                cvat_path = run.images / f"{image_id}.jpg"
                if not cv2.imwrite(str(cvat_path), small, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise RuntimeError(f"could not write {cvat_path}")
                fields = {
                    "native_width": width, "native_height": height,
                    "cvat_image_path": str(cvat_path), "cvat_width": size[0], "cvat_height": size[1],
                    "scale": size[0] / width,
                }

                mask = None
                if use_masks:
                    mask_path = existing_mask_path(pcfg.masks_root, item["path"], image_id)
                    if mask_path.exists():
                        mask = load_existing_mask(mask_path, (width, height), size)
                        counts["from_masks"] += 1
                    else:
                        counts["no_mask"] += 1
                        log.warning("No mask at %s; %s", mask_path,
                                    "using the model" if pcfg.missing_mask == "model" else "starts empty in CVAT")
                        if pcfg.missing_mask == "model":
                            mask = get_prelabeler().predict(image, size)
                elif pcfg.enable:
                    mask = get_prelabeler().predict(image, size)

                if mask is not None:
                    prelabel_path = run.masks / f"{image_id}.png"
                    if not cv2.imwrite(str(prelabel_path), mask.astype("uint8") * 255):
                        raise RuntimeError(f"could not write {prelabel_path}")
                    fields["prelabel_path"] = str(prelabel_path)
                    counts["prelabeled"] += 1

                ledger.update(source, image_id, status="prepared", error=None, **fields)
                counts["prepared"] += 1
            except Exception as exc:
                log.exception("prepare failed for %s", image_id)
                ledger.update(source, image_id, status="error", error=f"prepare: {exc}")
                counts["errors"] += 1
    return counts


def existing_mask_path(masks_root: str, source_path: str, image_id: str) -> Path:
    """<masks_root>/<batch>/segmentations/<image_id>.png, where <batch> is the
    folder above images/ in the source DB path (AgIR-CVToolkit's layout)."""
    batch = Path(source_path).parent.parent.name
    return Path(masks_root) / batch / "segmentations" / f"{image_id}.png"


def load_existing_mask(path: Path, native_size: tuple[int, int], size: tuple[int, int]):
    """Existing mask -> bool vegetation mask at the CVAT size. Masks may be
    class-coded (any nonzero value is vegetation)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"could not read {path}")
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if (mask.shape[1], mask.shape[0]) != native_size:
        raise RuntimeError(f"{path.name} is {mask.shape[1]}x{mask.shape[0]}, image is {native_size[0]}x{native_size[1]}")
    mask = (mask > 0).astype("uint8")
    if (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)
