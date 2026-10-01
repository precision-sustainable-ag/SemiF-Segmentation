"""prepare: make the CVAT frames of each fetched image, and their pre-labels.

Full frames (up to 13368x9520) are too big to annotate comfortably in the
browser. By default each image is downscaled until its longest side is at most
`label.prepare.max_side`. With `label.prepare.split=true` it's instead cut, at
full resolution and without overlap, into the smallest grid of tiles that are
all at most `max_side` (2x2 for 9560x6368, 3x2 for 13368x9520), named
<image_id>_r<row>c<col>. Frames go to the run folder's images/ and their
pre-labels, at the same size, to masks/ (0/255), as AgIR-CVToolkit's seg-infer
wrote images/ and masks/ for its CVAT upload. The ledger's `tiles` table records
where each frame sits in the full image, so pull_cvat can put the masks back
together. With `label.prelabel.source=masks` the pre-label is taken from an
existing mask instead of the model (see `existing_mask_path`).
"""

import logging
import math
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
        split = bool(lcfg.prepare.split)
        counts = {"prepared": 0, "frames": 0, "prelabeled": 0, "from_masks": 0, "no_mask": 0, "errors": 0}
        for item in tqdm(items, desc="prepare"):
            image_id = item["image_id"]
            try:
                image = cv2.imread(item["native_path"], cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"could not read {item['native_path']}")
                height, width = image.shape[:2]
                if split:
                    scale, size, canvas = 1.0, (width, height), image
                    boxes = tile_grid(width, height, max_side)
                    if len(boxes) > 1:
                        boxes = [(f"{image_id}_r{r}c{c}", x, y, w, h) for (r, c, x, y, w, h) in boxes]
                    else:
                        boxes = [(image_id, 0, 0, width, height)]
                else:
                    scale = min(1.0, max_side / max(height, width))
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    canvas = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else image
                    boxes = [(image_id, 0, 0, width, height)]

                mask = None  # pre-label at `size`, the canvas the frames are cut from
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

                tiles = []
                for name, x, y, w, h in boxes:
                    # The canvas is the full image when splitting, or the whole downscaled image.
                    cy0, cx0 = (y, x) if split else (0, 0)
                    cy1, cx1 = (y + h, x + w) if split else (size[1], size[0])
                    tile = {"name": name, "x": x, "y": y, "width": w, "height": h,
                            "cvat_width": cx1 - cx0, "cvat_height": cy1 - cy0}
                    cvat_path = run.images / f"{name}.jpg"
                    if not cv2.imwrite(str(cvat_path), canvas[cy0:cy1, cx0:cx1], [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        raise RuntimeError(f"could not write {cvat_path}")
                    tile["cvat_image_path"] = str(cvat_path)
                    if mask is not None:
                        prelabel_path = run.masks / f"{name}.png"
                        if not cv2.imwrite(str(prelabel_path), mask[cy0:cy1, cx0:cx1].astype("uint8") * 255):
                            raise RuntimeError(f"could not write {prelabel_path}")
                        tile["prelabel_path"] = str(prelabel_path)
                    tiles.append(tile)
                if mask is not None:
                    counts["prelabeled"] += 1

                ledger.set_tiles(source, image_id, tiles)
                single = tiles[0] if len(tiles) == 1 else {}
                ledger.update(
                    source, image_id, status="prepared", error=None,
                    native_width=width, native_height=height, scale=scale,
                    cvat_image_path=single.get("cvat_image_path"), prelabel_path=single.get("prelabel_path"),
                    cvat_width=single.get("cvat_width"), cvat_height=single.get("cvat_height"),
                )
                counts["prepared"] += 1
                counts["frames"] += len(tiles)
            except Exception as exc:
                log.exception("prepare failed for %s", image_id)
                ledger.update(source, image_id, status="error", error=f"prepare: {exc}")
                counts["errors"] += 1
    return counts


def tile_grid(width: int, height: int, max_side: int) -> list[tuple[int, int, int, int, int, int]]:
    """(row, col, x, y, w, h) of the smallest even grid whose tiles are all at
    most `max_side` on each side, row by row."""
    rows, cols = math.ceil(height / max_side), math.ceil(width / max_side)
    xs = [round(c * width / cols) for c in range(cols + 1)]
    ys = [round(r * height / rows) for r in range(rows + 1)]
    return [(r, c, xs[c], ys[r], xs[c + 1] - xs[c], ys[r + 1] - ys[r]) for r in range(rows) for c in range(cols)]


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
