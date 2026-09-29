"""tile: cut the manifest's images and masks into training tiles.

Every image is first resized to `preprocess.tile.scale` of its native size
(inference uses the same factor, so the model sees plants at the scale it
trained on), then covered completely with tile_size tiles: the last tile in
each row/column is aligned to the image edge instead of being dropped.
Background-only tiles are kept at `keep_background` rate so the model also
learns bare soil and residue. Output: paths.split_dir/{train,val,test}/{images,masks}.
"""

import logging
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.tiling import tile_starts

log = logging.getLogger(__name__)


def tile_image(row: dict, out_dir: str, scale: float, tile_size: int, overlap: int,
               keep_background: float, seed: int) -> dict:
    image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return {"image_id": row["image_id"], "error": "could not read image or mask"}
    if image.shape[:2] != mask.shape:
        return {"image_id": row["image_id"], "error": f"image {image.shape[:2]} != mask {mask.shape}"}

    if scale != 1.0:
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # Images smaller than a tile are padded; padded pixels are background.
    pad_h, pad_w = max(0, tile_size - image.shape[0]), max(0, tile_size - image.shape[1])
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    split_dir = Path(out_dir) / row["split"]
    stem_prefix = f"{row['source']}__{row['image_id']}"
    rng = np.random.default_rng([seed, zlib.crc32(stem_prefix.encode())])
    stride = tile_size - overlap
    kept = background = 0
    for y in tile_starts(image.shape[0], tile_size, stride):
        for x in tile_starts(image.shape[1], tile_size, stride):
            mask_tile = mask[y:y + tile_size, x:x + tile_size]
            if not mask_tile.any():
                if rng.random() >= keep_background:
                    continue
                background += 1
            stem = f"{stem_prefix}_y{y}_x{x}"
            cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"), image[y:y + tile_size, x:x + tile_size],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(split_dir / "masks" / f"{stem}.png"), mask_tile)
            kept += 1
    return {"image_id": row["image_id"], "split": row["split"], "tiles": kept, "background_tiles": background}


def main(cfg: DictConfig) -> None:
    tcfg = cfg.preprocess.tile
    if not 0 <= tcfg.overlap < tcfg.tile_size:
        raise ValueError("preprocess.tile.overlap must be in [0, tile_size)")
    manifest = pd.read_csv(cfg.paths.dataset_manifest)
    out_dir = Path(cfg.paths.split_dir)
    if out_dir.exists():
        # Stale tiles from an earlier build would leak into the new split.
        log.info("Removing previous tiles in %s", out_dir)
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict(orient="records")
    args = (str(out_dir), float(tcfg.scale), int(tcfg.tile_size), int(tcfg.overlap),
            float(tcfg.keep_background), int(tcfg.seed))
    with ProcessPoolExecutor(max_workers=int(tcfg.workers)) as pool:
        results = list(tqdm(pool.map(tile_image, rows, *[[a] * len(rows) for a in args]),
                            total=len(rows), desc="tile"))

    errors = [r for r in results if "error" in r]
    for r in errors:
        log.warning("Skipped %s: %s", r["image_id"], r["error"])
    done = pd.DataFrame([r for r in results if "error" not in r])
    if done.empty:
        raise RuntimeError("No tiles were written")
    summary = done.groupby("split")[["tiles", "background_tiles"]].sum()
    log.info("Wrote %d-px tiles at scale %s to %s:\n%s", tcfg.tile_size, tcfg.scale, out_dir, summary.to_string())
