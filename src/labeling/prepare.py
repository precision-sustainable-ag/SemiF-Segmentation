"""prepare: make the CVAT copy of each fetched image, and its pre-label.

Full frames (up to 13368x9520) are too big to annotate comfortably in the
browser, so each image is downscaled until its longest side is at most
`label.prepare.max_side`, into the run folder's images/. The pre-label mask
is written at that same size into masks/ (0/255), as AgIR-CVToolkit's
seg-infer wrote images/ and masks/ for its CVAT upload.
"""

import logging

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

        prelabeler = None
        if lcfg.prelabel.enable:
            from src.labeling.prelabel import Prelabeler  # torch/smp only when needed

            prelabeler = Prelabeler(lcfg.prelabel)
            run.masks.mkdir(parents=True, exist_ok=True)

        max_side = int(lcfg.prepare.max_side)
        quality = int(lcfg.prepare.jpeg_quality)
        counts = {"prepared": 0, "prelabeled": 0, "errors": 0}
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

                if prelabeler is not None:
                    mask = prelabeler.predict(image, size)
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
