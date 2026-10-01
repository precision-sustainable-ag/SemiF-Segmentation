"""Pseudo-instances for the preprocessed tiles (preprocess task pseudo_instances).

For every tile of the configured splits (paths.split_dir/<split>/images +
masks): proposals of the configured generator (proposals=sam3),
filtered against the tile's semantic mask (conf/proposals filtering), turned
into instances whose pixels and class come from the semantic mask
(src/models/proposals/pseudo_labels.py). Train on them with
model=maskrcnn_pointrend model.instances.source=pseudo.

Writes to paths.split_dir/<split>/<proposals.pseudo_labels.dirname>/ (instances_sam3):
  <stem>.png, <stem>.json  the instances (src/utils/instance_annotations.py),
                           annotation_source "sam3", with the generator settings
  proposals/<stem>.json    every proposal: score, box, metrics, and
                           "kept" or why it was rejected (output.save_proposals)
  previews/<stem>.png      image | semantic mask | instances (output.save_previews tiles)
  summary.csv              per tile: proposals, instances, rejections by reason,
                           share of plant pixels left without an instance
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models import build_proposal_generator
from src.models.proposals.pseudo_labels import target_stats
from src.models.proposals.structures import ProposalResult
from src.utils.instance_annotations import empty_target, save_instance_annotation
from src.utils.viz_utils import draw_instances

log = logging.getLogger(__name__)


def proposal_records(result: ProposalResult) -> list[dict]:
    """JSON-ready record of every proposal (kept and rejected), by descending score."""
    entries = [(p, "kept") for p in result.proposals] + list(result.rejected)
    records = []
    for p, status in sorted(entries, key=lambda e: -e[0].score):
        records.append({
            "status": status,
            "prompt": p.prompt,
            "score": round(p.score, 4),
            "bbox": [round(float(v), 1) for v in p.bbox.tolist()],
            "metrics": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in p.metrics.items()},
        })
    return records


def save_preview(path: Path, image: np.ndarray, semantic: np.ndarray, target: dict) -> None:
    """image | semantic mask (plant pixels green) | pseudo-instances, side by side."""
    sem = image.copy()
    sem[semantic > 0] = (0.4 * sem[semantic > 0] + 0.6 * np.array([34, 200, 34])).astype(np.uint8)
    masks = target["masks"].bool().numpy()
    texts = [f"{s:.2f}" for s in target["scores"].tolist()]
    inst = draw_instances(image, masks, target["boxes"].numpy(), texts) if len(masks) else image
    gap = np.full((image.shape[0], 8, 3), 255, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([image, gap, sem, gap, inst]), cv2.COLOR_RGB2BGR))


def generate_split(generator, image_dir: Path, mask_dir: Path, out_dir: Path, scfg: dict, provenance: dict) -> pd.DataFrame:
    """Pseudo-instances for one split's tiles; returns this run's summary rows."""
    pcfg, ocfg = scfg["pseudo_labels"], scfg.get("output") or {}
    ignore_value = scfg["filtering"].get("ignore_value")
    images = sorted(image_dir.glob("*.jpg"))
    if pcfg.get("max_tiles"):
        images = images[: int(pcfg["max_tiles"])]
    rows, previews = [], 0
    for image_path in tqdm(images, desc=f"{generator.name} {image_dir.parent.name}"):
        stem = image_path.stem
        if not pcfg.get("overwrite") and (out_dir / f"{stem}.png").exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        semantic = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if semantic is None:
            log.warning("No mask for %s; skipping", image_path)
            continue
        foreground = (semantic > 0) if ignore_value is None else (semantic > 0) & (semantic != ignore_value)

        result = None
        if pcfg.get("skip_empty_tiles", True) and not foreground.any():
            target = {**empty_target(*semantic.shape), "prompts": []}
        else:
            target, result = generator.create_pseudo_target(image, semantic, return_result=True)
        save_instance_annotation(out_dir, stem, target, extra={"provenance": provenance})

        row = {"tile": stem, **target_stats(target, semantic, ignore_value)}
        if result is not None:
            row["proposals"] = len(result) + len(result.rejected)
            row["kept_proposals"] = len(result)
            row.update({f"rejected_{k}": v for k, v in Counter(r.split(":")[0] for _, r in result.rejected).items()})
            if ocfg.get("save_proposals", True):
                (out_dir / "proposals").mkdir(exist_ok=True)
                with open(out_dir / "proposals" / f"{stem}.json", "w") as f:
                    json.dump({"image_size": list(result.image_size), "proposals": proposal_records(result)}, f, indent=1)
            if previews < int(ocfg.get("save_previews") or 0):
                save_preview(out_dir / "previews" / f"{stem}.png", image, semantic, target)
                previews += 1
        rows.append(row)
    return pd.DataFrame(rows)


def main(cfg: DictConfig) -> None:
    scfg = OmegaConf.to_container(cfg.proposals, resolve=True)
    pcfg = scfg["pseudo_labels"]
    generator = build_proposal_generator(scfg["name"], scfg)
    provenance = {
        "generator": scfg["name"],
        "created": datetime.now().isoformat(timespec="seconds"),
        **{k: scfg[k] for k in ("checkpoint", "score_threshold", "prompts", "resolution", "points_on_foreground",
                                "mask_generator") if k in scfg},
        "filtering": scfg["filtering"],
        "pseudo_labels": {k: pcfg.get(k) for k in ("clip_to_semantic", "exclusive_masks")},
    }
    split_dir = Path(cfg.paths.split_dir)
    for split in pcfg["splits"]:
        image_dir, mask_dir = split_dir / split / "images", split_dir / split / "masks"
        if not image_dir.is_dir():
            log.warning("No tiles for split %s in %s; skipping", split, image_dir)
            continue
        out_dir = split_dir / split / pcfg["dirname"]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = generate_split(generator, image_dir, mask_dir, out_dir, scfg, provenance)
        summary_path = out_dir / "summary.csv"
        if summary_path.exists() and len(summary):  # resumed run: replace the rows of re-done tiles
            old = pd.read_csv(summary_path)
            summary = pd.concat([old[~old["tile"].isin(summary["tile"])], summary], ignore_index=True)
        if len(summary):
            summary.to_csv(summary_path, index=False)
            log.info("%s: %d tiles, %.1f instances per tile, %.1f%% of plant pixels without an instance -> %s",
                     split, len(summary), summary["instances"].mean(),
                     100 * summary.get("unassigned_foreground_fraction", pd.Series([0.0])).mean(), out_dir)
