"""Plant bounding boxes for whole images (preprocess task full_image_boxes).

Each full-size image (~9500 x 13400 px) is downscaled by proposals.full_image.scale,
its SAM 3 proposals (conf/proposals) are predicted on overlapping tiles and
merged across tiles (ProposalGenerator.predict_tiled), filtered against the
image's semantic mask and clipped to it (as for pseudo-instances), and the
boxes are written in full-resolution pixel coordinates.

Images: the manifest's full images of proposals.full_image.splits, with their
label masks; or every *.jpg in proposals.full_image.image_dir (no semantic
mask: proposals are only score/area/overlap filtered).

Writes to paths.project_preprocess_dir/data/full_image_boxes/<generator>_scale<scale>/ (sam3_scale0.5, ...):
  <image_id>.json  image size, scale, generator settings, and per instance:
                   bbox (full resolution, x0 y0 x1 y1 pixel edges), bbox_scaled
                   (at the working scale), score, label, area_scaled (px)
  boxes.csv        every instance of every image, one row each
  previews/        downscaled image with the boxes (first plot_previews images)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models import build_proposal_generator
from src.models.proposals.pseudo_labels import clip_instances_to_semantic

log = logging.getLogger(__name__)


def rescale_boxes(boxes: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    """(N, 4) xyxy pixel-edge boxes of an image of size from_hw, in pixels of the
    same image at size to_hw (per-axis factors, so rounding in the resize is
    accounted for), clipped to it."""
    fy, fx = to_hw[0] / from_hw[0], to_hw[1] / from_hw[1]
    out = np.asarray(boxes, np.float64).reshape(-1, 4) * np.array([fx, fy, fx, fy])
    out[:, 0::2] = out[:, 0::2].clip(0, to_hw[1])
    out[:, 1::2] = out[:, 1::2].clip(0, to_hw[0])
    return out


def image_rows(cfg: DictConfig, fcfg: dict) -> list[dict]:
    """{image_id, split, image_path, mask_path (or None)} of the images to box."""
    if fcfg.get("image_dir"):
        image_dir = Path(cfg.paths.persistent_dir, fcfg["image_dir"])
        return [{"image_id": p.stem, "split": None, "image_path": str(p), "mask_path": None}
                for p in sorted(image_dir.glob("*.jpg"))]
    manifest = pd.read_csv(cfg.paths.dataset_manifest)
    rows = manifest[manifest["split"].isin(fcfg["splits"])]
    return rows[["image_id", "split", "image_path", "mask_path"]].to_dict("records")


def load_scaled(image_path, mask_path, scale: float):
    """(full (H, W), RGB image at scale, label mask at scale or None)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    full_hw = image.shape[:2]
    size = (max(1, round(full_hw[1] * scale)), max(1, round(full_hw[0] * scale)))
    image = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    mask = None
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != full_hw:
            raise ValueError(f"mask {mask_path} is {mask.shape}, image is {full_hw}")
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return full_hw, image, mask


def save_preview(path: Path, image: np.ndarray, boxes: np.ndarray, max_side: int = 2400) -> None:
    """The working-scale image, further shrunk to max_side, with the boxes."""
    f = min(1.0, max_side / max(image.shape[:2]))
    view = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else image.copy()
    thickness = max(1, round(max(view.shape[:2]) / 800))
    for x0, y0, x1, y1 in np.asarray(boxes) * f:
        cv2.rectangle(view, (int(x0), int(y0)), (int(round(x1)), int(round(y1))), (255, 40, 200), thickness)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def predict_instances(generator, image: np.ndarray, semantic, full_hw: tuple[int, int], scfg: dict) -> list[dict]:
    """Instances of one image, given at the working scale (RGB, and its label
    mask or None), best first: bbox at full resolution full_hw, bbox_scaled,
    score, label, area_scaled."""
    tcfg = scfg["tiling"]
    inst = generator.predict_tiled(image, semantic, tcfg["tile_size"], tcfg["overlap"], tcfg["merge_iou"])
    if semantic is not None and scfg["pseudo_labels"].get("clip_to_semantic", True):
        inst = clip_instances_to_semantic(inst, semantic, scfg["filtering"].get("ignore_value"),
                                          scfg["filtering"].get("min_area", 0))
    order = np.argsort(-inst.scores) if len(inst) else np.zeros(0, np.int64)
    full_boxes = rescale_boxes(inst.boxes[order], image.shape[:2], full_hw) if len(inst) else np.zeros((0, 4))
    areas = inst.areas()[order] if len(inst) else []
    return [{
        "id": k + 1,
        "bbox": [round(float(v), 1) for v in full_boxes[k]],
        "bbox_scaled": [int(v) for v in inst.boxes[i]],
        "score": round(float(inst.scores[i]), 4),
        "label": int(inst.labels[i]),
        "area_scaled": int(areas[k]),
    } for k, i in enumerate(order)]


def box_image(generator, row: dict, scfg: dict) -> tuple[dict, np.ndarray]:
    """(JSON record of one image's instances, the image at the working scale)."""
    scale = float(scfg["full_image"]["scale"])
    full_hw, image, semantic = load_scaled(row["image_path"], row["mask_path"], scale)
    instances = predict_instances(generator, image, semantic, full_hw, scfg)
    return {
        "image_id": row["image_id"],
        "split": row["split"],
        "image_path": str(row["image_path"]),
        "mask_path": None if row["mask_path"] is None else str(row["mask_path"]),
        "image_size": list(full_hw),
        "scale": scale,
        "scaled_size": list(image.shape[:2]),
        "annotation_source": generator.source.value,
        "instances": instances,
    }, image


def main(cfg: DictConfig) -> None:
    scfg = OmegaConf.to_container(cfg.proposals, resolve=True)
    fcfg = scfg["full_image"]
    rows = image_rows(cfg, fcfg)
    if fcfg.get("max_images"):
        rows = rows[: int(fcfg["max_images"])]
    out_dir = Path(cfg.paths.project_preprocess_dir) / "data" / "full_image_boxes" / f"{scfg['name']}_scale{fcfg['scale']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = build_proposal_generator(scfg["name"], scfg)
    provenance = {
        "generator": scfg["name"],
        "created": datetime.now().isoformat(timespec="seconds"),
        **{k: scfg[k] for k in ("checkpoint", "score_threshold", "prompts", "points_on_foreground", "mask_generator")
           if k in scfg},
        "tiling": scfg["tiling"],
        "filtering": scfg["filtering"],
    }
    log.info("Boxing %d full images at scale %s with %s -> %s", len(rows), fcfg["scale"], scfg["name"], out_dir)
    previews = 0
    for row in tqdm(rows, desc=f"{scfg['name']} full images"):
        json_path = out_dir / f"{row['image_id']}.json"
        if json_path.exists() and not fcfg.get("overwrite"):
            continue
        record, image = box_image(generator, row, scfg)
        with open(json_path, "w") as f:
            json.dump({**record, "provenance": provenance}, f, indent=1)
        if previews < int(fcfg.get("plot_previews") or 0):
            save_preview(out_dir / "previews" / f"{row['image_id']}.jpg", image,
                         [i["bbox_scaled"] for i in record["instances"]])
            previews += 1

    table = []  # rebuilt from every image's JSON, so resumed runs stay complete
    for json_path in sorted(out_dir.glob("*.json")):
        with open(json_path) as f:
            record = json.load(f)
        for inst in record["instances"]:
            table.append({"image_id": record["image_id"], "split": record["split"], "instance": inst["id"],
                          "x0": inst["bbox"][0], "y0": inst["bbox"][1], "x1": inst["bbox"][2], "y1": inst["bbox"][3],
                          "score": inst["score"], "label": inst["label"], "scale": record["scale"],
                          "annotation_source": record["annotation_source"]})
    columns = ["image_id", "split", "instance", "x0", "y0", "x1", "y1", "score", "label", "scale", "annotation_source"]
    pd.DataFrame(table, columns=columns).to_csv(out_dir / "boxes.csv", index=False)
    log.info("%d boxes over %d images -> %s", len(table), len(list(out_dir.glob("*.json"))), out_dir / "boxes.csv")
