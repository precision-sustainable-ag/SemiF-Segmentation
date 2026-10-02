"""Model-assisted box pre-labels (label.prelabel.kind=sam3_boxes): SAM 3 plant
boxes per image, uploaded to CVAT as editable rectangles.

As the preprocess task full_image_boxes does: the full image is downscaled by
proposals.full_image.scale (or to proposals.full_image.max_side, whole), the generator of conf/proposals (proposals=sam3)
predicts on overlapping tiles (proposals.tiling) and merges plants cut by tile
edges, and the boxes are scaled back to full resolution. There is no semantic
mask here, so proposals are only score/area/overlap filtered.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.labeling.cvat_boxes import class_names
from src.models import build_proposal_generator
from src.preprocessing.full_image_boxes import predict_instances, working_scale

log = logging.getLogger(__name__)


class BoxPrelabeler:
    def __init__(self, cfg):
        self.scfg = OmegaConf.to_container(cfg.proposals, resolve=True)
        self.fcfg = self.scfg["full_image"]
        self.class_names = class_names(cfg.cvat)
        self.generator = build_proposal_generator(self.scfg["name"], self.scfg)
        self.provenance = {
            "generator": self.scfg["name"],
            **({"max_side": self.fcfg["max_side"]} if self.fcfg.get("max_side") else {"scale": self.fcfg["scale"]}),
            **{k: self.scfg[k] for k in ("checkpoint", "score_threshold", "prompts", "prompt_labels") if k in self.scfg},
            "tiling": self.scfg["tiling"],
        }
        log.info("Box pre-labels: %s, %s", self.scfg["name"], {k: v for k, v in self.provenance.items()
                                                                 if k in ("scale", "max_side")})

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        """Boxes of one full-resolution BGR image, best first: id, bbox (full
        resolution), score, class_id, and label (its CVAT label name)."""
        full_hw = image_bgr.shape[:2]
        scale = working_scale(full_hw, self.fcfg)
        size = (max(1, round(full_hw[1] * scale)), max(1, round(full_hw[0] * scale)))
        scaled = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else image_bgr
        instances = predict_instances(self.generator, cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB), None, full_hw, self.scfg)
        boxes = []
        for inst in instances:
            if inst["label"] not in self.class_names:
                raise ValueError(f"detection class {inst['label']} has no CVAT label; add it to cvat.detection.labels")
            boxes.append({"id": inst["id"], "bbox": inst["bbox"], "score": inst["score"],
                          "class_id": inst["label"], "label": self.class_names[inst["label"]]})
        return boxes
