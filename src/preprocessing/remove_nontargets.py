import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hydra
from omegaconf import DictConfig
import json
import logging

log = logging.getLogger(__name__)

class NonTargetRemover:
    def __init__(self, cfg: DictConfig):

        self.remove_colorchecker=cfg.preprocess.remove_nontargets.remove_colorchecker

        self.image_list_path = Path(cfg.paths.project_mode_dir) / "data" / "images.txt"
        self.image_list = [Path(x) for x in self._get_images_paths()]

        self.mask_output_dir = Path(cfg.paths.project_mode_dir) / "data" / "masks"
        self.mask_output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_images_paths(self) -> list:
        """
        Get the list of image paths from the images.txt file.
        """
        with open(self.image_list_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths
        
    
    def read_json(self, json_file: str):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    def process_file(self, mask_path: Path) -> int:
        """
        Process a single mask file and its corresponding image.
        Creates crops and saves them to the output folders.
        """
        metadata_path = mask_path.parent.parent.parent / "metadata" / f"{mask_path.stem}.json"
        
        if not mask_path.exists():
            log.warning(f"Image file corresponding to {mask_path} not found. Skipping.")
            return 0

        # Read metadata and mask only if the mask file name starts with NC, MD, or TX
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        
        if mask is None:
            log.error(f"Error reading {mask_path} or corresponding image. Skipping.")
            return 0
        
        if mask_path.stem.startswith("NC") or mask_path.stem.startswith("MD") or mask_path.stem.startswith("TX"):
            metadata = self.read_json(metadata_path)

            for annotation in metadata["annotations"]:
                if "non_target_weed" in annotation:
                    # if annotation["non_target_weed"] is not None:
                        # log.info(f'Non-target weed {annotation["non_target_weed"]} found in {mask_file}')
                    
                    if annotation["non_target_weed"] == True:
                        log.debug(f"Removing non-target weed from {mask_path}")
                        bbox_xywh = annotation["bbox_xywh"] # top left x, top left y, width, height in pixels
                        category_class_id = annotation["category_class_id"]
                        x1, y1, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
                        x2, y2 = x1 + w, y1 + h
                        mask[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] == category_class_id, 0, mask[y1:y2, x1:x2])
                    
                    if not isinstance(annotation["non_target_weed"], bool):
                        log.warning(f"Non-target weed value is not a boolean in {mask_path.name}. Invalid value: {annotation['non_target_weed']}")
                    
                if self.remove_colorchecker:
                    if annotation["category_class_id"] == 28:
                        log.debug(f"Removing colorchecker values from {mask_path}")
                        bbox_xywh = annotation["bbox_xywh"] # top left x, top left y, width, height in pixels
                        category_class_id = annotation["category_class_id"]
                        x1, y1, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
                        x2, y2 = x1 + w, y1 + h
                        mask[y1:y2, x1:x2] = 0

        output_mask_path = self.mask_output_dir / mask_path.name
        cv2.imwrite(str(output_mask_path), mask)

    def process_all(self):
        """
        Process all mask files in the mask folder using multiprocessing.
        """
        mask_files = sorted([
            image.parent.parent / "meta_masks/semantic_masks" / f"{image.stem}.png"
            for image in self.image_list
        ])
        mutliproces = True
        if mutliproces:
            with ProcessPoolExecutor(max_workers=22) as executor:
                futures = [executor.submit(self.process_file, mask_file) for mask_file in mask_files]
                for future in futures:
                    future.result()
        else:
            for mask_file in mask_files:
                self.process_file(mask_file)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting cropping of images and masks.")
    log.info("Cropping images and masks")
    processor = NonTargetRemover(cfg)
    processor.process_all()
    log.info("Cropping complete")

if __name__ == "__main__":
    main()
