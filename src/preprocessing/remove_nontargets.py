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
    def __init__(
        self,
        mask_folder: str,
        metadata_folders: str,
        remove_colorchecker: bool = False
    ):
        self.mask_folder = Path(mask_folder)
        self.metadata_folders = Path(metadata_folders)
        self.remove_colorchecker = remove_colorchecker
        
        
    def read_json(self, json_file: str):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    def process_file(self, mask_file: str) -> int:
        """
        Process a single mask file and its corresponding image.
        Creates crops and saves them to the output folders.
        """
        mask_path = self.mask_folder / mask_file
        metadata_path = self.metadata_folders / mask_file.replace(".png", ".json")
        
        if not mask_path.exists():
            log.warning(f"Image file corresponding to {mask_file} not found. Skipping.")
            return 0

        # Read metadata and mask only if the mask file name starts with NC, MD, or TX
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        
        if mask is None:
            log.error(f"Error reading {mask_file} or corresponding image. Skipping.")
            return 0
        
        if mask_file.startswith("NC") or mask_file.startswith("MD") or mask_file.startswith("TX"):
            metadata = self.read_json(metadata_path)

        
            save_flag = False
            for annotation in metadata["annotations"]:
                if "non_target_weed" in annotation:
                    # if annotation["non_target_weed"] is not None:
                        # log.info(f'Non-target weed {annotation["non_target_weed"]} found in {mask_file}')
                    if annotation["non_target_weed"] == True:
                        save_flag = True
                        log.info(f"Removing non-target weed from {mask_file}")
                        bbox_xywh = annotation["bbox_xywh"] # top left x, top left y, width, height in pixels
                        category_class_id = annotation["category_class_id"]
                        x1, y1, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
                        x2, y2 = x1 + w, y1 + h
                        mask[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] == category_class_id, 0, mask[y1:y2, x1:x2])
                if self.remove_colorchecker:
                    if annotation["category_class_id"] == 28:
                        save_flag = True
                        log.info(f"Removing colorchecker values from {mask_file}")
                        bbox_xywh = annotation["bbox_xywh"] # top left x, top left y, width, height in pixels
                        category_class_id = annotation["category_class_id"]
                        x1, y1, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
                        x2, y2 = x1 + w, y1 + h
                        mask[y1:y2, x1:x2] = 0
        
        temp_test_dir = Path("data/SEMIF_512/non_targets_removed_masks")
        temp_test_dir.mkdir(parents=True, exist_ok=True)
        temp_test_mask_path = temp_test_dir / mask_file
        cv2.imwrite(str(temp_test_mask_path), mask)


    def process_all(self):
        """
        Process all mask files in the mask folder using multiprocessing.
        """
        mask_files = [f for f in os.listdir(self.mask_folder) if f.endswith(".png")]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_file, mask_file) for mask_file in mask_files]
            for future in futures:
                future.result()



@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting cropping of images and masks.")
    log.info("Cropping images and masks")
    processor = NonTargetRemover(
        mask_folder=cfg.paths.preprocess.mask_dir,
        metadata_folders=cfg.paths.preprocess.metadata_dir,
        remove_colorchecker=cfg.preprocess.remove_nontargets.remove_colorchecker
        )
    processor.process_all()
    log.info("Cropping complete")

if __name__ == "__main__":
    main()
