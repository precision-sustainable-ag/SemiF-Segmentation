import os
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
from typing import Tuple
from src.utils.class_groupings import CLASSGROUPS

log = logging.getLogger(__name__)


class MaskProcessor:
    """Class to process and remap image masks."""

    def __init__(self, group_name: str):
        """
        Initialize the MaskProcessor with the specified group name.

        :param group_name: The key in CLASSGROUPS defining the desired mapping.
        """
        self.group_name = group_name
        self.species_map = CLASSGROUPS[group_name]

    def remap_mask(self, image: np.ndarray, image_path: str) -> Tuple[np.ndarray, bool]:
        """
        Remap the given image mask based on the mapping group.

        :param image: The input image mask as a numpy array.
        :param image_path: Path to the input image (used for logging skipped files).
        :return: The remapped image and a flag indicating if it should be skipped.
        """
        # Initialize lookup table (assumes 8-bit image with values 0-255)
        lookup_table = np.zeros(256, dtype=np.uint8)

        # Gather all valid class_ids from the species map
        valid_class_ids = set()
        for class_group, mapping in self.species_map.items():
            class_ids = mapping["class_ids"]
            new_value = mapping["values"]
            lookup_table[class_ids] = new_value
            valid_class_ids.update(class_ids)

        # Find all unique pixel values not in the valid set
        image_unique = np.unique(image)
        invalid_values = set(image_unique) - valid_class_ids

        if invalid_values:
            log.warning(
                f"Invalid class IDs found in {image_path}: {sorted(list(invalid_values))}. "
                f"These pixels will be set to unknown (27)."
            )

            return None, True

        # Apply the lookup table for remapping
        remapped_image = lookup_table[image]
        return remapped_image, False

    def process_image(self, image_path: str, output_directory: str):
        """
        Process a single image mask.

        :param image_path: Path to the input image.
        :param output_directory: Directory where processed images will be saved.
        """
        # Load the image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Remap the mask
        image, should_skip = self.remap_mask(image, image_path)
        if should_skip:
            log.info(f"Skipping {image_path} due to invalid values.")
            return

        # Save the processed image
        output_path = os.path.join(output_directory, os.path.basename(image_path))
        cv2.imwrite(output_path, image)


class SplitDirectoryProcessor:
    """Class to process train, val, and test directories of image masks."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the SplitDirectoryProcessor.

        :param split_dirs: Dictionary containing input and output directories for each split.
        :param group_name: Mapping group for remapping masks.
        :param process_concurrently: Whether to process images concurrently.
        """
        self.group_name=cfg.preprocess.remap_masks.group_name
        self.process_concurrently=cfg.preprocess.remap_masks.process_concurrently

        self.mask_processor = MaskProcessor(self.group_name)

        if Path(cfg.paths.cropped_mask_dir).exists():
            self.mask_dir = Path(cfg.paths.cropped_mask_dir)
            
            
        elif Path(cfg.paths.project_mode_dir, "data", "masks").exists():
            self.mask_dir = Path(cfg.paths.project_mode_dir, "data", "masks")
    
        else:
            raise FileNotFoundError(
                "No valid mask directory found. Please check your configuration."
            )
        self.remapped_mask_dir = Path(str(self.mask_dir) + "_remapped")
        self.remapped_mask_dir.mkdir(parents=True, exist_ok=True)

    def get_image_paths(self, input_dir: str):
        """Get paths to all valid images in the input directory."""
        return [str(p) for p in Path(input_dir).glob("*.png")] 

    def process(self):
        """Process a single data split."""

        mask_paths = self.get_image_paths(self.mask_dir)

        if self.process_concurrently:

            available_cpus = 28
            with concurrent.futures.ThreadPoolExecutor(max_workers=available_cpus) as executor:
                futures = [
                    executor.submit(self.mask_processor.process_image, img_path, self.remapped_mask_dir)
                    for img_path in mask_paths
                ]
                concurrent.futures.wait(futures)
        else:

            for mask_path in mask_paths:
                self.mask_processor.process_image(mask_path, self.remapped_mask_dir)


@hydra.main(version_base="1.3", config_path="configs", config_name="split_mask_processing_config")
def main(cfg: DictConfig):
    log.info("Starting mask processing for train, val, and test splits.")

    processor = SplitDirectoryProcessor(cfg)
        
    processor.process()
    log.info("Mask processing complete for all splits.")


if __name__ == "__main__":
    main()
