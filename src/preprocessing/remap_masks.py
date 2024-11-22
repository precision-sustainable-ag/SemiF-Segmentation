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

        # Populate lookup table
        for class_group, mapping in self.species_map.items():
            class_ids = mapping["class_ids"]
            new_value = mapping["values"]
            lookup_table[class_ids] = new_value

        # Check and log if any invalid IDs (> 47) exist
        if np.any(image > 47):
            unique_invalid_values = np.unique(image[image > 47])
            log.warning(
                f"Invalid class IDs found in {image_path}: {unique_invalid_values}. Skipping."
            )
            return None, True  # Skip processing

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

    def __init__(self, split_dirs, group_name, process_concurrently=True):
        """
        Initialize the SplitDirectoryProcessor.

        :param split_dirs: Dictionary containing input and output directories for each split.
        :param group_name: Mapping group for remapping masks.
        :param process_concurrently: Whether to process images concurrently.
        """
        self.split_dirs = split_dirs
        self.group_name = group_name
        self.process_concurrently = process_concurrently
        self.mask_processor = MaskProcessor(group_name)

    def ensure_directories_exist(self):
        """Ensure all output directories exist."""
        for _, dirs in self.split_dirs.items():
            Path(dirs["remapped_mask_dir"]).mkdir(parents=True, exist_ok=True)

    def get_image_paths(self, input_dir: str):
        """Get paths to all valid images in the input directory."""
        return [str(p) for p in Path(input_dir).glob("*.png")] 

    def process_split(self, split_name: str, input_dir: str, output_dir: str):
        """Process a single data split."""
        log.info(f"Processing {split_name} masks.")
        image_paths = self.get_image_paths(input_dir)

        if self.process_concurrently:
            log.info(f"Processing {split_name} masks concurrently.")
            available_cpus = max(int(len(os.sched_getaffinity(0)) / 4), 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=available_cpus) as executor:
                futures = [
                    executor.submit(self.mask_processor.process_image, img_path, output_dir)
                    for img_path in image_paths
                ]
                concurrent.futures.wait(futures)
        else:
            log.info(f"Processing {split_name} masks sequentially.")
            for img_path in image_paths:
                self.mask_processor.process_image(img_path, output_dir)

    def process_splits(self):
        """Process all splits (train, val, test)."""
        self.ensure_directories_exist()

        for split_name, dirs in self.split_dirs.items():
            input_dir = dirs["mask_dir"]
            output_dir = dirs["remapped_mask_dir"]
            self.process_split(split_name, input_dir, output_dir)


@hydra.main(version_base="1.3", config_path="configs", config_name="split_mask_processing_config")
def main(cfg: DictConfig):
    log.info("Starting mask processing for train, val, and test splits.")
    split_dirs = {
        "train": {
            "mask_dir": cfg.paths.split_data.train_mask_dir,
            "remapped_mask_dir": cfg.paths.split_data.train_remapped_mask_dir,
        },
        "val": {
            "mask_dir": cfg.paths.split_data.val_mask_dir,
            "remapped_mask_dir": cfg.paths.split_data.val_remapped_mask_dir,
        },
        "test": {
            "mask_dir": cfg.paths.split_data.test_mask_dir,
            "remapped_mask_dir": cfg.paths.split_data.test_remapped_mask_dir,
        },
    }

    processor = SplitDirectoryProcessor(
        split_dirs=split_dirs,
        group_name=cfg.task.remap_masks.group_name,
        process_concurrently=cfg.task.remap_masks.process_concurrently,
    )
    processor.process_splits()
    log.info("Mask processing complete for all splits.")


if __name__ == "__main__":
    main()
