import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
from typing import List, Dict, Union
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, project_name: str, query_result: str, primary_storage: str, secondary_storage: str, destination_dir: str, remove_unmatched: bool = False):
        self.query_result = Path(query_result)
        self.primary_storage = Path(primary_storage, "semifield-developed-images")
        self.secondary_storage = Path(secondary_storage, "semifield-developed-images")
        self.destination_dir = Path(destination_dir, project_name)
        self.remove_unmatched = remove_unmatched
        self.data = self.load_data()
        self.images = []
        self.masks = []

    def load_data(self) -> List[Dict[str, Union[str, int]]]:
        """Load the dataset from the JSON query result."""
        with self.query_result.open('r') as file:
            return json.load(file)

    def collect_paths(self, subdir: str, file_extension: str) -> List[Path]:
        """Collect file paths for images or masks."""
        collected_paths = []
        for record in self.data:
            batch = record["batch_id"]
            image_id = record["image_id"]
            primary_path = self.primary_storage / batch / subdir / f"{image_id}{file_extension}"
            secondary_path = self.secondary_storage / batch / subdir / f"{image_id}{file_extension}"

            if primary_path.exists():
                collected_paths.append(primary_path)
            elif secondary_path.exists():
                collected_paths.append(secondary_path)
            else:
                print(f"File not found: {primary_path}")
        return collected_paths

    def load_images_and_masks(self):
        """Load image and mask paths."""
        self.images = self.collect_paths("images", ".jpg")
        self.masks = self.collect_paths("meta_masks/semantic_masks", ".png")

    def copy_files(self, files: List[Path], destination_subdir: str):
        """Copy files to a destination subdirectory."""
        dest_dir = self.destination_dir / destination_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        def copy_file(file_path: Path):
            dest_path = dest_dir / file_path.name
            shutil.copy(file_path, dest_path)

        with ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(copy_file, files)

    def filter_common_files(self, image_dir: str, mask_dir: str):
        """Filter images and masks to keep only matching filenames."""
        image_dir = Path(self.destination_dir, "images")
        mask_dir = Path(self.destination_dir, "masks")

        new_images = list(image_dir.glob("*.jpg"))
        new_masks = list(mask_dir.glob("*.png"))

        image_filenames = {image.stem for image in new_images}
        mask_filenames = {mask.stem for mask in new_masks}

        common_filenames = image_filenames & mask_filenames

        filtered_images = [image for image in new_images if image.stem in common_filenames]
        filtered_masks = [mask for mask in new_masks if mask.stem in common_filenames]

        log.info(f"Filtered images count: {len(filtered_images)}")
        log.info(f"Filtered masks count: {len(filtered_masks)}")

        unmatched_images = [image for image in new_images if image.stem not in common_filenames]
        unmatched_masks = [mask for mask in new_masks if mask.stem not in common_filenames]

        log.warning(f"Unmatched images count: {len(unmatched_images)}")
        log.warning(f"Unmatched masks count: {len(unmatched_masks)}")

        # Optionally remove unmatched files
        if self.remove_unmatched:
            log.info("Removing unmatched files")
            for image in unmatched_images:
                image.unlink()
                log.info(f"Removed unmatched image: {image}")
            for mask in unmatched_masks:
                mask.unlink()
                log.info(f"Removed unmatched mask: {mask}")

        self.images = filtered_images
        self.masks = filtered_masks


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Moving full-sized images and masks")
    manager = DatasetManager(
        project_name=cfg.project_name,
        query_result=cfg.task.move_data.query_result,
        primary_storage=cfg.paths.primary_storage,
        secondary_storage=cfg.paths.secondary_storage,
        destination_dir=cfg.paths.data_dir,
        remove_unmatched=cfg.task.move_data.remove_unmatched
    )

    manager.load_images_and_masks()
    log.info(f"Images count: {len(manager.images)}")
    log.info(f"Masks count: {len(manager.masks)}")
    log.info("Copying files to the destination directory")
    manager.copy_files(manager.images, destination_subdir="images")
    manager.copy_files(manager.masks, destination_subdir="masks")
    log.info("Filtering common files")
    manager.filter_common_files(image_dir="images", mask_dir="masks")
    log.info("Moving complete")

if __name__ == "__main__":
    main()
