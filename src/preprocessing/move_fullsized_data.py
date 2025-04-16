import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shutil
from typing import List, Dict, Union
import hydra
from omegaconf import DictConfig
import logging
from tqdm import tqdm   
log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg: DictConfig):
        self.query_result = Path(cfg.preprocess.move_fullsized_data.query_result)
        self.primary_storage = Path(cfg.paths.primary_storage, "semifield-developed-images")
        self.secondary_storage = Path(cfg.paths.secondary_storage, "semifield-developed-images")
        self.tertiary_storage = Path(cfg.paths.tertiary_storage, "semifield-developed-images")

        self.destination_dir = Path(cfg.paths.data_dir, cfg.project.name)
        self.remove_unmatched = cfg.preprocess.move_fullsized_data.remove_unmatched
        self.data = self.load_data()
        self.images = []
        self.masks = []

    def load_data(self) -> List[Dict[str, Union[str, int]]]:
        """Load the dataset from the JSON query result."""
        with self.query_result.open('r') as file:
            return json.load(file)

    def collect_paths(self) -> List[Path]:
        """Collect file paths for images, masks, and metadata, ensuring all three are present."""
        images, masks, metadata = [], [], []
        storage_paths = [self.primary_storage, self.secondary_storage, self.tertiary_storage]

        for record in tqdm(self.data):
            batch = record["batch_id"]
            image_id = record["image_id"]

            found = False
            for storage in storage_paths:
                image_path = storage / batch / "images" / f"{image_id}.jpg"
                mask_path = storage / batch / "meta_masks/semantic_masks" / f"{image_id}.png"
                metadata_path = storage / batch / "metadata" / f"{image_id}.json"

                if image_path.exists() and mask_path.exists() and metadata_path.exists():
                    images.append(image_path)
                    masks.append(mask_path)
                    metadata.append(metadata_path)
                    found = True
                    break

            if not found:
                log.warning(f"File set not found for image_id: {image_id}")

        return images, masks, metadata

    def load_images_and_masks(self):
        """Load image, mask, and metadata paths, ensuring all three are present."""
        self.images, self.masks, self.metadata = self.collect_paths()
        
    @staticmethod
    def copy_file(file_path: Path, dest_dir: Path):
        """Static method to copy a single file."""
        dest_path = dest_dir / file_path.name
        log.info(f"Copying {file_path} to {dest_path}")
        shutil.copy(file_path, dest_path)

    def copy_files(self, files: List[Path], destination_subdir: str, parallel: bool = True):
        """Copy files to a destination subdirectory."""
        dest_dir = self.destination_dir / destination_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        if parallel:
            with ProcessPoolExecutor(max_workers=12) as executor:
                executor.map(self.copy_file, files, [dest_dir] * len(files))
        else:
            for file in files:
                self.copy_file(file, dest_dir)

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
    manager = DatasetManager(cfg)

    manager.load_images_and_masks()
    log.info(f"Images count: {len(manager.images)}")
    log.info(f"Masks count: {len(manager.masks)}")
    log.info(f"Metadata count: {len(manager.metadata)}")
    log.info("Copying files to the destination directory")
    manager.copy_files(manager.images, destination_subdir="images")
    manager.copy_files(manager.masks, destination_subdir="masks")
    manager.copy_files(manager.metadata, destination_subdir="metadata", parallel=False)
    log.info("Filtering common files")
    manager.filter_common_files(image_dir="images", mask_dir="masks")
    log.info("Moving complete")

if __name__ == "__main__":
    main()
