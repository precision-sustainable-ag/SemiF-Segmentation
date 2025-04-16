import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import shutil
from typing import List, Dict
import hydra
from omegaconf import DictConfig
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg: DictConfig):
        self.image_stems_file = Path("/home/mkutuga/SemiF-Segmentation/data/hairy_vetch/image_stems.txt")
        self.primary_storage = Path(cfg.paths.primary_storage, "semifield-developed-images")
        self.secondary_storage = Path(cfg.paths.secondary_storage, "semifield-developed-images")
        self.tertiary_storage = Path(cfg.paths.tertiary_storage, "semifield-developed-images")

        self.destination_dir = Path(cfg.paths.data_dir, cfg.project.name)
        self.remove_unmatched = True

        self.image_stems = self.load_image_stems()
        self.images = []
        self.masks = []
        self.metadata = []

    def load_image_stems(self) -> List[str]:
        """Read the list of image stems from a .txt file."""
        with self.image_stems_file.open('r') as f:
            return [line.strip() for line in f if line.strip()]

    def collect_paths(self) -> List[Path]:
        """Find image, mask, and metadata paths matching given stems."""
        images, masks, metadata = [], [], []
        storage_paths = [self.primary_storage, self.secondary_storage, self.tertiary_storage]

        for stem in tqdm(self.image_stems):
            found = False
            for storage in storage_paths:
                # Try finding a matching subfolder (e.g., batch) that contains the file
                batch_dirs = list(storage.glob("*/images"))
                for batch_images_dir in batch_dirs:
                    batch = batch_images_dir.parent.name

                    image_path = storage / batch / "images" / f"{stem}.jpg"
                    mask_path = storage / batch / "meta_masks/semantic_masks" / f"{stem}.png"
                    metadata_path = storage / batch / "metadata" / f"{stem}.json"

                    if image_path.exists() and mask_path.exists() and metadata_path.exists():
                        images.append(image_path)
                        masks.append(mask_path)
                        metadata.append(metadata_path)
                        found = True
                        break
                if found:
                    break

            if not found:
                log.warning(f"Files not found for image stem: {stem}")

        return images, masks, metadata

    def load_images_and_masks(self):
        self.images, self.masks, self.metadata = self.collect_paths()

    @staticmethod
    def copy_file(file_path: Path, dest_dir: Path):
        dest_path = dest_dir / file_path.name
        log.debug(f"Copying {file_path} to {dest_path}")
        shutil.copy(file_path, dest_path)

    def copy_files(self, files: List[Path], destination_subdir: str, parallel: bool = True):
        dest_dir = self.destination_dir / destination_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        if parallel:
            with ProcessPoolExecutor(max_workers=12) as executor:
                executor.map(self.copy_file, files, [dest_dir] * len(files))
        else:
            for file in files:
                self.copy_file(file, dest_dir)

    def filter_common_files(self, image_dir: str, mask_dir: str):
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


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Downloading full-sized images and masks using image stems")

    manager = DatasetManager(cfg)
    manager.load_images_and_masks()

    log.info(f"Images found: {len(manager.images)}")
    log.info(f"Masks found: {len(manager.masks)}")
    log.info(f"Metadata found: {len(manager.metadata)}")

    manager.copy_files(manager.images, destination_subdir="images")
    manager.copy_files(manager.masks, destination_subdir="masks")
    manager.copy_files(manager.metadata, destination_subdir="metadata", parallel=False)

    manager.filter_common_files(image_dir="images", mask_dir="masks")
    log.info("Download complete")

if __name__ == "__main__":
    main()
