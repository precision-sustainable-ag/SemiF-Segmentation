import json
from pathlib import Path
from typing import List, Dict, Union
import hydra
from omegaconf import DictConfig
import logging
from tqdm import tqdm   
log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg: DictConfig):
        self.query_result = Path(cfg.paths.project_mode_dir).parent / "query" / f"{cfg.project.name}.json"
        self.primary_storage = Path(cfg.paths.primary_storage, "semifield-developed-images")
        self.secondary_storage = Path(cfg.paths.secondary_storage, "semifield-developed-images")
        self.tertiary_storage = Path(cfg.paths.tertiary_storage, "semifield-developed-images")

        # self.destination_dir = Path(cfg.paths.data_dir, cfg.project.name)
        self.destination_dir = Path(cfg.paths.project_mode_dir) / "data"
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        self.data = self.load_data()
        self.images = []
        self.masks = []

    def load_data(self) -> List[Dict[str, Union[str, int]]]:
        """Load the dataset from the JSON query result."""
        with self.query_result.open('r') as file:
            data = json.load(file)
            if "cutouts" in data:
                return data["cutouts"]
            else:
                return data

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

    def write_data_2_text_file(self, data: List[Path], file_path: Path):
        """Write the list of paths to a text file."""
        # Remove duplicates
        data = list(set(data))
        with file_path.open('w') as f:
            for item in data:
                f.write(f"{item}\n")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Moving full-sized images and masks")
    manager = DatasetManager(cfg)
    manager.load_images_and_masks()
    log.info(f"Images count: {len(manager.images)}")
    log.info(f"Masks count: {len(manager.masks)}")
    log.info(f"Metadata count: {len(manager.metadata)}")
    log.info("Copying files to the destination directory")
    output_path = manager.destination_dir / "images.txt"
    manager.write_data_2_text_file(manager.images, output_path)
    log.info(f"Saved image paths to {output_path}")

if __name__ == "__main__":
    main()
