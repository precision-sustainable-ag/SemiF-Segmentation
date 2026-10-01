import json
from pathlib import Path
from typing import List, Dict, Union
import hydra
from omegaconf import DictConfig
import logging
from tqdm import tqdm   
log = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, cfg: DictConfig, inference: bool = False):
        self.query_dir = Path(cfg.paths.project_query_dir) if not inference else Path(cfg.paths.project_inference_dir) / "data"
        self.query_result = self.query_dir / f"{cfg.project.name}.json"
        self.primary_storage = Path(cfg.paths.primary_storage, "semifield-developed-images")
        self.secondary_storage = Path(cfg.paths.secondary_storage, "semifield-developed-images")
        self.tertiary_storage = Path(cfg.paths.tertiary_storage, "semifield-developed-images")
        self.use_synthetic = cfg.preprocess.use_synthetic.enable
        self.use_all_synthetic_subprojects = cfg.preprocess.use_synthetic.use_all_subprojects
        self.synthetic_project_dir = Path(cfg.paths.synthetic_project_dir)
        self.synthetic_subproject_dir = Path(self.synthetic_project_dir, cfg.preprocess.use_synthetic.sub_project) if cfg.preprocess.use_synthetic.sub_project is not None else None

        # self.destination_dir = Path(cfg.paths.data_dir, cfg.project.name)
        self.destination_dir = Path(cfg.paths.project_preprocess_dir) / "data" if not inference else Path(cfg.paths.project_inference_dir) / "data"
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        self.images = []
        self.masks = []
        self.metadata = []

    def load_data(self) -> List[Dict[str, Union[str, int]]]:
        """Load the dataset from the JSON query result."""
        with self.query_result.open('r') as file:
            data = json.load(file)
            if "cutouts" in data:
                return data["cutouts"]
            else:
                return data

    def collect_synthetic_paths(self) -> List[Path]:
        """Collect synthetic image paths from the specified directory."""
        images, masks = [], []
        if self.use_synthetic:
            if self.use_all_synthetic_subprojects:
                for subproject in self.synthetic_project_dir.iterdir():
                    
                    if subproject.is_dir():
                        image_dir = subproject / "results" / "images"
                        mask_dir = subproject / "results" / "semantic_masks"
                        images.extend(list(image_dir.glob("*.jpg")))
                        masks.extend(list(mask_dir.glob("*.png")))
            else:
                images.extend(list(self.synthetic_subproject_dir.glob("results/images/*.jpg")))
                masks.extend(list(self.synthetic_subproject_dir.glob("results/meta_masks/semantic_masks/*.png")))
        # Filter out images and masks that do not have a corresponding pair
        images = [img for img in images if img.stem in {mask.stem for mask in masks}]
        masks = [mask for mask in masks if mask.stem in {img.stem for img in images}]
        # Sort the lists to ensure they are in the same order
        images.sort(key=lambda x: x.stem)
        masks.sort(key=lambda x: x.stem)
        return images, masks
    
    def collect_paths(self) -> List[Path]:
        """Collect file paths for images, masks, and metadata, ensuring all three are present."""
        images, masks, metadata = [], [], []
        storage_paths = [self.primary_storage, self.secondary_storage, self.tertiary_storage]
        data = self.load_data()
        for record in tqdm(data, leave=False, desc="Collecting paths"):
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
        if self.use_synthetic:
            synthetic_images, synthetic_masks = self.collect_synthetic_paths()
            self.images.extend(synthetic_images)
            self.masks.extend(synthetic_masks)
        else:
            self.images, self.masks, self.metadata = self.collect_paths()

    def write_data_2_text_file(self, data: List[Path], file_path: Path):
        """Write the list of paths to a text file."""
        # Remove duplicates
        # Deduplicate the list of image paths
        data = list(set(data))
        # Sort the deduplicated list by the stem of the path
        data = sorted(data, key=lambda x: x.stem)
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
