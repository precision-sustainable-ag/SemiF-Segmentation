import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.preprocessing.find_images import DatasetManager
from src.query import CutoutQuerySampler
log = logging.getLogger(__name__)

class InferenceDatasetManager:
    def __init__(self, cfg: DictConfig):        
        
        self.image_list_path = Path(cfg.paths.project_inference_dir) / "data" / "images.txt"

        self.destination_dir = Path(cfg.paths.project_inference_dir) / "data"

        self.sample_size = cfg.inference.get_dataset.sample_size

        self.copy_images = cfg.inference.get_dataset.images
        self.copy_metadata = cfg.inference.get_dataset.metadata
        self.copy_masks = cfg.inference.get_dataset.masks

        self._initialize_dst_dirs()
    
    def _initialize_dst_dirs(self):
        
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        if self.copy_images:
            self.image_dst = self.destination_dir / "images"
            self.image_dst.mkdir(parents=True, exist_ok=True)
            
        if self.copy_masks:
            self.mask_dst = self.destination_dir / "masks"
            self.mask_dst.mkdir(parents=True, exist_ok=True)
            
        if self.copy_metadata:
            self.metadata_dst = self.destination_dir / "metadata"
            self.metadata_dst.mkdir(parents=True, exist_ok=True)
            
    def get_images_paths(self) -> list:
        """
        Get the list of image paths from the images.txt file.
        """
        with open(self.image_list_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        image_paths = sorted([Path(x) for x in image_paths])
        sample_image_paths = self._sample_image_paths(image_paths)
        return sample_image_paths

    def copy_file(self, src: Path, dst: Path) -> None:
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def _sample_image_paths(self, image_paths: List) -> List:
        """
        Sample image paths from the list of images.
        """
        image_paths = random.sample(image_paths, min(len(image_paths), self.sample_size))
        return image_paths
    
    def _prepare_copy_tasks(self, image_paths: List[Path]) -> List[Tuple[Path, Path]]:
        tasks = []
        for image_path in image_paths:
            if self.copy_images:
                tasks.append((image_path, self.image_dst / image_path.name))
            if self.copy_masks:
                mask_path = image_path.parent.parent / "meta_masks/semantic_masks" / f"{image_path.stem}.png"
                tasks.append((mask_path, self.mask_dst / mask_path.name))
            if self.copy_metadata:
                metadata_path = image_path.parent.parent / "metadata" / f"{image_path.stem}.json"
                tasks.append((metadata_path, self.metadata_dst / metadata_path.name))
        return tasks

    def copy_files(self, image_paths: List[Path]) -> None:
        tasks = self._prepare_copy_tasks(image_paths)
        max_workers = 12
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.copy_file, src, dst): (src, dst) for src, dst in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Copying files"):
                src, dst = futures[future]
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Failed to copy {src} to {dst}: {e}")

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.inference.get_dataset.query:
        log.info("Running get_dataset task")
        sampler = CutoutQuerySampler(cfg, inference=True)
        sampler.run()

    if cfg.inference.get_dataset.collect_paths:
        log.info("Moving full-sized images and masks")
        manager = DatasetManager(cfg, inference=True)
        manager.load_images_and_masks()
        log.info(f"Images count: {len(manager.images)}")
        log.info(f"Masks count: {len(manager.masks)}")
        log.info(f"Metadata count: {len(manager.metadata)}")
        log.info("Copying files to the destination directory")
        output_path = manager.destination_dir / "images.txt"
        manager.write_data_2_text_file(manager.images, output_path)
        log.info(f"Saved image paths to {output_path}")


    log.info("Cropping images and masks")
    idm = InferenceDatasetManager(cfg)
    # Sample image paths
    image_paths = idm.get_images_paths()
    # Copy files to the destination directory
    idm.copy_files(image_paths)

    log.info("Cropping complete")


if __name__ == "__main__":
    main()
