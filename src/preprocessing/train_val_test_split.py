import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import hydra
from omegaconf import DictConfig
from typing import Dict
import logging

log = logging.getLogger(__name__)


class FileHelper:
    """Helper class for file operations."""

    @staticmethod
    def move_concurrently(source_dest_pairs):
        """Move files concurrently."""
        available_cpus = max(int(len(os.sched_getaffinity(0)) / 4), 1)
        with ProcessPoolExecutor(max_workers=available_cpus) as executor:
            futures = [
                executor.submit(shutil.copy, src, dest)
                for src, dest in source_dest_pairs
            ]
            for future in as_completed(futures):
                future.result()

    @staticmethod
    def move_sequentially(source_dest_pairs, description="Copying files"):
        """Move files sequentially with a progress bar."""
        for src, dest in tqdm(source_dest_pairs, desc=description):
            shutil.copy(src, dest)

    @staticmethod
    def ensure_directories_exist(*dirs):
        """Ensure directories exist, creating them if necessary."""
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True, parents=True)


class DataSet:
    """Class to manage and process a dataset of images and masks."""

    def __init__(self, dirs, val_size=0.1, test_size=0.1, random_state=42, use_concurrency=True):
        """
        Initialize the dataset handler.
        Args:
            dirs (Dict): Dictionary containing directory paths.
            test_size (float): Proportion of the data to use for testing.
            val_size (float): Proportion of the data to use for validation.
            random_state (int): Random seed for splitting data.
            use_concurrency (bool): Whether to use concurrent file operations.
        """
        self.dirs = {k: Path(v) for k, v in dirs.items()}
        
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_concurrency = use_concurrency

    def gather_files(self, image_dir: Path, mask_dir: Path, model_testing: Dict):
        """Gather image and mask files from the given directories."""
        log.info("Gathering images and masks.")

        self.image_list = sorted(image_dir.glob("*.jpg"))
        self.mask_list = sorted(mask_dir.glob("*.png"))
        if model_testing.status:
            log.info("Using a subset of the data for quick and dirty model testing.")
            num_images = int(len(self.image_list) * model_testing.factor)
            self.image_list = self.image_list[:num_images]
            self.mask_list = self.mask_list[:num_images]

        assert self.image_list, "No images found in the images directory."
        assert self.mask_list, "No masks found in the masks directory."

    def split_data(self):
        """Split the data into training, validation, and test sets."""
        log.info("Splitting data.")
        log.info(f"Train size: \t{1.0 - self.val_size - self.test_size}")
        log.info(f"Validation size: \t{self.val_size}")
        log.info(f"Test size: \t\t{self.test_size}")
        # Split into train+val and test sets
        if self.test_size == 0:
            log.warning("Test size is 0. Splitting data into train and validation only.")
            self.test_img_paths = []
            self.test_mask_paths = []
            train_val_img_paths = self.image_list
            train_val_mask_paths = self.mask_list
        else:
            log.info(f"Splitting data into train, validation, and test sets.")
            train_val_img_paths, self.test_img_paths, train_val_mask_paths, self.test_mask_paths = train_test_split(
            self.image_list,
            self.mask_list,
            test_size=self.test_size,
            random_state=self.random_state,
            )
        
        # Calculate validation size relative to train+val size
        relative_val_size = self.val_size / (1.0 - self.test_size)

        # Split train+val into train and validation sets
        self.train_img_paths, self.val_img_paths, self.train_labels_paths, self.val_labels_paths = train_test_split(
            train_val_img_paths,
            train_val_mask_paths,
            test_size=relative_val_size,
            random_state=self.random_state,
        )

        log.info(f"Training images: \t{len(self.train_img_paths)}")
        log.info(f"Validation images: \t{len(self.val_img_paths)}")
        log.info(f"Test images: \t{len(self.test_img_paths)}")

    def prepare_destination(self):
        """Prepare destination directories."""
        FileHelper.ensure_directories_exist(*self.dirs.values())

    def move_files(self):
        """Copying files from source to destination."""
        log.info("Copying files.")
        self.prepare_destination()

        def create_source_dest_pairs(src_paths, dest_dir):
            return [(src, dest_dir / src.name) for src in src_paths]

        pairs = {
            "train_img": create_source_dest_pairs(self.train_img_paths, self.dirs["train_image_dir"]),
            "train_mask": create_source_dest_pairs(self.train_labels_paths, self.dirs["train_mask_dir"]),
            "val_img": create_source_dest_pairs(self.val_img_paths, self.dirs["val_image_dir"]),
            "val_mask": create_source_dest_pairs(self.val_labels_paths, self.dirs["val_mask_dir"]),
            "test_img": create_source_dest_pairs(self.test_img_paths, self.dirs["test_image_dir"]),
            "test_mask": create_source_dest_pairs(self.test_mask_paths, self.dirs["test_mask_dir"]),
        }

        for key, pair_list in pairs.items():
            if not pair_list:  # Skip empty pairs (e.g., test set if unused)
                continue

            if self.use_concurrency:
                log.info(f"Copying {key} files concurrently.")
                FileHelper.move_concurrently(pair_list)
            else:
                log.info(f"Copying {key} files sequentially.")
                FileHelper.move_sequentially(pair_list, f"Copying {key} files")


@hydra.main(version_base="1.3", config_path="configs", config_name="dataset_config")
def main(cfg: DictConfig):
    log.info("Starting dataset split.")
    dataset = DataSet(
        dirs=cfg.paths.split_data,
        val_size=cfg.task.train_val_test_split.val_size,
        test_size=cfg.task.train_val_test_split.test_size,
        random_state=cfg.task.train_val_test_split.seed,
        use_concurrency=cfg.task.train_val_test_split.use_concurrency,
    )
    
    log.info("Gathering files.")
    
    image_dir = Path(cfg.paths.cropped_image_dir)
    mask_dir = Path(cfg.paths.cropped_mask_dir)
    model_testing = cfg.task.train_val_test_split.model_testing
    
    dataset.gather_files(image_dir, mask_dir, model_testing)
    
    log.info("Splitting data.")
    dataset.split_data()
    
    log.info("Preparing destination directories.")
    dataset.move_files()
    
    log.info("Dataset split complete.")


if __name__ == "__main__":
    main()
