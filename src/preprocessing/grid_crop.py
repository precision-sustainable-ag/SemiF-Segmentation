import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


class CropProcessor:
    def __init__(
        self,
        image_folder: str,
        mask_folder: str,
        output_cropped_image_folder: str,
        output_cropped_mask_folder: str,
        crop_height: int,
        crop_width: int,
    ):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.output_cropped_image_folder = Path(output_cropped_image_folder)
        self.output_cropped_mask_folder = Path(output_cropped_mask_folder)
        self.crop_height = crop_height
        self.crop_width = crop_width

        # Create output directories if they don't exist
        self.output_cropped_image_folder.mkdir(parents=True, exist_ok=True)
        self.output_cropped_mask_folder.mkdir(parents=True, exist_ok=True)

    def process_file(self, mask_file: str) -> int:
        """
        Process a single mask file and its corresponding image.
        Creates crops and saves them to the output folders.
        """
        image_path = self.image_folder / mask_file.replace(".png", ".jpg")  # Assuming images are .jpg
        mask_path = self.mask_folder / mask_file

        if not image_path.exists():
            log.warning(f"Image file corresponding to {mask_file} not found. Skipping.")
            return 0

        # Read image and mask
        image = cv2.imread(str(image_path))  # Load image in color
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        
        if mask is None or image is None:
            log.error(f"Error reading {mask_file} or corresponding image. Skipping.")
            return 0

        # Ensure the mask and image dimensions match
        if mask.shape[:2] != image.shape[:2]:
            log.debug(f"Dimension mismatch between mask and image for {mask_file}. Skipping.")
            return 0

        height, width = mask.shape
        crop_count = 0

        # Slide a window over the mask and image
        for y in range(0, height, self.crop_height):
            for x in range(0, width, self.crop_width):
                # Crop the mask and image
                mask_crop = mask[y:y + self.crop_height, x:x + self.crop_width]
                image_crop = image[y:y + self.crop_height, x:x + self.crop_width]

                # Ensure the cropped region matches the specified dimensions
                if mask_crop.shape[:2] == (self.crop_height, self.crop_width):
                    # Save the crop if the mask has any non-zero value
                    if np.any(mask_crop > 0):
                        crop_image_path = self.output_cropped_image_folder / f"{Path(mask_file).stem}_crop_{crop_count}.jpg"
                        crop_mask_path = self.output_cropped_mask_folder / f"{Path(mask_file).stem}_crop_{crop_count}.png"

                        cv2.imwrite(str(crop_image_path), image_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        cv2.imwrite(str(crop_mask_path), mask_crop)

                        crop_count += 1

        log.info(f"Processed {mask_file}, created {crop_count} crops.")
        return crop_count

    def process_all(self):
        """
        Process all mask files in the mask folder using multiprocessing.
        """
        mask_files = [f for f in os.listdir(self.mask_folder) if f.endswith(".png")]

        total_crops = 0
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_file, mask_file) for mask_file in mask_files]
            for future in futures:
                total_crops += future.result()

        log.info(f"Total crops created: {total_crops}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Cropping images and masks")
    processor = CropProcessor(
        image_folder=cfg.paths.image_dir,
        mask_folder=cfg.paths.mask_dir,
        output_cropped_image_folder=cfg.paths.cropped_image_dir,
        output_cropped_mask_folder=cfg.paths.cropped_mask_dir,
        crop_height=cfg.task.grid_crop.crop_height,
        crop_width=cfg.task.grid_crop.crop_width,
    )
    processor.process_all()
    log.info("Cropping complete")


if __name__ == "__main__":
    main()
