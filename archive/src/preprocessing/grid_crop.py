import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


class CropProcessor:
    def __init__(self, cfg: DictConfig):
        
        self.project_output_dir = Path(cfg.paths.project_preprocess_dir)
        self.local_masks_present, self.mask_folder = self._initialize_mask_folder()
        
        self.image_list_path = Path(cfg.paths.project_mode_dir) / "data" / "images.txt"
        self.image_list = sorted([Path(x) for x in self._get_images_paths()])

        self.output_cropped_image_folder=Path(cfg.paths.cropped_image_dir)
        self.output_cropped_mask_folder=Path(cfg.paths.cropped_mask_dir)
        self.crop_height=cfg.preprocess.grid_crop.crop_height
        self.crop_width=cfg.preprocess.grid_crop.crop_width


        # Create output directories if they don't exist
        self.output_cropped_image_folder.mkdir(parents=True, exist_ok=True)
        self.output_cropped_mask_folder.mkdir(parents=True, exist_ok=True)

    def _initialize_mask_folder(self):
        """
        Initialize the mask folder based on the configuration.
        """
        local_masks_present = False
        mask_folder = self.project_output_dir / "data" / "masks"
        
        if not mask_folder.exists():
            log.warning(f"Mask folder {mask_folder} does not exist.")
            return local_masks_present, None

        # Check if local masks are present
        if any(mask_folder.glob("*.png")):
            local_masks_present = True
            log.debug(f"Local masks found in {mask_folder}")

        return local_masks_present, mask_folder
    
    def _get_images_paths(self) -> list:
        """
        Get the list of image paths from the images.txt file.
        """
        with open(self.image_list_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths
    
    def process_file(self, image_path: Path, local_mask: bool = False) -> int:
        """
        Process a single mask file and its corresponding image.
        Creates crops and saves them to the output folders.
        """
        image_path = image_path
        
        if local_mask:
            mask_path = self.mask_folder / f"{image_path.stem}.png"
        else:
            mask_path = image_path.parent / "meta_masks/semantic_masks" / f"{image_path.stem}.png"

        # Read image and mask
        image = cv2.imread(str(image_path))  # Load image in color
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        
        if mask is None or image is None:
            log.error(f"Error reading {mask_path} or corresponding image. Skipping.")
            return 0

        # Ensure the mask and image dimensions match
        if mask.shape[:2] != image.shape[:2]:
            log.debug(f"Dimension mismatch between mask and image for {mask_path}. Skipping.")
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
                        crop_image_path = self.output_cropped_image_folder / f"{mask_path.stem}_crop_{crop_count}.jpg"
                        crop_mask_path = self.output_cropped_mask_folder / f"{mask_path.stem}_crop_{crop_count}.png"

                        forbidden_strings = ["screberg", "research-project"]
                        if any(s in str(crop_image_path) for s in forbidden_strings) or any(s in str(crop_mask_path) for s in forbidden_strings):
                            log.warning(f"Forbidden string found in {crop_image_path}. Skipping.")
                            continue
                        cv2.imwrite(str(crop_image_path), image_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        cv2.imwrite(str(crop_mask_path), mask_crop)

                        crop_count += 1

        log.debug(f"Processed {mask_path}, created {crop_count} crops.")
        return crop_count

    def process_all(self):
        """
        Process all mask files in the mask folder using multiprocessing.
        """
        

        total_crops = 0
        multiproc = True
        if multiproc:
            with ProcessPoolExecutor(max_workers=28) as executor:
                futures = [executor.submit(self.process_file, image_path, local_mask=self.local_masks_present) for image_path in self.image_list]
                for future in futures:
                    total_crops += future.result()
        else:
            for image_path in self.image_list:
                total_crops += self.process_file(image_path, local_mask=self.local_masks_present)

        log.info(f"Total crops created: {total_crops}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Cropping images and masks")
    
    processor = CropProcessor(cfg
        
    )
    processor.process_all()
    log.info("Cropping complete")


if __name__ == "__main__":
    main()
