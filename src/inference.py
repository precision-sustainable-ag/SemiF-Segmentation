import os
import cv2
import json
import torch
import random
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.model import SegmentationModule

log = logging.getLogger(__name__)


class MaskSaver:
    """Handles saving predicted masks and comparison plots."""
    def __init__(self, model, mean=None, std=None, use_normalization=True, rescale_factor=0.5, threshold=0.5, dpi=300, transparent=True):
        self.model = model
        self.mean = mean
        self.std = std
        self.use_normalization = use_normalization
        self.rescale_factor = rescale_factor
        self.threshold = threshold
        self.dpi = dpi
        self.transparent = transparent

    def save_masks(self, image_dir, output_dir, samplek=None, out_classes=1):
        mask_output_dir = Path(output_dir) / "masks"
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(Path(image_dir).glob("*.jpg"))
        if samplek:
            image_paths = random.sample(image_paths, samplek) if len(image_paths) > samplek else image_paths
            log.info(f"Sampling {samplek} images for inference.")

        # Initialize the dataframe to store inference statistics
        df = pd.DataFrame(columns=["image_name", "image_size", "rescaled_size", "inference_time", "model_name", "device", "out_classes"])

        for image_path in tqdm(image_paths):

            image, original_size, rescaled_size, load_time, preprocess_time = self._preprocess_image(image_path)
            
            if image is None:
                log.error(f"Error processing image: {image_path}")
                continue
            
            # Predict the mask
            mask, inference_time = self._predict_mask(image, out_classes)
            # Resize the predicted mask back to the original image size
            original_height, original_width = original_size
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            mask_filename = mask_output_dir / f"{image_path.stem}.png"
            if out_classes == 1:
                cv2.imwrite(str(mask_filename), mask * 255)
            else:
                cv2.imwrite(str(mask_filename), mask)

            # Add the inference statistics to the DataFrame
            df = pd.concat([df, pd.DataFrame([{
                "image_name": image_path.name,
                "image_size": original_size,
                "rescaled_size": rescaled_size,
                "load_time_s": round(load_time, 4),
                "preprocess_time_s": round(preprocess_time, 4),
                "inference_time_s": round(inference_time, 4),
                "model_name": self.model.__class__.__name__,
                "device": self.model.device.type,
                "out_classes": out_classes
            }])], ignore_index=True)

            log.info(f"Inference for {image_path.name} took {round(inference_time, 4)} seconds.")
        # Organize columns
        df = df[["image_name", "image_size", "rescaled_size", "load_time_s", "preprocess_time_s", "inference_time_s", "model_name", "device", "out_classes"]]
        df.sort_values("image_name", inplace=True)
        # Save the dataframe to a CSV
        csv_path = Path(output_dir, "inference_times.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"Inference times saved to: {csv_path}")
        return image_paths

    def _preprocess_image(self, image_path):
        start_time = time.time()
        image = cv2.imread(str(image_path))
        end_time = time.time()
        image_load_time = end_time - start_time

        start_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        height, width = original_size

        if self.rescale_factor != 1:
            new_height = int(height * self.rescale_factor)
            new_width = int(width * self.rescale_factor)
            # Ensure the new dimensions are divisible by 32 and at least 32
            new_height = max((new_height // 32) * 32, 32)
            new_width = max((new_width // 32) * 32, 32)
            image = cv2.resize(image, (new_width, new_height))
        else:  # when rescale_factor == 1
            if height % 32 != 0 or width % 32 != 0:
                new_height = max((height // 32) * 32, 32)
                new_width = max((width // 32) * 32, 32)
                try:
                    image = cv2.resize(image, (new_width, new_height))
                except Exception as e:
                    return None, None, None, None, None
        
        rescaled_size = image.shape[:2]
        if self.use_normalization:
            image = (image / 255.0 - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.model.device)
        end_time = time.time()
        preprocess_time = end_time - start_time
        return image_tensor, original_size, rescaled_size, image_load_time, preprocess_time

    def _predict_mask(self, image, out_classes):
        with torch.no_grad():
            self.model.eval()
            start_time = time.time()
            logits = self.model(image)
            end_time = time.time()
            inference_time = end_time - start_time
            if out_classes == 1:
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask = (probabilities > self.threshold).astype(np.uint8)
                return mask, inference_time
            elif out_classes == 3:
                pred_masks_softmax = torch.softmax(logits, dim=1)
                # This maybe wrong, need to check
                pred_masks_thresh = (pred_masks_softmax > self.threshold).float()
                pred_masks_argmax = pred_masks_thresh.argmax(dim=1)
                # pred_masks = torch.softmax(logits, dim=1).argmax(dim=1)
                return pred_masks_argmax.squeeze().cpu().numpy(), inference_time

    def save_comparison_plots(self, image_paths, mask_dir, output_dir, out_classes):
        output_dir = Path(output_dir) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Unified color mapping with both color values and names
        color_mapping = {
            0: {"color": [0, 0, 0], "name": 'Background (Black)'},
            1: {"color": [255, 182, 193], "name": 'Light Pink'},
            2: {"color": [173, 216, 230], "name": 'Light Blue'},
            3: {"color": [152, 251, 152], "name": 'Pale Green'},
            4: {"color": [255, 160, 122], "name": 'Light Salmon'},
            5: {"color": [238, 130, 238], "name": 'Violet'},
            255: {"color": [152, 251, 152], "name": 'Vegetation (pale green)'},
            # Add additional classes as needed
        }
        
        for image_path in tqdm(image_paths):
            mask_path = Path(mask_dir) / f"{image_path.stem}.png"
            if not mask_path.exists():
                continue
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Create a color mask image
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            if out_classes == 1:
                 # If binary, use one specific color for class 1
                color_mask[mask >= 1] = color_mapping[3]["color"]  # Use 'Pale Green' for binary segmentation
            
            elif out_classes > 1:
                for key, attributes in color_mapping.items():
                    color_mask[mask == key] = attributes["color"]
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(color_mask)

            # Get the unique values from the mask
            unique_values = [val for val in np.unique(mask) if val != 0]  # Ignore 0 if it's the background
            
            # Generate the title text for the unique values
            title_parts = []
            for value in unique_values:
                if value in color_mapping:
                    class_name = color_mapping[value]["name"]
                    title_parts.append(f"{class_name} (Value: {value})")
                else:
                    title_parts.append(f"Unknown Class (Value: {value})")
            
            # Join all the classes and colors into one title
            title = "Prediction: " + ", ".join(title_parts)
            
            axes[1].set_title(title)
            axes[1].axis("off")
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = output_dir / f"{image_path.stem}_comparison.png"
            log.debug(f"Saving comparison plot to: {plot_filename}")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight", transparent=True)
            plt.close()
        
        log.info(f"Comparison plots saved to: {output_dir}")

class InferencePipeline:
    """Manages the end-to-end inference pipeline."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_checkpoint = Path(cfg.paths.inference.checkpoints, "best.ckpt")
        
        self.image_dir = Path(cfg.inference.input_dir)
        assert self.image_dir.exists(), f"Input directory not found: {self.image_dir}"
        
        self.output_dir = Path(cfg.inference.inf_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mean = cfg.inference.mean
        self.std = cfg.inference.std
        self.out_classes = cfg.model.out_classes

    def run(self):
        mean, std = None, None
        if self.mean and self.std:
            mean = np.array(self.mean)
            std = np.array(self.std)
            # mean, std = load_mean_std_from_json(self.mean_std_file)

        model = SegmentationModule.load_from_checkpoint(
            arch_name=self.cfg.model.arch_name,
            encoder_name=self.cfg.model.encoder_name,
            encoder_weights=self.cfg.model.encoder_weights,
            in_channels=self.cfg.model.in_channels,
            out_classes=self.cfg.model.out_classes,
            mode=self.cfg.model.mode,
            ignore_index=self.cfg.model.ignore_index,
            checkpoint_path=self.model_checkpoint,
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        mask_saver = MaskSaver(model, mean, std, self.cfg.inference.use_normalization, self.cfg.inference.rescale_factor, self.cfg.inference.threshold, self.cfg.inference.plot.dpi, self.cfg.inference.plot.transparent)
        image_paths = mask_saver.save_masks(self.image_dir, self.output_dir, self.cfg.inference.samplek, out_classes=self.out_classes)
        mask_saver.save_comparison_plots(image_paths, self.output_dir / "masks", self.output_dir, self.out_classes)

def copy_hydra_config_to_subfolder(cfg : DictConfig):
    """Copies the entire .hydra directory (or just specific config files) to a target directory."""
    target_dir = Path(cfg.inference.inf_output_dir)
    source_hydra_dir = Path(HydraConfig.get().run.dir) / ".hydra"
    target_hydra_dir = target_dir / ".hydra"
    
    if not source_hydra_dir.exists():
        log.warning(f"No hydra directory found at {source_hydra_dir}")
        return

    log.debug(f"Copying .hydra directory from {source_hydra_dir} to {target_hydra_dir}")
    shutil.copytree(source_hydra_dir, target_hydra_dir, dirs_exist_ok=True)
    

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    nodes = cfg.inference.cuda_visible_devices
    devices_str = ",".join(map(str, nodes))
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    
    random.seed(cfg.inference.seed)
    pipeline = InferencePipeline(cfg)
    pipeline.run()

    copy_hydra_config_to_subfolder(cfg)

if __name__ == "__main__":
    main()
