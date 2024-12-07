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
from src.utils.model import SegmentationModule

log = logging.getLogger(__name__)

def load_mean_std_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    mean = np.array(data["mean"])
    std = np.array(data["std"])
    return mean, std

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
            image_paths = random.sample(image_paths, samplek)
            log.info(f"Sampling {samplek} images for inference.")

        # Initialize the dataframe to store inference statistics
        df = pd.DataFrame(columns=["image_name", "image_size", "rescaled_size", "inference_time", "model_name", "device", "out_classes"])

        for image_path in tqdm(image_paths):
            image, original_size, rescaled_size, load_time, preprocess_time = self._preprocess_image(image_path)
            mask, inference_time = self._predict_mask(image, out_classes)
            
            mask_filename = mask_output_dir / f"{image_path.stem}_mask.png"
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

        if self.rescale_factor != 1:
            height, width = image.shape[:2]
            new_height = int(height * self.rescale_factor)
            new_width = int(width * self.rescale_factor)
            new_height = (new_height // 32) * 32
            new_width = (new_width // 32) * 32
            image = cv2.resize(image, (new_width, new_height))
        
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
            start_time = time.time()
            logits = self.model(image)
            end_time = time.time()
            inference_time = end_time - start_time
            
            if out_classes == 1:
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
                return (probabilities > self.threshold).astype(np.uint8), inference_time
            elif out_classes > 1:
                pred_masks_softmax = torch.softmax(logits, dim=1)
                pred_masks_argmax = pred_masks_softmax.argmax(dim=1)
                return pred_masks_argmax.squeeze().cpu().numpy(), inference_time


def load_model(checkpoint_path, model_args):
    """Load the model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SegmentationModule(**model_args)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def run_inference(cfg):
    """Runs the inference pipeline."""
    checkpoint_path = cfg['model_checkpoint']
    model_args = cfg['model']
    model = load_model(checkpoint_path, model_args)
    mean, std = np.array(cfg['mean']), np.array(cfg['std'])

    mask_saver = MaskSaver(
        model=model,
        mean=mean,
        std=std,
        use_normalization=cfg['use_normalization'],
        rescale_factor=cfg['rescale_factor'],
        threshold=cfg['threshold'],
    )

    mask_saver.save_masks(cfg['image_dir'], cfg['output_dir'], samplek=cfg.get('samplek', None), out_classes=cfg['out_classes'])


if __name__ == "__main__":
    config = {
        'model_checkpoint': 'path/to/checkpoint.pth',
        'image_dir': 'path/to/images',
        'output_dir': 'path/to/output',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'use_normalization': True,
        'rescale_factor': 0.5,
        'threshold': 0.5,
        'out_classes': 3,
        'samplek': 10,
        'model': {
            'arch_name': 'UnetPlusPlus',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'in_channels': 3,
            'out_classes': 3,
            'mode': 'multiclass',
        }
    }
    run_inference(config)
