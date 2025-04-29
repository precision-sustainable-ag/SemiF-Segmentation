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


class ImageProcessor:
    def __init__(self, mean, std, rescale_factor=0.5, use_normalization=True):
        self.mean = mean
        self.std = std
        self.rescale_factor = rescale_factor
        self.use_normalization = use_normalization

    def preprocess(self, image_path, device):
        load_start = time.time()
        image = cv2.imread(str(image_path))
        load_time = time.time() - load_start

        preprocess_start = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        image = self._resize_image(image)
        rescaled_size = image.shape[:2]

        if self.use_normalization:
            image = (image / 255.0 - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
        preprocess_time = time.time() - preprocess_start

        return image_tensor, original_size, rescaled_size, load_time, preprocess_time

    def _resize_image(self, image):
        height, width = image.shape[:2]
        new_height = int(height * self.rescale_factor)
        new_width = int(width * self.rescale_factor)
        new_height = max((new_height // 32) * 32, 32)
        new_width = max((new_width // 32) * 32, 32)
        return cv2.resize(image, (new_width, new_height))


class Predictor:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, image_tensor, out_classes):
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            logits = self.model(image_tensor)
            inference_time = time.time() - start

            if out_classes == 1:
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                return (prob > self.threshold).astype(np.uint8), inference_time
            elif out_classes > 1:
                softmax = torch.softmax(logits, dim=1)
                argmax = (softmax > self.threshold).float().argmax(dim=1)
                return argmax.squeeze().cpu().numpy(), inference_time


class Visualizer:
    def __init__(self, output_dir, plot_cfg, dpi=300, transparent=True):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.transparent = transparent
        self.plot_comparisons = plot_cfg.comparisons
        self.plot_overlays = plot_cfg.overlays
        self._setup_dirs()
        self.color_mapping = {
            0: [0, 0, 0],
            1: [255, 182, 193],
            2: [173, 216, 230],
            3: [152, 251, 152],
            4: [255, 160, 122],
            5: [238, 130, 238],
            255: [152, 251, 152]
        }

    def _setup_dirs(self):
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True) if self.plot_comparisons else None
        (self.output_dir / "overlays").mkdir(parents=True, exist_ok=True) if self.plot_overlays else None

    def draw_overlay(self, image_path, mask, out_classes):
        
        if not self.plot_overlays and not self.plot_comparisons:
            return
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in self.color_mapping.items():
            overlay[mask == cls] = color

        if self.plot_overlays:
            # Ensure overlay is same shape as image
            if overlay.shape != image.shape:
                overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Blend image and overlay
            blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

            # Save overlay image
            overlay_path = self.output_dir / "overlays" / f"{image_path.stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        if self.plot_comparisons:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image)
            axes[0].axis("off")
            axes[0].set_title("Original")
            axes[1].imshow(overlay)
            axes[1].axis("off")
            axes[1].set_title("Prediction")

            plot_path = self.output_dir / "plots" / f"{image_path.stem}_comparison.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight", transparent=self.transparent)
            plt.close()


class InferenceRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_ckpt = Path(cfg.paths.inference.checkpoints) / "best.ckpt"
        self.image_dir = Path(cfg.inference.input_dir)
        self.output_dir = Path(cfg.paths.inference_output_dir)
        self.project_dir = Path(cfg.paths.project_mode_dir)
        self.output_mask_dir = self.output_dir / "masks"


        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)
        self.mean, self.std = self._load_stats()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_classes = cfg.model.out_classes

    def _load_stats(self):
        stats_file = self.project_dir / "preprocess/data/data_stats/rgb_mean_std.json"
        if stats_file.exists():
            with open(stats_file) as f:
                data = json.load(f)
                return np.array(data['mean']), np.array(data['std'])
        return self.cfg.inference.mean, self.cfg.inference.std

    def run(self):
        model = SegmentationModule.load_from_checkpoint(
            checkpoint_path=self.model_ckpt,
            cfg=self.cfg,
        ).to(self.device)

        processor = ImageProcessor(self.mean, self.std, self.cfg.inference.rescale_factor, self.cfg.inference.use_normalization)
        predictor = Predictor(model, self.cfg.inference.threshold)
        visualizer = Visualizer(self.output_dir,self.cfg.inference.plot, self.cfg.inference.plot.dpi, self.cfg.inference.plot.transparent)

        paths = sorted(self.image_dir.glob("*.jpg"))
        if self.cfg.inference.samplek:
            paths = random.sample(paths, self.cfg.inference.samplek)

        stats = []
        for path in tqdm(paths):
            tensor, original_size, rescaled_size, load_time, prep_time = processor.preprocess(path, self.device)
            if tensor is None:
                continue
            mask, inf_time = predictor.predict(tensor, self.out_classes)
            mask_path = self.output_mask_dir / f"{path.stem}.png"
            cv2.imwrite(str(mask_path), mask * 255 if self.out_classes == 1 else mask)

            stats.append({
                "image_name": path.name,
                "image_size": original_size,
                "rescaled_size": rescaled_size,
                "load_time_s": round(load_time, 4),
                "preprocess_time_s": round(prep_time, 4),
                "inference_time_s": round(inf_time, 4),
                "model_name": model.__class__.__name__,
                "device": self.device.type,
                "out_classes": self.out_classes
            })

            visualizer.draw_overlay(path, mask, self.out_classes)

        pd.DataFrame(stats).to_csv(self.output_dir / "inference_times.csv", index=False)
        log.info(f"Saved all predictions and stats to {self.output_dir}")


def copy_hydra_config(cfg):
    target = Path(cfg.paths.inference_output_dir) / ".hydra"
    source = Path(HydraConfig.get().run.dir) / ".hydra"
    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.inference.cuda_visible_devices))
    random.seed(cfg.inference.seed)
    InferenceRunner(cfg).run()
    copy_hydra_config(cfg)


if __name__ == "__main__":
    main()