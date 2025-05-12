import os
import cv2
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path
import shutil
import hydra
from hydra.core.hydra_config import HydraConfig


from omegaconf import DictConfig

from src.utils.model import SegmentationModule

log = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, mean, std, cfg):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.rescale_factor = cfg.inference.inference.rescale_factor 
        self.use_normalization = cfg.inference.inference.use_normalization
        self.bbox_crop = cfg.inference.inference.bbox_crop

    def preprocess(self, image_path, device):
        load_start = time.time()
        metadata_path = str(image_path).replace("images", "metadata").replace(".jpg", ".json")
        image = cv2.imread(str(image_path))
        load_time = time.time() - load_start

        if image is None:
            return None, None, None, load_time, 0.0

        preprocess_start = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.bbox_crop and Path(metadata_path).exists():
            crops = self._get_bbox_crops(image, metadata_path)
            if not crops:
                return None, None, None, load_time, time.time() - preprocess_start

            # Stack into tensor: [N, C, H, W]
            # tensor_batch = torch.stack([self._to_tensor(crop, device) for crop in crops])
            tensors = [self._to_tensor(crop, device).unsqueeze(0) for crop in crops]  # keep [1, C, H, W] per crop
            preprocess_time = time.time() - preprocess_start
            # return tensors, original_size, None, load_time, preprocess_time
            return tensors, crops, original_size, load_time, preprocess_time  # return crops too
        else:
            image = self._resize_image(image)
            tensor = self._to_tensor(image, device).unsqueeze(0)  # Shape: [1, C, H, W]
            preprocess_time = time.time() - preprocess_start
            return tensor, image.shape[:2], original_size, load_time, preprocess_time

    def _resize_image(self, image):
        height, width = image.shape[:2]
        new_height = int(height * self.rescale_factor)
        new_width = int(width * self.rescale_factor)
        new_height = max((new_height // 32) * 32, 32)
        new_width = max((new_width // 32) * 32, 32)
        return cv2.resize(image, (new_width, new_height))
    
    def _to_tensor(self, image, device):
        if self.use_normalization:
            image = (image / 255.0 - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32, device=device)

    def _get_bbox_crops(self, image, metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        crops = []
        for ann in metadata.get("annotations", []):
            bbox = ann.get("bbox_xywh")
            if bbox:
                x_min, y_min, width, height = bbox
                x_max, y_max = x_min + width, y_min + height
                x_min, y_min = max(int(x_min), 0), max(int(y_min), 0)
                x_max, y_max = min(int(x_max), image.shape[1]), min(int(y_max), image.shape[0])

                crop = image[y_min:y_max, x_min:x_max]
                if crop.size > 0:
                    crop = self._resize_image(crop)
                    crops.append(crop)
        return crops


class Predictor:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, image_tensor: torch.Tensor, out_classes: int):
        """
        image_tensor: shape [B, C, H, W] (B=1 for full image, or B=N for bbox crops)
        Returns: mask(s) in shape [H, W] if B=1, or list of [H, W] if B > 1
        """
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            outputs = self.model(image_tensor)
            inference_time = time.time() - start

            # If model returns logits, apply softmax/sigmoid as needed
            if out_classes == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > self.threshold).float()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            # preds shape: [B, H, W]
            preds = preds.squeeze(1) if out_classes == 1 else preds

            if preds.shape[0] == 1:
                # Single image case: return HxW numpy array
                mask = preds[0].cpu().numpy().astype(np.uint8)
                return mask, inference_time
            else:
                # Batch of bbox crops: return list of HxW numpy arrays
                masks = [pred.cpu().numpy().astype(np.uint8) for pred in preds]
                return masks, inference_time


class Visualizer:
    def __init__(self, output_dir: Path, plot_cfg, dpi: int = 300, transparent: bool = False):

        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.transparent = transparent
        self.plot_cfg = plot_cfg
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
        self.plots_dst = self.output_dir / "plots"
        self.plots_dst.mkdir(parents=True, exist_ok=True) if self.plot_comparisons else None
        self.overlays_dst = self.output_dir / "overlays"
        self.overlays_dst.mkdir(parents=True, exist_ok=True) if self.plot_overlays else None

    def draw_overlay(self, image_path: Path, mask, out_classes: int, bbox_idx: int = None, crop_image=None):
        """
        Draw and save overlay (only if plot_overlays is True)
        """
        if not self.plot_overlays:
            return  # Skip if overlays are disabled

        if crop_image is not None:
            image = crop_image
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Failed to load {image_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != mask.shape:
                # resize the image to match the mask shape
                image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        overlay_img = self._apply_mask_overlay(image, mask, out_classes)

        # Plot with matplotlib
        fig, ax = plt.subplots(figsize=(image.shape[1] / self.dpi, image.shape[0] / self.dpi), dpi=self.dpi)
        ax.imshow(overlay_img)
        ax.axis('off')


        # Save overlay
        suffix = f"_bbox{bbox_idx}" if bbox_idx is not None else ""
        out_path = self.overlays_dst / f"{image_path.stem}{suffix}_overlay.png"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=self.transparent)
        plt.close(fig)

    def draw_comparison(self, image_path: Path, mask, out_classes: int, bbox_idx: int = None, crop_image=None):
        """
        Draw and save input/mask side-by-side (only if plot_comparisons is True)
        """
        if not self.plot_comparisons:
            return

        if crop_image is not None:
            image = crop_image
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Failed to load {image_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != mask.shape:
                # resize the image to match the mask shape
                image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image shape {image.shape[:2]} does not match mask shape {mask.shape}")

        overlay_img = self._apply_mask_overlay(image, mask, out_classes)

        # Comparison plot: side by side
        fig, axs = plt.subplots(1, 2, figsize=(2 * image.shape[1] / self.dpi, image.shape[0] / self.dpi), dpi=self.dpi)
        axs[0].imshow(image)
        # axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(overlay_img)
        # axs[1].set_title("Overlay")
        axs[1].axis('off')

        # Save comparison
        suffix = f"_bbox{bbox_idx}" if bbox_idx is not None else ""
        out_path = self.plots_dst / f"{image_path.stem}{suffix}_comparison.png"
        
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=self.transparent)
        plt.close(fig)

    def _apply_mask_overlay(self, image, mask, out_classes):
        """
        Apply the mask overlay onto the image and return the overlayed image.
        """
        overlay_img = image.copy()
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image shape {image.shape[:2]} does not match mask shape {mask.shape}")


        if out_classes == 1:
            green_overlay = np.zeros_like(image)
            green_overlay[:, :, 1] = 255  # Green mask
            mask_bool = mask.astype(bool)
            if mask_bool.any():  # <-- Skip empty masks
                overlay_img[mask_bool] = cv2.addWeighted(image[mask_bool], 0.7, green_overlay[mask_bool], 0.3, 0)
            
        else:
            colors = np.random.randint(0, 255, (out_classes, 3), dtype=np.uint8)
            for cls in range(1, out_classes):  # skip background
                mask_bool = (mask == cls)
                if mask_bool.any():  # <-- Skip empty masks
                    overlay_img[mask_bool] = cv2.addWeighted(image[mask_bool], 0.7, colors[cls], 0.4, 0)

        return overlay_img


class InferenceRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_ckpt = Path(cfg.paths.inference.checkpoints) / "best.ckpt"
        self.image_dir = Path(cfg.paths.project_inference_dir) / "data" / "images"
        self.metadata_dir = Path(cfg.paths.project_inference_dir) / "data" / "metadata"
        self.output_dir = Path(cfg.paths.inference_output_dir) / "results"
        self.project_preprocess_dir = Path(cfg.paths.project_preprocess_dir)
        self.output_mask_dir = self.output_dir / "masks"


        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)
        self.mean, self.std = self._load_stats()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_classes = cfg.model.out_classes

        self.bbox_crop = self.cfg.inference.inference.bbox_crop

    def _load_stats(self):
        stats_file = self.project_preprocess_dir / "data/data_stats/rgb_mean_std.json"
        if stats_file.exists():
            with open(stats_file) as f:
                data = json.load(f)
                return np.array(data['mean']), np.array(data['std'])
        return self.cfg.inference.inference.mean, self.cfg.inference.inference.std

    def run(self):
        model = SegmentationModule.load_from_checkpoint(
            checkpoint_path=self.model_ckpt,
            cfg=self.cfg,
        ).to(self.device)

        processor = ImageProcessor(self.mean, self.std, self.cfg)
        predictor = Predictor(model, self.cfg.inference.inference.threshold)
        visualizer = Visualizer(self.output_dir,self.cfg.inference.inference.plot, self.cfg.inference.inference.plot.dpi, self.cfg.inference.inference.plot.transparent)

        paths = sorted(self.image_dir.glob("*.jpg"))
        
        stats = []
        for path in tqdm(paths):
            tensor_result = processor.preprocess(path, self.device)
            if tensor_result[0] is None:
                log.warning(f"Skipping {path} due to None tensor result.")
                continue

            tensors, crops, original_size, load_time, prep_time = tensor_result

            if isinstance(tensors, list):
                masks = []
                inf_times = []
                for t in tensors:
                    mask, inf_time = predictor.predict(t, self.out_classes)
                    masks.append(mask)
                    inf_times.append(inf_time)
                inf_time = sum(inf_times)
                for idx, (mask, crop_img) in enumerate(zip(masks, crops)):
                    mask_path = self.output_mask_dir / f"{path.stem}_bbox{idx}.png"
                    cv2.imwrite(str(mask_path), mask * 255 if self.out_classes == 1 else mask)
                    visualizer.draw_overlay(path, mask, self.out_classes, bbox_idx=idx, crop_image=crop_img)
                    visualizer.draw_comparison(path, mask, self.out_classes, bbox_idx=idx, crop_image=crop_img)

            else:
                # Single full image
                mask, inf_time = predictor.predict(tensors, self.out_classes)
                mask_path = self.output_mask_dir / f"{path.stem}.png"
                cv2.imwrite(str(mask_path), mask * 255 if self.out_classes == 1 else mask)
                visualizer.draw_overlay(path, mask, self.out_classes)
                visualizer.draw_comparison(path, mask, self.out_classes)
        
            crop_shapes = [crop.shape for crop in crops] if self.bbox_crop else None
            stats.append({
                "image_name": path.name,
                "image_size": original_size,
                "rescaled/cropped_size": crop_shapes if self.bbox_crop else tensors[0].shape[2:],
                "load_time_s": round(load_time, 4),
                "preprocess_time_s": round(prep_time, 4),
                "inference_time_s": round(inf_time, 4),
                "model_name": model.__class__.__name__,
                "device": self.device.type,
                "out_classes": self.out_classes
            })

            # visualizer.draw_overlay(path, mask, self.out_classes)

        pd.DataFrame(stats).to_csv(self.output_dir / "inference_times.csv", index=False)
        log.info(f"Saved all predictions and stats to {self.output_dir}")

def copy_hydra_config(cfg):
    target = Path(cfg.paths.inference_output_dir) / ".hydra"
    source = Path(HydraConfig.get().run.dir) / ".hydra"
    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.inference.inference.cuda_visible_devices))
    random.seed(cfg.inference.inference.seed)
    InferenceRunner(cfg).run()
    copy_hydra_config(cfg)

if __name__ == "__main__":
    main()