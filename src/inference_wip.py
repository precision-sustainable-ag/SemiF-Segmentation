import os
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

import hydra
from omegaconf import DictConfig


from src.utils.model import SegmentationModule

log = logging.getLogger(__name__)


class MaskPredictor:
    """Handles saving predicted masks and comparison plots."""
    def __init__(self, model, mean=None, std=None, use_normalization=True, rescale_factor=0.5, threshold=0.5):
        self.model = model
        self.mean = mean if mean is not None else 0
        self.std = std if std is not None else 1
        self.use_normalization = use_normalization
        self.rescale_factor = rescale_factor
        self.threshold = threshold

    def _preprocess_image(self, image_path):
        image = cv2.imread(str(image_path))

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
                    return None, None
        
        if self.use_normalization:
            image = (image / 255.0 - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.model.device)
        
        return image_tensor, original_size
    
    def _inference_model(self, image, out_classes):
        with torch.no_grad():
            self.model.eval()
            
            logits = self.model(image)
            
            if out_classes == 1:
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
                return (probabilities > self.threshold).astype(np.uint8)
            elif out_classes == 3:
                pred_masks_softmax = torch.softmax(logits, dim=1)
                pred_masks_thresh = (pred_masks_softmax > self.threshold).float()
                pred_masks_argmax = pred_masks_thresh.argmax(dim=1)
                return pred_masks_argmax.squeeze().cpu().numpy()
    
    def predict(self, image_path, out_classes=1):        
        image, original_size = self._preprocess_image(image_path)
        
        if image is None:
            log.error(f"Error processing image: {image_path}")
            return None
        
        # Predict the mask
        mask, _ = self._inference_model(image, out_classes)
        # Resize the predicted mask back to the original image size
        original_height, original_width = original_size
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        return mask

        
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    nodes = cfg.inference.cuda_visible_devices
    devices_str = ",".join(map(str, nodes))
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    
    model_checkpoint = Path(cfg.paths.inference.checkpoints, "best.ckpt")
    
    model = SegmentationModule.load_from_checkpoint(
            arch_name=cfg.model.arch_name,
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            out_classes=cfg.model.out_classes,
            mode=cfg.model.mode,
            ignore_index=cfg.model.ignore_index,
            checkpoint_path=model_checkpoint,
        )

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    mean = cfg.inference.mean
    std = cfg.inference.std

    predictor = MaskPredictor(model, mean, std, use_normalization=True)

    image_dir = Path(cfg.inference.input_dir)
    out_classes = cfg.model.out_classes
    for image_path in tqdm(image_dir.glob("*")):
            mask = predictor.predict(image_path, out_classes)
            if mask is None:
                continue

if __name__ == "__main__":
    main()
