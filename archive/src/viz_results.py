import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.utils.datasets import SegmentationDataset  # Ensure this points to your dataset class
# from src.utils.model import SegmentationModel
import logging
import uuid

log = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Class to handle the evaluation of a trained segmentation model.
    """

    def __init__(self, model_path: str, classes: list, device: str = "cpu"):
        """
        Initialize the evaluator with the model path and device.
        
        :param model_path: Path to the pre-trained model.
        :param classes: List of class names for segmentation.
        :param device: Device to run the evaluation (e.g., "cpu" or "cuda").
        """
        self.model_path = model_path
        self.classes = ["monocot", "dicot"]
        self.device = torch.device(device)
        self.model = None

    def load_model(self):
        """
        Load the pre-trained model from the specified path.
        """
        log.info(f"Loading model from {self.model_path}")
        
        self.model = smp.create_model(
            arch="FPN",  # Specify architecture if not saved in checkpoint
            encoder_name="resnext50_32x4d",  # Specify encoder name
            in_channels=3,
            classes=len(self.classes),
        )
        
        # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded successfully.")

    def predict(self, images):
        """
        Predict masks for the given batch of images.
        
        :param images: Batch of images.
        :return: Predicted masks.
        """

        images = images.to(self.device).float()
        with torch.no_grad():
            logits = self.model(images.to(self.device))
        
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        return predictions


class Visualizer:
    """
    Class to handle visualization of images, ground truth masks, and predictions.
    """

    @staticmethod
    def visualize(image, gt_mask, pr_mask, viz_output_dir="."):
        """
        Visualize a single sample (image, ground truth mask, and predicted mask).
        
        :param image: Original image.
        :param gt_mask: Ground truth mask.
        :param pr_mask: Predicted mask.
        """
        # Ensure gt_mask and pr_mask are 2D
        if len(gt_mask.shape) == 3:
            log.info("Reducing ground truth mask to 2D.")
            gt_mask = gt_mask.argmax(axis=0)  # For multi-class, reduce to 2D
        if len(pr_mask.shape) == 3:
            log.info("Reducing predicted mask to 2D.")
            pr_mask = pr_mask.argmax(axis=0)  # For multi-class, reduce to 2D

        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image.transpose(1, 2, 0))  # Convert CHW to HWC
        plt.title("Image")
        plt.axis("off")

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="tab20")
        plt.title("Ground Truth")
        plt.axis("off")

        # Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask, cmap="tab20")
        plt.title("Prediction")
        plt.axis("off")
        output_file_path = Path(viz_output_dir, f"output_{uuid.uuid4().hex}.png")
        plt.savefig(output_file_path)


def get_validation_augmentation():
    """
    Add paddings to make image shape divisible by 32.
    """
    return A.Compose([A.PadIfNeeded(384, 480)])

def get_model_dir(model_dir, n):
    if n is None:
        return model_dir
    
    if n in [0, 1]:
        return model_dir
    return model_dir + f"n{n}"

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Initializing evaluation pipeline.")
    
    # Initialize dataset and dataloader
    test_dataset = SegmentationDataset(
        images_dir=cfg.paths.test_image_dir,
        masks_dir=cfg.paths.test_remapped_mask_dir,
        classes=cfg.train.classes,
        # augmentation=get_validation_augmentation(),
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Initialize model evaluator
    model_dir = Path(get_model_dir(cfg.paths.model_dir, cfg.preprocess.viz_results.model_dir_n))
    if not model_dir.exists():
        log.error(f"Model directory {model_dir} does not exist. Stopping the visualization.")
        return
    else:
        log.info(f"Model directory for viz: {model_dir}")
    
    evaluator = ModelEvaluator(
        model_path = model_dir,
        classes=cfg.train.classes, 
        device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
        # device=cfg.eval.device)
    evaluator.load_model()
    
    # Fetch a batch of test samples
    images, gt_masks, _ = next(iter(test_loader))
    
    # Predict masks
    log.info("Running predictions on test samples.")
    predictions = evaluator.predict(images)

    # Visualize results
    log.info("Visualizing predictions.")
    viz_output_dir = Path(model_dir, "viz_results")
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, gt_masks, predictions)):
        if idx < 32:
            Visualizer.visualize(
                image=image.cpu().numpy(),
                gt_mask=gt_mask.cpu().numpy(),
                pr_mask=pr_mask.cpu().numpy(),
                viz_output_dir=viz_output_dir,
            )
        else:
            break


if __name__ == "__main__":
    main()
