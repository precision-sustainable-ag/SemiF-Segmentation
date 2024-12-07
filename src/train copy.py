import os
import hydra
from pathlib import Path
from omegaconf import DictConfig
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.augs import default_train_aug, default_val_aug
from src.utils.datasets import Dataset
from src.utils.model import SegmentationModule
from src.utils.viz_utils import viz_batch


def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Training Pipeline
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Global Constants
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    data_root = "data/COMBINED_VEG_TEST"
    project_name = "COMBINED_VEG_TEST"
    size = 256
    ARCH_NAME="DeepLabV3Plus"
    ENCODER_NAME="resnet152"
    ENCODER_WEIGHTS="imagenet"
    AUGMENT = False
    EPOCHS = 50
    IGNORE_INDEX = None # Label that indicates ignored pixels (does not contribute to loss) int or None
    IN_CHANNELS = 3
    OUT_CLASSES = 1
    MODE = "binary" # "binary" or "multiclass"
    DIST_TRAIN = False
    BATCH_SIZE = 12
    
    IMG_HW = (256,256)
    NORMALIZE = True

    NUM_WORKERS = 12
    DATA_SEED = 42
    
    
    assert OUT_CLASSES == 1 if MODE == "binary" else OUT_CLASSES > 1, "OUT_CLASSES must be 1 if MODE is 'binary'"

    mean = np.array([
        0.3966197412901707,
        0.4072120533419924,
        0.26327527369052733
        ])
    std = np.array([
        0.17277786815898163,
        0.1814243603695206,
        0.14668307851972828
        ])
    
    kwargs = {
        "img_hw": IMG_HW,
        "normalize_image": NORMALIZE,
        "mean": mean,
        "std": std, 
        "data_seed": DATA_SEED
    }
    
    set_seed(DATA_SEED)

    # Initialize CSV logger
    logger = CSVLogger(
        save_dir="lightning_logs",  # Set the directory to save logs
        name=project_name,        # Name of the experiment
    )

    # Directories
    
    train_image_dir = Path(f"{data_root}/train_val_test_{size}x{size}/train/images")
    train_mask_dir = Path(f"{data_root}/train_val_test_{size}x{size}/train/remapped_masks")

    val_image_dir = Path(f"{data_root}/train_val_test_{size}x{size}/val/images")
    val_mask_dir = Path(f"{data_root}/train_val_test_{size}x{size}/val/remapped_masks")
    
    # Use the logger's directory for checkpoints
    checkpoint_dir = Path(logger.log_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,         # Base directory for checkpoints
        monitor="valid_dataset_iou",    # Metric to monitor
        mode="max",                     # Save the model with the highest IoU
        save_top_k=1,                   # Save only the best model
        filename="best",
    )

    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,     # Base directory for checkpoints
        save_last=True,             # Save the last model
        save_top_k=0,               # Do not save any other models
    )
    
    # Datasets and Dataloaders
    train_dataset = Dataset(
        train_image_dir,
        train_mask_dir,
        augmentation=default_train_aug(img_hw=IMG_HW) if AUGMENT else None,
        normalize_image=NORMALIZE,
        mean=mean,
        std=std
    )

    val_dataset = Dataset(
        val_image_dir,
        val_mask_dir,
        augmentation=default_val_aug(img_hw=IMG_HW) if AUGMENT else None,
        normalize_image=NORMALIZE,
        mean=mean,
        std=std
    )

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    example_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    viz_batch(example_loader, output_dir=logger.log_dir)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    # Model
    model = SegmentationModule(
        arch_name=ARCH_NAME,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        mode=MODE,
        ignore_index=IGNORE_INDEX,
        kwargs=kwargs
    )

    # Trainer
    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[best_checkpoint, last_checkpoint],
        deterministic=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=logger,
        enable_model_summary=True,
        strategy=DDPStrategy(find_unused_parameters=True) if DIST_TRAIN else "auto", # For multi-GPU training
        num_sanity_val_steps=0,
        )
    
    # Train
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
        )

    # Validate
    val_result = trainer.validate(model, val_loader, verbose=False)
    valid_per_image_iou = val_result[0]["valid_per_image_iou"]
    valid_dataset_iou = val_result[0]["valid_dataset_iou"]
    print(f"Validation IoU per images: {valid_per_image_iou}")
    print(f"Validation dataset IoU: {valid_dataset_iou}")

    # Load and Save best full model
    best_model_ckpt_path = best_checkpoint.best_model_path
    best_model = SegmentationModule.load_from_checkpoint(
        arch_name=ARCH_NAME,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        mode=MODE,
        ignore_index=IGNORE_INDEX,
        checkpoint_path=best_model_ckpt_path,
    )
    # Create a directory to save the model
    model_dir = Path(logger.log_dir, "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Best model being saved to: {Path(model_dir, 'best_model')}.pth")
    best_model.save_model(model_dir, "best_model")
    
if __name__ == "__main__":
    main()