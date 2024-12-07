import os
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from src.utils.datasets import Dataset
from src.utils.model import SegmentationModule
from src.utils.viz_utils import viz_batch
from src.utils.augs import train_aug, val_aug

def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print full config for debugging
    
    nodes = cfg.train.cuda_visible_devices 
    devices_str = ",".join(map(str, nodes))    
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    
    set_seed(cfg.dataset.data_seed)
    
    # Logger
    logger = CSVLogger(save_dir=cfg.logger.save_dir, name=None)

    # Directories
    split_data_dir = Path(cfg.paths.train.split_data.split_dir)
    train_image_dir = split_data_dir / "train" / "images"
    train_mask_dir = split_data_dir / "train" / "remapped_masks"
    val_image_dir = split_data_dir / "val" / "images"
    val_mask_dir = split_data_dir / "val" / "remapped_masks"

    checkpoint_dir = Path(logger.log_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.train.callbacks.monitor,
        mode=cfg.train.callbacks.mode,
        save_top_k=cfg.train.callbacks.save_top_k,
        filename="best",
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=cfg.train.callbacks.save_last,
    )
    
    # Datasets and Dataloaders
    train_dataset = Dataset(
        train_image_dir,
        train_mask_dir,
        augmentation=train_aug(cfg),
        normalize_image=cfg.dataset.normalize,
        mean=np.array(cfg.dataset.mean),
        std=np.array(cfg.dataset.std),
    )

    val_dataset = Dataset(
        val_image_dir,
        val_mask_dir,
        augmentation=val_aug(cfg),
        normalize_image=cfg.dataset.normalize,
        mean=np.array(cfg.dataset.mean),
        std=np.array(cfg.dataset.std),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)
    
    sample_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    viz_batch(sample_dataloader, output_dir=logger.log_dir)
    
    del sample_dataloader

    # Model
    model = SegmentationModule(
        arch_name=cfg.model.arch_name,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        out_classes=cfg.model.out_classes,
        mode=cfg.model.mode,
        ignore_index=cfg.model.ignore_index,
    )
    
    if cfg.train.train_strategy == "auto":
        train_strategy = "auto"
    elif cfg.train.train_strategy == "ddp":
        train_strategy = DDPStrategy(find_unused_parameters=True)
    
    # Trainer
    trainer = Trainer(
        **cfg.train.trainer,
        strategy=train_strategy,
        callbacks=[best_checkpoint, last_checkpoint],
        logger=logger,
        # num_nodes=len(nodes)
    )
    trainer.fit(model, train_loader, val_loader)
    
    # Validate
    val_result = trainer.validate(model, val_loader, verbose=False)
    valid_per_image_iou = val_result[0]["valid_per_image_iou"]
    valid_dataset_iou = val_result[0]["valid_dataset_iou"]
    print(f"Validation IoU per images: {valid_per_image_iou}")
    print(f"Validation dataset IoU: {valid_dataset_iou}")

    # Load and Save best full model
    best_model_ckpt_path = best_checkpoint.best_model_path
    best_model = SegmentationModule.load_from_checkpoint(
        arch_name=cfg.model.arch_name,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        out_classes=cfg.model.out_classes,
        mode=cfg.model.mode,
        ignore_index=cfg.model.ignore_index,
        checkpoint_path=best_model_ckpt_path,
    )
    # Create a directory to save the model
    model_dir = Path(logger.log_dir, "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Best model being saved to: {Path(model_dir, 'best_model')}.pth")
    best_model.save_model(model_dir, "best_model")

if __name__ == "__main__":
    main()
