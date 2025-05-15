import os
import hydra
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import numpy as np
import json
import torch
import logging
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from src.utils.datasets import Dataset
from src.utils.model import SegmentationModule
from src.utils.viz_utils import viz_batch, side_by_side_plot, pad_to_multiple
from src.utils.augs import train_aug, val_aug

log = logging.getLogger(__name__)


def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

def load_stats(stats_file: Path):
        # stats_file = self.project_dir / "preprocess/data/data_stats/rgb_mean_std.json"
    if stats_file.exists():
        with open(stats_file) as f:
            data = json.load(f)
            return np.array(data['mean']), np.array(data['std'])
    return None, None

@rank_zero_only
def log_run_info(logger: CSVLogger, cfg: DictConfig, val_result: list[dict]):
    """
    Log run information to the logger for github actions.
    """
    valid_per_image_iou = val_result[0]["valid_per_image_iou"]
    valid_dataset_iou = val_result[0]["valid_dataset_iou"]

    run_info = {
        "project_name": cfg.project.name,
        "timestamp": datetime.now().isoformat(),
        "version_dir": str(logger.log_dir),
        
        "model_name": cfg.model.arch_name,
        "encoder": cfg.model.encoder_name,
        "encoder_weights": cfg.model.encoder_weights,
        "loss_function": cfg.train.loss.name,
        "base_lr": cfg.train.lr.base_lr,
        "num_GPUs": len(list(cfg.train.cuda_visible_devices)),
        "num_epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "image_size": f"{cfg.preprocess.grid_crop.crop_height}x{cfg.preprocess.grid_crop.crop_width}",
        "training_images": len(list(Path(cfg.paths.split_dir, "train", "images").glob("*.jpg"))),
        "validation_images": len(list(Path(cfg.paths.split_dir, "val", "images").glob("*.jpg"))),

        "valid_dataset_iou": valid_dataset_iou,
        "valid_per_image_iou": valid_per_image_iou,
        
        }

    Path("logs").mkdir(exist_ok=True)
    with open("logs/run_info.json", "w") as f:
        json.dump(run_info, f, indent=4)
    print(f"Run info saved to: logs/run_info.json")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print full config for debugging
    
    nodes = cfg.train.cuda_visible_devices 
    devices_str = ",".join(map(str, nodes))    
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    
    set_seed(cfg.train.seed)
    stats_file = Path(cfg.paths.project_preprocess_dir) / "data/data_stats/rgb_mean_std.json"
    mean, std = load_stats(stats_file)
    # Logger
    logger = CSVLogger(save_dir=cfg.paths.project_mode_dir, name=None)

    # Directories
    split_data_dir = Path(cfg.paths.split_dir)
    train_image_dir = split_data_dir / "train" / "images"
    train_mask_dir = split_data_dir / "train" / "masks"
    val_image_dir = split_data_dir / "val" / "images"
    val_mask_dir = split_data_dir / "val" / "masks"
    test_image_dir = split_data_dir / "test" / "images"
    test_mask_dir = split_data_dir / "test" / "masks"

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
        normalize_image=cfg.train.dataset.normalize,
        mean=mean,
        std=std,
    )

    val_dataset = Dataset(
        val_image_dir,
        val_mask_dir,
        augmentation=val_aug(cfg),
        normalize_image=cfg.train.dataset.normalize,
        mean=mean,
        std=std,
    )

    test_dataset = Dataset(
        test_image_dir,
        test_mask_dir,
        augmentation=val_aug(cfg),
        normalize_image=cfg.train.dataset.normalize,
        mean=mean,
        std=std,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.dataset.workers,
        pin_memory=True,
        drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.dataset.workers
        )
    
    
    sample_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.dataset.workers
        )
    viz_batch(sample_dataloader, output_dir=logger.log_dir)
    
    del sample_dataloader

    # Model
    model = SegmentationModule(cfg)
    
    if cfg.train.train_strategy == "auto":
        train_strategy = "auto"
    elif cfg.train.train_strategy == "ddp":
        train_strategy = DDPStrategy(find_unused_parameters=False)
    
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
    
    try: 
        log_run_info(logger, cfg, val_result)
    except Exception as e:
        print(f"Error logging run info: {e}")

    # Load and Save best full model
    best_model_ckpt_path = best_checkpoint.best_model_path
    best_model = SegmentationModule.load_from_checkpoint(
        cfg=cfg,
        checkpoint_path=best_model_ckpt_path,
    )
    # Create a directory to save the model
    model_dir = Path(logger.log_dir, "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Best model being saved to: {Path(model_dir, 'best_model')}.pth")
    best_model.save_model(model_dir, "best_model")

    # === Sample Inference on 5 Validation Images === #

    class_colors = {
        0: [0, 0, 0],          # background -> black
        1: [255, 182, 193],        # monocot -> green
        2: [173, 216, 230],        # dicot -> red
    }

    class_labels = {
        0: "Background/soil",
        1: "Grass",
        2: "Hairy vetch",
    }
    log.info("Running sample inference on validation images...")

    best_model.eval()  # set model to eval mode

    output_dir = Path(logger.log_dir) / "sample_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    sample = test_dataset[0]
    image = sample[0].unsqueeze(0).to(best_model.device)  # add batch dim
    mask_gt = sample[1].cpu().numpy().squeeze()
    mask_stem = sample[2]
    img_path = test_image_dir / f"{mask_stem}.jpg"

    with torch.no_grad():
        output = best_model(image)
        mask_pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Save side-by-side plot
    output_path = output_dir / f"test_prediction.png"

    side_by_side_plot(img_path, mask_gt, mask_pred, output_path, class_colors=class_colors, class_labels=class_labels)

    log.info(f"Saved sample predictions to {output_dir}")

    # === Sample Inference on Unlabeled Images === #
    log.info("Running inference on sample unlabeled images...")

    inference_images = [Path(cfg.paths.persistent_dir, x) for x in cfg.train.sample_inference]  # List of image paths

    # Dummy mask paths just to satisfy Dataset interface
    dummy_masks = inference_images  # dummy paths, ignored during inference

    # Load Dataset
    inference_dataset = Dataset(
        images_dir=inference_images,
        masks_dir=dummy_masks,  # dummy
        augmentation=None,      # no augmentation needed
        normalize_image=cfg.train.dataset.normalize,
        mean=mean,
        std=std
    )

    # Output folder for inference results
    output_dir_infer = Path(logger.log_dir) / "sample_inference"
    output_dir_infer.mkdir(parents=True, exist_ok=True)
  
    # Inference loop
    for idx in range(len(inference_dataset)):
        image_tensor, _, mask_stem = inference_dataset[idx]  # ignore dummy mask
        # Pad to multiple of 16
        if "deeplab" in cfg.model.arch_name.lower():
            multiple = 16
        elif "segformer" in cfg.model.arch_name.lower():
            multiple = 32
        else:
            multiple = 16
            log.info(f"Unknown model architecture: {cfg.model.arch_name}. Using default padding multiple of 16.")
        
        image_padded = pad_to_multiple(image_tensor, multiple=multiple)

        image = image_padded.unsqueeze(0).to(best_model.device)

        with torch.no_grad():
            output = best_model(image)
            mask_pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Save visualization
        output_path = output_dir_infer / f"inference{idx}.png"
        img_path = Path(inference_images[0]).parent / f"{mask_stem}.jpg"
        side_by_side_plot(img_path, None, mask_pred, output_path, class_colors=class_colors, class_labels=class_labels)

    log.info(f"Saved unlabeled inference predictions to {output_dir_infer}")

if __name__ == "__main__":
    main()
