from pathlib import Path
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import logging
from src.utils.dataset import SegmentationDataset
from src.utils.model import SegmentationModel

log = logging.getLogger(__name__)

def get_model_directory(model_dir: Path):
    if model_dir.exists():
        counter = 2
        new_model_dir = model_dir.with_name(f"{model_dir.name}n{counter}")
        while new_model_dir.exists():
            counter += 1
            new_model_dir = model_dir.with_name(f"{model_dir.name}n{counter}")
        model_dir = new_model_dir
    model_dir.mkdir(parents=True)
    return model_dir

# Visualization Utility
def visualize_images(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace("_", " ").title())
        plt.imshow(image.transpose(1, 2, 0) if name == "image" else image, cmap="tab20")
    plt.show()

# Augmentation Functions
def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True),
        A.RandomCrop(height=512, width=512, always_apply=True),
        A.GaussNoise(p=0.2),
        A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.9),
        A.OneOf([A.Blur(p=1), A.MotionBlur(p=1)], p=0.9),
    ])

def get_validation_augmentation():
    return A.Compose([A.PadIfNeeded(384, 480)])

# Training Pipeline
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Initializing datasets and dataloaders.")
    
    train_dataset = SegmentationDataset(
        cfg.paths.split_data.train_image_dir,
        cfg.paths.split_data.train_remapped_mask_dir,
        # augmentation=get_training_augmentation(),
        classes=cfg.train.classes,
    )
    val_dataset = SegmentationDataset(
        cfg.paths.split_data.val_image_dir,
        cfg.paths.split_data.val_remapped_mask_dir,
        # augmentation=get_validation_augmentation(),
        classes=cfg.train.classes
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=cfg.train.shuffle, 
        num_workers=cfg.train.num_workers
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers
        )

    model = SegmentationModel(
        cfg.model.arch, 
        cfg.model.encoder, 
        in_channels=cfg.model.in_channels, 
        out_classes=len(train_dataset.CLASSES),
        lr=cfg.train.lr,
        )
    
    trainer = pl.Trainer(max_epochs=cfg.train.epochs, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    # Save the model
    save_model_dir = get_model_directory(Path(cfg.paths.model_dir))

    try:
        # save the model
        model.model.save_pretrained(save_model_dir)
        log.info(f"Model saved to {save_model_dir} using model.model.save_pretrained method.")
    except Exception as e:
        log.exception("Failed to save model using model.model.save_pretrained method.")


if __name__ == "__main__":
    main()
