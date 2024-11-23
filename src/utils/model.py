import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from pathlib import Path
import torch
import logging

log = logging.getLogger(__name__)

# Model Class
class SegmentationModel(pl.LightningModule):
    """Custom PyTorch Lightning Module for Segmentation."""
    
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters() # might be too big
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            in_channels=in_channels, 
            classes=out_classes,
            **kwargs
        )
        
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        # Learning rate
        self.lr = lr
        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=False)
        
        self.number_of_classes = out_classes

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize the image and return the logits
        return self.model((image - self.mean) / self.std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1,
            },
                }

    def shared_step(self, batch, stage):
        image, mask = batch

        # log.info(f"Processing {stage} batch with image shape {image.shape} and mask shape {mask.shape}")
        # log.info(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}")
        # log.info(f"Mask min: {mask.min()}, Mask max: {mask.max()}")
        # log.info(f"Mask unique values: {mask.unique()}")
        
        # Ensure the image dimensions are correct
        assert image.dim() == 4, f"Expected 4D input, got {image.dim()}" # [batch_size, channels, H, W]

        # Ensure the mask is a long tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]
        
        # Forward pass (predict the mask logits)
        logits_mask = self.forward(image)

        assert (logits_mask.shape[1] == self.number_of_classes)  # [batch_size, number_of_classes, H, W]
        
        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)
        
        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
        # loss = self.loss_fn(logits_mask, mask.long())
        # return {"loss": loss, "logits_mask": logits_mask, "mask": mask}

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
    
    def save_model(self, save_dir: Path):
        """
        Save the model state dictionary and configuration.
        
        :param save_dir: Directory to save the model.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "arch": self.model.__class__.__name__,
                "encoder_name": self.model.encoder_name,
                "in_channels": self.model.in_channels,
                "out_classes": self.number_of_classes,
            },
            save_dir / "model.pth",
        )

    @classmethod
    def load_model(cls, checkpoint_path: Path):
        """
        Load the model from a saved checkpoint.
        
        :param checkpoint_path: Path to the checkpoint file.
        :return: Instantiated model.
        """
        checkpoint = torch.load(checkpoint_path)
        model = cls(
            checkpoint["arch"],
            checkpoint["encoder_name"],
            checkpoint["in_channels"],
            checkpoint["out_classes"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model