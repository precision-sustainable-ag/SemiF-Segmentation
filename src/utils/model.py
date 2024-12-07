import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

log = logging.getLogger(__name__)

class SegmentationModule(pl.LightningModule):
    def __init__(self, arch_name, encoder_name, encoder_weights, in_channels, out_classes, mode='binary', ignore_index=None, **kwargs):
        """
        Unified segmentation module supporting multiple model architectures.
        
        Args:
            arch (str): Type of model, e.g., 'Unet', 'UnetPlusPlus', 'DeepLabV3Plus', etc.
            encoder_name (str): Name of the encoder, e.g., 'resnet34', 'efficientnet-b3'.
            in_channels (int): Number of input channels.
            out_classes (int): Number of output classes.
            mode (str): Loss and metric computation mode, 'binary' or 'multiclass'.
            **kwargs: Additional arguments for model configuration.
        """
        super().__init__()
        self.save_hyperparameters()
        self.arch_name = arch_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.mode = mode
        self.ignore_index = ignore_index
        # Threshold for binary classification
        self.threshold = 0.5

        # Dynamically initialize the model
        self.model = smp.create_model(
            self.arch_name,
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.out_classes,
            **kwargs,
        )

        # Preprocessing parameters for normalization
        params = smp.encoders.get_preprocessing_params(self.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Configure loss function
        if mode == "binary":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True, ignore_index=self.ignore_index)
        elif mode == "multiclass":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=self.ignore_index)
        else:
            raise ValueError("Invalid mode. Choose 'binary' or 'multiclass'.")

        # Metrics aggregation
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": 1e-3},
            {"params": self.model.decoder.parameters(), "lr": 1e-3},
        ])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-2,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

    def shared_step(self, batch, stage):
        images, masks, _ = batch
        masks = masks.long()
        
        # Forward pass
        logits = self.forward(images)
        
        # Compute the loss
        loss = self.loss_fn(logits, masks)

        if self.mode == "binary":
            prob_masks = torch.sigmoid(logits)
            pred_masks = (prob_masks > self.threshold).long()
        else:
            prob_masks = torch.softmax(logits, dim=1)
            pred_masks = prob_masks.argmax(keepdim=True, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks, mode=self.mode, num_classes=self.out_classes)
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{stage}_per_image_iou", per_image_iou, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_dataset_iou", dataset_iou, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(result)
        return result

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.plot_train_val_metrics()
        # Clear the training step outputs
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.log("valid_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def save_model(self, save_dir: Path, save_name: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "arch": self.model.__class__.__name__}, save_dir / f"{save_name}.pth")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def plot_train_val_metrics(self):
        """
        Plots training and validation metrics (e.g., loss, IoU) from a metrics CSV file after each epoch.
        Loss and IoU are separated into subplots.
        
        Args:
            metrics_file (str): Path to the metrics CSV file.
            metrics (list): List of metric column names to plot (e.g., ["train_loss", "valid_loss"]).
            output_file (str, optional): If specified, saves the plot to this file. Default is None.
        
        Returns:
            None: Displays the plot (and optionally saves it to a file).
        """
        # Load the metrics CSV into a pandas DataFrame
        metrics_file = Path(self.logger.log_dir, "metrics.csv")
        if not metrics_file.exists():
            log.warning("No metrics file found.")
            return
        
        metrics_df = pd.read_csv(metrics_file)

        metrics = list(metrics_df.columns)
        
        if metrics_df.empty:
            log.warning("No metrics to plot.")
            return
        
        
        # Separate rows for training and validation metrics
        if "train_loss" in metrics:
            train_df = metrics_df.dropna(subset=["train_loss"])
        if "valid_loss" in metrics:
            val_df = metrics_df.dropna(subset=["valid_loss"])

        # Define colors for training (dark) and validation (light) metrics
        colors = {
            "train": {"loss": "#1f77b4", "dataset_iou": "#2ca02c", "per_image_iou": "#9467bd"},  # Dark colors for training
            "valid": {"loss": "#ffa500", "dataset_iou": "#98df8a", "per_image_iou": "#c5b0d5"},  # Light colors for validation
        }

        # Initialize the subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot the loss metrics
        for metric in metrics:
            if "loss" in metric:
                if "train" in metric:
                    axs[0].plot(
                        train_df["epoch"], train_df[metric],
                        label=f"Training {metric.split('_')[-1].capitalize()}",
                        marker="o", color=colors["train"]["loss"]
                    )
                elif "valid" in metric:
                    axs[0].plot(
                        val_df["epoch"], val_df[metric],
                        label=f"Validation {metric.split('_')[-1].capitalize()}",
                        marker="s", color=colors["valid"]["loss"]
                    )

        axs[0].set_title("Loss")
        axs[0].set_ylabel("Loss")
        axs[0].legend(loc="upper right")
        axs[0].grid(True)

        # Plot the IoU metrics
        for metric in metrics:
            if "iou" in metric:
                key = "dataset_iou" if "dataset" in metric else "per_image_iou"
                if "train" in metric:
                    axs[1].plot(
                    train_df["epoch"], train_df[metric],
                    label=f"Training {key.replace('_', ' ').capitalize()}",
                    marker="o", color=colors["train"].get(key, "#2ca02c")
                    )
                elif "valid" in metric:
                    axs[1].plot(
                    val_df["epoch"], val_df[metric],
                    label=f"Validation {key.replace('_', ' ').capitalize()}",
                    marker="s", color=colors["valid"].get(key, "#98df8a")
                    )

        axs[1].set_title("IoU")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("IoU")
        axs[1].legend(loc="lower right")
        axs[1].grid(True)

        # Force integer x-axis ticks
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))


        # Adjust layout
        plt.tight_layout()

        # Show or save the plot
        output_file = Path(self.logger.log_dir, "metrics.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()