import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator

log = logging.getLogger(__name__)


def deterministic_get_stats(pred: torch.Tensor, target: torch.Tensor, mode: str, num_classes: int = 2, ignore_index: int = None):
    """
    Deterministic replacement for smp.metrics.get_stats using torch.bincount instead of torch.histc.
    Returns correct shape (batch_size, num_classes) for compatibility with iou_score.
    """
    batch_size = pred.shape[0]

    if mode == "binary":
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        if ignore_index is not None:
            mask = target != ignore_index
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
 
        tp = []
        fp = []
        fn = []
        tn = []

        for i in range(batch_size):
            p = pred[i][mask[i]]
            t = target[i][mask[i]]

            # For class 0 (background) and class 1 (object)
            for c in [0, 1]:
                tp.append(((p == c) & (t == c)).sum())
                fp.append(((p == c) & (t != c)).sum())
                fn.append(((p != c) & (t == c)).sum())
                tn.append(((p != c) & (t != c)).sum())
        
         # Reshape to (batch_size, 2)
        tp = torch.tensor(tp, dtype=torch.float32).view(batch_size, 2)
        fp = torch.tensor(fp, dtype=torch.float32).view(batch_size, 2)
        fn = torch.tensor(fn, dtype=torch.float32).view(batch_size, 2)
        tn = torch.tensor(tn, dtype=torch.float32).view(batch_size, 2)


    elif mode == "multiclass":
        tp = []
        fp = []
        fn = []
        tn = []  # TN stays zero-filled in multiclass for IoU

        for i in range(batch_size):
            pred_i = pred[i].view(-1)
            target_i = target[i].view(-1)

            if ignore_index is not None:
                mask = target_i != ignore_index
                pred_i = pred_i[mask]
                target_i = target_i[mask]

            matched = pred_i[target_i == pred_i]
            tp_i = torch.bincount(matched, minlength=num_classes)

            fp_i = torch.bincount(pred_i[target_i != pred_i], minlength=num_classes)
            fn_i = torch.bincount(target_i[target_i != pred_i], minlength=num_classes)

            tn_i = torch.zeros_like(tp_i)  # TN is often unused for multiclass IoU

            tp.append(tp_i)
            fp.append(fp_i)
            fn.append(fn_i)
            tn.append(tn_i)

        tp = torch.stack(tp, dim=0)  # Shape: (batch_size, num_classes)
        fp = torch.stack(fp, dim=0)
        fn = torch.stack(fn, dim=0)
        tn = torch.stack(tn, dim=0)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return tp, fp, fn, tn

# Monkey patch SMP metrics
smp.metrics.get_stats = deterministic_get_stats

class FastBoundaryLoss(torch.nn.Module):
    def __init__(self, dilation_ratio=0.02):
        super().__init__()
        self.dilation_ratio = dilation_ratio

    def forward(self, logits, targets):
        # probs = torch.sigmoid(logits)
        assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf!"
        assert targets.min() >= 0 and targets.max() <= 1, "Target mask values are invalid!"
        probs = torch.sigmoid(torch.clamp(logits, -10, 10))

        batch_size, _, height, width = probs.shape

        targets = targets.float()

        # Compute soft "edge strength" via gradients
        grad_targets_x = F.pad(targets[:, :, 1:, :] - targets[:, :, :-1, :], (0, 0, 0, 1))
        grad_targets_y = F.pad(targets[:, :, :, 1:] - targets[:, :, :, :-1], (0, 1, 0, 0))

        grad_probs_x = F.pad(probs[:, :, 1:, :] - probs[:, :, :-1, :], (0, 0, 0, 1))
        grad_probs_y = F.pad(probs[:, :, :, 1:] - probs[:, :, :, :-1], (0, 1, 0, 0))

        edge_targets = (grad_targets_x**2 + grad_targets_y**2 + 1e-4).sqrt()
        edge_probs = (grad_probs_x**2 + grad_probs_y**2 + 1e-4).sqrt()
        

        # Use soft edge overlap
        intersection = (edge_targets * edge_probs).sum(dim=(1,2,3))
        union = (edge_targets + edge_probs).sum(dim=(1,2,3))

        boundary_dice = (2. * intersection) / (union + 1e-4)

        return 1 - boundary_dice.mean()

class DiceWithBoundaryLoss(torch.nn.Module):
    def __init__(self, boundary_weight=0.2):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.boundary_loss = FastBoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        loss = (1 - self.boundary_weight) * dice + self.boundary_weight * boundary
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf!"
        return loss


class DiceBCELoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # BCE Loss (built-in from_logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float())

        # Dice Loss (manual)
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 0.5 * bce_loss + 0.5 * dice_loss


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, mode="binary", dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.mode = mode
        self.dice_loss = smp_losses.DiceLoss(mode=mode, from_logits=True)  # <- KEEP from_logits
        self.focal_loss = smp_losses.FocalLoss(mode=mode)  # <- REMOVE from_logits here
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal

    
class SegmentationModule(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
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
        self.cfg = cfg
        self.arch_name=cfg.model.arch_name
        self.encoder_name=cfg.model.encoder_name
        self.encoder_weights=cfg.model.encoder_weights
        self.in_channels=cfg.model.in_channels
        self.out_classes=cfg.model.out_classes
        self.mode=cfg.model.mode
        self.ignore_index=cfg.model.ignore_index
        
        self.base_lr = cfg.train.lr.base_lr
        self.encoder_lr = self.base_lr * 0.1

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

        # Configure loss function
        self.loss_strategy = cfg.train.loss.name
        self.loss_fn = self.configure_loss()

        # Metrics aggregation
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_loss(self):
        """
        Configures the loss function based on the mode (binary or multiclass).
        """
        if self.mode == "binary":

            if self.loss_strategy == "DiceFocalLoss":
                return DiceFocalLoss(mode="binary", dice_weight=0.5, focal_weight=0.5)
            
            if self.loss_strategy == "DiceBCELoss":
                return DiceBCELoss()
            
            if self.loss_strategy == "DiceWithBoundaryLoss":
                return DiceWithBoundaryLoss(boundary_weight=0.2)
            
            if self.loss_strategy == "smpDiceLoss":
                return smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True, ignore_index=self.ignore_index)
            
        elif self.mode == "multiclass":
            if self.loss_strategy == "DiceCrossEntropy":
                dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=self.ignore_index)
                ce = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                return lambda logits, targets: 0.5 * dice(logits, targets) + 0.5 * ce(logits, targets)
            else:
                self.cfg.train.loss.name = "smpDiceLoss"
                return smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=self.ignore_index)
            
        else:
            raise ValueError("Invalid mode. Choose 'binary' or 'multiclass'.")
        

    def configure_optimizers(self):
        
        head_lr = self.base_lr         # For segmentation head (often like decoder)
        # Safely check if model has a segmentation head
        if hasattr(self.model, 'segmentation_head'):
            segmentation_head_params = list(self.model.segmentation_head.parameters())
        else:
            segmentation_head_params = []

        optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": self.encoder_lr}, # used to be 1e-3
            {"params": self.model.decoder.parameters(), "lr": self.base_lr}, # used to be 1e-3
            {"params": segmentation_head_params, "lr": head_lr},
        ])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.encoder_lr * 10, self.base_lr * 10, head_lr * 10],  # 10x schedule for each group
            epochs=self.trainer.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        images, masks, _ = batch
        masks = masks.long()
        
        if self.mode == "multiclass" and masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # (B, H, W)
        
        # Forward pass
        logits = self.forward(images)
        # Compute the loss
        loss = self.loss_fn(logits, masks)

        if self.mode == "binary":
            prob_masks = torch.sigmoid(logits)
            pred_masks = (prob_masks > self.threshold).long()
        elif self.mode == "multiclass":
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

        if self.mode == "binary":
            class_iou = self.compute_classwise_iou(tp, fp, fn, tn)
            object_iou = class_iou[1]
            background_iou = class_iou[0]
            self.log(f"{stage}_background_iou", background_iou, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_object_iou", object_iou, on_epoch=True, prog_bar=True)

        elif self.mode == "multiclass":
            # Compute per-class IoU for multiclass
            class_iou = self.compute_classwise_iou(tp, fp, fn, tn)
            for idx, iou in enumerate(class_iou):
                self.log(f"{stage}_class_{idx}_iou", iou.item(), on_epoch=True, prog_bar=False)

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

    def compute_classwise_iou(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor):
        """
        Compute per-class IoU from aggregated tp, fp, fn.

        Args:
            tp (torch.Tensor): Shape (N, C)
            fp (torch.Tensor): Shape (N, C)
            fn (torch.Tensor): Shape (N, C)
            tn (torch.Tensor): Shape (N, C)

        Returns:
            torch.Tensor: Class-wise IoU, shape (C,)
        """
        tp_total = tp.sum(dim=0)
        fp_total = fp.sum(dim=0)
        fn_total = fn.sum(dim=0)

        class_iou = tp_total / (tp_total + fp_total + fn_total + 1e-7)

        return class_iou

    def plot_train_val_metrics(self):
        """
        Plots training and validation metrics (e.g., loss, IoU) from metrics CSV file.
        Handles binary and multiclass metrics dynamically.
        """
        metrics_file = Path(self.logger.log_dir, "metrics.csv")
        if not metrics_file.exists():
            log.warning("No metrics file found.")
            return

        metrics_df = pd.read_csv(metrics_file)
        if metrics_df.empty:
            log.warning("No metrics to plot.")
            return

        metrics = list(metrics_df.columns)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # === Plot Loss Metrics ===
        for metric in metrics:
            if "loss" in metric:
                df = metrics_df.dropna(subset=[metric])
                if "train" in metric:
                    axs[0].plot(df["epoch"], df[metric], label=f"Train {metric.split('_')[-1]}", marker="o")
                elif "valid" in metric:
                    axs[0].plot(df["epoch"], df[metric], label=f"Valid {metric.split('_')[-1]}", marker="s")

        axs[0].set_title("Loss")
        axs[0].set_ylabel("Loss")
        axs[0].legend(loc="upper right")
        axs[0].grid(True)

        # === Plot IoU Metrics ===
        # Predefine some common colors
        base_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

        color_idx = 0
        for metric in metrics:
            if "iou" in metric:
                df = metrics_df.dropna(subset=[metric])
                if "train" in metric:
                    color = base_colors[color_idx % len(base_colors)]
                    axs[1].plot(df["epoch"], df[metric], label=f"Train {metric.replace('train_', '').replace('_', ' ')}", marker="o", color=color)
                    color_idx += 1
                elif "valid" in metric:
                    color = base_colors[color_idx % len(base_colors)]
                    axs[1].plot(df["epoch"], df[metric], label=f"Valid {metric.replace('valid_', '').replace('_', ' ')}", marker="s", color=color)
                    color_idx += 1

        axs[1].set_title("IoU")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("IoU")
        axs[1].legend(loc="best", fontsize='small')
        axs[1].grid(True)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        output_file = Path(self.logger.log_dir, "metrics.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()