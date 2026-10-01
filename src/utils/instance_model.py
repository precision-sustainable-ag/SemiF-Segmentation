import logging
import math

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision

from src.models import build_model
from src.utils.model import SemanticMetricsMixin, detail_metrics_config
from src.utils.seg_metrics import detail_metrics

log = logging.getLogger(__name__)

# Keys torchvision's detectors read from a target; InstanceDataset adds others.
MODEL_TARGET_KEYS = ("boxes", "labels", "masks")
# COCO-style AP logged per stage, as {stage}_{key}
AP_KEYS = ("bbox_map", "bbox_map_50", "bbox_map_75", "segm_map", "segm_map_50", "segm_map_75")


def instances_to_semantic(output: dict, image_hw: tuple[int, int], score_threshold: float = 0.5,
                          mask_threshold: float = 0.5) -> torch.Tensor:
    """(H, W) class indices from one image's detections: each instance with
    score >= score_threshold paints its label where its mask probability
    > mask_threshold, higher scores on top."""
    semantic = torch.zeros(image_hw, dtype=torch.long, device=output["scores"].device)
    keep = torch.where(output["scores"] >= score_threshold)[0]
    for i in keep[output["scores"][keep].argsort()]:
        semantic[output["masks"][i, 0] > mask_threshold] = output["labels"][i]
    return semantic


class InstanceSegmentationModule(SemanticMetricsMixin, pl.LightningModule):
    def __init__(self, cfg):
        """
        Lightning wrapper for the torchvision-style instance-segmentation
        models of src.models (cfg.model.name, e.g. maskrcnn or
        maskrcnn_pointrend).

        Trains on the models' own loss dict. Validation and test log:
        - instance metrics: COCO box and mask AP (bbox_map, segm_map, ...,
          over all detections) and instance counts (count_mae, count_ratio:
          detections with score >= predict.score_threshold vs gt instances);
        - semantic metrics, on the detections painted into a semantic mask:
          the same as SegmentationModule (valid_dataset_iou, ...) plus the
          detail metrics, so the two kinds of model compare directly.
        """
        super().__init__()
        self.save_hyperparameters({
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "train": OmegaConf.to_container(cfg.train, resolve=True),
        })
        self.cfg = cfg
        mcfg = cfg.model
        self.mode = mcfg.mode
        self.out_classes = mcfg.out_classes
        self.score_threshold = mcfg.predict.score_threshold
        self.mask_threshold = mcfg.predict.mask_threshold
        self.detail_cfg = detail_metrics_config(cfg)

        options = OmegaConf.to_container(mcfg.get("pointrend") or {}, resolve=True) if mcfg.name == "maskrcnn_pointrend" else {}
        self.model = build_model(
            mcfg.name,
            num_classes=mcfg.num_classes,
            pretrained=mcfg.pretrained,
            backbone=mcfg.backbone,
            trainable_backbone_layers=mcfg.trainable_backbone_layers,
            **OmegaConf.to_container(mcfg.get("detector") or {}, resolve=True),
            **options,
        )

        self.valid_ap = MeanAveragePrecision(iou_type=("bbox", "segm"), backend="pycocotools")
        self.test_ap = MeanAveragePrecision(iou_type=("bbox", "segm"), backend="pycocotools")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        """SGD with linear warmup then cosine decay per step (torchvision's detection recipe)."""
        ocfg = self.cfg.model.optim
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=ocfg.lr, momentum=ocfg.momentum, weight_decay=ocfg.weight_decay)
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = min(int(ocfg.warmup_steps), total)

        def lr_factor(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def forward(self, images, targets=None):
        """torchvision detection API: loss dict in train mode (with targets), detections in eval mode."""
        return self.model(list(images), targets)

    @torch.no_grad()
    def predict_instances(self, images) -> list[dict[str, torch.Tensor]]:
        """Per-image {"boxes", "labels", "scores", "masks" (N, 1, H, W) probabilities}."""
        return self.model(list(images))

    def semantic_from_instances(self, outputs: list[dict], images) -> torch.Tensor:
        """(B, H, W) class indices painted from predict_instances' outputs."""
        return torch.stack([
            instances_to_semantic(o, img.shape[-2:], self.score_threshold, self.mask_threshold)
            for o, img in zip(outputs, images)
        ])

    @torch.no_grad()
    def predict_semantic(self, images) -> torch.Tensor:
        """(B, H, W) class indices for a (B, C, H, W) batch (or list of (C, H, W))."""
        return self.semantic_from_instances(self.predict_instances(images), images)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: t[k] for k in MODEL_TARGET_KEYS} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        batch_size = len(images)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        self.plot_train_val_metrics()

    def shared_eval_step(self, batch, stage):
        images, targets, _ = batch
        outputs = self.predict_instances(images)

        getattr(self, f"{stage}_ap").update(
            [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"],
              "masks": o["masks"][:, 0] > self.mask_threshold} for o in outputs],
            [{"boxes": t["boxes"], "labels": t["labels"], "masks": t["masks"].bool()} for t in targets],
        )
        counts = torch.tensor([[int((o["scores"] >= self.score_threshold).sum()), len(t["boxes"])]
                               for o, t in zip(outputs, targets)], dtype=torch.float32)

        pred = self.semantic_from_instances(outputs, images)
        gt = torch.stack([t["semantic_mask"] for t in targets]).long()
        if self.mode == "binary":
            stats = smp.metrics.get_stats(pred[:, None], gt[:, None], mode="binary")
        else:
            stats = smp.metrics.get_stats(pred, gt, mode="multiclass", num_classes=self.out_classes)
        tp, fp, fn, tn = stats
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "detail": detail_metrics(pred, gt, self.detail_cfg),
                "counts": counts}

    def log_instance_metrics(self, outputs, stage):
        """AP from the stage's MeanAveragePrecision, and instance counts (predicted, gt) per image."""
        ap = getattr(self, f"{stage}_ap")
        results = ap.compute()
        ap.reset()
        for key in AP_KEYS:
            self.log(f"{stage}_{key}", results[key].float(), on_epoch=True, prog_bar=key == "segm_map")
        if outputs:
            counts = torch.cat([x["counts"] for x in outputs])
            self.log(f"{stage}_count_mae", (counts[:, 0] - counts[:, 1]).abs().mean(), on_epoch=True)
            self.log(f"{stage}_count_ratio", counts[:, 0].sum() / counts[:, 1].sum().clamp(min=1), on_epoch=True)

    def on_validation_epoch_end(self):
        self.log_instance_metrics(self.validation_step_outputs, "valid")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self.log_instance_metrics(self.test_step_outputs, "test")
        super().on_test_epoch_end()

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self.shared_eval_step(batch, "valid"))

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_eval_step(batch, "test"))
