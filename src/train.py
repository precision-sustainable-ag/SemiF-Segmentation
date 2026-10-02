import os
import hydra
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import cv2
import numpy as np
import pandas as pd
import json
import torch
import logging
import albumentations as A
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from src.utils.datasets import Dataset, InstanceDataset, collate_instances
from src.utils.instance_model import InstanceSegmentationModule
from src.utils.model import SegmentationModule
from src.utils.viz_utils import viz_batch, side_by_side_plot, pad_to_multiple, instance_plot
from src.utils.augs import train_aug, val_aug

log = logging.getLogger(__name__)


def set_seed(seed=2**3, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)

def is_instance_model(cfg: DictConfig) -> bool:
    """Instance-segmentation models (src/models, conf/model/maskrcnn*.yaml) vs the smp semantic ones."""
    return cfg.model.get("task", "semantic") == "instance"

def instances_dir(cfg: DictConfig, image_dir) -> Path | None:
    """Where a split's instance files are (model.instances.source=pseudo: split_dir/<split>/<pseudo_dirname>, e.g. instances_sam3,
    written by the preprocess task pseudo_instances), or None to use the semantic mask's connected blobs."""
    icfg = cfg.model.instances
    source = icfg.get("source", "connected_components")
    if source == "connected_components":
        return None
    if source != "pseudo":
        raise ValueError(f"model.instances.source must be connected_components or pseudo, got {source!r}")
    path = Path(image_dir).parent / icfg.pseudo_dirname
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found; generate it with mode=preprocess preprocess.tasks.pseudo_instances=true")
    return path

def make_dataset(cfg: DictConfig, image_dir, mask_dir, augmentation, mean, std):
    """The dataset the configured model trains on: semantic masks, or instances derived from them."""
    kwargs = dict(augmentation=augmentation, normalize_image=cfg.train.dataset.normalize, mean=mean, std=std)
    if is_instance_model(cfg):
        return InstanceDataset(image_dir, mask_dir, min_area=cfg.model.instances.min_area,
                               max_instances=cfg.model.instances.max_instances,
                               instances_dir=instances_dir(cfg, image_dir), **kwargs)
    return Dataset(image_dir, mask_dir, **kwargs)

def load_stats(stats_file: Path):
        # stats_file = self.project_dir / "preprocess/data/data_stats/rgb_mean_std.json"
    if stats_file.exists():
        with open(stats_file) as f:
            data = json.load(f)
            return np.array(data['mean']), np.array(data['std'])
    return None, None

def pick_sample_tiles(mask_paths, n: int, seed: int, min_vegetation: float = 0.01) -> list[int]:
    """Indices of up to n tiles to show predictions for: three quarters with
    vegetation (at least min_vegetation of the tile), the rest mostly bare, and
    spread over as many source images as possible (round-robin by image)."""
    rng = np.random.default_rng(seed)
    by_kind = {True: {}, False: {}}
    for idx, path in enumerate(mask_paths):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        image = Path(path).stem.rsplit("_y", 1)[0]  # tiles are <source>__<image_id>_y<y>_x<x>
        by_kind[bool((mask > 0).mean() >= min_vegetation)].setdefault(image, []).append(idx)

    def round_robin(groups: dict, k: int) -> list[int]:
        queues = [list(rng.permutation(groups[key])) for key in sorted(groups)]
        queues = [queues[i] for i in rng.permutation(len(queues))]
        picked = []
        while len(picked) < k and any(queues):
            for queue in queues:
                if queue and len(picked) < k:
                    picked.append(int(queue.pop()))
        return picked

    vegetated = round_robin(by_kind[True], n - n // 4)
    return vegetated + round_robin(by_kind[False], n - len(vegetated))


@rank_zero_only
def log_run_info(logger: CSVLogger, cfg: DictConfig, val_result: list[dict], test_result: list[dict] | None):
    """
    Log run information (metrics, and which labels the model was trained on).
    """
    valid_per_image_iou = val_result[0]["valid_per_image_iou"]
    valid_dataset_iou = val_result[0]["valid_dataset_iou"]
    manifest_path = Path(cfg.paths.dataset_manifest)
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None
    run_info = {
        "project_name": cfg.project.name,
        "timestamp": datetime.now().isoformat(),
        "version_dir": str(Path(logger.log_dir)),
        "model_name": cfg.model.arch_name,
        "encoder": cfg.model.get("encoder_name", cfg.model.get("backbone")),
        "encoder_weights": cfg.model.get("encoder_weights", cfg.model.get("pretrained")),
        "loss_function": cfg.train.loss.name,
        "base_lr": cfg.train.lr.base_lr,
        "num_GPUs": len(list(cfg.train.cuda_visible_devices)),
        "num_epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "image_size": f"{cfg.preprocess.tile.tile_size}x{cfg.preprocess.tile.tile_size}",
        "train_scale": cfg.preprocess.tile.scale,
        "training_images": len(list(Path(cfg.paths.split_dir, "train", "images").glob("*.jpg"))),
        "validation_images": len(list(Path(cfg.paths.split_dir, "val", "images").glob("*.jpg"))),
        "test_images": len(list(Path(cfg.paths.split_dir, "test", "images").glob("*.jpg"))),
        "labeled_images_per_split": manifest.groupby("split").size().to_dict() if manifest is not None else None,
        "label_rounds": sorted(manifest["round"].unique().tolist()) if manifest is not None else None,
        "valid_dataset_iou": valid_dataset_iou,
        "valid_per_image_iou": valid_per_image_iou,
        "test_dataset_iou": test_result[0]["test_dataset_iou"] if test_result else None,
        "test_per_image_iou": test_result[0]["test_per_image_iou"] if test_result else None,
        "valid_metrics": val_result[0],
        "test_metrics": test_result[0] if test_result else None,
        }

    run_info_path = Path(logger.log_dir) / "run_info.json"
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=4)
    print(f"Run info saved to: {run_info_path}")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print full config for debugging
    
    nodes = cfg.train.cuda_visible_devices 
    devices_str = ",".join(map(str, nodes))    
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    
    instance = is_instance_model(cfg)
    # torchvision's roi_align and grid_sample have no deterministic CUDA backward
    deterministic = bool(cfg.train.trainer.deterministic) and not instance
    if instance and cfg.train.trainer.deterministic:
        log.warning("Instance models can't train deterministically; running with train.trainer.deterministic=false.")
    set_seed(cfg.train.seed, deterministic)
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
    train_dataset = make_dataset(cfg, train_image_dir, train_mask_dir, train_aug(cfg), mean, std)
    val_dataset = make_dataset(cfg, val_image_dir, val_mask_dir, val_aug(cfg), mean, std)
    test_dataset = make_dataset(cfg, test_image_dir, test_mask_dir, val_aug(cfg), mean, std)
    collate_fn = collate_instances if instance else None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.dataset.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.dataset.workers,
        collate_fn=collate_fn,
        )
    
    sample_dataloader = DataLoader(  # viz_batch shows images with their semantic masks
        Dataset(train_image_dir, train_mask_dir, augmentation=train_aug(cfg),
                normalize_image=cfg.train.dataset.normalize, mean=mean, std=std),
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.dataset.workers
        )

    # Model
    module_cls = InstanceSegmentationModule if instance else SegmentationModule
    model = module_cls(cfg)
    
    if cfg.train.train_strategy == "auto":
        train_strategy = "auto"
    elif cfg.train.train_strategy == "ddp":
        train_strategy = DDPStrategy(find_unused_parameters=False)
    
    # Trainer
    trainer = Trainer(
        **{**cfg.train.trainer, "deterministic": deterministic},
        strategy=train_strategy,
        callbacks=[best_checkpoint, last_checkpoint],
        logger=logger,
        # num_nodes=len(nodes)
    )
    is_rank_zero = trainer.is_global_zero
    
    if is_rank_zero:
        viz_batch(sample_dataloader, output_dir=logger.log_dir)
    del sample_dataloader

    trainer.fit(model, train_loader, val_loader)
    
    # Validate and test the best checkpoint (the model saved below and used for inference)
    val_result = trainer.validate(model, val_loader, ckpt_path="best", verbose=False)
    valid_per_image_iou = val_result[0]["valid_per_image_iou"]
    valid_dataset_iou = val_result[0]["valid_dataset_iou"]
    print(f"Validation IoU per images: {valid_per_image_iou}")
    print(f"Validation dataset IoU: {valid_dataset_iou}")

    # The test split never influences training or checkpoint selection
    test_result = None
    if len(test_dataset):
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.dataset.workers,
            collate_fn=collate_fn,
            )
        test_result = trainer.test(model, test_loader, ckpt_path="best", verbose=False)
        print(f"Test IoU per images: {test_result[0]['test_per_image_iou']}")
        print(f"Test dataset IoU: {test_result[0]['test_dataset_iou']}")
    else:
        log.warning("No test tiles found; skipping the held-out test evaluation.")

    try:
        log_run_info(logger, cfg, val_result, test_result)
    except Exception as e:
        print(f"Error logging run info: {e}")

    # Load and Save best full model
    best_model_ckpt_path = best_checkpoint.best_model_path
    best_model = module_cls.load_from_checkpoint(
        cfg=cfg,
        checkpoint_path=best_model_ckpt_path,
    )
    best_model.eval()
    # Create a directory to save the model
    if is_rank_zero:
        model_dir = Path(logger.log_dir, "model")
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Best model being saved to: {Path(model_dir, 'best_model')}.pth")
        best_model.save_model(model_dir, "best_model")

    # === Sample Inference on a Test Tile === #

    class_labels = dict(enumerate(cfg.train.class_names))
    palette = [[0, 0, 0], [34, 139, 34], [173, 216, 230], [255, 160, 122], [238, 130, 238]]
    class_colors = {idx: palette[idx % len(palette)] for idx in class_labels}

    scfg = cfg.train.sample_predictions
    if is_rank_zero and len(test_dataset) and scfg.n_test_tiles:
        picks = pick_sample_tiles(test_dataset.masks_fps, int(scfg.n_test_tiles), int(scfg.seed),
                                  float(scfg.min_vegetation))
        log.info("Running sample inference on %d test tiles...", len(picks))
        best_model.eval()
        output_dir = Path(logger.log_dir) / "sample_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        instance_dir = output_dir / "instances"  # instance models: boxes, masks and scores per instance
        if instance:
            instance_dir.mkdir(exist_ok=True)
        rows = []
        for n, idx in enumerate(picks):
            image_tensor, mask_gt, mask_stem = test_dataset[idx]
            batch = image_tensor.unsqueeze(0).to(best_model.device)
            row = {"tile": mask_stem}
            if instance:
                target, mask_gt = mask_gt, mask_gt["semantic_mask"]
                detections = best_model.predict_instances(batch)
                mask_pred = best_model.semantic_from_instances(detections, batch)
                row["gt_instances"] = len(target["boxes"])
                row["pred_instances"] = int((detections[0]["scores"] >= best_model.score_threshold).sum())
            else:
                mask_pred = best_model.predict_semantic(batch)
            mask_pred = mask_pred.squeeze(0).cpu().numpy()
            mask_gt = mask_gt.cpu().numpy().squeeze()
            union = np.count_nonzero((mask_gt > 0) | (mask_pred > 0))
            iou = np.count_nonzero((mask_gt > 0) & (mask_pred > 0)) / union if union else 1.0
            rows.append({**row, "vegetation_frac": float((mask_gt > 0).mean()), "iou": iou})
            if instance:
                instance_plot(test_image_dir / f"{mask_stem}.jpg", target, detections[0],
                              instance_dir / f"{n:02d}_iou{iou:.2f}_{mask_stem}.png",
                              best_model.score_threshold, best_model.mask_threshold, class_labels)
            side_by_side_plot(test_image_dir / f"{mask_stem}.jpg", mask_gt, mask_pred,
                              output_dir / f"{n:02d}_iou{iou:.2f}_{mask_stem}.png",
                              class_colors=class_colors, class_labels=class_labels)
        pd.DataFrame(rows).to_csv(output_dir / "sample_predictions.csv", index=False)
        log.info("Saved %d sample predictions (mean IoU %.3f) to %s", len(rows),
                 float(np.mean([r["iou"] for r in rows])), output_dir)

    # === Sample Inference on Unlabeled Images === #
    inference_images = [Path(cfg.paths.persistent_dir, x) for x in cfg.train.sample_inference]  # List of image paths
    missing = [p for p in inference_images if not p.exists()]
    if missing:
        log.warning(f"train.sample_inference images not found, skipping them: {missing}")
    inference_images = [p for p in inference_images if p.exists()]

    if is_rank_zero and inference_images:
        log.info("Running inference on sample unlabeled images...")

        # Dummy mask paths just to satisfy Dataset interface
        dummy_masks = inference_images  # dummy paths, ignored during inference

        # Resize to the scale the model was trained at
        scale = cfg.preprocess.tile.scale
        rescale = A.RandomScale(scale_limit=(scale - 1, scale - 1), p=1.0) if scale != 1 else None

        # Load Dataset
        inference_dataset = Dataset(
            images_dir=inference_images,
            masks_dir=dummy_masks,  # dummy
            augmentation=rescale,
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

            if instance:
                detections = best_model.predict_instances(image)
                mask_pred = best_model.semantic_from_instances(detections, image).squeeze(0).cpu().numpy()
                instance_plot(inference_dataset.images_fps[idx], None, detections[0],
                              output_dir_infer / f"inference{idx}_instances.png",
                              best_model.score_threshold, best_model.mask_threshold, class_labels,
                              image_scale=scale)
            else:
                mask_pred = best_model.predict_semantic(image).squeeze(0).cpu().numpy()

            # Save visualization
            output_path = output_dir_infer / f"inference{idx}.png"
            img_path = Path(inference_dataset.images_fps[idx])
            side_by_side_plot(img_path, None, mask_pred, output_path, class_colors=class_colors, class_labels=class_labels)

        log.info(f"Saved unlabeled inference predictions to {output_dir_infer}")

if __name__ == "__main__":
    main()
