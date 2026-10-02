"""Run sam3.train's Trainer on our box dataset, locally, on one GPU.

The trainer config follows sam3's box-only fine-tuning recipe
(sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml in the
SAM 3 repo; configs aren't part of the installed package): no segmentation
head or mask loss, Hungarian-matched box (L1 + GIoU) and classification /
presence losses, AdamW with an inverse-square-root schedule, bf16 autocast.
The model is build_trainable_model (sam3_model.py), so frozen parameters keep
their SAM 3 values. sam3.train.train isn't imported: it needs submitit, only
used for SLURM launches.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

NORM = [0.5, 0.5, 0.5]  # SAM 3's input normalization


def _scheduler(base_lr: float, warmup: int, cooldown: int, timescale: int) -> dict:
    return {"_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler", "base_lr": base_lr,
            "timescale": timescale, "warmup_steps": warmup, "cooldown_steps": cooldown}


def trainer_config(dataset_dir: str | Path, out_dir: str | Path, checkpoint_path: str, tcfg: dict) -> dict:
    """The Trainer's config (as sam3.train.train builds from YAML), for the
    COCO dataset in dataset_dir/train (written by box_dataset.py)."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    resolution = int(tcfg["resolution"])
    train_transforms = [
        {"_target_": "sam3.train.transforms.basic_for_api.ComposeAPI", "transforms": [
            {"_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
             "query_filter": {"_target_": "sam3.train.transforms.filter_query_transforms.FilterCrowds"}},
            {"_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
             "sizes": {"_target_": "sam3.train.transforms.basic.get_random_resize_scales",
                       "size": resolution, "min_size": int(tcfg["min_resize"]), "rounded": False},
             "max_size": {"_target_": "sam3.train.transforms.basic.get_random_resize_max_size", "size": resolution},
             "square": True, "consistent_transform": False},
            {"_target_": "sam3.train.transforms.basic_for_api.PadToSizeAPI", "size": resolution,
             "consistent_transform": False},
            {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
            {"_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
             "query_filter": {"_target_": "sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"}},
            {"_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI", "mean": NORM, "std": NORM},
            {"_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
             "query_filter": {"_target_": "sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"}},
        ]},
        {"_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
         "query_filter": {"_target_": "sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut",
                          "max_num_objects": int(tcfg["max_ann_per_img"])}},
    ]
    matcher = {"_target_": "sam3.train.matcher.BinaryHungarianMatcherV2", "focal": True, "cost_class": 2.0,
               "cost_bbox": 5.0, "cost_giou": 2.0, "alpha": 0.25, "gamma": 2, "stable": False}
    loss = {
        "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
        "matcher": matcher,
        "o2m_weight": 2.0,
        "o2m_matcher": {"_target_": "sam3.train.matcher.BinaryOneToManyMatcher", "alpha": 0.3, "threshold": 0.4,
                        "topk": 4},
        "use_o2m_matcher_on_o2m_aux": False,
        "loss_fns_find": [
            {"_target_": "sam3.train.loss.loss_fns.Boxes", "weight_dict": {"loss_bbox": 5.0, "loss_giou": 2.0}},
            {"_target_": "sam3.train.loss.loss_fns.IABCEMdetr", "weak_loss": False,
             "weight_dict": {"loss_ce": 20.0, "presence_loss": 20.0}, "pos_weight": 10.0, "alpha": 0.25,
             "gamma": 2, "use_presence": True, "pos_focal": False, "pad_n_queries": 200, "pad_scale_pos": 1.0},
        ],
        "loss_fn_semantic_seg": None,
        "scale_by_find_batch_size": True,
    }
    sched = {k: int(tcfg[k]) for k in ("warmup", "cooldown", "timescale")}
    lr = float(tcfg["lr"])
    return {
        "_target_": "sam3.train.trainer.Trainer",
        "skip_saving_ckpts": False,
        "empty_gpu_mem_cache_after_eval": True,
        "skip_first_val": True,
        "max_epochs": int(tcfg["epochs"]),
        "accelerator": "cuda",
        "seed_value": int(tcfg["seed"]),
        "val_epoch_freq": int(tcfg["epochs"]),
        "mode": "train_only",  # evaluation is the sam3_finetune evaluate task
        "gradient_accumulation_steps": int(tcfg["gradient_accumulation_steps"]),
        "distributed": {"backend": "nccl", "find_unused_parameters": True, "gradient_as_bucket_view": True},
        "loss": {"all": loss, "default": {"_target_": "sam3.train.loss.sam3_loss.DummyLoss"}},
        "data": {
            "train": {
                "_target_": "sam3.train.data.torch_dataset.TorchDataset",
                "dataset": {
                    "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
                    "limit_ids": None,
                    "transforms": train_transforms,
                    "load_segmentation": False,
                    "max_ann_per_img": 500000,
                    "multiplier": 1,
                    "max_train_queries": 50000,
                    "max_val_queries": 50000,
                    "training": True,
                    "use_caching": False,
                    "img_folder": str(dataset_dir / "train") + "/",
                    "ann_file": str(dataset_dir / "train" / "_annotations.coco.json"),
                },
                "shuffle": True,
                "batch_size": int(tcfg["batch_size"]),
                "num_workers": int(tcfg["num_workers"]),
                "pin_memory": True,
                "drop_last": True,
                "collate_fn": {"_target_": "sam3.train.data.collator.collate_fn_api", "_partial_": True,
                               "repeats": 1, "dict_key": "all", "with_seg_masks": False},
            },
        },
        "model": {
            "_target_": "src.finetune.sam3_model.build_trainable_model",
            "checkpoint_path": checkpoint_path,
            "freeze": list(tcfg["freeze"]),
            "enable_segmentation": False,
        },
        "meters": None,
        "optim": {
            "amp": {"enabled": True, "amp_dtype": "bfloat16"},
            "optimizer": {"_target_": "torch.optim.AdamW"},
            "gradient_clip": {"_target_": "sam3.train.optim.optimizer.GradientClipper", "max_norm": 0.1,
                              "norm_type": 2},
            "param_group_modifiers": None,
            "options": {
                "lr": [
                    {"scheduler": _scheduler(lr, **sched)},
                    {"scheduler": _scheduler(lr * float(tcfg["backbone_lr_factor"]), **sched),
                     "param_names": ["backbone.*"]},
                ],
                "weight_decay": [
                    {"scheduler": {"_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                   "value": float(tcfg["weight_decay"])}},
                    {"scheduler": {"_target_": "fvcore.common.param_scheduler.ConstantParamScheduler", "value": 0.0},
                     "param_names": ["*bias*"], "module_cls_names": ["torch.nn.LayerNorm"]},
                ],
            },
        },
        # Frozen weights are SAM 3's own; export takes them from the base checkpoint.
        "checkpoint": {"save_dir": str(out_dir / "checkpoints"), "save_freq": 0,
                       "save_list": [int(e) for e in tcfg.get("snapshots") or []],  # checkpoint_<epoch>.pt
                       "skip_saving_parameters": list(tcfg["freeze"])},
        "logging": {
            "tensorboard_writer": {"_target_": "sam3.train.utils.logger.make_tensorboard_logger",
                                   "log_dir": str(out_dir / "tensorboard"), "flush_secs": 120, "should_log": True},
            "wandb_writer": None,
            "log_dir": str(out_dir / "logs"),
            "log_freq": int(tcfg["log_freq"]),
        },
    }


def _patch_focal_loss() -> None:
    """Route gamma=0 focal losses (sam3's presence loss: presence_gamma=0) to
    sam3's PyTorch implementation. Its Triton kernel's backward computes
    -gamma * (1 - p_t) ** (gamma - 1), which is 0 * inf = NaN once the presence
    head is confident (p_t == 1): every step after the first few gets NaN
    gradients. With gamma=0 the loss is alpha-weighted BCE either way."""
    from sam3.train.loss import loss_fns

    if getattr(loss_fns.sigmoid_focal_loss, "_gamma0_safe", False):
        return
    original = loss_fns.sigmoid_focal_loss

    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2, loss_on_multimask=False,
                           reduce=True, triton=True):
        return original(inputs, targets, num_boxes, alpha, gamma, loss_on_multimask, reduce,
                        triton and gamma != 0)

    sigmoid_focal_loss._gamma0_safe = True
    loss_fns.sigmoid_focal_loss = sigmoid_focal_loss


def run_trainer(config: dict, out_dir: str | Path) -> Path:
    """Train in this process (one GPU, as sam3.train.train's single_proc_run);
    returns the trainer's last checkpoint."""
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(config)
    (out_dir / "trainer_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(random.randint(20000, 60000)),
                      RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    try:
        register_omegaconf_resolvers()
    except Exception as exc:  # already registered in this process
        log.debug("omegaconf resolvers: %s", exc)
    _patch_focal_loss()
    trainer = instantiate(cfg, _recursive_=False)
    try:
        trainer.run()
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    checkpoint = out_dir / "checkpoints" / "checkpoint.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"training finished without writing {checkpoint}")
    return checkpoint
