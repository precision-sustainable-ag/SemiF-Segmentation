"""mode=sam3_finetune: fine-tune SAM 3's detector on the project's labels (the
corrected boxes of box rounds, or plant boxes from the masks of semantic
rounds), and check it beats zero-shot SAM 3 (conf/sam3_finetune).

    python main.py mode=sam3_finetune project.name=<project> sam3_finetune.name=<run>

Tasks, in <paths.sam3_finetune_dir> = outputs/runs/<project.name>/sam3_finetune/<name>/:
  dataset   dataset/: tiled COCO train/valid sets (src/finetune/box_dataset.py)
  train     train/: sam3.train on dataset/train, backbone frozen (src/finetune/sam3_trainer.py)
  export    sam3_finetuned.pt: the trained detector weights and the checkpoint they
            apply to (src/finetune/sam3_model.py); use it as proposals.checkpoint,
            with proposals.tiling.box_from=model
  evaluate  eval.json: box AP and precision/recall on dataset/valid, zero-shot
            vs fine-tuned; detections/, and previews/ (ground truth | zero-shot |
            fine-tuned) of a few valid images (src/finetune/evaluate.py)
"""

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def run_dir(cfg) -> Path:
    return Path(cfg.paths.sam3_finetune_dir)


def dataset(cfg: DictConfig) -> dict:
    from src.finetune.box_dataset import build, labeled_images

    fcfg = cfg.sam3_finetune
    images = labeled_images(cfg.paths.labels_db, fcfg.rounds, fcfg.labels)
    if not images:
        raise ValueError(f"No labeled images with pulled {fcfg.labels} in {cfg.paths.labels_db}; "
                         "pull finished jobs first")
    log.info("%d labeled images (%s) from %s", len(images), fcfg.labels, cfg.paths.labels_db)
    return build(images, run_dir(cfg) / "dataset", OmegaConf.to_container(fcfg.prompts, resolve=True),
                 OmegaConf.to_container(fcfg.dataset, resolve=True),
                 OmegaConf.to_container(fcfg.masks, resolve=True))


def train(cfg: DictConfig) -> dict:
    from src.finetune.sam3_model import base_checkpoint_path
    from src.finetune.sam3_trainer import run_trainer, trainer_config

    tcfg = OmegaConf.to_container(cfg.sam3_finetune.train, resolve=True)
    out = run_dir(cfg) / "train"
    if (out / "checkpoints" / "checkpoint.pt").exists():
        # sam3's Trainer resumes from its last checkpoint; a finished run would do nothing.
        log.warning("%s exists: the trainer resumes from it (delete %s to train from scratch)",
                    out / "checkpoints" / "checkpoint.pt", out)
    config = trainer_config(run_dir(cfg) / "dataset", out, base_checkpoint_path(tcfg["checkpoint"]), tcfg)
    checkpoint = run_trainer(config, out)
    return {"checkpoint": str(checkpoint)}


def export(cfg: DictConfig) -> dict:
    """sam3_finetuned.pt from the last epoch, and sam3_finetuned_e<N>.pt from each
    train.snapshots epoch (whose large trainer checkpoint_<N>.pt is then removed)."""
    from src.finetune.sam3_model import export_checkpoint

    checkpoints = run_dir(cfg) / "train" / "checkpoints"
    trained = checkpoints / "checkpoint.pt"
    if not trained.exists():
        raise FileNotFoundError(f"{trained} doesn't exist; run the train task first")
    base = cfg.sam3_finetune.train.checkpoint
    out = run_dir(cfg) / "sam3_finetuned.pt"
    summary = {"final": {"checkpoint": str(out), **export_checkpoint(base, trained, out)}}
    for snapshot in sorted(checkpoints.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        epoch = int(snapshot.stem.split("_")[1])
        target = run_dir(cfg) / f"sam3_finetuned_e{epoch}.pt"
        summary[f"e{epoch}"] = {"checkpoint": str(target), **export_checkpoint(base, snapshot, target)}
        snapshot.unlink()  # optimizer state included: ~3x the export; the export is all inference needs
    log.info("Pre-label with it: proposals.checkpoint=%s proposals.tiling.box_from=model", out)
    return summary


def evaluate(cfg: DictConfig) -> dict:
    """Zero-shot SAM 3, each exported snapshot, and the final model on dataset/valid.
    The best by evaluate.select_by is linked as sam3_finetuned_best.pt and plotted
    against zero-shot in previews/. (It's chosen on the valid set its scores come
    from, so they're slightly optimistic.)"""
    from src.finetune.evaluate import evaluate as run_eval
    from src.models import build_proposal_generator

    fcfg = cfg.sam3_finetune
    run = run_dir(cfg)
    checkpoints = {"zero_shot": str(fcfg.train.checkpoint)}
    snapshots = sorted(run.glob("sam3_finetuned_e*.pt"), key=lambda p: int(p.stem.rsplit("_e", 1)[1]))
    checkpoints.update({f"finetuned_e{p.stem.rsplit('_e', 1)[1]}": str(p) for p in snapshots})
    if (run / "sam3_finetuned.pt").exists():
        checkpoints["finetuned"] = str(run / "sam3_finetuned.pt")
    else:
        log.warning("%s doesn't exist; evaluating zero-shot SAM 3 only", run / "sam3_finetuned.pt")
    scfg = OmegaConf.to_container(cfg.proposals, resolve=True)
    ecfg = OmegaConf.to_container(fcfg.evaluate, resolve=True)
    results = run_eval(run / "dataset", checkpoints, scfg, ecfg,
                       lambda c: build_proposal_generator(c["name"], c), out_dir=run)
    tuned = {k: v for k, v in results["checkpoints"].items() if k != "zero_shot"}
    if tuned:
        key = ecfg.get("select_by", "AP")
        best = max(tuned, key=lambda k: tuned[k][key])
        results["best"] = {"name": best, "by": key, "checkpoint": tuned[best]["checkpoint"]}
        link = run / "sam3_finetuned_best.pt"
        link.unlink(missing_ok=True)
        os.link(tuned[best]["checkpoint"], link)  # a hard link: no second copy on disk
        log.info("Best by %s: %s (%.3f vs zero-shot %.3f) -> %s", key, best, tuned[best][key],
                 results["checkpoints"]["zero_shot"][key], link)
    (run / "eval.json").write_text(json.dumps(results, indent=1))
    return results


TASK_REGISTRY = {"dataset": dataset, "train": train, "export": export, "evaluate": evaluate}


def main(cfg: DictConfig) -> None:
    devices = cfg.sam3_finetune.cuda_visible_devices
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({
        "project": {"name": cfg.project.name},  # subname is a labeling round's
        "sam3_finetune": OmegaConf.to_container(cfg.sam3_finetune, resolve=True),
        "proposals": OmegaConf.to_container(cfg.proposals, resolve=True)})))
    log.info("SAM 3 fine-tuning run %s", out)
    summaries = {}
    for task, enabled in cfg.sam3_finetune.tasks.items():
        if not enabled:
            log.info("Skipping task %s", task)
            continue
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown sam3_finetune task {task!r}; choose from {sorted(TASK_REGISTRY)}")
        log.info("Running task %s", task)
        summaries[task] = TASK_REGISTRY[task](cfg)
    previous = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    (out / "summary.json").write_text(json.dumps({**previous, **summaries}, indent=1, default=str))
