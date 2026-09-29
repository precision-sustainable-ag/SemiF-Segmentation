import logging
import os

from omegaconf import DictConfig

from src.labeling.fetch import main as fetch
from src.labeling.ledger import LabelLedger
from src.labeling.prepare import main as prepare
from src.labeling.pull_cvat import main as pull_cvat
from src.labeling.push_cvat import main as push_cvat
from src.labeling.run_folder import RunFolder, start_log, stop_log, write_cfg, write_manifest, write_metrics
from src.labeling.select import main as select

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "select": select,
    "fetch": fetch,
    "prepare": prepare,
    "push_cvat": push_cvat,
    "pull_cvat": pull_cvat,
}


def main(cfg: DictConfig) -> None:
    """ Runs the enabled labeling tasks for one round (conf/label/<round>.yaml),
    in its AgIR-CVToolkit-style run folder, outputs/runs/<project.name>/<round>/ """
    devices = cfg.label.prelabel.cuda_visible_devices
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))

    run = RunFolder.from_cfg(cfg)
    handler = start_log(run)
    summaries = {}
    try:
        write_cfg(cfg, run)
        log.info(f"Labeling round {cfg.label.round} (source: {cfg.label.source}) in {run.root}")
        for task, enabled in cfg.label.tasks.items():
            if not enabled:
                log.info(f"Skipping task {task}. Enabled: {enabled}")
                continue
            if task not in TASK_REGISTRY:
                log.error(f"Task {task} not found in labeling task registry")
                return
            log.info(f"Running task {task}")
            summaries[task] = TASK_REGISTRY[task](cfg)
    finally:
        # manifest.csv and metrics.json reflect the ledger even if a task failed.
        with LabelLedger(cfg.paths.labels_db) as ledger:
            items = ledger.items(round_name=cfg.label.round)
            counts = ledger.status_counts(cfg.label.round)
        write_manifest(run, items)
        write_metrics(run, cfg, counts, summaries)
        log.info(f"Round {cfg.label.round} status counts: {counts}")
        stop_log(handler)
