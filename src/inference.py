import logging
import os
import random
import shutil
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.inferencing.get_dataset import main as get_dataset
from src.inferencing.inference import main as inference

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "get_dataset": get_dataset,
    "inference": inference,
    # Add more tasks here as needed
}


def copy_hydra_config_to_subfolder(cfg : DictConfig, task_name: str):
    """Copies the entire .hydra directory (or just specific config files) to a target directory."""
    target_dir = Path(cfg.paths.project_mode_dir) / "data" / ".hydra" / task_name
    source_hydra_dir = Path(HydraConfig.get().run.dir) / ".hydra"
    target_hydra_dir = target_dir 
    
    if not source_hydra_dir.exists():
        log.warning(f"No hydra directory found at {source_hydra_dir}")
        return

    log.info(f"Copying .hydra directory from {source_hydra_dir} to {target_hydra_dir}")
    shutil.copytree(source_hydra_dir, target_hydra_dir, dirs_exist_ok=True)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    log.info(f"Starting preprocessing tasks...")
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.inference.inference.cuda_visible_devices))
    random.seed(cfg.inference.inference.seed)
    
    task_dict = cfg.inference.tasks

    for task, enabled in task_dict.items():

        if enabled:
            log.info(f"Running task {task}")

            if task in TASK_REGISTRY:
                log.info(f"Running task {task}")
                TASK_REGISTRY[task](cfg)
                copy_hydra_config_to_subfolder(cfg, task)

            else:
                log.error(f"Task {task} not found in inferencing task registry")
                return
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    
    log.info("PreproceInferencing complete.")

if __name__ == "__main__":
    main()