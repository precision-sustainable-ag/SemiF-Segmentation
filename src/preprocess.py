import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
import shutil

from hydra.core.hydra_config import HydraConfig

# Import the task functions
from src.preprocessing.grid_crop import main as grid_crop
from src.preprocessing.train_val_test_split import main as train_val_test_split
from src.preprocessing.remap_masks import main as remap_masks
from src.preprocessing.data_stats import main as data_stats

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "grid_crop": grid_crop,
    "train_val_test_split": train_val_test_split,
    "remap_masks": remap_masks,
    "data_stats": data_stats,
    # Add more tasks here as needed
}

def copy_hydra_config_to_subfolder(cfg : DictConfig, task_name: str):
    """Copies the entire .hydra directory (or just specific config files) to a target directory."""
    target_dir = Path(cfg.project.output_dir) / "data" / ".hydra" / task_name
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
    
    task_dict = cfg.preprocess.tasks

    for task, enabled in task_dict.items():

        if enabled:
            log.info(f"Running task {task}")

            if task in TASK_REGISTRY:
                log.info(f"Running task {task}")
                TASK_REGISTRY[task](cfg)
                copy_hydra_config_to_subfolder(cfg, task)

            else:
                log.error(f"Task {task} not found in preprocessing task registry")
                return
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    
    log.info("Preprocessing complete.")

if __name__ == "__main__":
    main()