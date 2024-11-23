import logging
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING

# Import the task functions
from src.preprocessing.move_fullsized_data import main as move_fullsized_data
from src.preprocessing.grid_crop import main as grid_crop
from src.preprocessing.train_val_test_split import main as train_val_test_split
from src.preprocessing.remap_masks import main as remap_masks


from src.train import main as train
from src.viz_results import main as viz_results
from src.data_stats import main as data_stats

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "move_fullsized_data": move_fullsized_data,
    "grid_crop": grid_crop,
    "train_val_test_split": train_val_test_split,
    "remap_masks": remap_masks,
    "train": train,
    "viz_results": viz_results,
    "data_stats": data_stats,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    for tsk in cfg.tasks:
        try:            
            TASK_REGISTRY[tsk](cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()