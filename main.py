import logging
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING

# Import the task functions
from src.preprocess import main as preprocess
from src.inference import main as inference
from src.train import main as train
from src.query import main as query
from src.sync import main as sync


log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "sync": sync,
    "preprocess": preprocess,
    "train": train,
    "inference": inference,
    "query": query,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    mode = cfg.mode
    log.info(f"Starting {mode}")
    
    if mode not in TASK_REGISTRY:
        log.error(f"Task {mode} not found in task registry")
        return
    
    TASK_REGISTRY[mode](cfg)

if __name__ == "__main__":
    main()