import logging
import os
import random

import hydra
from omegaconf import DictConfig

from src.inferencing.inference import main as inference

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "inference": inference,
    # Add more tasks here as needed
}


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
                TASK_REGISTRY[task](cfg)  # saves its config in its run folder

            else:
                log.error(f"Task {task} not found in inferencing task registry")
                return
        else:
            log.info(f"Skipping task {task}. Enabled: {enabled}")
    
    
    log.info("PreproceInferencing complete.")

if __name__ == "__main__":
    main()