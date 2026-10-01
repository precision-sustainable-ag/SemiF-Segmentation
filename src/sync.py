"""
Refreshes the local snapshots of the source databases that labeling rounds
select from (conf/sources). `select` already snapshots its own source when
needed; this mode refreshes all of them at once, e.g. after a new DB release.
"""

import logging

from omegaconf import DictConfig

from src.labeling.snapshot import source_db_path

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    for name, source in cfg.sources.items():
        if source.snapshot == "off":
            log.info(f"{name}: snapshot is off, queried in place ({source.db})")
            continue
        path = source_db_path(source.db, cfg.paths.source_db_dir, source.snapshot)
        log.info(f"{name}: {path}")
    log.info("File synchronization complete.")
