"""select: pick new images for a labeling round from the round's source DB.

Tops the round up to `label.select.n_images`, skipping every image this
source already has in labels.db (in any round), so re-running is a no-op
once the round is full.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.labeling.ledger import LabelLedger
from src.labeling.snapshot import source_db_path
from src.labeling.sources import balanced_sample, get_source

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    lcfg = cfg.label
    source_name = lcfg.source
    scfg = cfg.sources[source_name]

    with LabelLedger(cfg.paths.labels_db) as ledger:
        ledger.ensure_round(lcfg.round, source_name, OmegaConf.to_yaml(lcfg, resolve=True))
        have = len(ledger.items(round_name=lcfg.round))
        need = lcfg.select.n_images - have
        if need <= 0:
            log.info("Round %s already has %d images (n_images=%d); nothing to select.",
                     lcfg.round, have, lcfg.select.n_images)
            return

        db_path = source_db_path(scfg.db, cfg.paths.source_db_dir, scfg.snapshot)
        filters = OmegaConf.to_container(lcfg.select.filters, resolve=True) or {}
        candidates = get_source(source_name, db_path).candidates(filters)
        log.info("%d images in %s match filters %s", len(candidates), Path(db_path).name, filters)

        known = ledger.known_image_ids(source_name)
        candidates = candidates[~candidates["image_id"].isin(known)]
        balance_by = list(lcfg.select.balance_by or [])
        picked = balanced_sample(candidates, need, balance_by, seed=lcfg.select.seed)

        added = ledger.add_items(lcfg.round, source_name, picked.to_dict(orient="records"))
        log.info("Selected %d new images for round %s (%d were already in it).", added, lcfg.round, have)
        if balance_by and added:
            counts = picked.groupby(balance_by, dropna=False).size()
            log.info("Picks per %s:\n%s", balance_by, counts.to_string())
