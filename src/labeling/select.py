"""select: pick new images for a labeling round from the round's source DB.

Tops the round up to `label.select.n_images`, skipping every image this
source already has in the project's labels.db (in any round), so re-running is
a no-op once the round is full. Like `agir-cv query`, it writes the picks to
query/query.csv (image_path is the path the source DB stores) and the query
itself to query/query_spec.json.
"""

import logging
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.labeling.ledger import LabelLedger
from src.labeling.run_folder import RunFolder, run_id, who, write_query
from src.labeling.snapshot import source_db_path
from src.labeling.sources import balanced_sample, get_source

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> dict | None:
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
            return None

        db_path = source_db_path(scfg.db, cfg.paths.source_db_dir, scfg.snapshot)
        filters = OmegaConf.to_container(lcfg.select.filters, resolve=True) or {}
        source = get_source(source_name, db_path)
        candidates = source.candidates(filters)
        matched = len(candidates)
        log.info("%d images in %s match filters %s", matched, Path(db_path).name, filters)

        known = ledger.known_image_ids(source_name)
        candidates = candidates[~candidates["image_id"].isin(known)]
        balance_by = list(lcfg.select.balance_by or [])
        picked = balanced_sample(candidates, need, balance_by, seed=lcfg.select.seed)

        added = ledger.add_items(lcfg.round, source_name, picked.to_dict(orient="records"))
        log.info("Selected %d new images for round %s (%d were already in it).", added, lcfg.round, have)
        if balance_by and added:
            counts = picked.groupby(balance_by, dropna=False).size()
            log.info("Picks per %s:\n%s", balance_by, counts.to_string())

    run = RunFolder.from_cfg(cfg)
    spec = query_spec(cfg, run, source, db_path, filters, need, balance_by,
                      matched=matched, already_in_project=matched - len(candidates), returned=len(picked))
    write_query(run, picked.drop(columns=["meta"]).rename(columns={"path": "image_path"}), spec)
    log.info("Query results saved to %s", run.query_csv)
    return {"matched": matched, "already_in_project": matched - len(candidates),
            "selected": added, "round_images": have + added}


def query_spec(cfg, run, source, db_path, filters, n, balance_by, *, matched, already_in_project, returned) -> dict:
    """query_spec.json in agir-cv query's layout. Selection is a balanced
    sample: round-robin over the balance_by groups, each in a seeded order."""
    seed = cfg.label.select.seed
    sample_raw = f"balanced:n={n},seed={seed}" + (f",by={'|'.join(balance_by)}" if balance_by else "")
    db = Path(cfg.sources[cfg.label.source].db)
    return {
        "query_metadata": {"run_id": run_id(cfg), "timestamp": datetime.now().isoformat(), **who()},
        "database": {
            "type": cfg.label.source,
            "db_path": str(db),
            "snapshot": None if Path(db_path) == db else str(db_path),
            "table": source.table,
        },
        "query_parameters": {
            "filters": {"raw": filters, "parsed": filters},
            "projection": None,
            "sort": {"raw": None, "parsed": None},
            "limit": n,
            "offset": None,
            "sample": {"raw": sample_raw,
                       "parsed": {"strategy": "balanced", "n": n, "seed": seed, "by": balance_by}},
        },
        "execution": {
            "preview_mode": False,
            "preview_count": 0,
            "output_format": "csv",
            "output_path": str(run.query_csv),
            "rows_matched": matched,
            "rows_already_in_project": already_in_project,
            "rows_returned": returned,
        },
    }
