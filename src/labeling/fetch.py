"""fetch: move the round's selected full-resolution images to this machine.

Globus by default (transfer.backend=globus); transfer.backend=copy does a plain
file copy from mounted roots instead. Images already on disk from an earlier,
interrupted run aren't transferred again.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.labeling.ledger import LabelLedger
from src.labeling.transfer import make_backend, plan_fetch

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    lcfg = cfg.label
    source = lcfg.source
    roots = OmegaConf.to_container(cfg.sources[source].roots, resolve=True)
    dest_dir = Path(cfg.paths.label_images_dir)

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items_for_task(lcfg.round, "selected", "fetch", lcfg.retry_errors)
        if not items:
            log.info("No selected images waiting to be fetched in round %s.", lcfg.round)
            return

        jobs, unresolved = plan_fetch(items, roots, dest_dir)
        for item, reason in unresolved:
            log.warning("Can't fetch %s: %s", item["image_id"], reason)
            ledger.update(source, item["image_id"], status="error", error=f"fetch: {reason}")

        pending = [job for job in jobs if not job.dest.exists()]
        log.info("%d images to fetch (%d already on disk, %d unresolved).",
                 len(pending), len(jobs) - len(pending), len(unresolved))
        if lcfg.fetch.dry_run:
            for job in pending:
                log.info("[dry-run] %s: %s/%s -> %s", job.image_id, job.root, job.rel_path, job.dest)
            return

        task_ids = {}
        if pending:
            tcfg = OmegaConf.to_container(cfg.transfer, resolve=True)
            backend = make_backend(tcfg, roots, dest_dir)
            task_ids = backend.run(pending, label=f"semif-segmentation {lcfg.round}")

        pending_ids = {job.image_id for job in pending}
        fetched = 0
        for job in jobs:
            task_id = task_ids.get(job.root) if job.image_id in pending_ids else None
            if job.dest.exists():
                fields = {"globus_task_id": task_id} if task_id else {}
                ledger.update(source, job.image_id, status="fetched", root=job.root,
                              native_path=str(job.dest), error=None, **fields)
                fetched += 1
            else:
                where = f" (Globus task {task_id})" if task_id else ""
                ledger.update(source, job.image_id, status="error", root=job.root,
                              error=f"fetch: {job.rel_path} missing after transfer{where}")
        log.info("Fetched %d/%d images for round %s.", fetched, len(items), lcfg.round)
