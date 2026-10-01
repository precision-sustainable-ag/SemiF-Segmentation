"""fetch: move the round's selected full-resolution images into its run folder.

Globus by default (transfer.backend=globus, submitted with the globus CLI as
AgIR-CVToolkit's scinet-transfer does); transfer.backend=copy does a plain file
copy from mounted roots instead. Files keep their layout below their source
root. The run folder gets the batch file(s) and transfer_manifest.json. Images
already on disk from an earlier, interrupted run aren't transferred again.
"""

import logging

from omegaconf import DictConfig, OmegaConf

from src.labeling.ledger import LabelLedger
from src.labeling.run_folder import RunFolder, run_id
from src.labeling.transfer import FINAL_STATES, make_backend, plan_fetch, write_transfer_manifest

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> dict | None:
    lcfg = cfg.label
    source = lcfg.source
    roots = OmegaConf.to_container(cfg.sources[source].roots, resolve=True)
    run = RunFolder.from_cfg(cfg)

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items_for_task(lcfg.round, "selected", "fetch", lcfg.retry_errors)
        if not items:
            log.info("No selected images waiting to be fetched in round %s.", lcfg.round)
            return None

        jobs, unresolved = plan_fetch(items, roots, run.root)
        for item, reason in unresolved:
            log.warning("Can't fetch %s: %s", item["image_id"], reason)
            ledger.update(source, item["image_id"], status="error", error=f"fetch: {reason}")

        pending = [job for job in jobs if not job.dest.exists()]
        log.info("%d images to fetch (%d already on disk, %d unresolved).",
                 len(pending), len(jobs) - len(pending), len(unresolved))
        if lcfg.fetch.dry_run:
            for job in pending:
                log.info("[dry-run] %s: %s/%s -> %s", job.image_id, job.root, job.rel_path, job.dest)
            return {"dry_run": True, "to_fetch": len(pending), "unresolved": len(unresolved)}

        records = []
        if pending:
            backend = make_backend(OmegaConf.to_container(cfg.transfer, resolve=True), roots, run, run_id(cfg))
            records = backend.submit(pending)
            write_transfer_manifest(run, backend, records)  # task IDs are on disk before the long wait
            backend.wait(records)
            write_transfer_manifest(run, backend, records)

        record_of = {record["root"]: record for record in records}
        pending_ids = {job.image_id for job in pending}
        counts = {"fetched": 0, "still_transferring": 0, "errors": len(unresolved)}
        for job in jobs:
            record = record_of.get(job.root) if job.image_id in pending_ids else None
            task_id = (record or {}).get("task_id")
            fields = {"globus_task_id": task_id} if task_id else {}
            if job.dest.exists():
                ledger.update(source, job.image_id, status="fetched", root=job.root,
                              native_path=str(job.dest), error=None, **fields)
                counts["fetched"] += 1
            elif task_id and record["status"] not in FINAL_STATES:
                # Still on its way: stays selected, so the next fetch picks it up.
                ledger.update(source, job.image_id, root=job.root, **fields)
                counts["still_transferring"] += 1
            else:
                where = f" (Globus task {task_id})" if task_id else ""
                ledger.update(source, job.image_id, status="error", root=job.root,
                              error=f"fetch: {job.rel_path} missing after transfer{where}", **fields)
                counts["errors"] += 1
        log.info("Fetched %d/%d images for round %s.", counts["fetched"], len(items), lcfg.round)
        transfers = [{key: r[key] for key in ("root", "task_id", "status", "num_items") if key in r} for r in records]
        return {**counts, "transfers": transfers}
