"""Moving a round's full-resolution images into its run folder.

Each `sources.<source>.roots` entry maps a directory as the source DB sees it
(`local_path`, when it's mounted here) to a Globus collection (`endpoint`, a key
in `transfer.globus.endpoints`) and that same directory's path on the collection
(`endpoint_path`). As with AgIR-CVToolkit's scinet-transfer, files keep their
layout below the root: <root>/semifield-developed-images/<batch>/images/<id>.jpg
lands at <run folder>/semifield-developed-images/<batch>/images/<id>.jpg.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from src.labeling.run_folder import RunFolder

log = logging.getLogger(__name__)

# Globus task states after which no more files will arrive.
FINAL_STATES = ("SUCCEEDED", "FAILED")


@dataclass
class FetchJob:
    source: str
    image_id: str
    root: str          # name of the roots entry the file is fetched from
    rel_path: str      # path relative to that root, and to the run folder it lands in
    dest: Path         # local destination


def plan_fetch(items: list[dict], roots: list[dict], run_root: Path) -> tuple[list[FetchJob], list[tuple[dict, str]]]:
    """Resolve each item to a root. Returns (jobs, [(item, reason), ...] for unresolved items)."""
    jobs: list[FetchJob] = []
    unresolved: list[tuple[dict, str]] = []
    for item in items:
        root, rel_or_reason = _resolve(PurePosixPath(item["path"]), roots)
        if root is None:
            unresolved.append((item, rel_or_reason))
            continue
        dest = Path(run_root) / rel_or_reason
        jobs.append(FetchJob(item["source"], item["image_id"], root["name"], rel_or_reason, dest))
    return jobs, unresolved


def _resolve(path: PurePosixPath, roots: list[dict]) -> tuple[dict | None, str]:
    """(root, path relative to it) or (None, reason).

    A root whose local_path is mounted here is only used if the file is
    actually there; roots without a local_path (reachable only through Globus)
    are assumed to have it, and the transfer reports it otherwise.
    """
    if path.is_absolute():
        for root in roots:
            local = root.get("local_path")
            if local and _is_under(path, PurePosixPath(local)):
                rel = path.relative_to(local)
                if Path(local).is_dir() and not (Path(local) / rel).exists():
                    return None, f"{path} not found under {root['name']}"
                return root, str(rel)
        return None, f"{path} is not under any configured root's local_path"

    fallback = None
    for root in roots:
        local = root.get("local_path")
        if local and Path(local).is_dir():
            if (Path(local) / path).exists():
                return root, str(path)
        elif fallback is None:
            # Not mounted here: only reachable through Globus.
            fallback = root
    if fallback is not None:
        return fallback, str(path)
    return None, f"{path} not found under any mounted root, and no Globus-only root is configured"


def _is_under(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _by_root(jobs: list[FetchJob]) -> dict[str, list[FetchJob]]:
    by_root: dict[str, list[FetchJob]] = defaultdict(list)
    for job in jobs:
        by_root[job.root].append(job)
    return dict(by_root)


def _batch_path(path: str) -> str:
    # The CLI splits each line like a shell would: quote only paths that need it.
    return shlex.quote(path) if re.search(r"[\s'\"#\\]", path) else path


def write_batch_file(pairs: list[tuple[str, str]], path: Path) -> None:
    """A `globus transfer --batch` file: "<source path> <destination path>" per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{_batch_path(src)} {_batch_path(dst)}\n" for src, dst in pairs))
    log.info("Batch file written: %s (%d items)", path, len(pairs))


def globus_folder_url(endpoint: str, path: str) -> str:
    """Globus web-app link that opens `path` on `endpoint`."""
    return f"https://app.globus.org/file-manager?origin_id={endpoint}&origin_path={quote(path, safe='')}"


def write_transfer_manifest(run: RunFolder, backend, records: list[dict]) -> None:
    """transfer_manifest.json: scinet-transfer's manifest, with one entry per transfer."""
    manifest = {**backend.describe(), "num_items": sum(r["num_items"] for r in records), "transfers": records}
    run.root.mkdir(parents=True, exist_ok=True)
    run.transfer_manifest.write_text(json.dumps(manifest, indent=2))


class CopyBackend:
    """Plain file copy from mounted roots, for testing or before Globus is set up."""

    def __init__(self, roots: list[dict], run: RunFolder):
        self.roots = {root["name"]: root for root in roots}
        self.run = run

    def describe(self) -> dict:
        return {"backend": "copy", "dst_root": str(self.run.root) + "/"}

    def submit(self, jobs: list[FetchJob]) -> list[dict]:
        records = []
        for root_name, root_jobs in _by_root(jobs).items():
            local = self.roots[root_name].get("local_path")
            record = {"root": root_name, "src_root": local, "num_items": len(root_jobs), "copied": 0}
            records.append(record)
            if not local:
                log.warning("Root %s isn't mounted here; the copy backend can't fetch its %d images",
                            root_name, len(root_jobs))
                continue
            for job in root_jobs:
                src = Path(local) / job.rel_path
                job.dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, job.dest)
                    record["copied"] += 1
                except OSError as exc:
                    log.warning("Copy failed for %s: %s", src, exc)
        return records

    def wait(self, records: list[dict]) -> None:
        """Copies finish in submit()."""


class GlobusBackend:
    """One Globus transfer per source root, submitted with the globus CLI the
    way AgIR-CVToolkit's scinet-transfer does it: run `globus login` once and
    fetch reuses that session. Each transfer's batch file stays in the run folder.
    """

    def __init__(self, gcfg: dict, roots: list[dict], run: RunFolder, run_id: str):
        destination = gcfg.get("destination") or {}
        if not destination.get("endpoint") or not destination.get("dst_root"):
            raise ValueError("Set transfer.globus.destination.endpoint and .dst_root")
        self.gcfg = gcfg
        self.options = gcfg.get("transfer") or {}
        self.roots = {root["name"]: root for root in roots}
        self.run = run
        self.run_id = run_id
        self.cli = os.path.expanduser(gcfg.get("cli") or "globus")
        self.dst_name = destination.get("name") or "destination"
        self.dst_endpoint = destination["endpoint"]
        self.dst_root_shared = destination["dst_root"].rstrip("/") + "/"
        # Namespaced by the run, as scinet-transfer does: <dst_root>/<project.name>/<project.subname>/
        self.dst_root = self.dst_root_shared + run_id.strip("/") + "/"

    def describe(self) -> dict:
        return {
            "backend": "globus",
            "dst_name": self.dst_name,
            "dst_endpoint": self.dst_endpoint,
            "dst_root": self.dst_root,
            "dst_root_shared": self.dst_root_shared,
            "project_run_id": self.run_id,
            "sync_level": self.options.get("sync_level", "checksum"),
        }

    def submit(self, jobs: list[FetchJob]) -> list[dict]:
        by_root = _by_root(jobs)
        # Check the config for every root before submitting anything.
        sources = {name: self._source(self.roots[name]) for name in by_root}
        self._check_logged_in()

        records = []
        for root_name, root_jobs in by_root.items():
            endpoint, endpoint_path = sources[root_name]
            batch_path = self.run.batch_file(root_name if len(by_root) > 1 else None)
            pairs = [(posixpath.join(endpoint_path, job.rel_path), self.dst_root + job.rel_path) for job in root_jobs]
            write_batch_file(pairs, batch_path)
            label = f"{self.options.get('label_prefix', 'semif-segmentation fetch')} dst={self.dst_name} run={self.run_id}"
            if len(by_root) > 1:
                label += f" root={root_name}"
            task_id = self._submit(endpoint, batch_path, label)
            log.info("Globus task %s: %d files from %s. Monitor: https://app.globus.org/activity/%s",
                     task_id, len(pairs), root_name, task_id)
            records.append({
                "root": root_name,
                "src_endpoint": endpoint,
                "src_root": endpoint_path.rstrip("/") + "/",
                "num_items": len(pairs),
                "batch_file": str(batch_path),
                "task_id": task_id,
                "status": "ACTIVE",
            })
        log.info("Files land in %s (on %s): %s", self.dst_root, self.dst_name,
                 globus_folder_url(self.dst_endpoint, self.dst_root))
        return records

    def wait(self, records: list[dict]) -> None:
        """Poll each task until it ends, or until transfer.timeout_s passes
        (the transfer keeps going then; re-run fetch once it's done)."""
        poll = float(self.options.get("poll_interval_s", 60))
        deadline = time.monotonic() + float(self.options.get("timeout_s", 48 * 3600))
        for record in records:
            task_id = record["task_id"]
            while True:
                task = self._task(task_id)
                record["status"] = task.get("status", "UNKNOWN")
                if record["status"] in FINAL_STATES:
                    break
                if time.monotonic() > deadline:
                    log.warning("Globus task %s is still %s after transfer.globus.transfer.timeout_s; "
                                "re-run fetch when it's done: https://app.globus.org/activity/%s",
                                task_id, record["status"], task_id)
                    break
                log.info("Globus task %s: %s (%s of %s files transferred)",
                         task_id, record["status"], task.get("files_transferred"), task.get("files"))
                time.sleep(poll)
            if record["status"] == "FAILED":
                log.error("Globus task %s FAILED: https://app.globus.org/activity/%s", task_id, task_id)
            elif task.get("subtasks_skipped_errors"):
                log.warning("Globus task %s skipped %s files it couldn't read: https://app.globus.org/activity/%s",
                            task_id, task["subtasks_skipped_errors"], task_id)

    def _source(self, root: dict) -> tuple[str, str]:
        name = root.get("endpoint")
        endpoint = (self.gcfg.get("endpoints") or {}).get(name) if name else None
        if not endpoint:
            raise ValueError(f"Root {root['name']!r}: set transfer.globus.endpoints.{name} to its collection ID")
        if not root.get("endpoint_path"):
            raise ValueError(f"Set endpoint_path for root {root['name']!r} in conf/sources")
        return endpoint, root["endpoint_path"]

    def _globus(self, *args: str) -> subprocess.CompletedProcess:
        binary = shutil.which(self.cli)
        if binary is None:
            raise RuntimeError(
                f"The globus CLI ({self.cli}) isn't on PATH. Install it with `uv tool install globus-cli`, "
                "run `globus login`, or set transfer.globus.cli to its full path."
            )
        return subprocess.run([binary, *args], capture_output=True, text=True)

    def _check_logged_in(self) -> None:
        result = self._globus("session", "show", "--format", "json")
        if result.returncode != 0:
            raise RuntimeError(f"Not logged into Globus ({result.stderr.strip()}). Run: globus login")

    def _submit(self, endpoint: str, batch_path: Path, label: str) -> str:
        args = [
            "transfer", endpoint, self.dst_endpoint,
            "--batch", str(batch_path),
            "--label", label[:128],
            "--sync-level", self.options.get("sync_level", "checksum"),
            "--verify-checksum" if self.options.get("verify_checksum", True) else "--no-verify-checksum",
            "--format", "json",
        ]
        if self.options.get("skip_source_errors", True):
            args.append("--skip-source-errors")
        if self.options.get("notify"):
            args += ["--notify", str(self.options["notify"])]
        result = self._globus(*args)
        if result.returncode != 0:
            # e.g. a collection that needs a data_access consent: the CLI says what to run.
            raise RuntimeError(f"globus transfer failed for {batch_path.name}:\n"
                               f"{result.stderr.strip() or result.stdout.strip()}")
        try:
            return json.loads(result.stdout)["task_id"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError(f"Could not read the task_id from globus transfer's output:\n{result.stdout}") from exc

    def _task(self, task_id: str) -> dict:
        result = self._globus("task", "show", task_id, "--format", "json")
        if result.returncode != 0:
            log.warning("globus task show %s failed: %s", task_id, result.stderr.strip())
            return {"status": "UNKNOWN"}
        try:
            return json.loads(result.stdout)
        except ValueError:
            return {"status": "UNKNOWN"}


def make_backend(tcfg: dict, roots: list[dict], run: RunFolder, run_id: str):
    if tcfg["backend"] == "copy":
        return CopyBackend(roots, run)
    if tcfg["backend"] == "globus":
        return GlobusBackend(tcfg["globus"], roots, run, run_id)
    raise ValueError(f"transfer.backend must be globus or copy, got {tcfg['backend']!r}")
