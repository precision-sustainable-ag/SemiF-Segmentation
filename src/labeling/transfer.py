"""Moving full-resolution images from where the source DB says they live to
this machine.

Each `sources.<source>.roots` entry maps a directory as the source DB sees it
(`local_path`, when it's mounted here) to a Globus collection (`endpoint`, a key
in `transfer.globus.endpoints`) and that same directory's path on the collection
(`endpoint_path`). Fetched files land in `paths.label_images_dir/<source>/`.
"""

from __future__ import annotations

import logging
import posixpath
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)


@dataclass
class FetchJob:
    source: str
    image_id: str
    root: str          # name of the roots entry the file is fetched from
    rel_path: str      # path relative to that root
    dest: Path         # local destination


def plan_fetch(items: list[dict], roots: list[dict], dest_dir: Path) -> tuple[list[FetchJob], list[tuple[dict, str]]]:
    """Resolve each item to a root. Returns (jobs, [(item, reason), ...] for unresolved items)."""
    jobs: list[FetchJob] = []
    unresolved: list[tuple[dict, str]] = []
    for item in items:
        path = PurePosixPath(item["path"])
        root, rel_or_reason = _resolve(path, roots)
        if root is None:
            unresolved.append((item, rel_or_reason))
            continue
        suffix = path.suffix.lower() or ".jpg"
        dest = Path(dest_dir) / item["source"] / f"{item['image_id']}{suffix}"
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


class CopyBackend:
    """Plain file copy from mounted roots, for testing or before Globus is set up."""

    def __init__(self, roots: list[dict]):
        self.roots = {root["name"]: root for root in roots}

    def run(self, jobs: list[FetchJob], label: str) -> dict[str, str]:
        for job in jobs:
            local = self.roots[job.root].get("local_path")
            if not local:
                log.warning("Root %s isn't mounted here; the copy backend can't fetch %s", job.root, job.image_id)
                continue
            src = Path(local) / job.rel_path
            job.dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, job.dest)
            except OSError as exc:
                log.warning("Copy failed for %s: %s", src, exc)
        return {}


class GlobusBackend:
    """One Globus transfer per source root, into the configured destination collection.

    Uses a native-app login (transfer.globus.client_id); the first run prints a
    login URL, after which tokens are cached under ~/.globus/app.
    """

    def __init__(self, gcfg: dict, roots: list[dict], dest_dir: Path):
        try:
            import globus_sdk
        except ImportError as exc:
            raise ImportError("transfer.backend=globus needs globus-sdk: pip install globus-sdk") from exc
        self.globus_sdk = globus_sdk

        if not gcfg.get("client_id"):
            raise ValueError(
                "Set transfer.globus.client_id (register a native app at "
                "https://app.globus.org/settings/developers), or use transfer.backend=copy."
            )
        self.gcfg = gcfg
        self.roots = {root["name"]: root for root in roots}
        self.dest_dir = Path(dest_dir)
        self.destination = gcfg["destination"]
        if not self.destination.get("id") or not self.destination.get("path"):
            raise ValueError("Set transfer.globus.destination.id and .path")

    def _endpoint(self, root: dict) -> dict:
        name = root.get("endpoint")
        endpoint = self.gcfg["endpoints"].get(name) if name else None
        if not endpoint or not endpoint.get("id"):
            raise ValueError(f"Root {root['name']!r}: set transfer.globus.endpoints.{name}.id")
        if not root.get("endpoint_path"):
            raise ValueError(f"Set endpoint_path for root {root['name']!r} in conf/sources")
        return endpoint

    def _client(self, endpoints: list[dict]):
        app = self.globus_sdk.UserApp(
            "semif-segmentation",
            client_id=self.gcfg["client_id"],
            config=self.globus_sdk.GlobusAppConfig(request_refresh_tokens=True),
        )
        client = self.globus_sdk.TransferClient(app=app)
        # Globus Connect Server v5 mapped collections need a data_access consent.
        collections = [e["id"] for e in endpoints + [self.destination] if e.get("data_access")]
        if collections:
            client.add_app_data_access_scope(collections)
        return client

    def run(self, jobs: list[FetchJob], label: str) -> dict[str, str]:
        by_root: dict[str, list[FetchJob]] = defaultdict(list)
        for job in jobs:
            by_root[job.root].append(job)
        endpoints = {name: self._endpoint(self.roots[name]) for name in by_root}
        client = self._client(list(endpoints.values()))

        task_ids: dict[str, str] = {}
        for root_name, root_jobs in by_root.items():
            root = self.roots[root_name]
            data = self.globus_sdk.TransferData(
                source_endpoint=endpoints[root_name]["id"],
                destination_endpoint=self.destination["id"],
                label=f"{label} from {root_name}"[:128],
                sync_level=self.gcfg.get("sync_level"),
                verify_checksum=bool(self.gcfg.get("verify_checksum", True)),
                skip_source_errors=bool(self.gcfg.get("skip_source_errors", True)),
                notify_on_succeeded=False,
            )
            for job in root_jobs:
                src = posixpath.join(root["endpoint_path"], job.rel_path)
                dst = posixpath.join(self.destination["path"], job.dest.relative_to(self.dest_dir).as_posix())
                data.add_item(src, dst)
            task_id = client.submit_transfer(data)["task_id"]
            log.info("Submitted Globus task %s: %d files from %s", task_id, len(root_jobs), root_name)
            task_ids[root_name] = task_id

        for root_name, task_id in task_ids.items():
            self._wait(client, task_id)
        return task_ids

    def _wait(self, client, task_id: str) -> None:
        poll = int(self.gcfg.get("poll_seconds", 60))
        deadline = time.monotonic() + float(self.gcfg.get("timeout_hours", 48)) * 3600
        while not client.task_wait(task_id, timeout=poll, polling_interval=min(poll, 30)):
            task = client.get_task(task_id)
            log.info("Globus task %s: %s (%s of %s files done)",
                     task_id, task["status"], task.get("files_transferred"), task.get("files"))
            if time.monotonic() > deadline:
                raise TimeoutError(f"Globus task {task_id} still running after timeout_hours")
        task = client.get_task(task_id)
        if task["status"] != "SUCCEEDED":
            raise RuntimeError(f"Globus task {task_id} ended {task['status']}: {task.get('nice_status_details')}")
        for skipped in client.task_skipped_errors(task_id):
            log.warning("Globus task %s skipped %s: %s", task_id, skipped.get("source_path"), skipped.get("error_code"))


def make_backend(tcfg: dict, roots: list[dict], dest_dir: Path):
    if tcfg["backend"] == "copy":
        return CopyBackend(roots)
    if tcfg["backend"] == "globus":
        return GlobusBackend(tcfg["globus"], roots, dest_dir)
    raise ValueError(f"transfer.backend must be globus or copy, got {tcfg['backend']!r}")
