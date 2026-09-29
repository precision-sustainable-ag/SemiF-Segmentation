"""Minimal CVAT REST client (self-hosted or app.cvat.ai) scoped to one organization.

Covers only what a labeling round needs. Every request carries `org_id`, so
projects and tasks are created in, and looked up from, that organization.
Endpoint paths follow the ones cvat-sdk generates from CVAT's API schema.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
import yaml

log = logging.getLogger(__name__)

# cvat-sdk's cap for one bulk (multipart) upload request.
MAX_REQUEST_BYTES = 100 * 2**20


class CvatError(RuntimeError):
    pass


class CvatClient:
    def __init__(
        self,
        url: str,
        org_id: int | None,
        *,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: float = 300,
        poll_seconds: float = 5,
        session: requests.Session | None = None,
    ):
        if not url:
            raise ValueError("CVAT url is not set (cvat.url, or cvat.url in the credentials file)")
        self.base = url.rstrip("/")
        self.org_id = org_id
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self.session = session or requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        elif username and password:
            self.session.auth = (username, password)
        else:
            raise ValueError("CVAT credentials need a token, or a username and password")

    @classmethod
    def from_config(cls, ccfg) -> "CvatClient":
        """cvat.url / cvat.org_id fall back to the `cvat:` section of the credentials file."""
        keys: dict = {}
        credentials = Path(ccfg.credentials_file)
        if credentials.exists():
            keys = (yaml.safe_load(credentials.read_text()) or {}).get("cvat") or {}
        else:
            log.warning("CVAT credentials file not found: %s", credentials)
        org_id = ccfg.org_id if ccfg.org_id is not None else keys.get("org_id")
        if org_id is None:
            raise ValueError("Set cvat.org_id (or cvat.org_id in the credentials file)")
        return cls(
            ccfg.url or keys.get("url"),
            int(org_id),
            username=keys.get("username"),
            password=keys.get("password"),
            token=keys.get("token"),
            timeout=ccfg.request_timeout_s,
            poll_seconds=ccfg.poll_seconds,
        )

    # ---------- plumbing ----------

    def request(self, method: str, path: str, *, params: dict | None = None,
                expect: Iterable[int] = (200, 201, 202, 204), **kwargs) -> requests.Response:
        query = dict(params or {})
        if self.org_id is not None:
            query["org_id"] = self.org_id
        url = f"{self.base}/api/{path.lstrip('/')}"
        response = self.session.request(method, url, params=query, timeout=self.timeout, **kwargs)
        if response.status_code not in tuple(expect):
            raise CvatError(f"{method} {url} -> HTTP {response.status_code}: {response.text[:500]}")
        return response

    def _json(self, method: str, path: str, **kwargs) -> Any:
        response = self.request(method, path, **kwargs)
        return response.json() if response.content else {}

    def list_all(self, path: str, params: dict | None = None) -> list[dict]:
        results: list[dict] = []
        page = 1
        while True:
            data = self._json("GET", path, params={**(params or {}), "page": page, "page_size": 100})
            results.extend(data.get("results", []))
            if not data.get("next"):
                return results
            page += 1

    # ---------- projects & labels ----------

    def find_project(self, name: str) -> dict | None:
        return next((p for p in self.list_all("projects", {"name": name}) if p["name"] == name), None)

    def create_project(self, name: str, labels: list[dict]) -> dict:
        return self._json("POST", "projects", json={"name": name, "labels": labels}, expect=(201,))

    def label_ids(self, project_id: int) -> dict[str, int]:
        return {label["name"]: label["id"] for label in self.list_all("labels", {"project_id": project_id})}

    # ---------- tasks ----------

    def create_task(self, name: str, project_id: int, segment_size: int) -> dict:
        spec = {"name": name, "project_id": project_id, "segment_size": segment_size}
        return self._json("POST", "tasks", json=spec, expect=(201,))

    def upload_images(self, task_id: int, paths: list[Path], *, image_quality: int,
                      max_request_bytes: int = MAX_REQUEST_BYTES) -> None:
        """CVAT's upload session: Upload-Start, one Upload-Multiple request per
        batch of files, then Upload-Finish, which starts building the task."""
        endpoint = f"tasks/{task_id}/data/"
        self.request("POST", endpoint, headers={"Upload-Start": ""}, expect=(202,))
        for batch in _batches(paths, max_request_bytes):
            files = [(f"client_files[{i}]", (p.name, p.read_bytes(), "image/jpeg")) for i, p in enumerate(batch)]
            self.request("POST", endpoint, headers={"Upload-Multiple": ""}, files=files, expect=(200,))
            log.info("Uploaded %d images to task %s", len(batch), task_id)
        response = self.request(
            "POST", endpoint, headers={"Upload-Finish": ""},
            data={"image_quality": image_quality, "sorting_method": "natural"}, expect=(202,),
        )
        rq_id = (response.json() if response.content else {}).get("rq_id")
        if rq_id:
            self.wait_for_request(rq_id)
        else:  # servers that predate the requests API
            self._wait_for_task_status(task_id)

    def wait_for_request(self, rq_id: str, timeout_s: float = 3 * 3600) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            data = self._json("GET", f"requests/{rq_id}")
            status = data.get("status")
            if status == "finished":
                return data
            if status == "failed":
                raise CvatError(f"CVAT request {rq_id} failed: {data.get('message')}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"CVAT request {rq_id} still {status} after {timeout_s:.0f}s")
            time.sleep(self.poll_seconds)

    def _wait_for_task_status(self, task_id: int, timeout_s: float = 3 * 3600) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            data = self._json("GET", f"tasks/{task_id}/status")
            state = data.get("state")
            if state == "Finished":
                return
            if state == "Failed":
                raise CvatError(f"CVAT task {task_id} data processing failed: {data.get('message')}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"CVAT task {task_id} still {state} after {timeout_s:.0f}s")
            time.sleep(self.poll_seconds)

    def task_frames(self, task_id: int) -> dict:
        """Frame metadata: {"frames": [{"name", "width", "height"}, ...], "deleted_frames": [...], ...}."""
        return self._json("GET", f"tasks/{task_id}/data/meta")

    def add_task_annotations(self, task_id: int, shapes: list[dict], tags: list[dict] | None = None) -> None:
        payload = {"version": 0, "tags": tags or [], "shapes": shapes, "tracks": []}
        self.request("PATCH", f"tasks/{task_id}/annotations/", params={"action": "create"},
                     json=payload, expect=(200,))

    # ---------- jobs & users ----------

    def task_jobs(self, task_id: int) -> list[dict]:
        jobs = self.list_all("jobs", {"task_id": task_id})
        # Ground-truth (honeypot) jobs aren't part of the annotation work.
        return sorted((j for j in jobs if j.get("type", "annotation") == "annotation"),
                      key=lambda j: j["start_frame"])

    def job_annotations(self, job_id: int) -> dict:
        return self._json("GET", f"jobs/{job_id}/annotations/")

    def assign_job(self, job_id: int, user_id: int) -> None:
        self._json("PATCH", f"jobs/{job_id}", json={"assignee": user_id})

    def find_user_id(self, username: str) -> int | None:
        users = self.list_all("users", {"search": username})
        return next((u["id"] for u in users if u.get("username") == username), None)


def _batches(paths: list[Path], max_bytes: int) -> Iterator[list[Path]]:
    batch: list[Path] = []
    size = 0
    for path in paths:
        n = path.stat().st_size
        if n > max_bytes:
            raise CvatError(
                f"{path} is {n / 2**20:.0f} MiB, over the {max_bytes / 2**20:.0f} MiB upload limit; "
                "lower label.prepare.max_side or jpeg_quality"
            )
        if batch and size + n > max_bytes:
            yield batch
            batch, size = [], 0
        batch.append(path)
        size += n
    if batch:
        yield batch
