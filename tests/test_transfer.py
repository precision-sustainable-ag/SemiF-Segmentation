import sys
import types
from pathlib import Path

import pytest

from src.labeling.transfer import CopyBackend, GlobusBackend, plan_fetch


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpg")
    return path


@pytest.fixture
def roots(tmp_path):
    first, second = tmp_path / "lts1", tmp_path / "lts2"
    first.mkdir()
    second.mkdir()
    _touch(second / "semifield-developed-images/B1/images/img1.jpg")
    _touch(first / "field-batches/NC_1/developed-images/A1.jpg")
    return [
        {"name": "lts1", "local_path": str(first), "endpoint": "ep1", "endpoint_path": "/lts1"},
        {"name": "lts2", "local_path": str(second), "endpoint": "ep1", "endpoint_path": "/lts2"},
        {"name": "juno", "local_path": None, "endpoint": "juno", "endpoint_path": "/LTS/project/x"},
    ]


def test_plan_fetch_resolves_roots(tmp_path, roots):
    items = [
        {"source": "semif", "image_id": "img1", "path": "semifield-developed-images/B1/images/img1.jpg"},
        {"source": "semif", "image_id": "img9", "path": "semifield-developed-images/B9/images/img9.jpg"},
        {"source": "field", "image_id": "A1", "path": f"{roots[0]['local_path']}/field-batches/NC_1/developed-images/A1.jpg"},
        {"source": "field", "image_id": "A2", "path": f"{roots[0]['local_path']}/field-batches/NC_1/developed-images/A2.jpg"},
        {"source": "field", "image_id": "Z1", "path": "/elsewhere/Z1.jpg"},
    ]
    jobs, unresolved = plan_fetch(items, roots, tmp_path / "images")
    by_id = {job.image_id: job for job in jobs}

    # Relative path: first mounted root that has the file.
    assert by_id["img1"].root == "lts2"
    # Not on any mount: falls back to the Globus-only root.
    assert by_id["img9"].root == "juno"
    assert by_id["img9"].rel_path == "semifield-developed-images/B9/images/img9.jpg"
    # Absolute path under a root's local_path.
    assert by_id["A1"].root == "lts1"
    assert by_id["A1"].rel_path == "field-batches/NC_1/developed-images/A1.jpg"
    assert by_id["A1"].dest == tmp_path / "images/field/A1.jpg"
    # Missing on its (mounted) root, and outside every root.
    assert sorted(item["image_id"] for item, _ in unresolved) == ["A2", "Z1"]


def test_copy_backend(tmp_path, roots):
    items = [{"source": "semif", "image_id": "img1", "path": "semifield-developed-images/B1/images/img1.jpg"}]
    jobs, _ = plan_fetch(items, roots, tmp_path / "images")
    CopyBackend(roots).run(jobs, label="test")
    assert (tmp_path / "images/semif/img1.jpg").read_bytes() == b"jpg"


@pytest.fixture
def fake_globus(monkeypatch):
    """A stand-in globus_sdk module that records what GlobusBackend submits."""
    record = {"transfers": []}

    class TransferData:
        def __init__(self, **kwargs):
            self.kwargs, self.items = kwargs, []
            record["transfers"].append(self)

        def add_item(self, src, dst):
            self.items.append((src, dst))

    class TransferClient:
        def __init__(self, app):
            record["app"] = app

        def add_app_data_access_scope(self, ids):
            record["scopes"] = sorted(ids)

        def submit_transfer(self, data):
            return {"task_id": f"task-{len(record['transfers'])}"}

        def task_wait(self, task_id, timeout, polling_interval):
            return True

        def get_task(self, task_id):
            return {"status": "SUCCEEDED"}

        def task_skipped_errors(self, task_id):
            return []

    module = types.SimpleNamespace(
        UserApp=lambda name, client_id, config: {"name": name, "client_id": client_id},
        GlobusAppConfig=lambda **kwargs: kwargs,
        TransferClient=TransferClient,
        TransferData=TransferData,
    )
    monkeypatch.setitem(sys.modules, "globus_sdk", module)
    return record


def _globus_cfg(**overrides):
    gcfg = {
        "client_id": "cid",
        "endpoints": {"ep1": {"id": "EP1", "data_access": True}, "juno": {"id": "JUNO", "data_access": True}},
        "destination": {"id": "DEST", "path": "/~/labels/images", "data_access": False},
        "sync_level": "checksum", "verify_checksum": True, "skip_source_errors": True,
        "poll_seconds": 1, "timeout_hours": 1,
    }
    gcfg.update(overrides)
    return gcfg


def test_globus_backend_submits_one_transfer_per_root(tmp_path, roots, fake_globus):
    items = [
        {"source": "semif", "image_id": "img1", "path": "semifield-developed-images/B1/images/img1.jpg"},
        {"source": "semif", "image_id": "img9", "path": "semifield-developed-images/B9/images/img9.jpg"},
    ]
    jobs, _ = plan_fetch(items, roots, tmp_path / "images")
    task_ids = GlobusBackend(_globus_cfg(), roots, tmp_path / "images").run(jobs, label="semif-segmentation r1")

    assert task_ids == {"lts2": "task-1", "juno": "task-2"}
    assert fake_globus["scopes"] == ["EP1", "JUNO"]  # destination is GCP: no data_access consent
    lts2, juno = fake_globus["transfers"]
    assert (lts2.kwargs["source_endpoint"], lts2.kwargs["destination_endpoint"]) == ("EP1", "DEST")
    assert lts2.items == [("/lts2/semifield-developed-images/B1/images/img1.jpg", "/~/labels/images/semif/img1.jpg")]
    assert juno.items == [("/LTS/project/x/semifield-developed-images/B9/images/img9.jpg", "/~/labels/images/semif/img9.jpg")]
    assert lts2.kwargs["sync_level"] == "checksum" and lts2.kwargs["verify_checksum"] is True
    assert lts2.kwargs["skip_source_errors"] is True


def test_globus_backend_config_errors(tmp_path, roots, fake_globus):
    with pytest.raises(ValueError, match="client_id"):
        GlobusBackend(_globus_cfg(client_id=None), roots, tmp_path)
    items = [{"source": "semif", "image_id": "img1", "path": "semifield-developed-images/B1/images/img1.jpg"}]
    jobs, _ = plan_fetch(items, roots, tmp_path / "images")
    backend = GlobusBackend(_globus_cfg(endpoints={"ep1": {"id": None}}), roots, tmp_path / "images")
    with pytest.raises(ValueError, match="endpoints.ep1.id"):
        backend.run(jobs, label="x")
