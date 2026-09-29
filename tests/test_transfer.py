import json
import subprocess
from pathlib import Path

import pytest

from src.labeling import transfer
from src.labeling.run_folder import RunFolder
from src.labeling.transfer import CopyBackend, GlobusBackend, plan_fetch, write_transfer_manifest


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


@pytest.fixture
def run(tmp_path):
    return RunFolder(tmp_path / "outputs/runs/proj/r1")


SEMIF_ITEMS = [
    {"source": "semif", "image_id": "img1", "path": "semifield-developed-images/B1/images/img1.jpg"},
    {"source": "semif", "image_id": "img9", "path": "semifield-developed-images/B9/images/img9.jpg"},
]


def test_plan_fetch_resolves_roots_and_keeps_their_layout(roots, run):
    items = SEMIF_ITEMS + [
        {"source": "field", "image_id": "A1", "path": f"{roots[0]['local_path']}/field-batches/NC_1/developed-images/A1.jpg"},
        {"source": "field", "image_id": "A2", "path": f"{roots[0]['local_path']}/field-batches/NC_1/developed-images/A2.jpg"},
        {"source": "field", "image_id": "Z1", "path": "/elsewhere/Z1.jpg"},
    ]
    jobs, unresolved = plan_fetch(items, roots, run.root)
    by_id = {job.image_id: job for job in jobs}

    # Relative path: first mounted root that has the file.
    assert by_id["img1"].root == "lts2"
    assert by_id["img1"].dest == run.root / "semifield-developed-images/B1/images/img1.jpg"
    # Not on any mount: falls back to the Globus-only root.
    assert by_id["img9"].root == "juno"
    assert by_id["img9"].rel_path == "semifield-developed-images/B9/images/img9.jpg"
    # Absolute path under a root's local_path: lands at its path below the root.
    assert by_id["A1"].root == "lts1"
    assert by_id["A1"].dest == run.root / "field-batches/NC_1/developed-images/A1.jpg"
    # Missing on its (mounted) root, and outside every root.
    assert sorted(item["image_id"] for item, _ in unresolved) == ["A2", "Z1"]


def test_copy_backend(roots, run):
    jobs, _ = plan_fetch(SEMIF_ITEMS[:1], roots, run.root)
    backend = CopyBackend(roots, run)
    records = backend.submit(jobs)
    assert (run.root / "semifield-developed-images/B1/images/img1.jpg").read_bytes() == b"jpg"
    assert records == [{"root": "lts2", "src_root": roots[1]["local_path"], "num_items": 1, "copied": 1}]

    write_transfer_manifest(run, backend, records)
    manifest = json.loads(run.transfer_manifest.read_text())
    assert manifest["backend"] == "copy" and manifest["num_items"] == 1


@pytest.fixture
def globus_cli(monkeypatch):
    """Stands in for the globus CLI: records every command, answers like it."""
    calls = []
    state = {"logged_in": True, "task_status": "SUCCEEDED"}

    def run(cmd, capture_output, text):
        calls.append(cmd)
        args = cmd[1:]
        if args[:2] == ["session", "show"]:
            code = 0 if state["logged_in"] else 1
            return subprocess.CompletedProcess(cmd, code, stdout="{}", stderr="" if code == 0 else "No session")
        if args[0] == "transfer":
            n = sum(1 for c in calls if c[1] == "transfer")
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"task_id": f"task-{n}"}), stderr="")
        if args[:2] == ["task", "show"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"status": state["task_status"]}), stderr="")
        raise AssertionError(f"unexpected globus command {args}")

    monkeypatch.setattr(transfer.subprocess, "run", run)
    monkeypatch.setattr(transfer.shutil, "which", lambda cli: f"/usr/bin/{cli}")
    return calls, state


def _globus_cfg(**overrides):
    gcfg = {
        "cli": "globus",
        "endpoints": {"ep1": "EP1", "juno": "JUNO"},
        "destination": {"name": "sunny", "endpoint": "DEST", "dst_root": "/~/outputs/runs"},
        "transfer": {"sync_level": "checksum", "verify_checksum": True, "skip_source_errors": True,
                     "notify": "failed,inactive", "label_prefix": "semif-segmentation fetch",
                     "poll_interval_s": 0, "timeout_s": 60},
    }
    gcfg.update(overrides)
    return gcfg


def _submitted(calls):
    return [c[1:] for c in calls if c[1] == "transfer"]


def test_globus_backend_one_transfer_per_root(roots, run, globus_cli):
    calls, _ = globus_cli
    jobs, _ = plan_fetch(SEMIF_ITEMS, roots, run.root)
    backend = GlobusBackend(_globus_cfg(), roots, run, "proj/r1")
    records = backend.submit(jobs)

    # Two source collections -> two transfers, each with its own batch file,
    # landing under <dst_root>/<project>/<round>/ with the source layout.
    assert [r["task_id"] for r in records] == ["task-1", "task-2"]
    assert (run.root / "globus_batch_lts2.txt").read_text() == (
        "/lts2/semifield-developed-images/B1/images/img1.jpg "
        "/~/outputs/runs/proj/r1/semifield-developed-images/B1/images/img1.jpg\n"
    )
    assert (run.root / "globus_batch_juno.txt").read_text().startswith(
        "/LTS/project/x/semifield-developed-images/B9/images/img9.jpg /~/outputs/runs/proj/r1/")
    first, second = _submitted(calls)
    assert first[:3] == ["transfer", "EP1", "DEST"] and second[:3] == ["transfer", "JUNO", "DEST"]
    for flag in ("--skip-source-errors", "--verify-checksum"):
        assert flag in first
    assert first[first.index("--sync-level") + 1] == "checksum"
    assert first[first.index("--notify") + 1] == "failed,inactive"
    assert first[first.index("--label") + 1] == "semif-segmentation fetch dst=sunny run=proj/r1 root=lts2"

    backend.wait(records)
    assert {r["status"] for r in records} == {"SUCCEEDED"}
    write_transfer_manifest(run, backend, records)
    manifest = json.loads(run.transfer_manifest.read_text())
    assert manifest["dst_root"] == "/~/outputs/runs/proj/r1/"
    assert manifest["project_run_id"] == "proj/r1" and manifest["num_items"] == 2
    assert [t["src_endpoint"] for t in manifest["transfers"]] == ["EP1", "JUNO"]


def test_globus_backend_single_root_uses_globus_batch_txt(roots, run, globus_cli):
    jobs, _ = plan_fetch(SEMIF_ITEMS[:1], roots, run.root)
    records = GlobusBackend(_globus_cfg(), roots, run, "proj/r1").submit(jobs)
    assert records[0]["batch_file"] == str(run.root / "globus_batch.txt")
    assert (run.root / "globus_batch.txt").exists()


def test_globus_backend_stops_waiting_at_timeout(roots, run, globus_cli):
    _, state = globus_cli
    state["task_status"] = "ACTIVE"
    jobs, _ = plan_fetch(SEMIF_ITEMS[:1], roots, run.root)
    backend = GlobusBackend(_globus_cfg(transfer={"poll_interval_s": 0, "timeout_s": 0}), roots, run, "proj/r1")
    records = backend.submit(jobs)
    backend.wait(records)
    assert records[0]["status"] == "ACTIVE"


def test_globus_backend_errors(roots, run, globus_cli):
    calls, state = globus_cli
    with pytest.raises(ValueError, match="destination"):
        GlobusBackend(_globus_cfg(destination={"endpoint": None}), roots, run, "proj/r1")
    jobs, _ = plan_fetch(SEMIF_ITEMS[:1], roots, run.root)
    with pytest.raises(ValueError, match="endpoints.ep1"):
        GlobusBackend(_globus_cfg(endpoints={"ep1": None}), roots, run, "proj/r1").submit(jobs)
    state["logged_in"] = False
    with pytest.raises(RuntimeError, match="globus login"):
        GlobusBackend(_globus_cfg(), roots, run, "proj/r1").submit(jobs)
    assert not _submitted(calls)
