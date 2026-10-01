"""Local snapshots of the source databases.

The source DBs live on shared NFS and can be written while we read them, so
select works from a local copy made with SQLite's online backup API (which
yields a consistent copy even if the source changes mid-copy).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

SNAPSHOT_MODES = ("auto", "always", "off")


def source_db_path(db: str | Path, snapshot_dir: str | Path, mode: str = "auto") -> Path:
    """Path select should read: the DB itself (mode "off") or a fresh local snapshot."""
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f"snapshot must be one of {SNAPSHOT_MODES}, got {mode!r}")
    db = Path(db)
    if mode == "off":
        if not db.exists():
            raise FileNotFoundError(f"Source DB not found: {db}")
        return db
    return ensure_snapshot(db, Path(snapshot_dir) / db.name, refresh=mode)


def ensure_snapshot(src: Path, dst: Path, refresh: str = "auto") -> Path:
    """Copy `src` to `dst` unless `dst` already matches src's current size and mtime.

    refresh="always" re-copies unconditionally.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source DB not found: {src}")

    stamp_path = dst.with_name(dst.name + ".source.json")
    stat = src.stat()
    stamp = {"source": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    if dst.exists() and refresh == "auto":
        try:
            previous = json.loads(stamp_path.read_text())
        except (OSError, ValueError):
            previous = None
        if previous == stamp:
            log.info("Snapshot is up to date: %s", dst)
            return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    log.info("Snapshotting %s -> %s (%.1f GB)", src, dst, stat.st_size / 1e9)

    src_conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(tmp)
    try:
        src_conn.backup(dst_conn, pages=16384)
    finally:
        dst_conn.close()
        src_conn.close()

    tmp.replace(dst)
    stamp_path.write_text(json.dumps(stamp))
    log.info("Snapshot complete: %s", dst)
    return dst
