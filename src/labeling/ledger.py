"""labels.db: the local record of every image pulled into one of a project's
labeling rounds. It lives next to the rounds' run folders, in
outputs/runs/<project.name>/labels.db (see run_folder.py).

One row per (source, image_id). Each `mode=label` task only acts on rows in the
status it expects and advances them, so re-running a task is safe:

    selected -> fetched -> prepared -> in_cvat -> labeled | excluded

A task that fails on an image sets status "error" and an `error` message
prefixed with the task name ("fetch: ..."); `label.retry_errors=true` makes
that task pick those rows up again.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

STATUSES = ("selected", "fetched", "prepared", "in_cvat", "labeled", "excluded", "error")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    round       TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    config_yaml TEXT
);

CREATE TABLE IF NOT EXISTS items (
    source                 TEXT NOT NULL,
    image_id               TEXT NOT NULL,
    round                  TEXT NOT NULL REFERENCES rounds(round),
    status                 TEXT NOT NULL,
    path                   TEXT NOT NULL,  -- as the source DB stores it (relative or absolute)
    batch                  TEXT,
    location               TEXT,
    meta_json              TEXT,
    root                   TEXT,           -- sources.<source>.roots entry the image was fetched from
    native_path            TEXT,           -- local full-resolution copy, in the round's run folder
    native_width           INTEGER,
    native_height          INTEGER,
    cvat_image_path        TEXT,           -- downscaled copy uploaded to CVAT (images/)
    cvat_width             INTEGER,
    cvat_height            INTEGER,
    scale                  REAL,           -- cvat_width / native_width
    prelabel_path          TEXT,           -- model mask at CVAT resolution (masks/)
    globus_task_id         TEXT,
    cvat_task_id           INTEGER,
    cvat_job_id            INTEGER,
    cvat_frame             INTEGER,
    cvat_prelabel_uploaded INTEGER NOT NULL DEFAULT 0,
    mask_path              TEXT,           -- human mask at CVAT resolution (cvat_downloads/<task>/masks/)
    changed_frac           REAL,           -- fraction of pixels the annotator changed vs the prelabel
    split                  TEXT,           -- train | val | test, assigned once by build_dataset
    error                  TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    PRIMARY KEY (source, image_id)
);

CREATE INDEX IF NOT EXISTS idx_items_round_status ON items(round, status);

-- The frames an image is uploaded to CVAT as: one downscaled copy, or (with
-- label.prepare.split) full-resolution pieces of it. x/y/width/height place the
-- piece in the full-resolution image.
CREATE TABLE IF NOT EXISTS tiles (
    source                 TEXT NOT NULL,
    image_id               TEXT NOT NULL,
    name                   TEXT NOT NULL,  -- CVAT frame name (file stem)
    x                      INTEGER NOT NULL,
    y                      INTEGER NOT NULL,
    width                  INTEGER NOT NULL,
    height                 INTEGER NOT NULL,
    cvat_image_path        TEXT NOT NULL,  -- the file uploaded to CVAT (images/)
    cvat_width             INTEGER,
    cvat_height            INTEGER,
    prelabel_path          TEXT,           -- pre-label at CVAT resolution (masks/)
    cvat_task_id           INTEGER,
    cvat_job_id            INTEGER,
    cvat_frame             INTEGER,
    cvat_prelabel_uploaded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, image_id, name),
    FOREIGN KEY (source, image_id) REFERENCES items(source, image_id)
);
"""

# Ledgers from before `tiles` existed: each prepared image was one frame.
_MIGRATE = """
INSERT OR IGNORE INTO tiles (
    source, image_id, name, x, y, width, height, cvat_image_path, cvat_width, cvat_height,
    prelabel_path, cvat_task_id, cvat_job_id, cvat_frame, cvat_prelabel_uploaded
)
SELECT source, image_id, image_id, 0, 0, native_width, native_height, cvat_image_path, cvat_width,
       cvat_height, prelabel_path, cvat_task_id, cvat_job_id, cvat_frame, cvat_prelabel_uploaded
FROM items
WHERE cvat_image_path IS NOT NULL AND native_width IS NOT NULL
  AND status IN ('prepared', 'in_cvat', 'labeled', 'excluded')
  AND NOT EXISTS (SELECT 1 FROM tiles t WHERE t.source = items.source AND t.image_id = items.image_id);
"""

_TILE_COLUMNS = (
    "name", "x", "y", "width", "height", "cvat_image_path", "cvat_width", "cvat_height", "prelabel_path",
)
_TILE_UPDATABLE = {"cvat_task_id", "cvat_job_id", "cvat_frame", "cvat_prelabel_uploaded", "prelabel_path"}

_UPDATABLE = {
    "round", "status", "path", "batch", "location", "meta_json", "root",
    "native_path", "native_width", "native_height",
    "cvat_image_path", "cvat_width", "cvat_height", "scale", "prelabel_path",
    "globus_task_id", "cvat_task_id", "cvat_job_id", "cvat_frame", "cvat_prelabel_uploaded",
    "mask_path", "changed_frac", "split", "error",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LabelLedger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.executescript(_MIGRATE)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LabelLedger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- rounds ----------

    def ensure_round(self, round_name: str, source: str, config_yaml: str | None = None) -> None:
        row = self.conn.execute("SELECT source FROM rounds WHERE round = ?", (round_name,)).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO rounds (round, source, created_at, config_yaml) VALUES (?, ?, ?, ?)",
                (round_name, source, _now(), config_yaml),
            )
            self.conn.commit()
        elif row["source"] != source:
            raise ValueError(
                f"Round {round_name!r} already exists with source {row['source']!r}, not {source!r}"
            )

    # ---------- items ----------

    def add_items(self, round_name: str, source: str, rows: Iterable[dict]) -> int:
        """Insert newly selected images; images already in the ledger are left alone."""
        now = _now()
        values = [
            (
                source, row["image_id"], round_name, "selected", row["path"],
                row.get("batch"), row.get("location"),
                json.dumps(row.get("meta") or {}, default=str),
                row.get("width"), row.get("height"), now, now,
            )
            for row in rows
        ]
        before = self.conn.total_changes
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO items (
                source, image_id, round, status, path, batch, location, meta_json,
                native_width, native_height, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def items(
        self,
        round_name: str | None = None,
        status: str | Iterable[str] | None = None,
        source: str | None = None,
    ) -> list[dict]:
        clauses, params = [], []
        if round_name is not None:
            clauses.append("round = ?")
            params.append(round_name)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            clauses.append(f"status IN ({', '.join('?' * len(statuses))})")
            params.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM items {where} ORDER BY source, image_id", params)
        return [dict(r) for r in rows]

    def items_for_task(self, round_name: str, status: str, task: str, retry_errors: bool) -> list[dict]:
        """Rows ready for `task`, plus that task's own failures when retrying."""
        rows = self.items(round_name=round_name, status=status)
        if retry_errors:
            rows += [
                r for r in self.items(round_name=round_name, status="error")
                if (r["error"] or "").startswith(f"{task}:")
            ]
        return rows

    def update(self, source: str, image_id: str, **fields) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Unknown ledger columns: {sorted(unknown)}")
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{col} = ?" for col in fields)
        self.conn.execute(
            f"UPDATE items SET {assignments} WHERE source = ? AND image_id = ?",
            [*fields.values(), source, image_id],
        )
        self.conn.commit()

    def known_image_ids(self, source: str) -> set[str]:
        rows = self.conn.execute("SELECT image_id FROM items WHERE source = ?", (source,))
        return {r["image_id"] for r in rows}

    def status_counts(self, round_name: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM items WHERE round = ? GROUP BY status", (round_name,)
        )
        return {r["status"]: r["n"] for r in rows}

    # ---------- tiles ----------

    def set_tiles(self, source: str, image_id: str, tiles: Iterable[dict]) -> None:
        """Replace an image's tiles (prepare)."""
        self.conn.execute("DELETE FROM tiles WHERE source = ? AND image_id = ?", (source, image_id))
        self.conn.executemany(
            f"INSERT INTO tiles (source, image_id, {', '.join(_TILE_COLUMNS)}) "
            f"VALUES (?, ?, {', '.join('?' * len(_TILE_COLUMNS))})",
            [(source, image_id, *(tile.get(col) for col in _TILE_COLUMNS)) for tile in tiles],
        )
        self.conn.commit()

    def tiles(self, source: str, image_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tiles WHERE source = ? AND image_id = ? ORDER BY y, x", (source, image_id)
        )
        return [dict(r) for r in rows]

    def update_tile(self, source: str, image_id: str, name: str, **fields) -> None:
        unknown = set(fields) - _TILE_UPDATABLE
        if unknown:
            raise ValueError(f"Unknown tile columns: {sorted(unknown)}")
        assignments = ", ".join(f"{col} = ?" for col in fields)
        self.conn.execute(
            f"UPDATE tiles SET {assignments} WHERE source = ? AND image_id = ? AND name = ?",
            [*fields.values(), source, image_id, name],
        )
        self.conn.commit()

    def pending_prelabel_tiles(self, round_name: str) -> list[dict]:
        """Tiles of images already in CVAT whose pre-label hasn't been uploaded."""
        rows = self.conn.execute(
            """
            SELECT t.* FROM tiles t JOIN items i ON i.source = t.source AND i.image_id = t.image_id
            WHERE i.round = ? AND i.status = 'in_cvat'
              AND t.prelabel_path IS NOT NULL AND t.cvat_prelabel_uploaded = 0 AND t.cvat_task_id IS NOT NULL
            ORDER BY t.cvat_task_id, t.cvat_frame
            """,
            (round_name,),
        )
        return [dict(r) for r in rows]
