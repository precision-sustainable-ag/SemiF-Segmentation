"""Source-database adapters: turn a round's filters into candidate images.

Both adapters return one row per image with the same columns:

    image_id, path, batch, location, width, height, meta (dict)

plus the metadata columns listed in `meta`, so they can be used in
`label.select.balance_by`. `path` is exactly what the source DB stores:
relative to one of the configured roots for `semif`, absolute (NFS) for `field`.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_where(
    filters: Mapping | None, allowed: set[str], column_prefix: str = ""
) -> tuple[list[str], list]:
    """{"col": value | [values] | "pattern%" | None} -> SQL clauses and parameters.

    Column names are checked against `allowed` since they can't be bound as
    parameters.
    """
    clauses: list[str] = []
    params: list = []
    for column, value in (filters or {}).items():
        if column not in allowed:
            raise ValueError(f"Unknown filter column {column!r}. Available: {sorted(allowed)}")
        ref = f"{column_prefix}{column}"
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Filter {column!r} has an empty list")
            clauses.append(f"{ref} IN ({', '.join('?' * len(value))})")
            params.extend(value)
        elif value is None:
            clauses.append(f"{ref} IS NULL")
        elif isinstance(value, str) and "%" in value:
            clauses.append(f"{ref} LIKE ?")
            params.append(value)
        else:
            clauses.append(f"{ref} = ?")
            params.append(value)
    return clauses, params


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _with_meta(df: pd.DataFrame, meta_columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["meta"] = df[list(meta_columns)].to_dict(orient="records") if len(df) else []
    return df


class SemifSource:
    """AgIR_DB_v2: one flat `semif` table with one row per cutout."""

    name = "semif"
    meta_columns = ("state", "season", "bbot_version", "species")

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def candidates(self, filters: Mapping | None = None) -> pd.DataFrame:
        with closing(_connect_ro(self.db_path)) as conn:
            clauses, params = build_where(filters, _table_columns(conn, "semif"))
            clauses.insert(0, "image_path IS NOT NULL AND image_path != ''")
            # Filters apply per cutout, so an image qualifies if any of its
            # cutouts match; species lists the matching cutouts' species.
            sql = f"""
                SELECT image_id,
                       MIN(image_path)    AS path,
                       MIN(batch_id)      AS batch,
                       MIN(state)         AS location,
                       MIN(state)         AS state,
                       MAX(fullres_width)  AS width,
                       MAX(fullres_height) AS height,
                       MIN(season)        AS season,
                       MIN(bbot_version)  AS bbot_version,
                       GROUP_CONCAT(DISTINCT category_common_name) AS species
                FROM semif
                WHERE {" AND ".join(clauses)}
                GROUP BY image_id
            """
            df = pd.read_sql_query(sql, conn, params=params)
        return _with_meta(df, self.meta_columns)


class FieldSource:
    """field_exploration.db: file_status (one row per image) joined to its
    processed JPG on NFS in file_locations."""

    name = "field"
    meta_columns = (
        "plant_type", "species", "growth_stage", "crop_or_fallow",
        "cover_crop_family", "flower_fruit_or_seeds",
    )

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def candidates(self, filters: Mapping | None = None) -> pd.DataFrame:
        with closing(_connect_ro(self.db_path)) as conn:
            clauses, params = build_where(
                filters, _table_columns(conn, "file_status"), column_prefix="fs."
            )
            clauses.insert(0, "fs.processed_jpg_in_nfs = 1")
            meta_select = ", ".join(f"fs.{col}" for col in self.meta_columns)
            sql = f"""
                SELECT fs.base_name         AS image_id,
                       MIN(fl.path)         AS path,
                       MIN(fl.batch_label)  AS batch,
                       fs.location_code     AS location,
                       {meta_select}
                FROM file_status fs
                JOIN file_locations fl
                  ON fl.base_name = fs.base_name
                 AND fl.artifact_kind = 'processed_jpg'
                 AND fl.storage_location = 'nfs'
                WHERE {" AND ".join(clauses)}
                GROUP BY fs.base_name
            """
            df = pd.read_sql_query(sql, conn, params=params)
        # field_exploration.db doesn't store image dimensions; prepare reads them.
        df["width"] = np.nan
        df["height"] = np.nan
        return _with_meta(df, self.meta_columns)


SOURCES = {cls.name: cls for cls in (SemifSource, FieldSource)}


def get_source(name: str, db_path: str | Path):
    if name not in SOURCES:
        raise ValueError(f"Unknown source {name!r}; expected one of {sorted(SOURCES)}")
    return SOURCES[name](db_path)


def balanced_sample(
    df: pd.DataFrame, n: int, balance_by: Sequence[str] | None, seed: int
) -> pd.DataFrame:
    """Up to `n` rows spread as evenly as possible across the groups in
    `balance_by`: round-robin over the groups, each in a seeded random order.
    Without `balance_by` it's a plain seeded random sample."""
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    shuffled = df.sample(frac=1.0, random_state=seed)
    if not balance_by:
        return shuffled.head(n)

    missing = [col for col in balance_by if col not in df.columns]
    if missing:
        raise ValueError(f"balance_by columns {missing} not in candidates: {sorted(df.columns)}")

    groups = [g.index for _, g in shuffled.groupby(list(balance_by), dropna=False, sort=True)]
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]

    picked: list = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for index in groups:
            if depth < len(index):
                picked.append(index[depth])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        depth += 1
    return df.loc[picked]
