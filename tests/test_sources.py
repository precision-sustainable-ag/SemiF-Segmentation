import sqlite3

import pandas as pd
import pytest

from src.labeling.sources import FieldSource, SemifSource, balanced_sample, build_where


def make_semif_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE semif (batch_id TEXT, image_id TEXT, cutout_id TEXT, state TEXT, season TEXT,
           bbot_version TEXT, category_common_name TEXT, image_path TEXT,
           fullres_width INTEGER, fullres_height INTEGER)"""
    )
    rows = [
        ("B1", "img1", "c1", "MD", "s22", "2.0", "velvetleaf", "semifield-developed-images/B1/images/img1.jpg", 9560, 6368),
        ("B1", "img1", "c2", "MD", "s22", "2.0", "cocklebur", "semifield-developed-images/B1/images/img1.jpg", 9560, 6368),
        ("B2", "img2", "c3", "NC", "s25", "3.1", "cocklebur", "semifield-developed-images/B2/images/img2.jpg", None, None),
        ("B2", "img3", "c4", "NC", "s25", "3.1", "waterhemp", None, None, None),
    ]
    conn.executemany("INSERT INTO semif VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def make_field_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE file_status (base_name TEXT PRIMARY KEY, location_code TEXT, plant_type TEXT, species TEXT,
            growth_stage TEXT, crop_or_fallow TEXT, cover_crop_family TEXT, flower_fruit_or_seeds TEXT,
            processed_jpg_in_nfs BOOLEAN);
        CREATE TABLE file_locations (id INTEGER PRIMARY KEY, base_name TEXT, extension TEXT, artifact_kind TEXT,
            storage_location TEXT, path TEXT, batch_label TEXT);
        INSERT INTO file_status VALUES
            ('A1', 'NC', 'COVERCROPS', 'Crimson clover', NULL, NULL, NULL, 'True', 1),
            ('A2', 'MD', 'WEEDS', 'Palmer amaranth', NULL, NULL, NULL, 'False', 1),
            ('A3', 'MD', 'COVERCROPS', 'Cereal rye', NULL, NULL, NULL, 'False', 0);
        INSERT INTO file_locations (base_name, extension, artifact_kind, storage_location, path, batch_label) VALUES
            ('A1', 'jpg', 'processed_jpg', 'nfs', '/lts/field-batches/NC_1/developed-images/A1.jpg', 'NC_1'),
            ('A1', 'arw', 'raw', 'nfs', '/lts/field-batches/NC_1/raw/A1.ARW', 'NC_1'),
            ('A2', 'jpg', 'processed_jpg', 'nfs', '/lts/field-batches/MD_1/developed-images/A2.jpg', 'MD_1'),
            ('A3', 'jpg', 'processed_jpg', 'nfs', '/lts/field-batches/MD_1/developed-images/A3.jpg', 'MD_1');
        """
    )
    conn.close()


def test_build_where():
    clauses, params = build_where({"a": 1, "b": ["x", "y"], "c": "%pat%", "d": None}, {"a", "b", "c", "d"}, "t.")
    assert clauses == ["t.a = ?", "t.b IN (?, ?)", "t.c LIKE ?", "t.d IS NULL"]
    assert params == [1, "x", "y", "%pat%"]
    with pytest.raises(ValueError, match="Unknown filter column"):
        build_where({"nope": 1}, {"a"})


def test_semif_candidates_one_row_per_image(tmp_path):
    db = tmp_path / "agir.db"
    make_semif_db(db)
    df = SemifSource(db).candidates({}).set_index("image_id")
    assert sorted(df.index) == ["img1", "img2"]  # img3 has no image_path
    assert df.loc["img1", "path"] == "semifield-developed-images/B1/images/img1.jpg"
    assert df.loc["img1", "batch"] == "B1" and df.loc["img1", "location"] == "MD"
    assert set(df.loc["img1", "species"].split(",")) == {"velvetleaf", "cocklebur"}
    assert df.loc["img1", "meta"]["bbot_version"] == "2.0"

    filtered = SemifSource(db).candidates({"category_common_name": "cocklebur", "state": "NC"})
    assert list(filtered["image_id"]) == ["img2"]


def test_field_candidates_processed_jpgs_on_nfs(tmp_path):
    db = tmp_path / "field.db"
    make_field_db(db)
    df = FieldSource(db).candidates({}).set_index("image_id")
    assert sorted(df.index) == ["A1", "A2"]  # A3 isn't processed_jpg_in_nfs
    assert df.loc["A1", "path"] == "/lts/field-batches/NC_1/developed-images/A1.jpg"
    assert df.loc["A1", "batch"] == "NC_1"
    assert df.loc["A1", "meta"]["plant_type"] == "COVERCROPS"

    crimson = FieldSource(db).candidates({"species": "%crimson%", "plant_type": ["COVERCROPS"]})
    assert list(crimson["image_id"]) == ["A1"]


def test_balanced_sample_spreads_across_groups():
    df = pd.DataFrame({"image_id": [f"i{n}" for n in range(12)], "batch": ["a"] * 8 + ["b"] * 3 + ["c"]})
    picked = balanced_sample(df, 6, ["batch"], seed=0)
    assert len(picked) == 6
    counts = picked["batch"].value_counts().to_dict()
    # Round-robin: every group once, then a and b again, then whichever of
    # a/b comes first in the seeded group order.
    assert counts["c"] == 1 and sorted([counts["a"], counts["b"]]) == [2, 3]
    # Deterministic for a given seed.
    assert list(balanced_sample(df, 6, ["batch"], seed=0)["image_id"]) == list(picked["image_id"])
    assert len(balanced_sample(df, 50, ["batch"], seed=0)) == 12
    with pytest.raises(ValueError, match="balance_by"):
        balanced_sample(df, 3, ["species"], seed=0)
