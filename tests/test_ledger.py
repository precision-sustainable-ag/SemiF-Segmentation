import pytest

from src.labeling.ledger import LabelLedger


def _rows(*ids):
    return [{"image_id": i, "path": f"a/{i}.jpg", "batch": "b1", "location": "NC", "meta": {"k": 1}} for i in ids]


def test_add_items_is_idempotent(tmp_path):
    with LabelLedger(tmp_path / "labels.db") as ledger:
        ledger.ensure_round("r1", "field")
        assert ledger.add_items("r1", "field", _rows("x", "y")) == 2
        assert ledger.add_items("r1", "field", _rows("y", "z")) == 1
        assert [i["image_id"] for i in ledger.items(round_name="r1")] == ["x", "y", "z"]
        assert ledger.known_image_ids("field") == {"x", "y", "z"}
        assert ledger.status_counts("r1") == {"selected": 3}


def test_round_source_is_fixed(tmp_path):
    with LabelLedger(tmp_path / "labels.db") as ledger:
        ledger.ensure_round("r1", "field")
        ledger.ensure_round("r1", "field")
        with pytest.raises(ValueError, match="already exists"):
            ledger.ensure_round("r1", "semif")


def test_update_and_retry_errors(tmp_path):
    with LabelLedger(tmp_path / "labels.db") as ledger:
        ledger.ensure_round("r1", "field")
        ledger.add_items("r1", "field", _rows("x", "y", "z"))
        ledger.update("field", "x", status="error", error="fetch: missing")
        ledger.update("field", "y", status="error", error="prepare: bad jpeg")
        ids = lambda rows: sorted(r["image_id"] for r in rows)
        assert ids(ledger.items_for_task("r1", "selected", "fetch", retry_errors=False)) == ["z"]
        assert ids(ledger.items_for_task("r1", "selected", "fetch", retry_errors=True)) == ["x", "z"]
        with pytest.raises(ValueError, match="Unknown ledger columns"):
            ledger.update("field", "x", bogus=1)
