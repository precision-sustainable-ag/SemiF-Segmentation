from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.labeling.ledger import LabelLedger
from src.preprocessing import build_dataset, data_stats, tile
from src.preprocessing.build_dataset import assign_splits
from src.utils.tiling import tile_starts

TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _items(groups: dict[str, int], split=None):
    return [
        {"source": "field", "image_id": f"{batch}_{n}", "batch": batch, "location": "NC", "split": split}
        for batch, count in groups.items() for n in range(count)
    ]


def test_tile_starts_cover_edges():
    assert tile_starts(100, 32, 32) == [0, 32, 64, 68]
    assert tile_starts(96, 32, 32) == [0, 32, 64]
    assert tile_starts(20, 32, 32) == [0]
    assert tile_starts(100, 40, 30) == [0, 30, 60]


def test_splits_follow_groups_and_targets():
    items = _items({f"b{n}": 10 for n in range(10)})
    assignments = assign_splits(items, TARGETS, "batch")
    split_of_batch = {}
    for (_, image_id), split in assignments.items():
        batch = image_id.split("_")[0]
        assert split_of_batch.setdefault(batch, split) == split  # a batch never straddles splits
    counts = pd.Series(assignments).value_counts().to_dict()
    assert counts == {"train": 80, "val": 10, "test": 10}


def test_existing_splits_never_change():
    old = _items({"b0": 5, "b1": 5})
    old[0]["split"] = old[1]["split"] = "test"
    new = _items({"b0": 2}) + _items({"b9": 3})
    for n, item in enumerate(new):
        item["image_id"] = f"{item['batch']}_new{n}"
    assignments = assign_splits(old + new, TARGETS, "batch")
    assert ("field", "b0_0") not in assignments  # already had a split
    # New images from an existing group join that group's split.
    assert {assignments[("field", i["image_id"])] for i in new if i["batch"] == "b0"} == {"test"}


def _labeled_ledger(cfg, n_images=6):
    """A round's run folder with n_images labeled 150x100 images, their human
    masks at CVAT resolution (120x80), and the last image excluded."""
    rng = np.random.default_rng(1)
    run_root = Path(cfg.paths.project_runs_dir) / "r1"
    image_dir = run_root / "field-batches/B/developed-images"
    mask_dir = run_root / "cvat_downloads/vegetation_001_r1/masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    with LabelLedger(cfg.paths.labels_db) as ledger:
        ledger.ensure_round("r1", "field")
        rows = [{"image_id": f"IMG{n}", "path": f"/x/IMG{n}.jpg", "batch": f"B{n}", "location": "NC"} for n in range(n_images)]
        ledger.add_items("r1", "field", rows)
        for n in range(n_images):
            image = rng.integers(0, 255, (100, 150, 3), dtype=np.uint8)
            mask = np.zeros((80, 120), np.uint8)
            mask[8:32, 16:56] = 1
            cv2.imwrite(str(image_dir / f"IMG{n}.jpg"), image)
            cv2.imwrite(str(mask_dir / f"IMG{n}.png"), mask)
            ledger.update("field", f"IMG{n}", status="labeled", native_path=str(image_dir / f"IMG{n}.jpg"),
                          mask_path=str(mask_dir / f"IMG{n}.png"))
        ledger.update("field", "IMG5", status="excluded")


def test_build_tile_and_stats(make_cfg):
    cfg = make_cfg("mode=preprocess", "preprocess.tile.scale=0.5", "preprocess.tile.tile_size=32",
                   "preprocess.tile.workers=2", "preprocess.data_stats.workers=2",
                   "preprocess.build_dataset.val_size=0.2", "preprocess.build_dataset.test_size=0.2")
    _labeled_ledger(cfg)

    build_dataset.main(cfg)
    manifest = pd.read_csv(cfg.paths.dataset_manifest)
    assert sorted(manifest["image_id"]) == ["IMG0", "IMG1", "IMG2", "IMG3", "IMG4"]  # excluded image left out
    assert set(manifest["split"]) == {"train", "val", "test"}
    with LabelLedger(cfg.paths.labels_db) as ledger:
        assert all(i["split"] for i in ledger.items(status="labeled"))

    tile.main(cfg)
    split_dir = Path(cfg.paths.split_dir)
    images = sorted((split_dir / "train" / "images").glob("*.jpg"))
    masks = sorted((split_dir / "train" / "masks").glob("*.png"))
    assert images and [p.stem for p in images] == [p.stem for p in masks]
    # 150x100 at scale 0.5 -> 75x50 (the 120x80 mask too): x origins 0, 32, 43; y origins 0, 18 -> 6 tiles per image.
    n_train = (manifest["split"] == "train").sum()
    assert len(images) == 6 * n_train
    tile_mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    assert tile_mask.shape == (32, 32) and set(np.unique(tile_mask)) <= {0, 1}

    data_stats.main(cfg)
    stats_dir = Path(cfg.paths.project_preprocess_dir) / "data" / "data_stats"
    assert set(OmegaConf.load(stats_dir / "rgb_mean_std.json")) == {"mean", "std"}
    assert (stats_dir / "class_distribution.png").exists()


def test_tile_scales_cvat_masks_and_rejects_mismatched_ones(tmp_path):
    image = np.zeros((100, 150, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "img.jpg"), image)
    mask = np.zeros((40, 60), np.uint8)  # CVAT resolution: same aspect as the image
    mask[:, 30:] = 1
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    cv2.imwrite(str(tmp_path / "portrait.png"), np.zeros((60, 40), np.uint8))
    for split in ("train",):
        for sub in ("images", "masks"):
            (tmp_path / "tiles" / split / sub).mkdir(parents=True)
    row = {"source": "field", "image_id": "img", "split": "train", "image_path": str(tmp_path / "img.jpg")}
    args = (str(tmp_path / "tiles"), 1.0, 150, 0, 1.0, 0)

    result = tile.tile_image({**row, "mask_path": str(tmp_path / "mask.png")}, *args)
    assert result["tiles"] == 1
    tile_mask = cv2.imread(str(next((tmp_path / "tiles/train/masks").glob("*.png"))), cv2.IMREAD_GRAYSCALE)
    assert tile_mask.shape == (150, 150)  # padded from 150x100
    assert tile_mask[:100, :75].max() == 0 and tile_mask[:100, 75:].min() == 1

    result = tile.tile_image({**row, "mask_path": str(tmp_path / "portrait.png")}, *args)
    assert "aspect ratio" in result["error"]
