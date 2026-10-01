"""Stitching tile predictions into full-image instances, and full-image COCO AP."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

import numpy as np  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

from src.inferencing.tiled_instances import (  # noqa: E402
    coco_ap,
    crop_to_rle,
    instances_from_mask,
    paint_semantic,
    predict_tiled_instances,
    tile_starts,
)
from src.utils.datasets import masks_to_instances  # noqa: E402


def test_tile_starts_cover_with_overlap():
    assert tile_starts(100, 128, 32) == [0]
    starts = tile_starts(300, 100, 30)
    assert starts[0] == 0 and starts[-1] == 200
    assert all(b - a <= 70 for a, b in zip(starts, starts[1:]))  # neighbours share >= 30 px
    with pytest.raises(ValueError):
        tile_starts(300, 100, 100)


class PerfectTileDetector(torch.nn.Module):
    """Stands in for a detector: returns each tile's connected blobs of image
    channel 0 (the test puts the label mask there), cut at the tile edge."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, crops):
        outputs = []
        for crop in crops:
            t = masks_to_instances((crop[0] > 0.5).numpy().astype(np.uint8))
            n = len(t["labels"])
            outputs.append({"boxes": t["boxes"], "labels": t["labels"], "scores": torch.linspace(0.9, 0.8, n),
                            "masks": t["masks"][:, None].float()})
        return outputs


def _scene():
    mask = np.zeros((300, 420), np.uint8)
    mask[20:60, 30:70] = 1      # inside one tile
    mask[120:170, 80:140] = 1   # crosses the seam at x = 100..128 (tiles of 128, overlap 40)
    mask[200:215, 10:400] = 1   # long stem spanning several tiles
    mask[100:115, 300:315] = 1  # inside an overlap: seen whole by two tiles
    return mask


def test_stitched_tiles_reproduce_full_image_instances():
    mask = _scene()
    image = torch.from_numpy(mask).float()[None].expand(3, -1, -1)
    dt = predict_tiled_instances(PerfectTileDetector(), image, tile_size=128, overlap=40)
    gt = instances_from_mask(mask)
    assert len(dt) == len(gt) == 4
    order = lambda inst: np.lexsort((inst.boxes[:, 1], inst.boxes[:, 0]))  # noqa: E731
    for i, j in zip(order(dt), order(gt)):
        assert dt.boxes[i].tolist() == gt.boxes[j].tolist()
        assert np.array_equal(dt.masks[i], gt.masks[j])
    assert np.array_equal(paint_semantic(dt), mask)
    ap = coco_ap([gt], [dt])
    assert ap["segm_map"] == pytest.approx(1.0) and ap["bbox_map"] == pytest.approx(1.0)


def test_without_merging_seams_split_instances():
    mask = _scene()
    image = torch.from_numpy(mask).float()[None].expand(3, -1, -1)
    unmerged = predict_tiled_instances(PerfectTileDetector(), image, tile_size=128, overlap=40, merge_iou=1.01)
    assert len(unmerged) > 4


def test_crop_to_rle_matches_full_mask_encoding():
    rng = np.random.default_rng(0)
    full = np.zeros((37, 53), bool)
    full[5:20, 10:40] = rng.random((15, 30)) > 0.4
    rle = crop_to_rle(full[5:20, 10:40], (10, 5, 40, 20), 37, 53)
    assert np.array_equal(mask_utils.decode(rle).astype(bool), full)
    expected = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
    assert rle["counts"] == expected["counts"]
    empty = crop_to_rle(np.zeros((3, 3), bool), (0, 0, 3, 3), 10, 10)
    assert not mask_utils.decode(empty).any()


def test_coco_ap_penalizes_split_instances():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1
    gt = instances_from_mask(mask)
    half = np.zeros_like(mask)
    half[10:90, 10:50] = 1
    half[10:90, 51:90] = 1  # predicted as two pieces
    dt = instances_from_mask(half)
    ap = coco_ap([gt], [dt])
    assert ap["segm_map"] < 0.5


def test_low_score_detections_do_not_chain_merges():
    """A plant piece and a low-score fragment agreeing across a seam stay separate,
    so the fragment can't glue itself (and whatever it touches) to the plant."""
    from src.inferencing.tiled_instances import Instances, merge_across_tiles

    inst = Instances(100, 200)
    piece = np.ones((20, 40), bool)
    inst.boxes = np.array([[80, 10, 120, 30], [80, 10, 120, 30]])
    inst.labels = np.array([1, 1])
    inst.masks = [piece, piece.copy()]
    tile_ids, rects = np.array([0, 1]), [(0, 0, 120, 100), (80, 0, 200, 100)]
    inst.scores = np.array([0.9, 0.2], np.float32)
    assert len(merge_across_tiles(inst, tile_ids, rects, min_score=0.5)) == 2
    inst.scores = np.array([0.9, 0.8], np.float32)
    assert len(merge_across_tiles(inst, tile_ids, rects, min_score=0.5)) == 1
