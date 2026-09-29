import numpy as np
import pytest

from src.labeling.cvat_masks import decode_mask, encode_mask, mask_shape, rasterize


def test_encode_matches_cvat_format():
    bitmap = np.zeros((4, 6), dtype=bool)
    bitmap[1, 2:4] = True
    bitmap[2, 3:5] = True
    # Tight box x 2..4, y 1..2 -> rows [1 1 0], [0 1 1] -> runs: 0 (no leading
    # background), 2 fg, 2 bg, 2 fg; then the inclusive box.
    assert encode_mask(bitmap) == [0, 2, 2, 2, 2, 1, 4, 2]


def test_encode_leading_background_run():
    bitmap = np.zeros((3, 3), dtype=bool)
    bitmap[0, 2] = True
    bitmap[2, 0] = True
    # Box is the whole image; flat = [0 0 1 0 0 0 1 0 0]
    assert encode_mask(bitmap) == [2, 1, 3, 1, 2, 0, 0, 2, 2]


def test_encode_empty_returns_none():
    assert encode_mask(np.zeros((5, 5), dtype=bool)) is None
    assert mask_shape(np.zeros((5, 5), dtype=bool), frame=0, label_id=1) is None


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random_masks(seed):
    rng = np.random.default_rng(seed)
    bitmap = rng.random((37, 53)) > 0.6
    points = encode_mask(bitmap)
    assert np.array_equal(decode_mask(points, 37, 53), bitmap)


def test_decode_clips_box_to_image():
    bitmap = np.ones((4, 4), dtype=bool)
    points = encode_mask(bitmap)
    # Same run lengths shifted two pixels past the right/bottom edge.
    shifted = points[:-4] + [2, 2, 5, 5]
    out = decode_mask(shifted, 4, 4)
    assert out[2:, 2:].all() and out.sum() == 4


def test_rasterize_mask_polygon_rectangle():
    shapes = [
        mask_shape(np.pad(np.ones((2, 2), bool), ((1, 7), (1, 7))), frame=0, label_id=1),
        {"type": "polygon", "points": [6, 6, 9, 6, 9, 9, 6, 9]},
        {"type": "rectangle", "points": [0, 8, 2, 10], "rotation": 0},
        {"type": "points", "points": [1, 1]},
    ]
    mask, skipped = rasterize(shapes, 10, 10)
    assert mask[1:3, 1:3].all()
    assert mask[6:10, 6:10].all()
    assert mask[8:10, 0:2].all()
    assert skipped == {"points": 1}
