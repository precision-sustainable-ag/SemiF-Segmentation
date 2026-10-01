import cv2
import numpy as np
import pytest

from src.labeling.cvat_masks import component_shapes, decode_mask, encode_mask, mask_shape, rasterize, resize_mask


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


def test_component_shapes_one_shape_per_blob():
    mask = np.zeros((40, 60), bool)
    mask[2:12, 3:20] = True       # 170 px
    mask[20:38, 40:58] = True     # 324 px
    mask[30, 5] = True            # 1 px speck
    shapes = component_shapes(mask, frame=4, label_id=7, min_area_px=10)
    assert len(shapes) == 2 and all(s["frame"] == 4 and s["label_id"] == 7 for s in shapes)
    assert shapes[0]["points"][-4:] == [40, 20, 57, 37]  # largest first, box in image coordinates
    rebuilt, skipped = rasterize(shapes, 40, 60)
    expected = mask.copy()
    expected[30, 5] = False
    assert not skipped and np.array_equal(rebuilt, expected)
    assert len(component_shapes(mask, 0, 7)) == 3


def test_resize_mask_upscales_smoothly():
    def disk(h, w, scale):
        yy, xx = np.mgrid[:h, :w]
        return ((yy + 0.5) / scale - 20) ** 2 + ((xx + 0.5) / scale - 30) ** 2 <= 15 ** 2

    mask = disk(40, 60, 1)
    big = resize_mask(mask, (150, 100))  # 2.5x
    assert big.shape == (100, 150) and big.dtype == bool
    # Closer to the disk drawn directly at 2.5x than the nearest-neighbour staircase is.
    truth = disk(100, 150, 2.5)
    nearest = cv2.resize(mask.astype(np.uint8), (150, 100), interpolation=cv2.INTER_NEAREST).astype(bool)
    assert np.count_nonzero(big != truth) < np.count_nonzero(nearest != truth)
    assert np.array_equal(resize_mask(mask, (60, 40)), mask)
