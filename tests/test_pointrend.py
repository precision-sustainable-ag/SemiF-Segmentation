"""PointRend building blocks (src/models/pointrend) and the torchvision
detectors built from them (src/models)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn.functional as F  # noqa: E402

from src.models import MODEL_REGISTRY, build_model  # noqa: E402
from src.models.pointrend import CoarseMaskHead, PointHead, PointRendMaskHead, PointRendRoIHeads  # noqa: E402
from src.models.pointrend.losses import point_targets  # noqa: E402
from src.models.pointrend.sampling import (  # noqa: E402
    box_coords_to_image_coords,
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sample_fine_grained_features,
    sample_masks_at_points,
)
from src.models.pointrend.uncertainty import calculate_uncertainty, select_class_logits  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- sampling -------------------------------------------------------------

def test_point_sample_reads_pixel_centers():
    x = torch.randn(2, 3, 5, 7)
    i, j = torch.tensor([0, 4, 2]), torch.tensor([6, 0, 3])
    coords = torch.stack([(j + 0.5) / 7, (i + 0.5) / 5], dim=1).expand(2, -1, -1)
    assert torch.allclose(point_sample(x, coords), x[:, :, i, j], atol=1e-6)


def test_point_sample_is_bilinear_and_zero_outside():
    x = torch.tensor([[[[0.0, 2.0]]]])  # 1 x 2 image, pixel centers at x = 0.25, 0.75
    out = point_sample(x, torch.tensor([[[0.5, 0.5], [2.0, 0.5]]]))
    assert torch.allclose(out, torch.tensor([[[1.0, 0.0]]]))


def test_regular_grid_reproduces_the_grid():
    x = torch.randn(3, 2, 6, 6)
    coords = generate_regular_grid_point_coords(3, 6)
    assert coords.shape == (3, 36, 2)
    assert torch.allclose(point_sample(x, coords).view(3, 2, 6, 6), x, atol=1e-6)


def test_uncertain_points_on_grid_picks_top_cells():
    u = torch.full((2, 1, 4, 4), -1.0)
    u[0, 0, 1, 2] = 0.0   # most uncertain
    u[0, 0, 3, 0] = -0.5  # second
    u[1, 0, 0, 3] = 0.0
    u[1, 0, 2, 1] = -0.2
    idx, coords = get_uncertain_point_coords_on_grid(u, 2)
    assert idx.tolist() == [[1 * 4 + 2, 3 * 4 + 0], [0 * 4 + 3, 2 * 4 + 1]]
    assert torch.allclose(coords[0], torch.tensor([[2.5 / 4, 1.5 / 4], [0.5 / 4, 3.5 / 4]]))
    # never more points than cells
    assert get_uncertain_point_coords_on_grid(u, 100)[0].shape == (2, 16)


def test_uncertain_points_with_randomness_favors_the_boundary():
    # logits change sign at x = 0.5: the most uncertain points lie on that line
    xs = (torch.arange(28) + 0.5) / 28
    logits = (xs - 0.5).view(1, 1, 1, 28).expand(4, 1, 28, 28) * 20
    uncertainty = lambda l: calculate_uncertainty(l, None)  # noqa: E731
    coords = get_uncertain_point_coords_with_randomness(logits, uncertainty, 100, 3, 0.75)
    assert coords.shape == (4, 100, 2)
    assert ((coords >= 0) & (coords <= 1)).all()
    dist_uncertain = (coords[:, :75, 0] - 0.5).abs().mean()
    dist_random = (coords[:, 75:, 0] - 0.5).abs().mean()
    assert dist_uncertain < 0.1 < dist_random  # uniform: 0.25 on average

    only_random = get_uncertain_point_coords_with_randomness(logits, uncertainty, 50, 1, 0.0)
    assert only_random.shape == (4, 50, 2)
    with pytest.raises(ValueError):
        get_uncertain_point_coords_with_randomness(logits, uncertainty, 10, 0.5, 0.75)


def test_box_coords_to_image_coords():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    coords = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]]])
    assert box_coords_to_image_coords(boxes, coords).tolist() == [[[10, 20], [30, 60], [20, 30]]]


def test_sample_masks_at_points_matches_point_sample():
    masks = (torch.rand(3, 20, 30) > 0.5).to(torch.uint8)
    idx = torch.tensor([2, 0, 0, 1])
    coords = torch.rand(4, 50, 2) * torch.tensor([34.0, 24.0]) - 2  # some points outside the image
    expected = point_sample(masks[idx][:, None].float(), coords / torch.tensor([30.0, 20.0]))[:, 0]
    assert torch.allclose(sample_masks_at_points(masks, idx, coords), expected, atol=1e-5)


def test_fine_grained_features_follow_the_feature_stride():
    # stride-4 feature map whose value is the column index: at image x it reads x / 4 - 0.5
    feature = torch.arange(16.0).view(1, 1, 1, 16).expand(2, 2, 12, 16).contiguous()
    feature[1] += 100  # second image
    boxes = [torch.tensor([[8.0, 8.0, 40.0, 24.0]]), torch.tensor([[16.0, 0.0, 48.0, 16.0], [0.0, 0.0, 8.0, 8.0]])]
    coords = torch.tensor([[0.5, 0.5], [0.25, 1.0]]).expand(3, -1, -1)
    out = sample_fine_grained_features([feature], [(48, 64), (40, 60)], boxes, coords)
    assert out.shape == (3, 2, 2)
    abs_x = box_coords_to_image_coords(torch.cat(boxes), coords)[..., 0]
    expected = abs_x / 4 - 0.5 + torch.tensor([0.0, 100.0, 100.0])[:, None]
    assert torch.allclose(out[:, 0], expected, atol=1e-5)


# --- uncertainty ----------------------------------------------------------

def test_select_class_logits_and_uncertainty():
    logits = torch.randn(3, 4, 5)
    classes = torch.tensor([2, 0, 3])
    picked = select_class_logits(logits, classes)
    assert picked.shape == (3, 1, 5)
    assert torch.equal(picked[:, 0], logits[torch.arange(3), classes])
    agnostic = torch.randn(3, 1, 5)
    assert torch.equal(select_class_logits(agnostic, classes), agnostic)
    assert torch.equal(calculate_uncertainty(logits, classes), -picked.abs())


# --- heads ----------------------------------------------------------------

def test_coarse_head_shape():
    head = CoarseMaskHead(256, 14, num_mask_classes=3, out_resolution=7)
    assert head(torch.randn(5, 256, 14, 14)).shape == (5, 3, 7, 7)
    assert head(torch.randn(0, 256, 14, 14)).shape == (0, 3, 7, 7)
    with pytest.raises(ValueError):
        CoarseMaskHead(256, 13)


def test_point_head_shape():
    head = PointHead(256, num_mask_classes=2)
    assert head(torch.randn(4, 256, 30), torch.randn(4, 2, 30)).shape == (4, 2, 30)


def _features(batch=2, size=(64, 80)):
    h, w = size
    return {"0": torch.randn(batch, 256, h // 4, w // 4), "1": torch.randn(batch, 256, h // 8, w // 8)}


def test_mask_head_losses_and_empty_images():
    head = PointRendMaskHead(256, num_points=20)
    feats = _features()
    shapes = [(64, 80), (60, 70)]
    gt = [torch.zeros(1, 64, 80, dtype=torch.uint8), torch.zeros(0, 60, 70, dtype=torch.uint8)]
    gt[0][0, 10:40, 10:50] = 1
    proposals = [torch.tensor([[8.0, 8.0, 52.0, 42.0], [10.0, 10.0, 50.0, 40.0]]), torch.zeros(0, 4)]
    matched = [torch.tensor([0, 0]), torch.zeros(0, dtype=torch.long)]
    losses = head.losses(feats, proposals, shapes, gt, matched, None)
    assert set(losses) == {"loss_mask", "loss_mask_point"}
    assert all(torch.isfinite(v) and v > 0 for v in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in head.parameters())

    head.zero_grad()
    none = head.losses(feats, [torch.zeros(0, 4)] * 2, shapes, [gt[1], gt[1]], [matched[1]] * 2, None)
    assert all(v.item() == 0 for v in none.values())
    sum(none.values()).backward()  # every parameter still in the graph (DDP)
    assert all(p.grad is not None for p in head.parameters())


def test_point_targets_read_full_resolution_masks():
    gt = torch.zeros(1, 40, 40, dtype=torch.uint8)
    gt[0, :, :20] = 1  # left half
    boxes = [torch.tensor([[0.0, 0.0, 40.0, 40.0]])]
    coords = torch.tensor([[[0.1, 0.5], [0.4, 0.2], [0.6, 0.5], [0.9, 0.9]]])
    assert point_targets([gt], boxes, [torch.tensor([0])], coords).tolist() == [[1, 1, 0, 0]]


def test_subdivision_refines_only_uncertain_cells():
    head = PointRendMaskHead(256, subdivision_steps=1, subdivision_num_points=10).eval()
    feats = _features()
    boxes = [torch.tensor([[4.0, 4.0, 40.0, 40.0]]), torch.tensor([[0.0, 0.0, 20.0, 30.0]])]
    shapes = [(64, 80), (64, 80)]
    refined = head.inference(feats, boxes, shapes, None)
    assert refined.shape == (2, 1, 14, 14) and head.output_resolution == 14
    coarse = head._coarse_logits(feats, boxes, shapes)
    upsampled = F.interpolate(coarse, scale_factor=2, mode="bilinear", align_corners=False)
    changed = (refined != upsampled).flatten(1).sum(dim=1)
    assert (changed <= 10).all() and (changed > 0).all()

    empty = head.inference(feats, [torch.zeros(0, 4)] * 2, shapes, None)
    assert empty.shape == (0, 1, 14, 14)


def test_subdivision_output_resolution_and_skip_steps():
    head = PointRendMaskHead(256, subdivision_steps=3, subdivision_num_points=10_000).eval()
    out = head.inference(_features(1), [torch.tensor([[0.0, 0.0, 30.0, 30.0]])], [(64, 80)], None)
    assert out.shape == (1, 1, 56, 56)


# --- full models ----------------------------------------------------------

def _batch():
    images = [torch.rand(3, 96, 128), torch.rand(3, 64, 96)]
    masks = torch.zeros(2, 96, 128, dtype=torch.uint8)
    masks[0, 10:60, 20:90] = 1
    masks[1, 70:94, 100:126] = 1
    targets = [
        {"boxes": torch.tensor([[20.0, 10, 90, 60], [100, 70, 126, 94]]), "labels": torch.tensor([1, 2]), "masks": masks},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
         "masks": torch.zeros(0, 64, 96, dtype=torch.uint8)},
    ]
    return images, targets


@pytest.mark.parametrize("name,kwargs", [
    ("maskrcnn", {}),
    ("maskrcnn_pointrend", {}),
    ("maskrcnn_pointrend", {"class_agnostic_mask": False, "subdivision_steps": 2}),
])
def test_models_train_and_predict(name, kwargs):
    model = build_model(name, num_classes=3, pretrained=None, **kwargs)
    images, targets = _batch()
    losses = model(images, targets)
    expected = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    if name == "maskrcnn_pointrend":
        expected.add("loss_mask_point")
        assert isinstance(model.roi_heads, PointRendRoIHeads)
        assert model.roi_heads.point_mask_head.class_agnostic == kwargs.get("class_agnostic_mask", True)
    assert set(losses) == expected
    sum(losses.values()).backward()
    assert not [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]

    model.eval()
    outputs = model(images)
    for out, image in zip(outputs, images):
        assert set(out) == {"boxes", "labels", "scores", "masks"}
        n = out["boxes"].shape[0]
        assert out["masks"].shape == (n, 1, *image.shape[-2:])  # pasted at the input size (no resize)
        assert ((out["masks"] >= 0) & (out["masks"] <= 1)).all()


def test_build_model_registry():
    assert set(MODEL_REGISTRY) >= {"maskrcnn", "maskrcnn_pointrend"}
    with pytest.raises(ValueError):
        build_model("nope", num_classes=2)
    with pytest.raises(ValueError):
        build_model("maskrcnn", num_classes=2, pretrained="ade20k")


def test_pointrend_shares_the_baseline_detector():
    """Everything but the mask branch has the same modules (and so loads the same weights)."""
    base = build_model("maskrcnn", num_classes=2, pretrained=None).state_dict()
    point = build_model("maskrcnn_pointrend", num_classes=2, pretrained=None).state_dict()
    shared = {k for k in base if ".mask_" not in k}
    assert shared <= set(point)
    assert all(base[k].shape == point[k].shape for k in shared)
