"""Detail metrics, instance targets from semantic masks, and the Lightning
module that trains the instance-segmentation models."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("segmentation_models_pytorch")

import numpy as np  # noqa: E402

from src.utils.seg_metrics import DetailMetricsConfig, detail_metrics  # noqa: E402


# --- detail metrics -------------------------------------------------------

def _square(size=64, lo=16, hi=48):
    m = torch.zeros(1, size, size, dtype=torch.long)
    m[0, lo:hi, lo:hi] = 1
    return m


def test_detail_metrics_perfect_and_empty():
    m = _square()
    perfect = detail_metrics(m, m)
    assert all(v.item() == pytest.approx(1.0) for k, v in perfect.items() if k != "thin_recall")
    assert torch.isnan(perfect["thin_recall"]).all()  # no thin structure in a 32 px square

    empty = torch.zeros_like(m)
    both_empty = detail_metrics(empty, empty)
    assert both_empty["iou"].item() == 1 and both_empty["boundary_f1"].item() == 1
    missed = detail_metrics(empty, m)
    assert missed["dice"].item() == 0 and missed["boundary_iou"].item() == 0 and missed["boundary_f1"].item() == 0


def test_boundary_metrics_are_sensitive_to_contours():
    gt, shifted = _square(), _square(lo=17, hi=49)  # 1 px off
    out = detail_metrics(shifted, gt, DetailMetricsConfig(boundary_width=2, boundary_tolerance=2))
    assert out["boundary_f1"].item() == pytest.approx(1.0)  # within tolerance
    assert out["boundary_iou"].item() < out["iou"].item() < 1


def test_thin_recall_catches_missed_stems():
    gt = _square(128, 20, 100)
    gt[0, 0:20, 60:62] = 1  # a 2 px stem
    pred = _square(128, 20, 100)
    out = detail_metrics(pred, gt, DetailMetricsConfig(thin_width=7))
    assert out["iou"].item() > 0.99
    assert out["thin_recall"].item() < 0.1
    assert detail_metrics(gt, gt)["thin_recall"].item() == 1


def test_image_edges_are_not_boundaries():
    full = torch.ones(1, 32, 32, dtype=torch.long)
    out = detail_metrics(full, full)
    assert out["boundary_iou"].item() == 1


# --- instance targets -----------------------------------------------------

def test_masks_to_instances():
    from src.utils.datasets import masks_to_instances

    mask = np.zeros((40, 50), np.uint8)
    mask[5:15, 5:25] = 1    # 200 px
    mask[20:30, 30:35] = 2  # 50 px, class 2
    mask[35:37, 0:2] = 1    # 4 px
    t = masks_to_instances(mask, min_area=10)
    assert t["labels"].tolist() == [1, 2]
    assert t["boxes"].tolist() == [[5, 5, 25, 15], [30, 20, 35, 30]]
    assert t["masks"].shape == (2, 40, 50) and t["masks"].dtype == torch.uint8
    assert t["masks"].sum(dim=(1, 2)).tolist() == [200, 50]
    assert masks_to_instances(mask, max_instances=1)["labels"].tolist() == [1]
    empty = masks_to_instances(np.zeros((8, 8), np.uint8))
    assert empty["boxes"].shape == (0, 4) and empty["masks"].shape == (0, 8, 8)


def _write_tiles(root, n=4, size=96):
    import cv2

    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for i in range(n):
            image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size), np.uint8)
            if i % 2 == 0:
                mask[10:50, 20:70] = 1
                image[10:50, 20:70] = (40, 180, 40)
            cv2.imwrite(str(root / split / "images" / f"t{i}.jpg"), image)
            cv2.imwrite(str(root / split / "masks" / f"t{i}.png"), mask)


def test_instance_dataset(tmp_path):
    from src.utils.datasets import InstanceDataset, collate_instances

    _write_tiles(tmp_path, n=2)
    ds = InstanceDataset(tmp_path / "train/images", tmp_path / "train/masks", min_area=10)
    items = sorted((ds[i] for i in range(len(ds))), key=lambda item: item[2])
    image, target, stem = items[0]
    assert stem == "t0" and image.shape == (3, 96, 96)
    assert target["labels"].tolist() == [1] and target["semantic_mask"].shape == (96, 96)
    images, targets, stems = collate_instances(items)
    assert len(images) == len(targets) == 2 and targets[1]["boxes"].shape == (0, 4)


def test_instances_to_semantic_paints_higher_scores_on_top():
    from src.utils.instance_model import instances_to_semantic

    masks = torch.zeros(3, 1, 4, 4)
    masks[0, 0, :2] = 0.9
    masks[1, 0, 1:3] = 0.9
    masks[2, 0, 3] = 0.9
    out = {"scores": torch.tensor([0.9, 0.6, 0.3]), "labels": torch.tensor([1, 2, 1]), "masks": masks}
    sem = instances_to_semantic(out, (4, 4), score_threshold=0.5)
    assert sem[:, 0].tolist() == [1, 1, 2, 0]


# --- Lightning module -----------------------------------------------------

@pytest.mark.parametrize("model_name", ["maskrcnn", "maskrcnn_pointrend"])
def test_instance_module_fits_and_validates(make_cfg, tmp_path, model_name):
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader

    from src.train import make_dataset
    from src.utils.augs import val_aug
    from src.utils.datasets import collate_instances
    from src.utils.instance_model import InstanceSegmentationModule

    _write_tiles(tmp_path / "tiles")
    cfg = make_cfg("mode=train", f"model={model_name}", "model.pretrained=null",
                   "model.optim.warmup_steps=1", "model.instances.min_area=10")
    loaders = [
        DataLoader(make_dataset(cfg, tmp_path / f"tiles/{s}/images", tmp_path / f"tiles/{s}/masks",
                                val_aug(cfg), None, None), batch_size=2, collate_fn=collate_instances)
        for s in ("train", "val")
    ]
    module = InstanceSegmentationModule(cfg)
    trainer = Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    module.plot_train_val_metrics = lambda: None  # needs a CSV logger
    trainer.fit(module, *loaders)
    assert "train_loss" in trainer.logged_metrics
    result = trainer.validate(module, loaders[1], verbose=False)[0]
    for key in ("valid_dataset_iou", "valid_per_image_iou", "valid_object_iou", "valid_dice",
                "valid_boundary_iou", "valid_boundary_f1", "valid_thin_recall",
                "valid_bbox_map", "valid_bbox_map_50", "valid_segm_map", "valid_segm_map_75",
                "valid_count_mae", "valid_count_ratio"):
        assert key in result

    pred = module.eval().predict_semantic(torch.rand(2, 3, 96, 96))
    assert pred.shape == (2, 96, 96) and pred.dtype == torch.long


def test_instance_plot(tmp_path):
    import cv2

    from src.utils.viz_utils import instance_plot

    image = np.zeros((50, 70, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "img.jpg"), image)
    gt_masks = torch.zeros(1, 50, 70, dtype=torch.uint8)
    gt_masks[0, 10:30, 10:40] = 1
    gt = {"boxes": torch.tensor([[10.0, 10, 40, 30]]), "masks": gt_masks}
    masks = torch.zeros(2, 1, 64, 80)  # prediction padded to a multiple of 16
    masks[0, 0, 10:30, 10:40] = 0.9
    masks[1, 0, 40:45, 5:9] = 0.9
    pred = {"boxes": torch.tensor([[10.0, 10, 40, 30], [5, 40, 9, 45]]), "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.95, 0.3]), "masks": masks}
    instance_plot(tmp_path / "img.jpg", None, pred, tmp_path / "unlabeled.png")
    assert (tmp_path / "unlabeled.png").exists()
    pred["masks"] = masks[:, :, :50, :70]
    instance_plot(tmp_path / "img.jpg", gt, pred, tmp_path / "labeled.png", score_threshold=0.1)
    assert (tmp_path / "labeled.png").exists()
