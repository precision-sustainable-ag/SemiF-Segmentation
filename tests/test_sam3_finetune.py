"""mode=sam3_finetune without a GPU or SAM 3 weights: the box dataset from a
labeled box round, the trainer config, freezing and checkpoint export, the
evaluation metrics, and predict_tiled's box_from."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from src import sam3_finetune  # noqa: E402
from src.finetune import box_dataset, evaluate, sam3_model, sam3_trainer  # noqa: E402
from src.labeling.ledger import LabelLedger  # noqa: E402
from src.models.proposals.base import ProposalGenerator, model_box  # noqa: E402
from src.models.proposals.structures import InstanceProposal, ProposalResult  # noqa: E402

PROMPTS = {"plant": "plant", "colorchecker": "color chart"}
DCFG = {"val_fraction": 0.34, "seed": 0, "max_side": None, "scale": 0.5, "tile_size": 40, "overlap": 10, "min_visible": 0.5,
        "min_box_px": 2, "jpeg_quality": 95}


def labeled_round(root: Path, labels_db: Path, n: int = 3) -> None:
    """n labeled 160x120 images, each with a plant [20, 20, 60, 50] and a colorchecker
    [100, 60, 150, 110] (full resolution), as pull_cvat leaves a box round."""
    with LabelLedger(labels_db) as ledger:
        ledger.ensure_round("r_boxes", "field")
        ledger.add_items("r_boxes", "field", [{"image_id": f"IMG{k}", "path": f"IMG{k}.jpg"} for k in range(n)])
        for k in range(n):
            image = np.full((120, 160, 3), 60, np.uint8)
            native = root / f"IMG{k}.jpg"
            cv2.imwrite(str(native), image)
            boxes = root / f"IMG{k}.json"
            boxes.write_text(json.dumps({"image_id": f"IMG{k}", "image_size": [120, 160], "instances": [
                {"id": 1, "bbox": [20, 20, 60, 50], "label": "plant", "class_id": 1, "source": "sam3"},
                {"id": 2, "bbox": [100, 60, 150, 110], "label": "colorchecker", "class_id": 2, "source": "human"},
            ]}))
            ledger.update("field", f"IMG{k}", status="labeled", native_path=str(native), boxes_path=str(boxes))
        ledger.add_items("r_boxes", "field", [{"image_id": "WAITING", "path": "WAITING.jpg"}])  # still in CVAT


def test_box_dataset_tiles_like_the_prelabels(tmp_path):
    labeled_round(tmp_path, tmp_path / "labels.db")
    images = box_dataset.labeled_images(tmp_path / "labels.db", [])
    assert sorted(i["image_id"] for i in images) == ["IMG0", "IMG1", "IMG2"]

    out = tmp_path / "dataset"
    summary = box_dataset.build(images, out, PROMPTS, DCFG)
    split = json.loads((out / "split.json").read_text())
    assert sorted(split.values()) == ["train", "train", "valid"]
    train = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert train["categories"] == [{"id": 1, "name": "plant"}, {"id": 2, "name": "color chart"}]
    # 160x120 at scale 0.5 is 80x60: tiles of 40 with overlap >= 10 start at x 0, 30, 40 and y 0, 20.
    assert len(train["images"]) == 2 * 6 and summary["train"]["tiles"] == 12
    first = next(e for e in train["images"] if e["tile_xy"] == [0, 0])
    assert (first["width"], first["height"]) == (40, 40) and (out / "train" / first["file_name"]).exists()
    anns = [a for a in train["annotations"] if a["image_id"] == first["id"]]
    # The plant (10, 10, 30, 25 at scale 0.5) is whole in this tile; the colorchecker (50, 30, 75, 55) isn't in it.
    assert [(a["category_id"], a["bbox"]) for a in anns] == [(1, [10.0, 10.0, 20.0, 15.0])]
    # In the tile at x=30, y=20 the colorchecker is cut at x=70 (80% visible) and kept, clipped;
    # the plant has 0% left, and in the tile at y=20, x=0 only 1/3 of it: dropped.
    tile = next(e for e in train["images"] if e["tile_xy"] == [30, 20] and e["source_image"] == first["source_image"])
    assert [(a["category_id"], a["bbox"]) for a in train["annotations"] if a["image_id"] == tile["id"]] == [
        (2, [20.0, 10.0, 20.0, 25.0])]
    low = next(e for e in train["images"] if e["tile_xy"] == [0, 20] and e["source_image"] == first["source_image"])
    assert not [a for a in train["annotations"] if a["image_id"] == low["id"]]
    assert summary["train"]["source:human"] == 2 and summary["train"]["empty_tiles"] > 0

    # Re-building keeps each image's split, and new images are topped up.
    labeled_round(tmp_path, tmp_path / "labels.db", n=6)
    box_dataset.build(box_dataset.labeled_images(tmp_path / "labels.db", []), out, PROMPTS, DCFG)
    again = json.loads((out / "split.json").read_text())
    assert {k: again[k] for k in split} == split and len(again) == 6
    assert sum(v == "valid" for v in again.values()) == 2

    with pytest.raises(ValueError, match="no sam3_finetune.prompts entry"):
        box_dataset.build(images, tmp_path / "d2", {"plant": "plant"}, DCFG)


def test_box_dataset_whole_images(tmp_path):
    """max_side: each image is one input, downscaled whole; no tiles, no clipping."""
    labeled_round(tmp_path, tmp_path / "labels.db")
    images = box_dataset.labeled_images(tmp_path / "labels.db", [])
    summary = box_dataset.build(images, tmp_path / "ds", PROMPTS, {**DCFG, "max_side": 80})
    train = json.loads((tmp_path / "ds/train/_annotations.coco.json").read_text())
    assert summary["train"]["tiles"] == summary["train"]["images"] == 2 and summary["train"]["empty_tiles"] == 0
    assert all((e["width"], e["height"], e["tile_xy"]) == (80, 60, [0, 0]) for e in train["images"])
    first = train["images"][0]["id"]
    assert sorted(a["bbox"] for a in train["annotations"] if a["image_id"] == first) == [
        [10.0, 10.0, 20.0, 15.0], [50.0, 30.0, 25.0, 25.0]]  # both boxes whole, at scale 0.5


def test_working_scale():
    from src.preprocessing.full_image_boxes import working_scale

    assert working_scale((6368, 9560), {"scale": 0.325, "max_side": None}) == 0.325
    assert working_scale((6368, 9560), {"scale": 0.325, "max_side": 1008}) == pytest.approx(1008 / 9560)
    assert working_scale((500, 600), {"scale": 0.325, "max_side": 1008}) == 1.0  # never upscaled


def test_box_dataset_from_vegetation_masks(tmp_path):
    """A semantic round: plant boxes from the blobs of each pulled mask, plant only."""
    with LabelLedger(tmp_path / "labels.db") as ledger:
        ledger.ensure_round("r_masks", "semif")
        ledger.add_items("r_masks", "semif", [{"image_id": "M0", "path": "M0.jpg"}, {"image_id": "M1", "path": "M1.jpg"}])
        mask = np.zeros((120, 160), np.uint8)
        mask[20:60, 20:60] = 1      # a plant ...
        mask[62:66, 40:44] = 1      # ... and a leaf fragment 2 px below it (1 px at scale 0.5): joined
        mask[90:110, 100:150] = 1   # another plant
        mask[5:7, 150:152] = 1      # a speck: dropped
        for k in range(2):
            cv2.imwrite(str(tmp_path / f"M{k}.jpg"), np.full((120, 160, 3), 60, np.uint8))
            cv2.imwrite(str(tmp_path / f"M{k}.png"), mask)
            ledger.update("semif", f"M{k}", status="labeled", native_path=str(tmp_path / f"M{k}.jpg"),
                          mask_path=str(tmp_path / f"M{k}.png"))
    mcfg = {"label": "plant", "merge_px": 4, "min_area_px": 40}  # full resolution: 2 px and 10 px at scale 0.5
    assert box_dataset.mask_instances(tmp_path / "M0.png", (80, 60), mcfg) == [
        {"bbox": [10.0, 10.0, 30.0, 33.0], "label": "plant", "source": "mask_blob"},
        {"bbox": [50.0, 45.0, 75.0, 55.0], "label": "plant", "source": "mask_blob"}]
    # A leaf cut off inside the plant's box joins it rather than becoming its own box.
    nested = np.zeros((60, 80), np.uint8)
    nested[10:40, 10:40] = 1
    nested[20:30, 20:30] = 0
    nested[23:27, 23:27] = 1
    cv2.imwrite(str(tmp_path / "nested.png"), nested)
    apart = {**mcfg, "merge_px": 0, "min_area_px": 10}  # (mask at full size) 3 px gap: not joined by merge_px, only by containment
    assert [b["bbox"] for b in box_dataset.mask_instances(tmp_path / "nested.png", (80, 60), apart)] == [
        [10.0, 10.0, 40.0, 40.0]]
    assert len(box_dataset.mask_instances(tmp_path / "nested.png", (80, 60), {**apart, "absorb_contained": 0})) == 2
    images = box_dataset.labeled_images(tmp_path / "labels.db", [], "masks")
    assert [i["image_id"] for i in images] == ["M0", "M1"] and "mask_path" in images[0]
    assert box_dataset.labeled_images(tmp_path / "labels.db", [], "boxes") == []
    summary = box_dataset.build(images, tmp_path / "ds", PROMPTS, {**DCFG, "tile_size": 80, "val_fraction": 0.5}, mcfg)
    assert summary["categories"] == [{"id": 1, "name": "plant"}]  # no colorchecker: masks don't label it
    assert summary["train"]["boxes:plant"] == 2 and summary["train"]["source:mask_blob"] == 2
    with pytest.raises(ValueError, match="sam3_finetune.labels"):
        box_dataset.labeled_images(tmp_path / "labels.db", [], "polygons")


def test_trainer_config_is_box_only_and_freezes():
    tcfg = OmegaConf.to_container(OmegaConf.load("conf/sam3_finetune/default.yaml").train)
    config = sam3_trainer.trainer_config("/data/ds", "/runs/v1/train", "/ckpt/sam3.pt", tcfg)
    assert config["mode"] == "train_only" and config["max_epochs"] == 20
    assert config["model"] == {"_target_": "src.finetune.sam3_model.build_trainable_model",
                               "checkpoint_path": "/ckpt/sam3.pt", "freeze": ["backbone.*"],
                               "enable_segmentation": False}
    dataset = config["data"]["train"]["dataset"]
    assert dataset["ann_file"] == "/data/ds/train/_annotations.coco.json" and not dataset["load_segmentation"]
    losses = [f["_target_"].rsplit(".", 1)[1] for f in config["loss"]["all"]["loss_fns_find"]]
    assert losses == ["Boxes", "IABCEMdetr"]  # no Masks loss
    assert config["checkpoint"]["skip_saving_parameters"] == ["backbone.*"]
    assert config["optim"]["options"]["lr"][0]["scheduler"]["base_lr"] == pytest.approx(8e-5)


def test_freeze_patterns():
    names = ["backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight", "backbone.language_backbone.encoder.w",
             "transformer.decoder.layers.0.w", "segmentation_head.w"]
    assert sam3_model.frozen_names(names, ["backbone.*"]) == set(names[:2])
    assert sam3_model.frozen_names(names, ["backbone.vision_backbone.*", "transformer.encoder.*"]) == {names[0]}


def test_export_is_a_delta_on_the_base(tmp_path):
    # The trainer's checkpoint: no "detector." prefix, frozen backbone left out.
    torch.save({"model": {"transformer.w": torch.full((2, 2), 3.0)}, "epoch": 3}, tmp_path / "checkpoint.pt")
    counts = sam3_model.export_checkpoint("sam3", tmp_path / "checkpoint.pt", tmp_path / "out.pt")
    assert counts == {"tensors": 1, "base": "sam3"}
    delta = sam3_model.read_delta(tmp_path / "out.pt")
    assert delta["base"] == "sam3" and (delta["detector"]["transformer.w"] == 3).all()
    torch.save({"detector.w": torch.ones(1)}, tmp_path / "sam3.pt")  # a full sam3.pt isn't a delta
    assert sam3_model.read_delta(tmp_path / "sam3.pt") is None

    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.w = torch.nn.Parameter(torch.zeros(2, 2))
    model.backbone = torch.nn.Linear(2, 2)
    before = model.backbone.weight.detach().clone()
    sam3_model.apply_delta(model, delta)
    assert (model.transformer.w == 3).all() and torch.equal(model.backbone.weight, before)
    with pytest.raises(ValueError, match="aren't in the model"):
        sam3_model.apply_delta(model, {"detector": {"transformer.extra": torch.zeros(1)}})
    with pytest.raises(ValueError, match="shape"):
        sam3_model.apply_delta(model, {"detector": {"transformer.w": torch.zeros(3)}})


def test_presence_loss_has_finite_gradients_when_confident():
    """sam3's Triton focal loss returns NaN gradients for gamma=0 once p_t == 1;
    the patched loss doesn't (needs CUDA for Triton)."""
    if not torch.cuda.is_available():
        pytest.skip("Triton kernels need CUDA")
    sam3_trainer._patch_focal_loss()
    from sam3.train.loss import loss_fns

    logits = torch.tensor([[40.0, -40.0, 0.3]], device="cuda", requires_grad=True)  # saturated, as when confident
    targets = torch.tensor([[1.0, 0.0, 1.0]], device="cuda")
    loss_fns.sigmoid_focal_loss(logits, targets, num_boxes=1, alpha=0.5, gamma=0.0).backward()
    assert torch.isfinite(logits.grad).all()


def test_metrics():
    coco = {"images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "plant"}, {"id": 2, "name": "color chart"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
                            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0}]}
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in coco["annotations"]]
    assert evaluate.coco_ap(coco, perfect)["AP"] == pytest.approx(1.0)
    pr = evaluate.precision_recall(coco, perfect + [{"image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 10],
                                                      "score": 0.8}], 0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (2, 1, 0) and pr["precision"] == pytest.approx(2 / 3)
    assert evaluate.precision_recall(coco, perfect, 0.95)["recall"] == 0.0  # below the operating threshold
    assert evaluate.coco_ap(coco, [])["AP"] == 0.0


def test_previews(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    cv2.imwrite(str(valid / "a.jpg"), np.full((60, 80, 3), 90, np.uint8))
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 80, "height": 60}],
            "categories": [{"id": 1, "name": "plant"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
    dets = {"zero_shot": [{"image_id": 1, "category_id": 1, "bbox": [12, 10, 20, 20], "score": 0.9},
                          {"image_id": 1, "category_id": 1, "bbox": [50, 30, 9, 9], "score": 0.2}],
            "finetuned": []}
    assert evaluate.save_previews(tmp_path / "previews", valid, coco, dets, 0.5, 8, panel_width=200) == 1
    preview = cv2.imread(str(tmp_path / "previews/a.jpg"))
    assert preview.shape[1] == 3 * 200  # ground truth | zero_shot | finetuned


class BoxGenerator(ProposalGenerator):
    """One proposal per call whose predicted box is wider than its mask."""

    def predict_raw(self, image, semantic_mask=None, **kwargs):
        h, w = np.asarray(image).shape[:2]
        mask = torch.zeros((h, w), dtype=torch.bool)
        mask[10:20, 10:20] = True
        return ProposalResult([InstanceProposal(torch.tensor([5.0, 10.0, 25.5, 20.0]), mask, 0.9, "plant")], (h, w))


def test_predict_tiled_box_from():
    image = np.zeros((40, 40, 3), np.uint8)
    gen = BoxGenerator(device="cpu")
    assert gen.predict_tiled(image, tile_size=64).boxes.tolist() == [[10, 10, 20, 20]]
    inst = gen.predict_tiled(image, tile_size=64, box_from="model")
    assert inst.boxes.tolist() == [[5, 10, 26, 20]] and inst.masks[0].shape == (10, 21)
    assert model_box([-3, 2, 50, 2.0], 40, 40) is None and model_box([-3, 2, 50, 9], 40, 40) == (0, 2, 40, 9)
    with pytest.raises(ValueError, match="box_from"):
        gen.predict_tiled(image, box_from="both")


def test_mode_runs_the_dataset_task(tmp_path, make_cfg):
    cfg = make_cfg("mode=sam3_finetune", "project.name=boxes", "sam3_finetune.name=t1",
                   "sam3_finetune.tasks.train=false", "sam3_finetune.tasks.export=false",
                   "sam3_finetune.tasks.evaluate=false", "proposals.full_image.scale=0.5",
                   "proposals.tiling.tile_size=40", "proposals.tiling.overlap=10")
    labeled_round(tmp_path, Path(cfg.paths.labels_db))
    sam3_finetune.main(cfg)
    run = Path(cfg.paths.sam3_finetune_dir)
    assert run == tmp_path / "outputs/runs/boxes/sam3_finetune/t1"
    assert (run / "dataset/train/_annotations.coco.json").exists() and (run / "cfg.yaml").exists()
    assert json.loads((run / "summary.json").read_text())["dataset"]["train"]["images"] >= 1


class CheckpointQualityGenerator:
    """Finds the one plant exactly with checkpoints named *_e2*, offset elsewhere."""

    def __init__(self, checkpoint):
        self.good = "_e2" in str(checkpoint)
        self.prompts = []

    def predict(self, image):
        box = torch.tensor([10.0, 10.0, 30.0, 30.0]) if self.good else torch.tensor([14.0, 14.0, 34.0, 34.0])
        h, w = image.shape[:2]
        return ProposalResult([InstanceProposal(box, torch.zeros((h, w), dtype=torch.bool), 0.9, "plant")], (h, w))


def test_snapshots_are_exported_evaluated_and_the_best_linked(tmp_path, make_cfg, monkeypatch):
    cfg = make_cfg("mode=sam3_finetune", "project.name=boxes", "sam3_finetune.name=t2",
                   "sam3_finetune.tasks.dataset=false", "sam3_finetune.tasks.train=false")
    run = Path(cfg.paths.sam3_finetune_dir)
    (run / "train/checkpoints").mkdir(parents=True)
    for name in ("checkpoint.pt", "checkpoint_1.pt", "checkpoint_2.pt"):
        torch.save({"model": {"transformer.w": torch.zeros(2)}, "optimizer": {}}, run / "train/checkpoints" / name)
    valid = run / "dataset/valid"
    valid.mkdir(parents=True)
    cv2.imwrite(str(valid / "a.jpg"), np.full((60, 80, 3), 90, np.uint8))
    (valid / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 80, "height": 60}],
        "categories": [{"id": 1, "name": "plant"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}))
    import src.models

    monkeypatch.setattr(src.models, "build_proposal_generator", lambda name, c: CheckpointQualityGenerator(c["checkpoint"]))
    sam3_finetune.main(cfg)

    assert sorted(p.name for p in run.glob("sam3_finetuned*.pt")) == [
        "sam3_finetuned.pt", "sam3_finetuned_best.pt", "sam3_finetuned_e1.pt", "sam3_finetuned_e2.pt"]
    assert not list((run / "train/checkpoints").glob("checkpoint_*.pt"))  # snapshots replaced by their exports
    results = json.loads((run / "eval.json").read_text())
    assert list(results["checkpoints"]) == ["zero_shot", "finetuned_e1", "finetuned_e2", "finetuned"]
    assert results["best"]["name"] == "finetuned_e2" and results["checkpoints"]["finetuned_e2"]["AP"] == pytest.approx(1.0)
    assert (run / "sam3_finetuned_best.pt").stat().st_ino == (run / "sam3_finetuned_e2.pt").stat().st_ino
    preview = cv2.imread(str(run / "previews/a.jpg"))
    assert preview.shape[1] == 3 * 900  # ground truth | zero-shot | best only
