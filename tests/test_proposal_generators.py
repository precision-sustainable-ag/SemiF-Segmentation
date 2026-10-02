"""The SAM 3 generator with a fake processor, the instance files and
InstanceDataset, the pseudo_instances preprocess task, and Mask R-CNN /
PointRend on external proposals. No SAM checkpoint or package needed."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.models import build_model, build_proposal_generator  # noqa: E402
from src.models.external_proposals import predict_with_proposals  # noqa: E402
from src.models.proposals import SAM3ProposalGenerator  # noqa: E402
from src.models.proposals.base import to_pil  # noqa: E402
from src.utils.instance_annotations import (  # noqa: E402
    id_map_to_instances,
    load_instance_annotation,
    mask_box,
    save_instance_annotation,
)


class FakeSam3Processor:
    """Stands in for Sam3Processor: for any prompt, one proposal per connected
    blob of the image's red channel (> 127), scored scores[prompt] (default 0.9)."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.confidence_threshold = 0.5
        self.images = 0

    def set_image(self, image):
        self.images += 1
        return {"image": np.asarray(image)}

    def set_text_prompt(self, prompt, state):
        image = state["image"]
        n, comp = cv2.connectedComponents((image[..., 0] > 127).astype(np.uint8), connectivity=8)
        score = self.scores.get(prompt, 0.9)
        masks = [torch.from_numpy(comp == i) for i in range(1, n)] if score > self.confidence_threshold else []
        h, w = image.shape[:2]
        return {
            "masks": torch.stack(masks)[:, None] if masks else torch.zeros((0, 1, h, w), dtype=torch.bool),
            "boxes": torch.tensor([mask_box(m) for m in masks], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.full((len(masks),), score),
        }


def blob_image(h=128, w=160, blobs=((10, 10, 40, 40), (60, 20, 90, 60))):
    """RGB image with red rectangles (x0, y0, x1, y1) on green-ish soil, and its semantic mask."""
    image = np.full((h, w, 3), 60, np.uint8)
    semantic = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in blobs:
        image[y0:y1, x0:x1] = (200, 80, 40)
        semantic[y0:y1, x0:x1] = 1
    return image, semantic


def make_generator(**cfg):
    base = {"prompts": ["plant"], "score_threshold": 0.5, "device": "cpu",
            "filtering": {"min_area": 10, "mask_nms_threshold": 0.7, "containment_threshold": 0.9}}
    return SAM3ProposalGenerator.from_components(None, FakeSam3Processor(cfg.pop("scores", None)), {**base, **cfg})


# --- generator ------------------------------------------------------------

def test_generator_predict_pools_prompts_and_removes_duplicates():
    gen = make_generator(prompts=["plant", "weed"], scores={"weed": 0.8})
    assert gen.processor.confidence_threshold == 0.5
    image, _ = blob_image()
    res = gen.predict(image)
    assert len(res) == 2 and {p.prompt for p in res.proposals} == {"plant"}
    assert sorted(r for _, r in res.rejected) == ["duplicate_of:0", "duplicate_of:1"]  # weed repeats plant
    assert gen.processor.images == 1  # one image encoding shared by the prompts
    raw = gen.predict(image, apply_filtering=False)
    assert len(raw) == 4 and not raw.rejected


def test_generator_pseudo_target_and_empty_image():
    gen = make_generator()
    image, semantic = blob_image()
    target, result = gen.create_pseudo_target(image, semantic, return_result=True)
    assert sorted(target["boxes"].tolist()) == [[10, 10, 40, 40], [60, 20, 90, 60]]
    assert target["labels"].tolist() == [1, 1] and target["masks"].shape == (2, 128, 160)
    assert all(p.metrics["semantic_overlap"] == 1 for p in result.proposals)
    empty = gen.create_pseudo_target(np.full((32, 32, 3), 60, np.uint8), np.zeros((32, 32), np.uint8))
    assert empty["boxes"].shape == (0, 4) and empty["masks"].shape == (0, 32, 32)


def test_to_pil_inputs():
    image, _ = blob_image()
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    for x in (image, tensor, tensor.float() / 255, to_pil(image)):
        assert np.array_equal(np.asarray(to_pil(x)), image)
    with pytest.raises(ValueError):
        to_pil(tensor.float() / 255 - 0.5)  # mean/std-normalized
    with pytest.raises(ValueError):
        to_pil(image[..., 0])


def test_predict_tiled_merges_plants_cut_by_tiles():
    blobs = ((10, 10, 40, 40), (100, 90, 180, 150), (200, 20, 240, 230))  # the 2nd and 3rd span tiles
    image, semantic = blob_image(260, 300, blobs)
    gen = make_generator()
    inst = gen.predict_tiled(image, semantic, tile_size=128, overlap=48, merge_iou=0.5)
    assert gen.processor.images > 4
    assert len(inst) == 3
    boxes = sorted(tuple(int(v) for v in b) for b in inst.boxes)
    assert boxes == sorted(blobs)
    assert all(m.all() for m in inst.masks)  # each merged mask fills its rectangle


def test_build_proposal_generator_registry_and_missing_sam3(monkeypatch):
    with pytest.raises(ValueError):
        build_proposal_generator("nope", {})
    for name in ("sam3", "sam3.model_builder", "sam3.model", "sam3.model.sam3_image_processor"):
        monkeypatch.setitem(sys.modules, name, None)  # as if not installed
    with pytest.raises(ImportError, match="extra sam3"):
        build_proposal_generator("sam3", {"checkpoint": "random", "device": "cpu"})


def test_generators_from_the_hydra_config(make_cfg):
    cfg = make_cfg("model=maskrcnn_pointrend")
    assert cfg.proposals.name == "sam3" and cfg.proposals.tiling.tile_size == 1008
    gen = SAM3ProposalGenerator.from_components(None, FakeSam3Processor(), cfg.proposals)  # all yaml options are known
    assert gen.prompts == list(cfg.proposals.prompts)
    assert gen.filtering.min_semantic_overlap == cfg.proposals.filtering.min_semantic_overlap
    assert gen.pseudo_labels.min_area == cfg.proposals.filtering.min_area
    assert cfg.model.instances.source == "connected_components"
    assert cfg.model.instances.pseudo_dirname == "instances_sam3"
    assert "pseudo_instances" in cfg.preprocess.tasks



# --- instance files and InstanceDataset ----------------------------------

def test_instance_annotation_roundtrip(tmp_path):
    gen = make_generator()
    image, semantic = blob_image()
    target = gen.create_pseudo_target(image, semantic)
    save_instance_annotation(tmp_path, "tile", target, extra={"provenance": {"generator": "sam3"}})
    id_map, meta = load_instance_annotation(tmp_path, "tile")
    assert id_map.dtype == np.uint16 and id_map.max() == 2
    assert meta["provenance"]["generator"] == "sam3" and meta["image_size"] == [128, 160]
    assert [i["annotation_source"] for i in meta["instances"]] == ["sam3", "sam3"]
    back = id_map_to_instances(id_map, meta)
    assert torch.equal(back["masks"].sum(0), target["masks"].sum(0))
    assert back["annotation_source"] == ["sam3", "sam3"] and back["scores"].tolist() == pytest.approx([0.9, 0.9])
    assert len(id_map_to_instances(id_map, meta, min_area=1000)["boxes"]) == 1  # 900 px blob dropped

    meta["instances"][0]["annotation_source"] = "sam3_corrected"  # an annotator fixed instance 1
    assert id_map_to_instances(id_map, meta)["annotation_source"].count("sam3_corrected") == 1

    overlapping = {**target, "masks": torch.ones_like(target["masks"])}
    with pytest.raises(ValueError, match="overlap"):
        save_instance_annotation(tmp_path, "bad", overlapping)


def write_tiles(root, n=3, empty_last=True):
    """split_dir-like train/{images,masks} with n blob tiles (the last without plants)."""
    for d in ("images", "masks"):
        (root / "train" / d).mkdir(parents=True, exist_ok=True)
    for k in range(n):
        blobs = () if (empty_last and k == n - 1) else ((10 + k, 10, 40, 40), (60, 20, 90, 60))
        image, semantic = blob_image(blobs=blobs)
        cv2.imwrite(str(root / "train" / "images" / f"tile{k}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(root / "train" / "masks" / f"tile{k}.png"), semantic)
    return root / "train"


def test_instance_dataset_reads_pseudo_instances_with_augmentation(tmp_path):
    A = pytest.importorskip("albumentations")
    from src.utils.datasets import InstanceDataset

    split = write_tiles(tmp_path, n=2, empty_last=False)
    gen = make_generator()
    image, semantic = blob_image(blobs=((10, 10, 40, 40), (60, 20, 90, 60)))
    save_instance_annotation(split / "inst", "tile0", gen.create_pseudo_target(image, semantic))  # tile1: no file

    ds = InstanceDataset(split / "images", split / "masks", instances_dir=split / "inst")
    assert len(ds) == 1
    img, target, stem = ds[0]
    assert stem == "tile0" and img.shape == (3, 128, 160)
    assert sorted(target["boxes"].tolist()) == [[10, 10, 40, 40], [60, 20, 90, 60]]
    assert target["annotation_source"] == ["sam3", "sam3"] and target["semantic_mask"].shape == (128, 160)

    flipped = InstanceDataset(split / "images", split / "masks", instances_dir=split / "inst",
                              augmentation=A.HorizontalFlip(p=1.0))
    _, target, _ = flipped[0]
    assert sorted(target["boxes"].tolist()) == [[70, 20, 100, 60], [120, 10, 150, 40]]
    assert target["semantic_mask"][10, 120] == 1 and target["semantic_mask"][10, 10] == 0


def test_train_instances_dir_selection(tmp_path, make_cfg):
    pytest.importorskip("pytorch_lightning")
    from src.train import instances_dir

    split = write_tiles(tmp_path)
    cfg = make_cfg("model=maskrcnn_pointrend")
    assert instances_dir(cfg, split / "images") is None
    cfg = make_cfg("model=maskrcnn_pointrend", "model.instances.source=pseudo", "proposals=sam3")
    with pytest.raises(FileNotFoundError, match="pseudo_instances"):
        instances_dir(cfg, split / "images")
    (split / cfg.proposals.pseudo_labels.dirname).mkdir()
    assert instances_dir(cfg, split / "images") == split / "instances_sam3"


# --- preprocess task ------------------------------------------------------

def test_generate_split_writes_annotations_records_and_summary(tmp_path, make_cfg):
    from src.preprocessing.pseudo_instances import generate_split

    split = write_tiles(tmp_path, n=3)
    scfg = OmegaConf.to_container(make_cfg("proposals=sam3").proposals, resolve=True)
    scfg["output"]["save_previews"] = 1
    gen = SAM3ProposalGenerator.from_components(None, FakeSam3Processor(), {**scfg, "device": "cpu"})
    out = split / "instances_sam3"
    summary = generate_split(gen, split / "images", split / "masks", out, scfg, {"generator": "sam3"})

    assert len(summary) == 3 and summary["instances"].tolist() == [2, 2, 0]
    assert summary["unassigned_foreground_fraction"].fillna(0).max() == 0
    assert gen.processor.images == 2  # the plant-free tile never reached SAM 3
    for k in range(3):
        assert (out / f"tile{k}.png").exists() and (out / f"tile{k}.json").exists()
    assert len(list((out / "proposals").glob("*.json"))) == 2 and len(list((out / "previews").glob("*.png"))) == 1
    _, meta = load_instance_annotation(out, "tile0")
    assert meta["provenance"] == {"generator": "sam3"}

    again = generate_split(gen, split / "images", split / "masks", out, scfg, {})
    assert len(again) == 0 and gen.processor.images == 2  # resumed: nothing redone


# --- full-image boxes -----------------------------------------------------

def test_rescale_boxes_to_full_resolution():
    from src.preprocessing.full_image_boxes import rescale_boxes

    out = rescale_boxes(np.array([[10, 20, 30, 40], [0, 0, 60, 60]]), (50, 60), (500, 601))
    assert out[0].tolist() == pytest.approx([100.1666, 200, 300.5, 400], abs=1e-3)  # per-axis factors
    assert out[1].tolist() == pytest.approx([0, 0, 601, 500])
    assert rescale_boxes(np.zeros((0, 4)), (5, 5), (10, 10)).shape == (0, 4)


def test_clip_instances_to_semantic():
    from src.inferencing.tiled_instances import Instances
    from src.models.proposals.pseudo_labels import clip_instances_to_semantic

    semantic = np.zeros((50, 60), np.uint8)
    semantic[10:20, 10:30] = 2
    semantic[18:20, 10:12] = 1
    inst = Instances(50, 60, boxes=np.array([[5, 5, 40, 30], [45, 40, 55, 48]]), labels=np.array([1, 1]),
                     scores=np.array([0.9, 0.8], np.float32),
                     masks=[np.ones((25, 35), bool), np.ones((8, 10), bool)])  # 2nd lies on soil
    out = clip_instances_to_semantic(inst, semantic, min_area=10)
    assert len(out) == 1 and out.boxes.tolist() == [[10, 10, 30, 20]] and out.labels.tolist() == [2]
    assert out.masks[0].shape == (10, 20) and out.masks[0].all()


def test_full_image_boxes_task(tmp_path, make_cfg, monkeypatch):
    import json

    import pandas as pd

    from src.preprocessing import full_image_boxes

    blobs = ((20, 20, 80, 100), (200, 60, 380, 300))  # the 2nd spans tiles at scale 0.5
    cfg = make_cfg("mode=preprocess", "proposals.full_image.scale=0.5", "proposals.tiling.tile_size=128",
                   "proposals.tiling.overlap=48", "proposals.full_image.plot_previews=1")
    rows = []
    for k in range(2):
        image, semantic = blob_image(320, 400, blobs if k == 0 else ())
        cv2.imwrite(str(tmp_path / f"img{k}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(tmp_path / f"img{k}.png"), semantic)
        rows.append({"image_id": f"img{k}", "split": "val", "image_path": tmp_path / f"img{k}.jpg",
                     "mask_path": tmp_path / f"img{k}.png"})
    manifest = Path(cfg.paths.dataset_manifest)
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    generator = make_generator()
    monkeypatch.setattr(full_image_boxes, "build_proposal_generator", lambda name, config: generator)

    full_image_boxes.main(cfg)
    out = Path(cfg.paths.project_preprocess_dir) / "data" / "full_image_boxes" / "sam3_scale0.5"
    record = json.loads((out / "img0.json").read_text())
    assert record["image_size"] == [320, 400] and record["scaled_size"] == [160, 200]
    assert sorted(i["bbox"] for i in record["instances"]) == [list(map(float, b)) for b in blobs]  # full resolution
    assert record["annotation_source"] == "sam3" and record["provenance"]["generator"] == "sam3"
    assert json.loads((out / "img1.json").read_text())["instances"] == []
    table = pd.read_csv(out / "boxes.csv")
    assert len(table) == 2 and set(table["image_id"]) == {"img0"} and (table["scale"] == 0.5).all()
    assert len(list((out / "previews").glob("*.jpg"))) == 1

    calls = generator.processor.images
    full_image_boxes.main(cfg)  # resumed: nothing redone, table rebuilt
    assert generator.processor.images == calls and len(pd.read_csv(out / "boxes.csv")) == 2


# --- Mode 2: external proposals ------------------------------------------

@pytest.mark.parametrize("mode", ["replace", "union"])
def test_pointrend_runs_on_external_proposals(mode):
    torch.manual_seed(0)
    model = build_model("maskrcnn_pointrend", num_classes=2, pretrained=None, box_score_thresh=0.0,
                        subdivision_steps=2).eval()
    images = [torch.rand(3, 128, 128), torch.rand(3, 96, 128)]
    proposals = [torch.tensor([[10.0, 10, 50, 60], [60, 60, 120, 120]]), torch.zeros((0, 4))]
    out = predict_with_proposals(model, images, proposals, mode=mode)
    assert len(out) == 2
    for o, img in zip(out, images):
        assert set(o) >= {"boxes", "labels", "scores", "masks"}
        assert o["masks"].shape[1:] == (1, *img.shape[-2:])
    if mode == "replace":
        assert len(out[0]["boxes"]) <= 2 and len(out[1]["boxes"]) == 0  # detections only where proposals were
    with pytest.raises(ValueError):
        predict_with_proposals(model, images, proposals[:1])
    with pytest.raises(RuntimeError):
        predict_with_proposals(model.train(), images, proposals)
