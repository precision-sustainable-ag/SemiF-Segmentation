"""Proposal processing on mocked SAM outputs (no checkpoint, no SAM package):
normalization, score / area / semantic filtering, mask NMS and containment,
and pseudo-target generation."""

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from src.models.proposals.filtering import (  # noqa: E402
    FilterConfig,
    apply_filters,
    filter_by_score_area,
    filter_by_semantics,
    mask_iou,
    resolve_overlaps,
    semantic_metrics,
)
from src.models.proposals.structures import InstanceProposal, ProposalResult, normalize_output  # noqa: E402
from src.models.proposals.pseudo_labels import PseudoLabelConfig, create_pseudo_target, target_stats  # noqa: E402
from src.utils.instance_annotations import mask_box  # noqa: E402

H, W = 64, 80
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def rect(x0, y0, x1, y1, h=H, w=W):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def proposal(x0, y0, x1, y1, score, prompt="plant"):
    return InstanceProposal(torch.tensor([x0, y0, x1, y1], dtype=torch.float32), rect(x0, y0, x1, y1), score, prompt)


def raw_output(masks, scores, boxes=None):
    """What Sam3Processor.set_text_prompt returns: masks (N, 1, H, W), boxes (N, 4) xyxy, scores (N,)."""
    masks = torch.stack(masks)[:, None] if masks else torch.zeros((0, 1, H, W), dtype=torch.bool)
    if boxes is None:
        boxes = [mask_box(m[0]) for m in masks] if len(masks) else []
    return {"masks": masks, "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32)}


# --- normalization --------------------------------------------------------

def test_normalize_output_shapes_order_and_boxes():
    out = raw_output([rect(0, 0, 10, 10), rect(20, 30, 50, 60)], [0.6, 0.9],
                     boxes=[[0, 0, 10, 10], [20, 30, 95, 60]])  # second box runs past the image
    res = normalize_output(out, (H, W), prompt="weed")
    assert len(res) == 2 and res.image_size == (H, W)
    assert [p.score for p in res.proposals] == pytest.approx([0.9, 0.6])  # sorted by score
    assert all(p.prompt == "weed" for p in res.proposals)
    assert res.masks.shape == (2, H, W) and res.masks.dtype == torch.bool
    assert res.boxes.shape == (2, 4) and res.boxes.dtype == torch.float32
    assert res.boxes[0].tolist() == [20, 30, W, 60]  # clipped to the image width
    assert res.proposals[0].area() == 30 * 30


def test_normalize_accepts_probabilities_and_resizes_masks():
    probs = torch.zeros(1, 1, H // 2, W // 2)
    probs[0, 0, 5:10, 5:10] = 0.8
    probs[0, 0, 0:2, 0:2] = 0.3  # below mask_threshold
    res = normalize_output({"masks": probs, "boxes": torch.tensor([[10, 10, 20, 20]]), "scores": torch.tensor([0.7])},
                                (H, W))
    mask = res.proposals[0].mask
    assert mask.shape == (H, W) and mask.dtype == torch.bool
    assert mask_box(mask) == (10, 10, 20, 20) and mask.sum() == 100


def test_normalize_empty_and_inconsistent_outputs():
    res = normalize_output(raw_output([], []), (H, W))
    assert len(res) == 0 and res.masks.shape == (0, H, W) and res.boxes.shape == (0, 4) and res.scores.shape == (0,)
    with pytest.raises(ValueError):
        normalize_output({"masks": torch.zeros(2, 1, H, W), "boxes": torch.zeros(1, 4), "scores": torch.zeros(2)}, (H, W))
    with pytest.raises(ValueError):
        normalize_output({"masks": torch.zeros(1, 3, H, W), "boxes": torch.zeros(1, 4), "scores": torch.zeros(1)}, (H, W))


def test_mask_box_uses_pixel_edges():
    assert mask_box(rect(3, 4, 10, 12)) == (3, 4, 10, 12)
    assert mask_box(rect(3, 4, 10, 12).numpy()) == (3, 4, 10, 12)
    assert mask_box(torch.zeros(5, 5, dtype=torch.bool)) is None


# --- score / area ---------------------------------------------------------

def test_score_and_area_filters_record_reasons():
    res = ProposalResult([proposal(0, 0, 10, 10, 0.9), proposal(0, 0, 10, 10, 0.3), proposal(0, 0, 2, 2, 0.9),
                          proposal(0, 0, 60, 60, 0.9), InstanceProposal(torch.zeros(4), torch.zeros(H, W, dtype=torch.bool), 0.9)],
                         (H, W))
    out = filter_by_score_area(res, score_threshold=0.5, min_area=10, max_area=1000)
    assert [p.area() for p in out.proposals] == [100]
    assert [r for _, r in out.rejected] == ["score", "min_area", "max_area", "empty_mask"]
    assert out.proposals[0].metrics["area"] == 100


# --- semantic agreement ---------------------------------------------------

def semantic_with_plant(x0=10, y0=10, x1=30, y1=30, value=1):
    s = torch.zeros(H, W, dtype=torch.long)
    s[y0:y1, x0:x1] = value
    return s


def test_semantic_metrics_values():
    semantic = semantic_with_plant(10, 10, 30, 30)  # 20 x 20 plant
    exact = semantic_metrics(rect(10, 10, 30, 30), semantic)
    assert exact == pytest.approx({"area": 400, "semantic_overlap": 1, "background_fraction": 0,
                                   "foreground_iou": 1, "foreground_coverage": 1})
    half = semantic_metrics(rect(10, 10, 30, 20), semantic)  # top half of the plant
    assert half["semantic_overlap"] == 1 and half["foreground_coverage"] == 1  # box holds only that half
    spill = semantic_metrics(rect(20, 10, 40, 30), semantic)  # half on plant, half on soil
    assert spill["semantic_overlap"] == pytest.approx(0.5) and spill["background_fraction"] == pytest.approx(0.5)
    assert spill["foreground_iou"] == pytest.approx(0.5)


def test_ignore_value_separates_background_from_unlabeled():
    semantic = semantic_with_plant(10, 10, 20, 30)
    semantic[10:30, 20:30] = 255  # unlabeled
    m = semantic_metrics(rect(10, 10, 30, 30), semantic, ignore_value=255)
    assert m["semantic_overlap"] == pytest.approx(0.5) and m["background_fraction"] == 0


def test_semantic_filter_rejects_proposals_over_soil():
    semantic = semantic_with_plant(10, 10, 30, 30)
    res = ProposalResult([proposal(10, 10, 30, 30, 0.9), proposal(20, 10, 40, 30, 0.95), proposal(50, 40, 70, 60, 0.99)],
                         (H, W))
    cfg = FilterConfig(min_semantic_overlap=0.8, max_background_fraction=0.1)
    out = filter_by_semantics(res, semantic.numpy(), cfg)
    assert [p.score for p in out.proposals] == [0.9]
    assert sorted(r for _, r in out.rejected) == ["semantic_overlap", "semantic_overlap"]
    assert out.proposals[0].metrics["foreground_iou"] == 1
    with pytest.raises(ValueError):
        filter_by_semantics(res, torch.zeros(10, 10), cfg)


# --- overlaps -------------------------------------------------------------

def test_mask_iou():
    iou = mask_iou([rect(0, 0, 10, 10), rect(5, 0, 15, 10), rect(40, 40, 50, 50)])
    assert iou[0, 0] == 1 and iou[0, 1] == pytest.approx(50 / 150) and iou[0, 2] == 0 and iou[1, 0] == iou[0, 1]


def test_mask_nms_drops_duplicates_not_touching_plants():
    res = ProposalResult([proposal(10, 10, 30, 30, 0.8), proposal(11, 10, 30, 30, 0.9),  # near duplicate
                          proposal(30, 10, 50, 30, 0.7)], (H, W))  # touching neighbour
    out = resolve_overlaps(res, nms_threshold=0.7, containment_threshold=None)
    assert [p.score for p in out.proposals] == [0.9, 0.7]
    assert out.rejected[0][1] == "duplicate_of:0"


def test_nms_works_on_masks_not_boxes():
    # identical boxes, disjoint masks (two plants interleaved in one box): not duplicates
    a, b = torch.zeros(H, W, dtype=torch.bool), torch.zeros(H, W, dtype=torch.bool)
    a[10:30:2, 10:30] = True
    b[11:30:2, 10:30] = True
    box = torch.tensor([10, 10, 30, 30], dtype=torch.float32)
    res = ProposalResult([InstanceProposal(box, a, 0.9), InstanceProposal(box, b, 0.8)], (H, W))
    assert len(resolve_overlaps(res, nms_threshold=0.5, containment_threshold=None)) == 2


@pytest.mark.parametrize("policy, kept", [
    ("keep_higher_score", [0.9]),     # the whole plant scored higher
    ("keep_larger", [0.9]),
    ("keep_smaller", [0.8, 0.7]),     # the two leaves replace the plant
    ("keep_both", [0.9, 0.8, 0.7]),
])
def test_containment_policies(policy, kept):
    res = ProposalResult([proposal(10, 10, 40, 30, 0.9),               # plant
                          proposal(10, 10, 20, 30, 0.8), proposal(30, 10, 40, 30, 0.7)],  # two parts inside it
                         (H, W))
    out = resolve_overlaps(res, nms_threshold=0.7, containment_threshold=0.9, containment_policy=policy)
    assert [p.score for p in out.proposals] == kept
    assert len(out) + len(out.rejected) == 3


def test_keep_larger_replaces_a_higher_scored_part():
    res = ProposalResult([proposal(10, 10, 20, 30, 0.9), proposal(10, 10, 40, 30, 0.6)], (H, W))
    out = resolve_overlaps(res, containment_threshold=0.9, containment_policy="keep_larger")
    assert [p.score for p in out.proposals] == [0.6]
    assert out.rejected[0][1] == "contained_with:1"
    with pytest.raises(ValueError):
        resolve_overlaps(res, containment_policy="nope")
    with pytest.raises(ValueError):
        FilterConfig.from_dict({"containment_policy": "nope"})
    with pytest.raises(ValueError):
        FilterConfig.from_dict({"min_aera": 3})


def test_apply_filters_semantic_before_nms():
    # the top proposal spills onto soil; were NMS first it would suppress the good one
    semantic = semantic_with_plant(10, 10, 30, 30)
    res = ProposalResult([proposal(10, 10, 34, 30, 0.95), proposal(10, 10, 30, 30, 0.9)], (H, W))
    cfg = FilterConfig(min_semantic_overlap=0.9, mask_nms_threshold=0.7)
    out = apply_filters(res, cfg, score_threshold=0.5, semantic_mask=semantic)
    assert [p.score for p in out.proposals] == [0.9]
    assert len(apply_filters(ProposalResult([], (H, W)), cfg, 0.5, semantic)) == 0


# --- pseudo targets -------------------------------------------------------

def test_pseudo_target_format_classes_and_provenance():
    semantic = semantic_with_plant(10, 10, 30, 30, value=2)
    semantic[40:60, 40:60] = 1
    res = ProposalResult([proposal(8, 8, 32, 32, 0.9), proposal(40, 40, 60, 60, 0.8, prompt="weed")], (H, W))
    t = create_pseudo_target(res, semantic)
    assert t["boxes"].dtype == torch.float32 and t["boxes"].shape == (2, 4)
    assert t["labels"].dtype == torch.int64 and t["labels"].tolist() == [2, 1]  # class from the semantic mask
    assert t["masks"].dtype == torch.uint8 and t["masks"].shape == (2, H, W)
    assert t["boxes"][0].tolist() == [10, 10, 30, 30]  # clipped to the plant, box from the clipped mask
    assert t["annotation_source"] == ["sam3", "sam3"] and t["prompts"] == ["plant", "weed"]
    assert t["class_source"] == ["semantic", "semantic"] and t["scores"].tolist() == pytest.approx([0.9, 0.8])
    assert target_stats(t, semantic)["unassigned_foreground_fraction"] == 0


def test_pseudo_target_exclusive_masks_and_min_area():
    res = ProposalResult([proposal(10, 10, 30, 30, 0.9), proposal(20, 10, 40, 30, 0.8), proposal(28, 10, 32, 30, 0.7)],
                         (H, W))
    t = create_pseudo_target(res, None, PseudoLabelConfig(min_area=50))
    assert t["masks"].sum(0).max() == 1  # no pixel in two instances
    assert t["masks"].sum((1, 2)).tolist() == [400, 200]  # the second keeps its unclaimed half; the third nothing
    assert t["boxes"][1].tolist() == [30, 10, 40, 30]
    overlapping = create_pseudo_target(res, None, PseudoLabelConfig(exclusive_masks=False))
    assert overlapping["masks"].sum((1, 2)).tolist() == [400, 400, 80]


def test_pseudo_target_labels_from_prompts_without_semantics():
    res = ProposalResult([proposal(0, 0, 10, 10, 0.9, "grass"), proposal(20, 20, 30, 30, 0.8, "broadleaf plant")], (H, W))
    t = create_pseudo_target(res, None, PseudoLabelConfig(prompt_labels={"grass": 3}, default_label=1))
    assert t["labels"].tolist() == [3, 1] and t["class_source"] == ["prompt", "prompt"]


def test_pseudo_target_empty():
    t = create_pseudo_target(ProposalResult([], (H, W)), semantic_with_plant())
    assert t["boxes"].shape == (0, 4) and t["labels"].shape == (0,) and t["masks"].shape == (0, H, W)
    assert target_stats(t, semantic_with_plant())["unassigned_foreground_fraction"] == 1
    # a proposal entirely off the plant disappears when clipped
    off = create_pseudo_target(ProposalResult([proposal(50, 40, 70, 60, 0.9)], (H, W)), semantic_with_plant())
    assert len(off["boxes"]) == 0


# --- devices --------------------------------------------------------------

def test_result_to_device_roundtrip_cpu():
    res = ProposalResult([proposal(0, 0, 10, 10, 0.9)], (H, W), rejected=[(proposal(0, 0, 5, 5, 0.1), "score")])
    moved = res.to("cpu")
    assert moved.proposals[0].mask.device.type == "cpu" and moved.rejected[0][1] == "score"
    assert moved.proposals[0] is not res.proposals[0]


@cuda
def test_gpu_outputs_are_normalized_to_cpu_and_filters_run_on_gpu():
    out = {k: v.cuda() for k, v in raw_output([rect(10, 10, 30, 30), rect(11, 10, 30, 30)], [0.9, 0.8]).items()}
    on_cpu = normalize_output(out, (H, W))
    assert on_cpu.masks.device.type == "cpu" and on_cpu.boxes.device.type == "cpu"
    on_gpu = normalize_output(out, (H, W), device=None)
    assert on_gpu.masks.device.type == "cuda"
    filtered = apply_filters(on_gpu, FilterConfig(min_semantic_overlap=0.8), 0.5, semantic_with_plant())
    assert len(filtered) == 1 and filtered.proposals[0].mask.device.type == "cuda"
    t = create_pseudo_target(filtered, semantic_with_plant().cuda())
    assert t["masks"].device.type == "cpu" and len(t["boxes"]) == 1
    assert filtered.to("cpu").proposals[0].mask.device.type == "cpu"
