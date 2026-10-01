"""Point sampling for PointRend (Kirillov et al., 2020), after Detectron2's
projects/PointRend/point_rend/point_features.py.

Point coordinates are (x, y) in [0, 1] relative to some extent: a box (box-
relative coords, used by the mask heads) or an image (absolute pixel coords
divided by the image size). [0, 1] spans the outer edges of the extent, so
pixel i's center is at (i + 0.5) / size -- the align_corners=False convention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Bilinearly sample `input` (N, C, H, W) at `point_coords` (N, P, 2) in
    [0, 1] x [0, 1]; returns (N, C, P). Points outside [0, 1] read zeros."""
    kwargs.setdefault("align_corners", False)
    grid = 2.0 * point_coords.unsqueeze(2) - 1.0  # (N, P, 1, 2) in [-1, 1]
    return F.grid_sample(input, grid.to(input.dtype), **kwargs).squeeze(3)


def generate_regular_grid_point_coords(num_boxes: int, side: int, device=None) -> torch.Tensor:
    """(num_boxes, side * side, 2) coords of the cell centers of a side x side grid, row-major."""
    centers = (torch.arange(side, dtype=torch.float32, device=device) + 0.5) / side
    ys, xs = torch.meshgrid(centers, centers, indexing="ij")
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    return coords.unsqueeze(0).expand(num_boxes, -1, -1)


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Training-time point selection (PointRend sec. 3.1): draw
    num_points * oversample_ratio random points, keep the
    importance_sample_ratio * num_points most uncertain of them, and fill the
    rest with fresh uniform random points.

    Args:
        coarse_logits: (R, C, H, W) mask logits per box.
        uncertainty_func: maps (R, C, P) point logits to (R, 1, P) uncertainty
            (higher = more uncertain).
    Returns:
        (R, num_points, 2) box-relative coords.
    """
    if oversample_ratio < 1:
        raise ValueError(f"oversample_ratio must be >= 1, got {oversample_ratio}")
    if not 0 <= importance_sample_ratio <= 1:
        raise ValueError(f"importance_sample_ratio must be in [0, 1], got {importance_sample_ratio}")
    num_boxes, device = coarse_logits.shape[0], coarse_logits.device
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=device)
    point_uncertainties = uncertainty_func(point_sample(coarse_logits, point_coords))

    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain, dim=1)[1]
    point_coords = torch.gather(point_coords, 1, idx.unsqueeze(2).expand(-1, -1, 2))
    if num_random > 0:
        point_coords = torch.cat([point_coords, torch.rand(num_boxes, num_random, 2, device=device)], dim=1)
    return point_coords


@torch.no_grad()
def get_uncertain_point_coords_on_grid(uncertainty_map: torch.Tensor, num_points: int):
    """Inference-time point selection: the num_points most uncertain cells of
    an (R, 1, H, W) uncertainty map.

    Returns:
        point_indices: (R, P) flat indices into the H * W grid.
        point_coords: (R, P, 2) box-relative coords of those cells' centers.
    """
    R, _, H, W = uncertainty_map.shape
    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.stack(
        [((point_indices % W).float() + 0.5) / W, ((point_indices // W).float() + 0.5) / H], dim=2
    )
    return point_indices, point_coords


def box_coords_to_image_coords(boxes: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Box-relative (R, P, 2) coords to absolute image pixel coords, for (R, 4) xyxy boxes."""
    xy0 = boxes[:, None, 0:2]
    wh = (boxes[:, 2:4] - boxes[:, 0:2])[:, None, :]
    return xy0 + point_coords * wh


def _feature_scale(feature: torch.Tensor, image_shapes: list[tuple[int, int]]) -> float:
    """Feature-map stride as a scale (1/4 for FPN level p2), inferred the same
    way torchvision's MultiScaleRoIAlign does: nearest power of two of
    feature size / largest image size (images are padded up to it)."""
    max_h = max(h for h, _ in image_shapes)
    return 2.0 ** round(math.log2(feature.shape[-2] / max_h))


def sample_fine_grained_features(
    features: list[torch.Tensor],
    image_shapes: list[tuple[int, int]],
    boxes: list[torch.Tensor],
    point_coords: torch.Tensor,
) -> torch.Tensor:
    """Features at box-relative points, concatenated over feature levels.

    Args:
        features: per-level (B, C_l, H_l, W_l) maps of the padded image batch.
        image_shapes: (H, W) of each image before padding.
        boxes: per-image (R_i, 4) xyxy boxes, in image pixels.
        point_coords: (sum R_i, P, 2) box-relative coords, ordered as `boxes`.
    Returns:
        (sum R_i, sum C_l, P)
    """
    scales = [_feature_scale(f, image_shapes) for f in features]
    abs_coords = box_coords_to_image_coords(torch.cat(boxes), point_coords)
    num_points = point_coords.shape[1]
    per_image = []
    start = 0
    for i, img_boxes in enumerate(boxes):
        n = img_boxes.shape[0]
        coords = abs_coords[start:start + n]  # (n, P, 2)
        start += n
        levels = []
        for feature, scale in zip(features, scales):
            size = feature.new_tensor([feature.shape[-1], feature.shape[-2]])
            grid = (coords * scale / size).reshape(1, -1, 2)  # feature map covers size / scale px
            sampled = point_sample(feature[i:i + 1], grid)  # (1, C, n * P)
            levels.append(sampled.reshape(feature.shape[1], n, num_points).permute(1, 0, 2))
        per_image.append(torch.cat(levels, dim=1))
    return torch.cat(per_image)


def sample_masks_at_points(masks: torch.Tensor, mask_idx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of mask `mask_idx[r]` at absolute pixel coords
    `coords[r]`, for ground-truth point targets.

    Equal to point_sample(masks[mask_idx][:, None].float(), coords / (W, H))
    but reads only the four neighbouring pixels of each point, so it never
    builds the (R, H, W) float copy of the full-image masks.

    Args:
        masks: (G, H, W) instance masks (any dtype; values in [0, 1]).
        mask_idx: (R,) long, which mask each row samples.
        coords: (R, P, 2) absolute (x, y) pixel coords.
    Returns:
        (R, P) float.
    """
    _, H, W = masks.shape
    x = coords[..., 0] - 0.5  # pixel centers at integer positions
    y = coords[..., 1] - 0.5
    x0, y0 = x.floor(), y.floor()
    wx1, wy1 = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    out = coords.new_zeros(coords.shape[:2])
    rows = mask_idx[:, None].expand_as(x0)
    for dy, wy in ((0, 1 - wy1), (1, wy1)):
        for dx, wx in ((0, 1 - wx1), (1, wx1)):
            xi, yi = x0 + dx, y0 + dy
            inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            values = masks[rows, yi.clamp(0, H - 1), xi.clamp(0, W - 1)].to(out.dtype)
            out = out + values * wx * wy * inside
    return out
