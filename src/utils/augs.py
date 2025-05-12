import logging
import albumentations as A
from typing import List, Callable, Dict, Any
import random
log = logging.getLogger(__name__)


def get_noise_transforms(cfg) -> List[A.BasicTransform]:
    if not cfg.noise_transforms.enable:
        return []

    transforms = []

    for name, params in cfg.noise_transforms.transforms.items():
        if not params.enable:
            continue

        if name == "MultiplicativeNoise":
            transforms.append(A.MultiplicativeNoise(
                multiplier=tuple(params.multiplier),
                per_channel=params.per_channel,
                p=params.p
            ))

        elif name == "Downscale":
            transforms.append(A.Downscale(
                scale_min=params.scale_min,
                scale_max=params.scale_max,
                p=params.p
            ))

        elif name == "GridDistortion":
            transforms.append(A.GridDistortion(
                num_steps=params.num_steps,
                distort_limit=params.distort_limit,
                p=params.p
            ))

        elif name == "ElasticTransform":
            transforms.append(A.ElasticTransform(
                alpha=params.alpha,
                sigma=params.sigma,
                alpha_affine=params.alpha_affine,
                p=params.p
            ))

        elif name == "GaussNoise":
            transforms.append(A.GaussNoise(
                mean=(params.mean_range_min, params.mean_range_max),
                per_channel=params.per_channel,
                noise_scale_factor=params.noise_scale_factor,
                p=params.p
            ))

        elif name == "ISONoise":
            transforms.append(A.ISONoise(
                color_shift=(params.color_shift_min, params.color_shift_max),
                intensity=(params.intensity_min, params.intensity_max),
                p=params.p
            ))

        elif name == "JpegCompression":
            transforms.append(A.JpegCompression(
                quality_lower=params.quality_lower,
                quality_upper=params.quality_upper,
                p=params.p
            ))

    return transforms


def get_geometric_transforms(g_cfg, img_hw) -> List[A.BasicTransform]:
    transforms = []
    if g_cfg.horizontal_flip.enable:
        transforms.append(A.HorizontalFlip(p=g_cfg.horizontal_flip.p))

    if g_cfg.vertical_flip.enable:
        transforms.append(A.VerticalFlip(p=g_cfg.vertical_flip.p))

    if g_cfg.affine.enable:
        transforms.append(A.Affine(**g_cfg.affine))

    if g_cfg.padding.enable:
        transforms.append(A.PadIfNeeded(min_height=img_hw[0], min_width=img_hw[1], p=g_cfg.padding.p))

    if g_cfg.random_crop.enable:
        transforms.append(A.RandomCrop(height=img_hw[0], width=img_hw[1], p=g_cfg.random_crop.p))

    if g_cfg.perspective.enable:
        transforms.append(A.Perspective(scale=g_cfg.perspective.scale, p=g_cfg.perspective.p))

    if g_cfg.optical_distortion.enable:
        transforms.append(A.OpticalDistortion(distort_limit=g_cfg.optical_distortion.distort_limit, p=g_cfg.optical_distortion.p))

    return transforms


def get_photometric_transforms(cfg) -> List[A.BasicTransform]:
    transforms = []

    if cfg.rgb_shift.enable:
        transforms.append(A.RGBShift(
            r_shift_limit=cfg.rgb_shift.r_shift_limit,
            g_shift_limit=cfg.rgb_shift.g_shift_limit,
            b_shift_limit=cfg.rgb_shift.b_shift_limit,
            p=cfg.rgb_shift.p
        ))

    if cfg.random_gamma.enable:
        transforms.append(A.RandomGamma(
            gamma_limit=tuple(cfg.random_gamma.gamma_limit),
            p=cfg.random_gamma.p
        ))

    if cfg.to_gray.enable:
        transforms.append(A.ToGray(p=cfg.to_gray.p))

    if cfg.channel_shuffle.enable:
        transforms.append(A.ChannelShuffle(p=cfg.channel_shuffle.p))

    if cfg.invert_image.enable:
        transforms.append(A.InvertImg(p=cfg.invert_image.p))

    if cfg.color_jitter.enable:
        transforms.append(A.ColorJitter(p=cfg.color_jitter.p))

    if cfg.huesatval.enable:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=cfg.huesatval.hue_shift_limit,
            sat_shift_limit=cfg.huesatval.sat_shift_limit,
            val_shift_limit=cfg.huesatval.val_shift_limit,
            p=cfg.huesatval.p
        ))

    if cfg.clahe.enable:
        transforms.append(A.CLAHE(
            clip_limit=cfg.clahe.clip_limit,
            tile_grid_size=tuple(cfg.clahe.tile_grid_size),
            p=cfg.clahe.p
        ))

    if cfg.brightness_contrast.enable:
        transforms.append(A.RandomBrightnessContrast(p=cfg.brightness_contrast.p))

    if cfg.equalize.enable:
        transforms.append(A.Equalize(p=cfg.equalize.p))

    if cfg.random_tone_curve.enable:
        transforms.append(A.RandomToneCurve(p=cfg.random_tone_curve.p))

    return transforms

# === Main Augmentation Builders === #
def train_aug(cfg):
    img_hw = tuple(cfg.augment.train.img_hw)
    t_cfg = cfg.augment.train

    transforms = []

    # === Geometric Transforms ===
    geo_transforms = get_geometric_transforms(t_cfg.geometrics_transforms, img_hw)
    if geo_transforms:
        transforms.append(A.SomeOf(geo_transforms, n=min(len(geo_transforms), 3), replace=False, p=1.0))

    # === Photometric Transforms ===
    photo_transforms = get_photometric_transforms(t_cfg.photometrics_transforms)
    if photo_transforms:
        transforms.append(A.SomeOf(photo_transforms, n=min(len(photo_transforms), 2), replace=False, p=1.0))

    # === Noise Transforms ===
    noise_transforms = get_noise_transforms(t_cfg)
    if noise_transforms:
        transforms.append(A.SomeOf(noise_transforms, n=min(len(noise_transforms), 1), replace=False, p=1.0))

    # === Shuffle overall order ===
    random.shuffle(transforms)
    return A.Compose(transforms)


def val_aug(cfg):
    """
    Dynamically builds validation augmentations based on the configuration.
    """
    img_hw = tuple(cfg.augment.val.img_hw)
    val_transform = []

    if cfg.augment.val.padding.enable:
        val_transform.append(A.PadIfNeeded(min_height=img_hw[0], min_width=img_hw[1], p=cfg.augment.val.padding.p))
    
    return A.Compose(val_transform)
