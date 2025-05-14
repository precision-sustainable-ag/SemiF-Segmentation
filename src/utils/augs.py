import logging
import albumentations as A
from typing import List, Callable, Dict, Any
import random
log = logging.getLogger(__name__)


def build_transforms_from_config(cfg_section, albumentations_class_map, extra_args=None) -> List[A.BasicTransform]:
    transforms = []

    for key, albumentations_class in albumentations_class_map.items():
        if key in cfg_section and getattr(cfg_section[key], 'enable', False):
            params = cfg_section[key]
            cfg_dict = {k: v for k, v in params.items() if k != 'enable'}
            
            if extra_args and key in extra_args:
                cfg_dict.update(extra_args[key])

            transforms.append(albumentations_class(**cfg_dict))

    return transforms

def get_geometric_transforms(cfg, img_hw) -> List[A.BasicTransform]:
    g_cfg = cfg.geometrics_transforms
    if not g_cfg.enable:
        return []

    geometric_map = {
        "horizontal_flip": A.HorizontalFlip,
        "vertical_flip": A.VerticalFlip,
        "affine": A.Affine,
        "padding": A.PadIfNeeded,
        "crop_and_pad": A.CropAndPad,
        "elastic_transform": A.ElasticTransform,
        "grid_distortion": A.GridDistortion,
        "morphological": A.Morphological,
        "optical_distortion": A.OpticalDistortion,
        "perspective": A.Perspective,
    }

    extra_args = {
        # "padding": {"min_height": img_hw[0], "min_width": img_hw[1]},
        # "random_crop": {"height": img_hw[0], "width": img_hw[1]},
    }

    return build_transforms_from_config(g_cfg, geometric_map, extra_args)

def get_photometric_transforms(cfg) -> List[A.BasicTransform]:
    p_cfg = cfg.photometrics_transforms
    if not p_cfg.enable:
        return []

    photometric_map = {
        "rgb_shift": A.RGBShift,
        "random_gamma": A.RandomGamma,
        "to_gray": A.ToGray,
        "channel_shuffle": A.ChannelShuffle,
        "invert_image": A.InvertImg,
        "color_jitter": A.ColorJitter,
        "huesatval": A.HueSaturationValue,
        "clahe": A.CLAHE,
        "brightness_contrast": A.RandomBrightnessContrast,
        "equalize": A.Equalize,
        "random_tone_curve": A.RandomToneCurve,
    }

    return build_transforms_from_config(p_cfg, photometric_map)

def get_noise_transforms(cfg) -> List[A.BasicTransform]:
    n_cfg = cfg.noise_transforms
    if not n_cfg.enable:
        return []

    noise_map = {
        "MultiplicativeNoise": A.MultiplicativeNoise,
        "ISONoise": A.ISONoise,
        "ImageCompression": A.ImageCompression,
    }

    return build_transforms_from_config(n_cfg, noise_map)

def some_of_transforms(transforms: List[A.BasicTransform], some_cfg) -> A.SomeOf:
    if not transforms:
        return None
    return A.SomeOf(transforms, n=some_cfg.n, replace=some_cfg.replace, p=some_cfg.p)


# === Main Augmentation Builders === #
def train_aug(cfg):
    img_hw = tuple(cfg.augment.train.img_hw)
    t_cfg = cfg.augment.train

    transforms = []

    # Geometric Transforms
    geo_transforms = get_geometric_transforms(t_cfg, img_hw)
    geo_some = some_of_transforms(geo_transforms, t_cfg.geometrics_transforms.some_of)
    if geo_some:
        transforms.append(geo_some)

    # Photometric Transforms
    photo_transforms = get_photometric_transforms(t_cfg)
    photo_some = some_of_transforms(photo_transforms, t_cfg.photometrics_transforms.some_of)
    if photo_some:
        transforms.append(photo_some)

    # Noise Transforms
    noise_transforms = get_noise_transforms(t_cfg)
    noise_some = some_of_transforms(noise_transforms, t_cfg.noise_transforms.some_of)
    if noise_some:
        transforms.append(noise_some)

    # Shuffle block order
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
