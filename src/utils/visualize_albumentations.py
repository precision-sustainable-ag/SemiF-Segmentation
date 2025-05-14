import cv2
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

def load_images_from_folder(folder_path):
    image_paths = list(Path(folder_path).glob('*'))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    return image_paths

def build_transforms(config_section, albumentations_class_map, additional_kwargs=None):
    transforms = []

    for key, albumentations_class in albumentations_class_map.items():
        if key in config_section and getattr(config_section[key], 'enable', False):
            cfg_dict = OmegaConf.to_container(config_section[key], resolve=True)
            cfg_dict.pop('enable', None)

            # Inject additional kwargs if needed
            if additional_kwargs and key in additional_kwargs:
                cfg_dict.update(additional_kwargs[key])

            transforms.append((key, albumentations_class(**cfg_dict)))

    return transforms

def build_photometric_transforms(cfg):
    photometrics = cfg.train.photometrics_transforms

    photometric_map = {
        "huesatval": A.HueSaturationValue,
        "rgb_shift": A.RGBShift,
        "random_gamma": A.RandomGamma,
        "clahe": A.CLAHE,
        "to_gray": A.ToGray,
        "channel_shuffle": A.ChannelShuffle,
        "invert_image": A.InvertImg,
        "equalize": A.Equalize,
        "brightness_contrast": A.RandomBrightnessContrast,
        "color_jitter": A.ColorJitter,
        "random_tone_curve": A.RandomToneCurve,
    }

    return build_transforms(photometrics, photometric_map)

def build_geometric_transforms(cfg):
    geometrics = cfg.train.geometrics_transforms
    img_hw = OmegaConf.to_container(cfg.train.img_hw, resolve=True)

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
        # "mosaic": A.Mosaic, # need mosaic_metadata to implement
    }

    # Inject height/width into PadIfNeeded & RandomCrop
    extra_args = {
        # "padding": {"min_height": img_hw[0], "min_width": img_hw[1]},
        "random_crop": {"height": img_hw[0], "width": img_hw[1]},
    }

    return build_transforms(geometrics, geometric_map, additional_kwargs=extra_args)

def build_noise_transforms(cfg):
    noise_cfg = cfg.train.noise_transforms
    if not noise_cfg.get("enable", False):
        return []

    noise_map = {
        "MultiplicativeNoise": A.MultiplicativeNoise,
        "ISONoise": A.ISONoise,
        "ImageCompression": A.ImageCompression,
    }

    return build_transforms(noise_cfg, noise_map)

def apply_and_plot(image_path, output_dir, transforms):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig_cols = 4
    fig_rows = (len(transforms) + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(16, fig_rows * 3))
    axes = axes.flatten()

    for idx, (title, aug) in enumerate(transforms):
        print(f"Applying {title} ...")
        augmented = aug(image=image)['image']
        axes[idx].imshow(augmented)
        axes[idx].set_title(title)
        axes[idx].axis('off')

    for idx in range(len(transforms), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_augmentations.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # Important to free memory

def batch_process(input_folder, output_folder, cfg):
    image_paths = sorted(load_images_from_folder(input_folder))
    output_dir = Path(output_folder)
    
    
    transforms = [("Original", A.NoOp())]  # Add once, globally
    if cfg.train.photometrics_transforms.enable:
        transforms += build_photometric_transforms(cfg)
    if cfg.train.geometrics_transforms.enable:
        transforms += build_geometric_transforms(cfg)
    if cfg.train.noise_transforms.enable:
        transforms += build_noise_transforms(cfg)

    for img_path in image_paths:
        print(f"Processing {img_path.name}...")
        apply_and_plot(img_path, output_dir, transforms)
        exit()

if __name__ == "__main__":
    # === Usage ===
    input_folder = 'projects/hairy_vetch_001/inference/data/images'  # Update this path
    output_folder = 'data/aug_plots'  # Folder to save subplot images

    cfg = OmegaConf.load('conf/augment/color_invariant.yaml')

    batch_process(input_folder, output_folder, cfg)
