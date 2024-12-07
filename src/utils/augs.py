import albumentations as A

def train_aug(cfg):
    """
    Dynamically builds training augmentations based on the configuration.
    """
    img_hw = tuple(cfg.augment.train.img_hw)
    train_transform = []

    # Add augmentations based on config flags
    if cfg.augment.train.horizontal_flip:
        train_transform.append(A.HorizontalFlip(p=0.5))
    
    if cfg.augment.train.shift_scale_rotate.enable:
        sc_cfg = cfg.augment.train.shift_scale_rotate
        train_transform.append(
            A.ShiftScaleRotate(
                scale_limit=sc_cfg.scale_limit,
                rotate_limit=sc_cfg.rotate_limit,
                shift_limit=sc_cfg.shift_limit,
                border_mode=sc_cfg.border_mode,
                p=sc_cfg.p,
            )
        )

    if cfg.augment.train.padding.enable:
        p_cfg = cfg.augment.train.padding
        train_transform.append(A.PadIfNeeded(min_height=img_hw[0], min_width=img_hw[1], p=p_cfg.p))
    
    if cfg.augment.train.random_crop.enable:
        rc_cfg = cfg.augment.train.random_crop
        train_transform.append(A.RandomCrop(height=img_hw[0], width=img_hw[1], p=rc_cfg.p))
    
    if cfg.augment.train.gauss_noise.enable:
        gn_cfg = cfg.augment.train.gauss_noise
        train_transform.append(A.GaussNoise(p=gn_cfg.p, noise_scale_factor=gn_cfg.noise_scale_factor))
    
    if cfg.augment.train.perspective.enable:
        p_cfg = cfg.augment.train.perspective
        train_transform.append(A.Perspective(p=p_cfg.p, scale=p_cfg.scale))
    
    # Add color transformations
    if cfg.augment.train.color_transforms.enable:
        color_augs = []
        for aug in cfg.augment.train.color_transforms.transforms:
            if aug == "CLAHE":
                color_augs.append(A.CLAHE(p=1))
            elif aug == "RandomBrightnessContrast":
                color_augs.append(A.RandomBrightnessContrast(p=1))
            elif aug == "RandomGamma":
                color_augs.append(A.RandomGamma(p=1))
        train_transform.append(A.OneOf(color_augs, p=cfg.augment.train.color_transforms.p))

    # Add distortion transformations
    if cfg.augment.train.optical_distortion.enable:
        od_cfg = cfg.augment.train.optical_distortion
        train_transform.append(A.OpticalDistortion(distort_limit=od_cfg.distort_limit, p=od_cfg.p))

    # Add blur transformations
    if cfg.augment.train.blur_transforms.enable:
        blur_augs = []
        for aug in cfg.augment.train.blur_transforms.transforms:
            if aug == "Sharpen":
                blur_augs.append(A.Sharpen(p=1))
            elif aug == "Blur":
                blur_augs.append(A.Blur(blur_limit=cfg.augment.train.blur_transforms.blur_limit, p=1))
            elif aug == "MotionBlur":
                blur_augs.append(A.MotionBlur(blur_limit=cfg.augment.train.blur_transforms.blur_limit, p=1))
        train_transform.append(A.OneOf(blur_augs, p=cfg.augment.train.blur_transforms.p))

    # Add hue/saturation transformations
    if cfg.augment.train.hue_saturation.enable:
        hs_cfg = cfg.augment.train.hue_saturation
        hue_augs = []
        for aug in cfg.augment.train.hue_saturation.transforms:
            if aug == "RandomBrightnessContrast":
                hue_augs.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=hs_cfg.brightness_limit, 
                        contrast_limit=hs_cfg.contrast_limit,
                        p=1
                        )
                    )
                
            elif aug == "HueSaturationValue":
                hue_augs.append(
                    A.HueSaturationValue(
                        hue_shift_limit=hs_cfg.hue_shift_limit,
                        sat_shift_limit=hs_cfg.sat_shift_limit,
                        val_shift_limit=hs_cfg.val_shift_limit,
                        p=1))
        train_transform.append(A.OneOf(hue_augs, p=cfg.augment.train.hue_saturation.p))

    return A.Compose(train_transform)

def val_aug(cfg):
    """
    Dynamically builds validation augmentations based on the configuration.
    """
    img_hw = tuple(cfg.augment.val.img_hw)
    val_transform = []

    if cfg.augment.val.padding.enable:
        val_transform.append(A.PadIfNeeded(min_height=img_hw[0], min_width=img_hw[1], p=cfg.augment.val.padding.p))
    
    return A.Compose(val_transform)
