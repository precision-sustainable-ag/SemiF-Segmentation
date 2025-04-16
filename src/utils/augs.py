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
    
    if cfg.augment.train.affine.enable:
        a_cfg = cfg.augment.train.affine
        train_transform.append(
            A.Affine(
                scale=a_cfg.scale, 
                translate_percent=a_cfg.translate_percent, 
                translate_px=a_cfg.translate_px, 
                rotate=a_cfg.rotate, 
                shear=a_cfg.shear, 
                interpolation=a_cfg.interpolation, 
                mask_interpolation=a_cfg.mask_interpolation, 
                fit_output=a_cfg.fit_output, 
                keep_ratio=a_cfg.keep_ratio, 
                rotate_method=a_cfg.rotate_method, 
                balanced_scale=a_cfg.balanced_scale, 
                border_mode=a_cfg.border_mode, 
                fill=a_cfg.fill, 
                fill_mask=a_cfg.fill_mask,
                p=a_cfg.p
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
                color_augs.append(A.CLAHE(p=.5))
            elif aug == "RandomGamma":
                color_augs.append(A.RandomGamma(p=.5))
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

    # Add noise transformations
    if cfg.augment.train.noise_transforms.enable:
        noise_augs = []
        for aug, vals in cfg.augment.train.noise_transforms.transforms.items():
            if aug == "MultiplicativeNoise":
                noise_augs.append(A.MultiplicativeNoise(multiplier=vals["multiplier"], per_channel=vals["per_channel"], p=vals["p"]))
            elif aug == "JpegCompression":
                noise_augs.append(A.JpegCompression(quality_lower=vals["quality_lower"], quality_upper=vals["quality_upper"], p=vals["p"]))
            elif aug == "Downscale":
                noise_augs.append(A.Downscale(scale_min=vals["scale_min"], scale_max=vals["scale_max"], p=vals["p"]))
            elif aug == "GridDistortion":
                noise_augs.append(A.GridDistortion(num_steps=vals["num_steps"], distort_limit=vals["distort_limit"], p=vals["p"]))
            elif aug == "ElasticTransform":
                noise_augs.append(A.ElasticTransform(alpha=vals["alpha"], sigma=vals["sigma"], alpha_affine=vals["alpha_affine"], p=vals["p"]))
            elif aug == "GaussNoise":
                noise_augs.append(A.GaussNoise(mean_range=(vals["mean_range_min"], vals["mean_range_max"]), per_channel=vals['per_channel'], p=vals["p"], noise_scale_factor=vals["noise_scale_factor"]))
            elif aug == "ISONoise":
                noise_augs.append(A.ISONoise(color_shift=(vals["color_shift_min"], vals["color_shift_max"]), intensity=(vals["intensity_min"], vals["intensity_max"]), p=vals["p"]))
        train_transform.append(A.OneOf(noise_augs, p=cfg.augment.train.noise_transforms.one_of_p))
    
    if cfg.augment.train.perspective.enable:
        p_cfg = cfg.augment.train.perspective
        train_transform.append(A.Perspective(p=p_cfg.p, scale=p_cfg.scale))

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
