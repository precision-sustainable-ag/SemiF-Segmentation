import os
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import torch
from typing import Optional, List, Union

import logging

log = logging.getLogger(__name__)

# Dataset Class
class SegmentationDataset(BaseDataset):
    """Custom Dataset for Image Segmentation."""
    
    CLASSES = ["background", "monocot", "dicot"]

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = sorted([Path(f).stem for f in os.listdir(images_dir)])
        self.images_fps = [os.path.join(images_dir, f"{image_id}.jpg") for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, f"{image_id}.png") for image_id in self.ids]
        
        
        self.background_class = self.CLASSES.index("background")
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.class_values = (
            [self.CLASSES.index(cls.lower()) for cls in classes]
            if classes
            else list(range(len(self.CLASSES)))
        )
        
        self.class_map = {self.background_class: 0}
        
        self.class_map.update({v: i + 1 for i, v in enumerate(self.class_values) if v != self.background_class})
        
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask, self.masks_fps[i]

    def __len__(self):
        return len(self.ids)



class Dataset(BaseDataset):
    def __init__(
        self, 
        images_dir: Union[Path, List[Path]], 
        masks_dir: Union[Path, List[Path]], 
        augmentation=None, 
        normalize_image=False, 
        mean=None, 
        std=None
    ):
        """
        Dataset for image segmentation tasks.

        Args:
            images_dir (Union[Path, List[Path]]): Path to images directory or a list of image paths.
            masks_dir (Union[Path, List[Path]]): Path to masks directory or a list of mask paths.
            augmentation (callable, optional): Augmentation to apply to image-mask pairs.
            normalize_image (bool): Whether to normalize images.
            mean (list or np.array, optional): Mean for normalization.
            std (list or np.array, optional): Standard deviation for normalization.
        """
        if isinstance(images_dir, list):
            self.images_fps = images_dir
            self.masks_fps = masks_dir
        elif images_dir.is_dir() and masks_dir.is_dir():
            self.images_fps = list(Path(images_dir).glob("*.jpg"))
            self.masks_fps = [Path(masks_dir, f"{img.stem}.png") for img in self.images_fps]
        else:
            raise ValueError("images_dir and masks_dir must either be lists or valid directories.")

        self.augmentation = augmentation
        self.normalize_image = normalize_image
        self.mean = np.array(mean) if mean is not None else np.array([0.0, 0.0, 0.0])
        self.std = np.array(std) if std is not None else np.array([1.0, 1.0, 1.0])

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.images_fps[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load mask
        mask = cv2.imread(str(self.masks_fps[idx]), cv2.IMREAD_GRAYSCALE)

        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Normalize and convert to tensor
        image = self._preprocess_image(image)
        mask = self._preprocess_mask(mask)

        return image, mask, self.masks_fps[idx].stem

    def __len__(self):
        return len(self.images_fps)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the image: normalize and convert to tensor.

        Args:
            image (np.ndarray): The input image in HWC format.

        Returns:
            torch.Tensor: The processed image in CHW format.
        """
        image = image / 255.0  # Scale to [0, 1]
        if self.normalize_image:
            image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(image).float()

    def _preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        Preprocess the mask: convert to tensor.

        Args:
            mask (np.ndarray): The input mask in HW format.

        Returns:
            torch.Tensor: The processed mask.
        """
        if mask.ndim == 2:  # Add channel dimension if single-channel
            mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(mask).long()


class DatasetOLD(BaseDataset):
    def __init__(self, images_dir : Union[Path,List], masks_dir: Union[Path,List], augmentation=None, img2tensor=False, normalize_image=False, mean=None, std=None):
        # Get the image IDs
        if isinstance(images_dir, list):
            self.ids = [x.stem for x in images_dir]
            self.images_fps = images_dir
            self.masks_fps = masks_dir
        
        elif images_dir.is_dir() and masks_dir.is_dir():

            self.ids = [x.stem for x in Path(images_dir).glob("*.jpg")]
            self.images_fps = [Path(image) for image in images_dir.glob("*.jpg")]
            self.masks_fps = [Path(masks_dir, id + ".png") for id in self.ids]

        self.augmentation = augmentation

        self.img2tensor = img2tensor
        
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        if self.normalize_image:
            image = (image / 255.0 - self.mean) / self.std

        if self.img2tensor:
            image = self._img2tensor(image)
            print(image.shape)
            print(image.dtype)
            print(image.min(), image.max())
            mask = self._img2tensor(mask)
        
        else:
            image = image.transpose(2, 0, 1)
        
        return image, mask, self.masks_fps[i].stem

    def __len__(self):
        return len(self.ids)
    
    def _img2tensor(self, img, dtype: np.dtype = np.float32):
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        # img = np.transpose(img, (2, 0, 1))
        img = img.transpose(2, 0, 1)
        
        return torch.from_numpy(img.astype(dtype, copy=False))
    
