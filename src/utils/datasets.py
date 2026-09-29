from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import torch
from typing import List, Union

import logging

log = logging.getLogger(__name__)


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
            raise ValueError(f"images_dir and masks_dir must either be lists or valid directories not {images_dir} and {masks_dir}")

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
