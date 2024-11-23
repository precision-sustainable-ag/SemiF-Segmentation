import os
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset

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
        remapped_mask = np.zeros_like(mask)

        for class_value, new_value in self.class_map.items():
            remapped_mask[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=remapped_mask)
            image, remapped_mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), remapped_mask

    def __len__(self):
        return len(self.ids)