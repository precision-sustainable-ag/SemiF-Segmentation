from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import torch
from typing import List, Union

import logging

from src.utils.instance_annotations import id_map_to_instances, load_instance_annotation

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


def masks_to_instances(mask: np.ndarray, min_area: int = 0, max_instances: int | None = None) -> dict:
    """Instance targets (torchvision detection format) from a semantic mask:
    one instance per 8-connected blob of each class value > 0, largest first.

    Args:
        mask: (H, W) class indices, 0 = background.
        min_area: blobs with fewer pixels are dropped.
        max_instances: keep at most this many (the largest), None = all.
    Returns:
        {"boxes": (N, 4) float xyxy, "labels": (N,) int64 class values,
         "masks": (N, H, W) uint8}
    """
    height, width = mask.shape
    found = []  # (area, class value, component index, x, y, w, h, labels image)
    for value in np.unique(mask):
        if value == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == value).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                found.append((int(area), int(value), i, x, y, w, h, labels))
    found.sort(key=lambda f: -f[0])
    if max_instances is not None:
        found = found[:max_instances]

    masks = np.zeros((len(found), height, width), dtype=np.uint8)
    boxes = np.zeros((len(found), 4), dtype=np.float32)
    for k, (_, _, i, x, y, w, h, labels) in enumerate(found):
        masks[k] = labels == i
        boxes[k] = (x, y, x + w, y + h)  # pixel edges: the box covers columns x .. x + w - 1
    return {
        "boxes": torch.from_numpy(boxes),
        "labels": torch.tensor([f[1] for f in found], dtype=torch.int64),
        "masks": torch.from_numpy(masks),
    }


class InstanceDataset(Dataset):
    """Dataset for the instance-segmentation models: the same images,
    augmentation and normalization as Dataset. Instances come from
    - the semantic mask, one per connected blob (masks_to_instances), or
    - instance files in instances_dir (src.utils.instance_annotations, e.g.
      SAM 3 pseudo-instances from src/preprocessing/pseudo_instances.py);
      tiles without one are left out.
    Returns (image, target, stem), where target also holds the semantic mask
    as "semantic_mask" (H, W) for the semantic metrics, and, from instance
    files, "scores" and "annotation_source" (provenance) per instance."""

    def __init__(self, *args, min_area: int = 0, max_instances: int | None = None,
                 instances_dir: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_area = min_area
        self.max_instances = max_instances
        self.instances_dir = Path(instances_dir) if instances_dir is not None else None
        if self.instances_dir is not None:
            pairs = [(img, msk) for img, msk in zip(self.images_fps, self.masks_fps)
                     if (self.instances_dir / f"{Path(msk).stem}.png").exists()]
            skipped = len(self.images_fps) - len(pairs)
            if skipped:
                log.warning("%d of %d tiles have no instance file in %s; leaving them out",
                            skipped, len(self.images_fps), self.instances_dir)
            self.images_fps = [p[0] for p in pairs]
            self.masks_fps = [p[1] for p in pairs]

    def __getitem__(self, idx):
        if self.instances_dir is None:
            image, mask, stem = super().__getitem__(idx)
            semantic = mask[0]
            target = masks_to_instances(semantic.numpy().astype(np.uint8), self.min_area, self.max_instances)
            target["semantic_mask"] = semantic
            return image, target, stem

        stem = Path(self.masks_fps[idx]).stem
        image = cv2.cvtColor(cv2.imread(str(self.images_fps[idx])), cv2.COLOR_BGR2RGB)
        semantic = cv2.imread(str(self.masks_fps[idx]), cv2.IMREAD_GRAYSCALE)
        id_map, meta = load_instance_annotation(self.instances_dir, stem)
        if self.augmentation:  # the id map moves with the image exactly like the semantic mask
            augmented = self.augmentation(image=image, masks=[semantic, id_map])
            image, (semantic, id_map) = augmented["image"], augmented["masks"]
        target = id_map_to_instances(np.asarray(id_map), meta, self.min_area, self.max_instances)
        target["semantic_mask"] = self._preprocess_mask(np.asarray(semantic))[0]
        return self._preprocess_image(image), target, stem


def collate_instances(batch):
    """Batch InstanceDataset items as lists (images may differ in size, targets in length)."""
    images, targets, stems = zip(*batch)
    return list(images), list(targets), list(stems)
