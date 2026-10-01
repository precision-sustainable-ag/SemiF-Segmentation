from pathlib import Path
from omegaconf import DictConfig
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor

log = logging.getLogger(__name__)


def process_image(image_file):
    """
    Process a single image to compute its pixel mean and squared mean.

    Args:
        image_file (Path): Path to the image file.

    Returns:
        tuple: Mean, squared mean, and number of pixels for the image.
    """
    image = cv2.imread(str(image_file))  # Load image with OpenCV (BGR)
    if image is None:
        log.warning(f"Could not load image: {image_file}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize to [0, 1]

    # Compute mean and squared mean per channel
    mean = np.mean(image, axis=(0, 1))
    squared_mean = np.mean(image**2, axis=(0, 1))
    num_pixels = image.shape[0] * image.shape[1]

    return mean, squared_mean, num_pixels


def calculate_rgb_mean_std(image_files, num_workers=4):
    """
    Calculate the RGB mean and standard deviation over a list of images.

    Args:
        image_files (list): List of image file paths.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: Mean and standard deviation as numpy arrays.
    """
    total_mean = np.zeros(3, dtype=np.float64)
    total_squared_mean = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_image, image_files, chunksize=16)
        for result in tqdm(results, total=len(image_files), desc="Processing images"):
            if result is None:
                continue
            mean, squared_mean, num_pixels = result
            total_mean += mean * num_pixels
            total_squared_mean += squared_mean * num_pixels
            total_pixels += num_pixels

    overall_mean = total_mean / total_pixels
    overall_std = np.sqrt((total_squared_mean / total_pixels) - overall_mean**2)
    return overall_mean, overall_std


def save_mean_std(mean, std, output_file):
    """
    Save mean and std to a JSON file.

    Args:
        mean (numpy.ndarray): RGB mean values.
        std (numpy.ndarray): RGB std values.
        output_file (Path): Path to the output file.
    """
    data = {"mean": mean.tolist(), "std": std.tolist()}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    log.info(f"Saved RGB mean and std to {output_file}")


def count_mask_pixels(mask_file):
    """
    Count pixels per mask value in a single mask.

    Args:
        mask_file (Path): Path to the mask file.

    Returns:
        dict: {mask value: pixel count}
    """
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    if mask is None:
        log.warning(f"Could not load mask: {mask_file}")
        return {}
    values, counts = np.unique(mask, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def count_split_pixels(mask_files, num_workers=12):
    """
    Pixel counts per mask value over all masks of a split.

    Args:
        mask_files (list): List of mask file paths.
        num_workers (int): Number of worker processes.

    Returns:
        dict: {mask value: pixel count}
    """
    totals = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for counts in tqdm(executor.map(count_mask_pixels, mask_files, chunksize=16), total=len(mask_files), desc="Processing masks"):
            for value, count in counts.items():
                totals[value] = totals.get(value, 0) + count
    return totals


def plot_class_distributions(pixel_counts, class_names, output_file):
    """
    Bar plot of the fraction of pixels in each class, per split.
    """
    splits = list(pixel_counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(splits), 1)
    for i, split in enumerate(splits):
        total = sum(pixel_counts[split].values()) or 1
        fractions = [pixel_counts[split].get(value, 0) / total for value in range(len(class_names))]
        ax.bar(np.arange(len(class_names)) + i * width, fractions, width, label=split)
    ax.set_xticks(np.arange(len(class_names)) + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Fraction of pixels")
    ax.set_title("Mask class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def main(cfg: DictConfig):
    log.info("Starting data statistics calculation...")
    output_dir = Path(cfg.paths.project_preprocess_dir) / "data" / "data_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(cfg.paths.split_dir)
    workers = cfg.preprocess.data_stats.workers

    # Calculate and save RGB mean and std over the training tiles
    train_image_files = list((split_dir / "train" / "images").glob("*.jpg"))
    mean, std = calculate_rgb_mean_std(train_image_files, num_workers=workers)
    log.info(f"Calculated Mean: {mean}, Std: {std}")
    save_mean_std(mean, std, output_dir / "rgb_mean_std.json")

    # Pixel counts per class and split
    pixel_counts = {}
    for split in ("train", "val", "test"):
        mask_files = list((split_dir / split / "masks").glob("*.png"))
        if mask_files:
            log.info(f"Processing {split} masks...")
            pixel_counts[split] = count_split_pixels(mask_files, num_workers=workers)

    class_names = list(cfg.train.class_names)
    unexpected = {value for counts in pixel_counts.values() for value in counts if value >= len(class_names)}
    if unexpected:
        log.warning(f"Mask values {sorted(unexpected)} have no entry in train.class_names {class_names}")

    for split, counts in pixel_counts.items():
        total = sum(counts.values())
        summary = ", ".join(f"{class_names[v] if v < len(class_names) else v}: {c / total:.1%}" for v, c in sorted(counts.items()))
        log.info(f"{split} pixels: {summary}")

    with open(output_dir / "class_pixel_counts.json", "w") as f:
        json.dump(pixel_counts, f, indent=4)
    plot_class_distributions(pixel_counts, class_names, output_dir / "class_distribution.png")
    log.info(f"Saved class pixel counts and plot to {output_dir}")
