from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from src.utils.class_groupings import CLASSGROUPS
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
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


def calculate_rgb_mean_std_shared(image_files, num_workers=4):
    """
    Calculate the RGB mean and standard deviation using a shared list for multiprocessing.

    Args:
        image_files (list): List of image file paths.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: Mean and standard deviation as numpy arrays.
    """
    # Use Manager to create a shared list
    with Manager() as manager:
        shared_list = manager.list()

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, image_file): image_file for image_file in image_files}

            # Collect results
            for future in tqdm(futures, desc="Processing images"):
                result = future.result()
                if result is not None:
                    shared_list.append(result)

        # Aggregate results from the shared list
        total_mean = np.zeros(3, dtype=np.float64)
        total_squared_mean = np.zeros(3, dtype=np.float64)
        total_pixels = 0

        for mean, squared_mean, num_pixels in shared_list:
            total_mean += mean * num_pixels
            total_squared_mean += squared_mean * num_pixels
            total_pixels += num_pixels

        # Calculate overall mean and std
        overall_mean = total_mean / total_pixels
        overall_std = np.sqrt((total_squared_mean / total_pixels) - overall_mean**2)

    return overall_mean, overall_std

def save_mean_std(mean, std, output_file):
    """
    Save mean and std to a text file.

    Args:
        mean (numpy.ndarray): RGB mean values.
        std (numpy.ndarray): RGB std values.
        output_file (Path): Path to the output file.
    """
    data = {"mean": mean.tolist(), "std": std.tolist()}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    log.info(f"Saved RGB mean and std to {output_file}")

def process_mask(mask_file):
    """
    Process a single mask to extract unique values and class frequencies.

    Args:
        mask_file (Path): Path to the mask file.

    Returns:
        tuple: Set of unique values and a dictionary of class frequencies.
    """
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    if mask is None:
        log.warning(f"Could not load mask: {mask_file}")
        return set(), {}

    unique_values, counts = np.unique(mask, return_counts=True)
    class_frequencies = dict(zip(unique_values, counts))
    return set(unique_values), class_frequencies


def process_masks(mask_files, num_workers=12):
    """
    Process multiple masks to extract unique values and class frequencies concurrently.

    Args:
        mask_files (list): List of mask file paths.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: Combined set of unique values and combined class frequencies dictionary.
    """
    combined_unique_values = set()
    combined_class_frequencies = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_mask, mask_files), total=len(mask_files), desc="Processing masks"))

    for unique_values, class_frequencies in results:
        combined_unique_values.update(unique_values)
        for value, count in class_frequencies.items():
            combined_class_frequencies[value] = combined_class_frequencies.get(value, 0) + count

    return combined_unique_values, combined_class_frequencies

def plot_class_distributions(class_frequencies, title, group_name, output_dir=None):
    classes = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())

    clsint_to_str = {cls_value: str_cls for str_cls, cls_dict in CLASSGROUPS[group_name].items() for cls_value in classes if cls_value == cls_dict["values"]}

    class_names = [clsint_to_str[cls] for cls in classes]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, frequencies)
    plt.xlabel('Classes')
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.ylabel('Frequencies')
    plt.title(title)
    plt.xticks(class_names)
    if output_dir:
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(output_dir / filename)



@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Starting data statistics calculation...")
    data_stats_dir = Path(cfg.paths.project_mode_dir) / "data" / "data_stats"
    split_dir = Path(cfg.paths.split_dir)
    train_image_dir = split_dir / "train" / "images"
    train_remapped_mask_dir = split_dir / "train" / "masks"
    val_remapped_mask_dir = split_dir / "val" / "masks"
    test_remapped_mask_dir = split_dir / "test" / "masks"
    output_dir = Path(cfg.paths.project_mode_dir) / "data" / "data_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = cfg.preprocess.remap_masks.group_name

    # Get image files for mean and std calculation
    train_image_files = list(train_image_dir.glob("*.jpg"))

    # Calculate and save RGB mean and std
    mean, std = calculate_rgb_mean_std_shared(train_image_files, num_workers=18)
    log.info(f"Calculated Mean: {mean}, Std: {std}")
    # Save results
    output_file = data_stats_dir / "rgb_mean_std.json"
    save_mean_std(mean, std, output_file)

    # Process masks
    train_mask_files = list(train_remapped_mask_dir.glob("*.png"))
    val_mask_files = list(val_remapped_mask_dir.glob("*.png"))
    test_mask_files = list(test_remapped_mask_dir.glob("*.png"))

    # Process masks
    log.info("Processing training masks...")
    train_unique_values, train_class_frequencies = process_masks(train_mask_files, num_workers=18)

    log.info("Processing validation masks...")
    val_unique_values, val_class_frequencies = process_masks(val_mask_files, num_workers=18)

    log.info("Processing test masks...")
    test_unique_values, test_class_frequencies = process_masks(test_mask_files, num_workers=18)
    
    if cfg.preprocess.data_stats.ignore_background:
        train_class_frequencies.pop(0, None)
        val_class_frequencies.pop(0, None)
        test_class_frequencies.pop(0, None)

    log.info(f"Unique values in training masks: {train_unique_values}")
    log.info(f"Class frequencies in training masks: {train_class_frequencies}")

    log.info(f"Unique values in validation masks: {val_unique_values}")
    log.info(f"Class frequencies in validation masks: {val_class_frequencies}")

    log.info(f"Unique values in test masks: {test_unique_values}")
    log.info(f"Class frequencies in test masks: {test_class_frequencies}")


    log.info(f"Class frequencies in training masks: {train_class_frequencies}")
    log.info(f"Class frequencies in validation masks: {val_class_frequencies}")
    log.info(f"Class frequencies in test masks: {test_class_frequencies}")

    plot_class_distributions(train_class_frequencies, 'Training Mask Class Distribution', group_name, output_dir)
    plot_class_distributions(val_class_frequencies, 'Validation Mask Class Distribution', group_name, output_dir)
    plot_class_distributions(test_class_frequencies, 'Test Mask Class Distribution', group_name, output_dir)


    combined_class_frequencies = {
        value: train_class_frequencies.get(value, 0) + val_class_frequencies.get(value, 0) + test_class_frequencies.get(value, 0)
        for value in set(train_class_frequencies) | set(val_class_frequencies) | set(test_class_frequencies)
    }

    plot_class_distributions(combined_class_frequencies, 'Combined Mask Class Distribution', group_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
