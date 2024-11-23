from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint
from src.utils.class_groupings import CLASSGROUPS

log = logging.getLogger(__name__)

def plot_class_distributions(class_frequencies, title, group_name, output_dir=None):
    classes = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())

    clsint_to_str = {cls_value: str_cls for str_cls, cls_dict in CLASSGROUPS[group_name].items() for cls_value in classes if cls_value == cls_dict["values"]}
    
    class_names = [clsint_to_str[cls] for cls in classes]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, frequencies)
    plt.xlabel('Classes')
    plt.ylabel('Frequencies')
    plt.title(title)
    plt.xticks(class_names)  # Ensure x-axis labels are whole values
    if output_dir:
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(output_dir / filename)

def get_unique_values(mask_files):
    unique_values = set()
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        unique_values.update(np.unique(mask))
    return unique_values

def get_class_frequencies(mask_files, unique_values):
    class_frequencies = {value: 0 for value in unique_values}
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        unique_in_mask = np.unique(mask)
        for value in unique_in_mask:
            if value in unique_values:
                class_frequencies[value] += 1
    return class_frequencies

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    train_remapped_mask_dir = Path(cfg.paths.split_data.train_remapped_mask_dir)
    val_remapped_mask_dir = Path(cfg.paths.split_data.val_remapped_mask_dir)
    test_remapped_mask_dir = Path(cfg.paths.split_data.test_remapped_mask_dir)
    output_dir = Path(cfg.paths.model_dir).parent / "data_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = cfg.task.remap_masks.group_name
    
    train_mask_files = list(train_remapped_mask_dir.glob("*.png"))
    val_mask_files = list(val_remapped_mask_dir.glob("*.png"))
    test_mask_files = list(test_remapped_mask_dir.glob("*.png"))

    log.info(f"Number of training mask files: {len(train_mask_files)}")
    log.info(f"Number of validation mask files: {len(val_mask_files)}")
    log.info(f"Number of test mask files: {len(test_mask_files)}")

    train_unique_values = get_unique_values(train_mask_files)
    val_unique_values = get_unique_values(val_mask_files)
    test_unique_values = get_unique_values(test_mask_files)

    log.info(f"Unique values in training masks: {train_unique_values}")
    log.info(f"Unique values in validation masks: {val_unique_values}")
    log.info(f"Unique values in test masks: {test_unique_values}")

    train_class_frequencies = get_class_frequencies(train_mask_files, train_unique_values)
    val_class_frequencies = get_class_frequencies(val_mask_files, val_unique_values)
    test_class_frequencies = get_class_frequencies(test_mask_files, test_unique_values)
    
    if cfg.task.data_stats.ignore_background:
        train_class_frequencies.pop(0, None)
        val_class_frequencies.pop(0, None)
        test_class_frequencies.pop(0, None)

    log.info(f"Class frequencies in training masks: {train_class_frequencies}")
    log.info(f"Class frequencies in validation masks: {val_class_frequencies}")
    log.info(f"Class frequencies in test masks: {test_class_frequencies}")

    plot_class_distributions(train_class_frequencies, 'Training Mask Class Distribution',group_name=group_name, output_dir=output_dir)
    plot_class_distributions(val_class_frequencies, 'Validation Mask Class Distribution',group_name=group_name, output_dir=output_dir)
    plot_class_distributions(test_class_frequencies, 'Test Mask Class Distribution',group_name=group_name, output_dir=output_dir)

    combined_class_frequencies = {value: train_class_frequencies.get(value, 0) + val_class_frequencies.get(value, 0) + test_class_frequencies.get(value, 0) for value in set(train_class_frequencies) | set(val_class_frequencies) | set(test_class_frequencies)}

    plot_class_distributions(combined_class_frequencies, 'Combined Mask Class Distribution', group_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
