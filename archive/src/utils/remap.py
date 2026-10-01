from pathlib import Path
import cv2

import numpy as np
import shutil

image_dir = Path("hairy_vetch_crimson_clover") / "images"
mask_dir = Path("hairy_vetch_crimson_clover") / "masks"
mask_files = list(mask_dir.glob("*.png"))

output_mask_dir = Path("projects/SEMIF_512/preprocess/data/train_val_test_512x512/train/remapped_masks")
output_image_dir = Path("projects/SEMIF_512/preprocess/data/train_val_test_512x512/train/images")

for mask_file in mask_files:
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    
    
    # map the rgb mask to single channel grayscale where non zero pixels equal 29
    mask[mask != 1] = 1
    unique_values = np.unique(mask)
    # remove the 0 from unique values
    unique_values = unique_values[unique_values != 0]
    print(unique_values)
    
    output_mask_path = output_mask_dir / mask_file.name
    print(output_mask_path)
    # cv2.imwrite(str(output_mask_path), mask)

    input_image_path = image_dir / (mask_file.stem + ".jpg")
    output_image_path = output_image_dir / (mask_file.stem + ".jpg")
    print(input_image_path)
    print(output_image_path)
    # break
    # shutil.copy2(input_image_path, output_image_path)
    
    
    