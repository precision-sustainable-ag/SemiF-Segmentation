# SemiF-Segmentation

### **Script Descriptions**

#### **1. `move_fullsized_data.py`**
This script manages the movement of full-sized image and mask files from multiple storage locations to a centralized directory. It ensures that both image and mask files are matched correctly, filters out unmatched files (optionally deleting them), and organizes the data into a standardized structure. The script supports copying using multithreading.

---

#### **2. `grid_crop.py`**
This script processes a dataset of images and corresponding masks by cropping them into smaller, fixed-size tiles. It ensures that only tiles containing relevant data (non-zero mask values) are retained. This script supports multiprocessing.

---

#### **3. `train_val_test_split.py`**
This script splits a dataset of images and masks into training, validation, and optional test sets. The split proportions are configurable, and the files are copied into their respective directories for each set. The script ensures that all splits maintain proper image-mask correspondence and supports concurrent file copying.

---

#### **4. `remap_masks.py`**
This script remaps mask values in image files based on predefined class group mappings. It processes train, validation, and test splits separately, converting mask values into simplified categories for specific tasks. The remapped masks are saved into corresponding output directories. The script can process files sequentially or concurrently.

---

### Class Format Example (Palmer amaranth)
```json
"AMPA": {
            "class_id": 1,
            "USDA_symbol": "AMPA",
            "EPPO": "AMAPA",
            "group": "dicot",
            "class": "Magnoliopsida",
            "subclass": "Caryophyllidae",
            "order": "Caryophyllales",
            "family": "Amaranthaceae",
            "genus": "Amaranthus",
            "species": "palmeri",
            "common_name": "Palmer amaranth",
            "authority": "Watson",
            "growth_habit": "forb/herb",
            "duration": "annual",
            "collection_location": "NC",
            "category": "warm season weed",
            "collection_timing": "summer",
            "link": "https://plants.usda.gov/home/plantProfile?symbol=AMPA",
            "note": null,
            "hex": "#1d686e",
            "rgb": [
                29,
                104,
                110
            ]   ...
}
```
