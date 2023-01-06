# SemiF-Segmentation
SemiF-Segmentation provides a toolbox for developing models aimed at segmenting agronomically relevant plant images. This repo places an emphasis on data curation rather than archeticture development. We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) to apply the latest CNN and transformer approaches.

## Table of Contents  
### [Setup](#setup)  
### [Classes](#classes)  
### [Viz](#viz)

<br>

## Setup

### 1. Clone and move into this repo

```
git clone git@github.com:precision-sustainable-ag/SemiF-Segmentation.git
cd SemiF-Segmentation
```

### 2. [Install mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)


### 3. Create a `./data` folder in the `SemiF-Segmentation` project root

```
mkdir data
```

### 4. Download and place [`species_info.json`](https://github.com/precision-sustainable-ag/SemiF-AnnotationPipeline/blob/306f85ec966146c8adb985d5f82724a99990a3b9/data/semifield-utils/species_information/species_info.json) in `./data`



<br>

---
<br> 

## Classes

class_id 0 is background.

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

## Viz

[PSA Classes](mmsegmentation/mmseg/core/evaluation/class_names.py#L129)  
[PSA Palette](mmsegmentation/mmseg/core/evaluation/class_names.py#L141)


## TODOs:

1. Find dataset means and std


## Git subtree commands


### 1. Add subtree
```
git subtree add --prefix mmsegmentation git@github.com:open-mmlab/mmsegmentation.git master --squash
```

# 2. Add remote
```
git remote add -f mmsegmentation git@github.com:open-mmlab/mmsegmentation.git
```

# 3. Update subtree
```
git subtree pull --prefix mmsegmentation git@github.com:open-mmlab/mmsegmentation.git master --squash
```

## Contributing back upstream 
We can freely commit our fixes to the sub-project in our local working directory now. When it’s time to contribute back to the upstream project, we need to fork the project and add it as another remote:


git remote add mmsegmentation ssh://git@bitbucket.org/durdn/vim-surround.git


git fetch mmsegmentation master
git subtree pull --prefix mmsegmentation master --squash

