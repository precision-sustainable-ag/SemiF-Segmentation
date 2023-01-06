# SemiF-Segmentation

SemiF-Segmentation is a repository for training semantic segmentation models using agronomically relevant plant images! This repository is intended for scientific researchers and technical experts who are interested in using machine learning techniques to segment weeds, crops, and cover crops in agricultural images. 

In this repository, you will find a collection of data curation tools and approaches that are specifically designed for preparing and organizing agricultural image data, in particular the PSA SemiField-Image repository. In addition to these tools, we use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and their range of model architectures for training. We also provide examples and tutorials on how to use our tools and resources to solve challenging tasks in the field of agriculture, such as multispecies weed segementation.




### Table of Contents  
[Setup](#setup)  
[Classes](#classes)  
[Git subtree commands](#git-subtree-commands)  

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

[PSA Classes](mmsegmentation/mmseg/core/evaluation/class_names.py#L129)  

[PSA Palette](mmsegmentation/mmseg/core/evaluation/class_names.py#L141)

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


<br>


## [Git subtree commands](https://www.atlassian.com/git/tutorials/git-subtree)


### 1. Add subtree
```
git subtree add --prefix mmsegmentation git@github.com:open-mmlab/mmsegmentation.git master --squash
```

### 2. Add remote
```
git remote add -f mmsegmentation git@github.com:open-mmlab/mmsegmentation.git
```

### 3. Update subtree
```
git subtree pull --prefix mmsegmentation git@github.com:open-mmlab/mmsegmentation.git master --squash
```

### 4. Contributing back upstream 
We can freely commit our fixes to the sub-project in our local working directory now. When it’s time to contribute back to the upstream project, we need to fork the project and add it as another remote:

```
git remote add mmsegmentation ssh://git@bitbucket.org/durdn/vim-surround.git
```

#### Not sure what these do, not working for me. 
```
git fetch mmsegmentation master
git subtree pull --prefix mmsegmentation master --squash
```
