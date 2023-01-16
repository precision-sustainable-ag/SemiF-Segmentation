import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class PSAWeedsDataset(CustomDataset):

    CLASS_ID = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40
    ]
    USDA_SYMBOL = [
        "background", "AMPA", "AMAR2", "SEOB4", "XAST", "DISA", "ELIN3",
        "URPL2", "CYRO", "AMTU", "ECCR", "ECCO2", "URTE2", "BASC5", "HEAN3",
        "PAHY", "SOHA", "GLMA4", "AMHY", "CHAL7", "PADI", "DAST", "ABTH",
        "SEPU8", "SEFA", "ERCA20", "ZEA", "plant", "colorchecker", "VIVI",
        "PISA6", "TRIN3", "TRPR2", "BRASS2", "RASA2", "SECE", "TRITI2", "TRAE",
        "AVSA", "HORDE", "AVST2"
    ]

    ALL_CLASSES = ('background', 'Palmer amaranth', 'Common ragweed',
                   'Sicklepod', 'Cocklebur', 'Large crabgrass', 'Goosegrass',
                   'Broadleaf signalgrass', 'Purple nutsedge', 'Waterhemp',
                   'Barnyardgrass', 'Jungle rice', 'Texas millet', 'Kochia',
                   'Common sunflower', 'Ragweed parthenium', 'Johnsongrass',
                   'Soybean', 'Smooth pigweed', 'Common lambsquarters',
                   'Fall panicum', 'Jimson weed', 'Velvetleaf',
                   'Yellow foxtail', 'Giant foxtail', 'Horseweed', 'Maize',
                   'unknown', 'colorchecker', 'Hairy vetch', 'Winter pea',
                   'Crimson clover', 'Red clover', 'Mustards',
                   'cultivated radish', 'Cereal rye', 'Triticale',
                   'Winter wheat', 'Oats', 'Barley', 'Black oats')

    CLASSES = (
        'background',
        'Palmer amaranth',
        'Common ragweed',
        'Sicklepod',
        'Cocklebur',
        'Large crabgrass',
        'Goosegrass',
        'Broadleaf signalgrass',
        'Purple nutsedge',
        'Waterhemp',
        'Barnyardgrass',
        'Jungle rice',
        'Texas millet',
        'Kochia',
        'Common sunflower',
        'Ragweed parthenium',
        'Johnsongrass',
        'Soybean',
        'Smooth pigweed',
        'Common lambsquarters',
        'Fall panicum',
        'Jimson weed',
        'Velvetleaf',
        'Yellow foxtail',
        'Giant foxtail',
        'Horseweed',
        'Maize',
        'unknown',
        'colorchecker',
        'Hairy vetch',
        'Winter pea',
        'Crimson clover',
        'Red clover',
        'Mustards',
        'cultivated radish',
        'Cereal rye',
        'Triticale',
        'Winter wheat',
        'Oats',
        'Barley',
        'Black oats',
    )

    PALETTE = [[0, 0, 0], [29, 104, 110], [228, 82, 241], [96, 229, 78],
               [144, 59, 226], [158, 227, 63], [93, 76, 231], [219, 233, 60],
               [189, 54, 214], [63, 172, 56], [145, 43, 186], [87, 235, 147],
               [227, 66, 198], [143, 225, 117], [90, 66, 193], [227, 202, 59],
               [99, 100, 226], [160, 184, 55], [179, 96, 232], [107, 163, 60],
               [171, 42, 159], [207, 234, 124], [142, 66, 176], [64, 190, 123],
               [230, 54, 160], [90, 237, 195], [232, 60, 43], [85, 236, 227],
               [231, 59, 88], [158, 227, 160], [97, 127, 247], [238, 166, 38],
               [57, 103, 197], [232, 105, 40], [94, 151, 230], [186, 58, 28],
               [93, 207, 233], [234, 66, 126], [68, 151, 94], [225, 114, 216],
               [64, 112, 24]]

    def __init__(self, split, **kwargs):
        super(PSAWeedsDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)

        # class_indices = [
        #     list(self.ALL_CLASSES).index(x) if x != 'background' else 0
        #     for x in self.CLASSES
        # ]

        # self.label_map = dict(
        #     zip([x for x in class_indices],
        # [1 if x != 0 else 0 for x in range(0, 150)]))

        assert osp.exists(self.img_dir) and self.split is not None