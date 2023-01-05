import cv2
import numpy as np
from pathlib import Path
import random

path = "/home/mkutuga/mmsegmentation/example_data_same_size/labels"
paths = list(Path(path).rglob("*.png"))
idx = random.randint(0, 100)
img = cv2.imread(str(paths[idx]), 0)
print(img.shape)
print(np.unique(img))


class Test:
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

    CLASSES = ('background', 'Palmer amaranth', 'Common ragweed', 'Sicklepod',
               'Cocklebur', 'Large crabgrass', 'Goosegrass',
               'Broadleaf signalgrass', 'Purple nutsedge', 'Waterhemp',
               'Barnyardgrass', 'Jungle rice', 'Texas millet', 'Kochia',
               'Common sunflower', 'Ragweed parthenium', 'Johnsongrass',
               'Soybean', 'Smooth pigweed', 'Common lambsquarters',
               'Fall panicum', 'Jimson weed', 'Velvetleaf', 'Yellow foxtail',
               'Giant foxtail', 'Horseweed', 'Maize', 'unknown',
               'colorchecker', 'Hairy vetch', 'Winter pea', 'Crimson clover',
               'Red clover', 'Mustards', 'cultivated radish', 'Cereal rye',
               'Triticale', 'Winter wheat', 'Oats', 'Barley', 'Black oats')

    def __init__(self):

        # print(property_asel)
        class_indices = [
            list(self.ALL_CLASSES).index(x) if x != 'background' else 0
            for x in self.CLASSES
        ]
        print(class_indices)
        self.label_map = dict(
            zip([x for x in class_indices],
                [1 if x != 0 else 0 for x in range(0, 150)]))
        print(self.label_map)
        # print(self.label_map)


ts = Test()
