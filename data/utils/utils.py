import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

####### PARAMS

device = torch.device('cpu')
num_workers = 8
image_size = 512
batch_size = 1


class LeafData(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # import
        path = self.paths[idx]
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            trans = transforms.Compose([transforms.ToTensor()])
            image = trans(image)

        return image


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    count = 0
    for data in dataloader:
        count += 1
        print(count)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


cutoutdir = "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/semifield-cutouts"
df = pd.read_csv(
    "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/summer_weeds_2022/2022-12-20/02-24-54/filtered_cutouts.csv"
)

df["temp_path"] = cutoutdir + "/" + df["cutout_path"]
paths = [x.replace(".png", ".jpg") for x in list(df["temp_path"])]
print(len(paths))
# dataset
image_dataset = LeafData(paths=paths, transform=transforms.ToTensor())

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

print(get_mean_and_std(image_loader))