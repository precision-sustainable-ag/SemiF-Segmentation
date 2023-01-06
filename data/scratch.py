import pandas as pd
import numpy as np
# (tensor([0.2739, 0.4262, 0.4329]), tensor([0.1510, 0.2053, 0.1836]))
df = pd.read_csv(
    "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/summer_weeds_2022/2022-12-20/02-24-54/filtered_cutouts.csv"
)

df["g_mean_norm"] = df["g_mean"] / 255
# print(df["r_mean_norm"])
import cv2

img = cv2.imread(
    "../../psa_images/Pipeline/SemiF-AnnotationPipeline/data/semifield-cutouts/MD_2022-06-21/MD_Row-12_1655828089_57.png"
)

rmean = df["r_mean"].sum() / len(df)
gmean = df["g_mean"].sum() / len(df)
bmean = df["b_mean"].sum() / len(df)

print(rmean / 255, gmean / 255, bmean / 255)
