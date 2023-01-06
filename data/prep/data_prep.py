from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from natsort import index_natsorted


class PrepData:

    def __init__(self,
                 job_name,
                 csv,
                 cutoutdir,
                 ht=720,
                 wid=720,
                 is_resize=False):
        self.job_name = Path(job_name)
        self.job_name.mkdir(exist_ok=True, parents=True)

        self.imagedir = Path(job_name, "images")
        self.imagedir.mkdir(exist_ok=True, parents=True)

        self.maskdir = Path(job_name, "labels")
        self.maskdir.mkdir(exist_ok=True, parents=True)

        self.cutoutdir = cutoutdir

        self.df = self.get_cropouts(csv)

        self.paths = list(self.df["temp_path"])

        self.class_data = self.get_class_data()

        self.is_resize = is_resize

        self.ht, self.wid = ht, wid

    def get_cropouts(self, csv):
        df = pd.read_csv(csv)
        df["temp_path"] = self.cutoutdir + "/" + df["cutout_path"]
        exists = [x for x in list(df["temp_path"]) if Path(x).exists()]
        df = df[df.temp_path.isin(exists)]
        return df

    def get_class_data(self):
        # rs = self.df["r"]
        # gs = self.df["g"]
        # bs = self.df["b"]
        # cname = self.df["common_name"]
        # class_id = self.df["class_id"]
        # palette = [(int(cls_id), cn, r, g, b)
        #            for cls_id, cn, r, g, b in zip(class_id, cname, rs, gs, bs)]
        classes = self.df["common_name"].unique()
        # palette = list(dict.fromkeys(palette))
        classes = sorted(classes, reverse=False)
        # palette = [list(x) for x in palette]
        return classes

    def write_classes_data(self):
        # # Write class:species info

        with open(Path(self.job_name, "classes.txt"), 'w') as f:
            f.write(f"background\n")
            for cls in self.class_data:
                # for item in cls_id:
                f.write(f"{str(cls)}\n")
                # f.write(f"\n")

    def remap_labels(self, path_class):
        path = path_class[0]
        class_id = path_class[1]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if self.is_resize:
            mask = cv2.resize(mask, (self.ht, self.wid))

        mask = np.where(mask != 0, class_id, 0).astype(np.uint8)

        dst = Path(self.job_name, "labels",
                   Path(path).name.replace("_mask", ""))
        cv2.imwrite(str(dst), mask)

    def save_images(self, path):
        # print(path)
        src = str(path).replace(".png", ".jpg")
        dst = Path(self.job_name, "images", Path(src).name)

        if self.is_resize:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (self.ht, self.wid))
            cv2.imwrite(str(dst), img)
        else:
            shutil.copy2(src, dst)

    def copy_data(self):
        for crop_src in self.paths:
            parent = Path(crop_src).parent
            crop_name = Path(crop_src).stem + ".jpg"
            crop_src = Path(parent, crop_name)
            mask_src = str(crop_src).replace(".jpg", "_mask.png")

            crop_dst = Path(self.imagedir, crop_name)
            mask_dst = Path(self.maskdir,
                            str(crop_name).replace(".jpg", ".png"))

            shutil.copy2(crop_src, crop_dst)
            shutil.copy2(mask_src, mask_dst)

    def train_val_split(self):
        # # Train/val split
        imgs = [x.stem for x in Path(self.job_name, "images").glob("*.jpg")]
        masks = [x.stem for x in Path(self.job_name, "labels").glob("*.png")]

        img_train, img_test, _, _ = train_test_split(imgs,
                                                     masks,
                                                     test_size=0.025,
                                                     random_state=42)

        Path(self.job_name, "splits").mkdir(exist_ok=True, parents=True)
        train_txt = Path(self.job_name, "splits", "train.txt")
        val_txt = Path(self.job_name, "splits", "val.txt")

        with open(train_txt, 'w') as f:
            f.writelines(line + '\n' for line in img_train)
        with open(val_txt, 'w') as f:
            # select last 1/5 as train set
            f.writelines(line + '\n' for line in img_test)

    def resize(self, path):
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)


job_name = "all_species_720x720_64K"
csv = "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/summer_weeds_2022/2022-12-20/17-48-33/filtered_cutouts.csv"
cutoutdir = "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/semifield-cutouts"

prep = PrepData(job_name, csv, cutoutdir, is_resize=True)

# df = prep.df.sample(n=500, random_state=42)  # For testing TODO remove
df = prep.df
prep.df = df
prep.write_classes_data()

masks = [(path.replace(".png", "_mask.png"), clsid)
         for path, clsid in zip(list(df["temp_path"]), list(df["class_id"]))
         if Path(path).exists()]

print(f"{cpu_count()} cpus counted")
with Pool(cpu_count()) as p:
    p.map(prep.save_images, list(df["temp_path"]))
p.close()
p.join()

print(f"{cpu_count()} cpus counted")
with Pool(cpu_count()) as p:
    p.map(prep.remap_labels, masks)
p.close()
p.join()

# prep.copy_data()
prep.train_val_split()
