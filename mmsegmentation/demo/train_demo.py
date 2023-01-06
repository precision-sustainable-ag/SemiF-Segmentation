import os
import matplotlib.patches as mpatches
import torch, torchvision
import mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

os.chdir("./mmsegmentation")

print(torch.__version__, torch.cuda.is_available())
print(mmseg.__version__)

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# ## Train a semantic segmentation model on a new dataset
# To train on a customized dataset, the following steps are necessary.
# 1. Add a new dataset class.
# 2. Create a config file accordingly.
# 3. Perform training and evaluation.

# ### Add a new dataset
# convert dataset annotation to semantic segmentation map
data_root = 'test23473'
img_dir = 'images'
ann_dir = 'labels'
# define class and plaette for better visualization
# After downloading the data, we need to implement `load_annotations` function in the new dataset class `StanfordBackgroundDataset`.

cls_txt = f"/home/mkutuga/mmsegmentation/demo/mmsegmentation/{data_root}/classes.txt"
# Read metadata
classes = list()
palette = list()
with open(cls_txt, 'r') as f:
    for line in f:
        line = line.rstrip().split(",")
        classes.append(line[1])
        pal = [int(line[2]), int(line[3]), int(line[4])]
        palette.append(pal)
print("Number of classes: ", len(classes))


@DATASETS.register_module()
class StanfordBackgroundDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class PSAWeedsDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


# ### Create a config file

cfg = Config.fromfile(
    'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')

# Since the given config is used to train PSPNet on the cityscapes dataset, we need to modify it accordingly for our new dataset.

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 41
cfg.model.auxiliary_head.num_classes = 41

# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (720, 720)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(720, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 720),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = f'./work_dirs/{data_root}'

cfg.runner.max_iters = 100000
cfg.log_config.interval = 10
cfg.evaluation.interval = 500
cfg.checkpoint_config.interval = 500

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(
    model, datasets, cfg, distributed=False, validate=True, meta=dict())

# # Inference with trained model

# import pandas as pd
# import cv2

# csv = "/home/psa_images/Pipeline/SemiF-InspectCutouts/data/summer_weeds_2022/summer_weeds_2022.csv"
# df = pd.read_csv(csv, low_memory=False)
# ampa = df[df["USDA_symbol"] == "AMPA"]

# img = "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/semifield-developed-images/NC_2022-09-07/images/NC_1662556392.jpg"
# # img = mmcv.imread('iccv09Data/images/6000124.jpg')
# img = mmcv.imread(img)
# img = cv2.resize(img, (2000, 1000))

# model.cfg = cfg
# result = inference_segmentor(model, img)
# plt.figure(figsize=(8, 6))
# show_result_pyplot(model, img, result, palette)
