from mmseg.apis import set_random_seed
from mmseg.utils import get_device

# dataset settings
dataset_type = 'PSAWeedsDataset'
data_root = 'all_species_720x720_64K'
img_dir = 'images'
ann_dir = 'labels'
# workdir = './work_dirs/example_reset_classes'
num_classes = 20
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (720, 720)

_base_ = '/home/mkutuga/SemiF-Segmentation/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
# Since we use only one GPU, BN is used instead of SyncBN
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
# modify num classes of the model in decode/auxiliary head
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg, num_classes=num_classes),
    auxiliary_head=dict(norm_cfg=norm_cfg, num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(720, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 720),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split='splits/train.txt',
        classes=data_root + "/classes.txt",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='splits/val.txt',
        classes=data_root + "/classes.txt",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split='splits/val.txt',
        classes=data_root + "/classes.txt",
    ))

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
runner = dict(max_iters=100000)
log_config = dict(interval=10)
evaluation = dict(interval=300)
checkpoint_config = dict(interval=300)

# Set seed to facitate reproducing the result
seed = 0
set_random_seed(0, deterministic=False)
gpu_ids = range(1)
device = get_device()
