from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import mmcv
import os.path as osp
import argparse

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

import sys

sys.path.append("/home/mkutuga/mmsegmentation")

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


def timestamp():
    dt = datetime.timestamp(datetime.now())
    ts = datetime.fromtimestamp(dt)
    str_date_time = ts.strftime("%Y-%m-%d_%H:%M:%S")
    return str_date_time


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE


# build the model from a config file and a checkpoint file
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

############### Remove PALETTE Error
if not Path(checkpoint_file).exists():
    model = torch.load(old_checkpoint_file)
    model['meta']['PALETTE'] = palette
    torch.save(model, checkpoint_file)
    del model

############### Read csv
df = pd.read_csv(csv, low_memory=False)
df["temp_cropout_path"] = "/home/psa_images/Pipeline/SemiF-AnnotationPipeline/data/semifield-cutouts/" + df[
    "cutout_path"]
df = df[df["green_sum"] > 4000]
df = df.groupby(["common_name"]).sample(n=10)

for idx, row in df.iterrows():
    ######### Read image
    imgp = row["temp_cropout_path"].replace(".png", ".jpg")
    species = row["common_name"]
    if not Path(imgp).exists():
        continue

    ################## Create save file
    out_file = f"/home/mkutuga/mmsegmentation/assets/{timestamp()}_{species}_{Path(imgp).stem}_{model_name}_{backbone}_ {iters}_{dataset}.png"

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # test a single image
    result = inference_segmentor(model, imgp)

    model.show_result(
        imgp,
        result,
        # palette=get_palette('psa'),
        show=False,
        opacity=0.5,
        out_file=out_file)
