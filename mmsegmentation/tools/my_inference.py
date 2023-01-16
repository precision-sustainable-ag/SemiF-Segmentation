import sys
from pathlib import Path
from datetime import datetime
import argparse

import torch

sys.path.append("/home/mkutuga/SemiF-Segmentation/mmsegmentation")
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.utils import setup_multi_processes
from mmseg.datasets import build_dataset


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
    parser.add_argument('input', help='input image directory')
    parser.add_argument(
        '--output',
        default=f"output_{timestamp()}",
        help='output image directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    # Get images
    in_imgs = sorted(list(Path(args.input).glob("*.JPG")))
    # Set output path
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    # set multi-process settings
    setup_multi_processes(cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda:0')
    dataset = build_dataset(cfg.data.test)

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
    print(model.PALETTE)

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    for imgp in in_imgs:
        # test a single image
        result = inference_segmentor(model, imgp)
        ################## Create save file
        out_file = Path(out_dir, imgp.name)
        model.show_result(
            imgp, result, show=False, opacity=0.4, out_file=out_file)


if __name__ == '__main__':
    main()
