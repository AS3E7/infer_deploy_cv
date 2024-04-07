import argparse
import json
import glob
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from models.yolo import Model
from models.dynamic_yolo import Model as DModel
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    check_dataset, check_file, check_img_size, non_max_suppression,
    clip_coords, xywh2xyxy, box_iou, ap_per_class, set_logging,
    scale_coords, xyxy2xywh, coco80_to_coco91_class)
from utils.torch_utils import select_device, time_synchronized, intersect_dicts
from flops_counter import get_model_complexity_info
from utils.dynamic_general import bn_calibration_init


def calibrate_bn(model,
                 calib_loader,
                 iters):
    """Function to calibrate BN stats of a given model.
    """
    device = next(model.parameters()).device
    model.train()
    model.apply(lambda m: bn_calibration_init(m))
    cnt = 0
    while True:
        pbar = enumerate(calib_loader)
        num_batches = len(calib_loader)
        pbar = tqdm(pbar, total=num_batches)
        for i, (imgs, targets, paths, _) in pbar: 
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            model(imgs)
            cnt += 1
            if cnt > iters:
                break
        if cnt > iters:
            break
    return 


def reset_bn_params(model,
                    momentum,
                    eps):
    """Reset bn momentum and eps
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            m.eps = eps
    return


def extract_subnet(width, depth, opt, pth_path, cfg_path, model=None):
    device = select_device('0', batch_size=opt.batch_size)
    opt.data = check_file(opt.data)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if model is None:
        model = DModel(opt.cfg, ch=3, nc=data['nc']).to(device)
        state_dict = intersect_dicts(torch.load(opt.weights, map_location=device)['model'].state_dict(),
                                    model.state_dict()) 
        model.load_state_dict(state_dict, strict=False)

    _model = Model(opt.subnet_cfg, ch=3, nc=data['nc']).to(device)

    # get dataloader
    check_dataset(data)
    img_size = check_img_size(opt.img_size[0], s=model.stride.max())
    train_path = data['train']

    # model.set_active_subnet(width=width, depth=depth)
    layers = model.get_active_subnet(width=width, depth=depth)
    _yaml = model.get_active_yaml(width=width, depth=depth)
    _model.model = nn.Sequential(*layers)
    _model.yaml = _yaml
    _model.names = data['names']
    _model.to(device)

    if opt.calib_iters > 0:
        calib_loader = create_dataloader(train_path,
                                        img_size,
                                        opt.batch_size,
                                        model.stride.max(),
                                        opt,
                                        hyp=hyp,
                                        augment=True,
                                        cache=False,
                                        rect=False)[0]
        calibrate_bn(_model, calib_loader, opt.calib_iters)

    if opt.save_subnet_pth:
        torch.save(dict(model=_model, optimizer=None, epoch=-1),
                   osp.join(opt.save_subnet_pth, pth_path),
                   _use_new_zipfile_serialization=False)
    if opt.save_subnet_cfg:
        with open(osp.join(opt.save_subnet_cfg, cfg_path), 'w') as f:
            yaml.dump(_yaml, f, sort_keys=False)