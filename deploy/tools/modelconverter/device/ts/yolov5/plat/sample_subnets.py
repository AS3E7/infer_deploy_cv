import argparse
import json
import glob
from os import times

import torch
import torch.nn as nn
import yaml

import sys
sys.path.append("./")

from random import randint
from models.dynamic_yolo import Model as DModel
from models.yolo import Model
from utils.torch_utils import select_device, intersect_dicts
from flops_counter import get_model_complexity_info


def uniform_sample_subnet_config(width_map=(0.5, 0.75, 1), depth_map=(0.67, 1)):
    x = randint(0, 44)
    if x > 22:
        width_config = [2] * 22
        x = 44 - x
        while x > 0:
            i = randint(0, 21)
            if width_config[i] > 0:
                width_config[i] -= 1
                x -= 1
    else:
        width_config = [0] * 22
        while x > 0:
            i = randint(0, 21)
            if width_config[i] < 2:
                width_config[i] += 1
                x -= 1
    width_config = [width_map[_] for _ in width_config] + [1]

    x = randint(0, 8)
    depth_config = [0] * 8
    while x > 0:
        i = randint(0, 7)
        if depth_config[i] < 1:
            depth_config[i] += 1
            x -= 1
    depth_config = [depth_map[_] for _ in depth_config]

    return width_config, depth_config


def sample_subnets(model, counts, max_flops, min_flops, img_size=640):
    subnet_configs = []
    times = 0
    print('sampling subnets')
    print(' - max flops', max_flops)
    while len(subnet_configs) < counts:
        times += 1
        if times % 200 == 0:
            min_flops -= 1
            print(' - reduce min flops', min_flops)
        # width = model.sample_width('random')
        # depth = model.sample_depth('random')
        width, depth = uniform_sample_subnet_config()

        model.set_active_subnet(width, depth)
        macs, params = get_model_complexity_info(model, 
            (3, img_size, img_size),
            print_per_layer_stat=False,
            as_strings=False)
        if min_flops <= macs * 2 / 1e9 <= max_flops and (width, depth) not in subnet_configs:
            subnet_configs.append((width, depth))
    return subnet_configs


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-flops', type=float)
    parser.add_argument('--min-flops', type=float)
    parser.add_argument('--nc', type=int)
    parser.add_argument('--img-size', type=int)
    parser.add_argument('--counts', type=int)
    parser.add_argument('--save-dir', type=str)
    args = parser.parse_args()

    device = select_device('0', batch_size=16)

    # model = DModel('models/dynamic_yolov5s-pro-conv.yaml', ch=3, nc=args.nc).to(device)
    # state_dict = torch.load('weights/supermodel_s_conv_450e.pt',
    #                         map_location=device)['model'].state_dict()
    # state_dict = intersect_dicts(state_dict,
    #                              model.state_dict())
    # model.load_state_dict(state_dict, strict=False)
    # _model = Model('models/yolov5s-pro.yaml', ch=3, nc=args.nc).to(device)
    # subnet_configs = sample_subnets(model, args.counts, args.max_flops, args.min_flops, args.img_size)
    # os.makedirs(args.save_dir, exist_ok=True)
    # with open(os.path.join(args.save_dir, 'subnet_configs.json'), 'w') as f:
    #     json.dump(subnet_configs, f)
    # for i, (width, depth) in enumerate(subnet_configs):
    #     layers = model.get_active_subnet(width=width, depth=depth)
    #     _yaml = model.get_active_yaml(width=width, depth=depth)
    #     _model.model = nn.Sequential(*layers)
    #     _model.yaml = _yaml
    #     _model.to(device)
    #     with open(os.path.join(args.save_dir, 'subnet{}.yaml'.format(i)), 'w') as f:
    #         yaml.dump(_yaml, f, sort_keys=False)
    #     torch.save(dict(model=_model, epoch=-1, optimizer=None),
    #                os.path.join(args.save_dir, 'subnet{}.pth'.format(i)))

    ## sample test
    model = DModel('models/dynamic_yolov5s-pro-conv.yaml', ch=3, nc=args.nc).to(device)
    state_dict = torch.load('weights/supermodel_s_conv_450e.pt',
                            map_location=device)['model'].state_dict()
    subnet_configs = sample_subnets(model, 3, 24, 23)
    # import pdb; pdb.set_trace()