#Script for calculating subnet FLOPs
import os.path as osp
import sys
sys.path.append(osp.abspath('../'))
import argparse

from utils.torch_utils import select_device
from models.experimental import attempt_load
from flops_counter import get_model_complexity_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='get_subnet_flops.py')
    parser.add_argument('--weights', 
                        type=str,
                        required=True,
                        help='Weights to be loaded')
    parser.add_argument('--input-resolution',
                        type=int,
                        required=True,
                        help='Input resolution')
    parser.add_argument('--device', 
                        type=str,
                        default='0', 
                        help='CUDA device ID')
    parser.add_argument('--arc-file',
                        type=str,
                        default='subnet_arcs.txt',
                        help='File for subnet arcs')
   
    opt = parser.parse_args()

    device = select_device(opt.device, batch_size=1)

    with open(opt.arc_file) as f:
        lines = f.readlines()

    results = []

    width_list = []
    depth_list = []

    for i, line in enumerate(lines):
        line = line.strip() 
        line = line.split()        
        line = [float(x) for x in line]

        if i % 2 == 0:
            width_list.append(line)
        else:
            depth_list.append(line)

    for width, depth in zip(width_list, depth_list):
        model = attempt_load(opt.weights, map_location=device)   
        model.set_active_subnet(width=width, depth=depth)
        macs, params = get_model_complexity_info(model, 
            (3, opt.input_resolution, opt.input_resolution),
            print_per_layer_stat=False,
            as_strings=False)

        gflops = macs * 2 / 1e9

        results.append([width, depth, gflops])


    results = sorted(results, key=lambda res: res[-1])

    for item in results:
        print(*item[0])
        print(*item[1])
        print(item[2])
