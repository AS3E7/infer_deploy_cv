import yaml


def yaml2netconfig(subnet_yaml, supernet_yaml):
    with open(subnet_yaml) as f:
        subnet = yaml.load(f, Loader=yaml.FullLoader)
    with open(supernet_yaml) as f:
        supernet = yaml.load(f, Loader=yaml.FullLoader)

    width = []
    depth = []
    for i, (sub, sup) in enumerate(zip(subnet['backbone'], supernet['backbone'])):
        if i <= 1: continue
        if sub[2] == 'BottleneckCSP':
            depth.append(sub[1]*subnet['depth_multiple'] / round(sup[1]*supernet['depth_multiple']))
        width.append(sub[3][0]*subnet['width_multiple'] / int(sup[3][0]*supernet['width_multiple']))

    for sub, sup in zip(subnet['head'], supernet['head']):
        if sub[2] in ['Concat', 'nn.Upsample', 'Detect']:
            width.append(1)
            continue
        if sub[2] == 'BottleneckCSP':
            depth.append(sub[1]*subnet['depth_multiple'] / round(sup[1]*supernet['depth_multiple']))
        width.append(sub[3][0]*subnet['width_multiple'] / int(sup[3][0]*supernet['width_multiple']))
    return width, depth


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subnets', type=str, nargs='+')
    parser.add_argument('--supernet', type=str, default='models/yolov5s-pro.yaml')
    parser.add_argument('--save-json', type=str)
    args = parser.parse_args()

    configs = []
    for _ in args.subnets:
        cfg = yaml2netconfig(_, args.supernet)
        configs.append(cfg)
        print('width:', cfg[0])
        print('depth:', cfg[1])
        print()
    if args.save_json:
        import json
        with open(args.save_json, 'w') as f:
            json.dump(configs, f)