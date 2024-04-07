import argparse
import logging
import math
from copy import deepcopy
from pathlib import Path
import random
import torch
import torch.nn as nn

from models.dynamic_common import Upsample, Conv, DYBatchNorm2d, DYConv2d, Dynamic_Conv, Dynamic_Bottleneck, Dynamic_SPP, Focus, Dynamic_BottleneckCSP, Dynamic_Concat
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)



logger = logging.getLogger(__name__)


class Dynamic_Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Dynamic_Detect, self).__init__()
        self._anchors = anchors
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList()  # output conv
        # nn.Conv2d(x, self.no * self.na, 1)
        for x in ch:
            self.m.append(DYConv2d(x, self.no * self.na, 1))
    def forward(self, x, in_group):
        if getattr(self, 'thop', False):
            x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i], _ = self.m[i](x[i], in_group[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # self.no    number of classes +5; self.na number of anchors
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x), []
        # return (torch.cat(z, 1), x), [1]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.yolo import Detect as _Detect
        in_group = [_[1] for _ in in_group]
        _layer = _Detect(nc=self.nc, anchors=self._anchors, ch=in_group)
        _layer.i, _layer.f = self.i, self.f
        _layer.anchors, _layer.stride = self.anchors, self.stride
        # load weight
        for i, m, n in zip(in_group, _layer.m, self.m):
            m.weight.data.copy_(n.weight[:, :i, :, :].data)
            m.bias.data.copy_(n.bias.data)
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.yolo import Detect as _Detect
        return [self.f, 1, _Detect.__name__, ['nc', 'anchors']]


class Model(nn.Module):
    def __init__(self,
                 cfg='yolov5s.yaml',
                 ch=3,
                 nc=None,
                 width_choice=[32/64, 48/64, 64/64],
                 depth_choice=[2/3, 3/3]): 
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.width_choice = width_choice
        self.depth_choice = depth_choice
        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        self.apply(lambda m: setattr(m, 'out_width_mult', 80 / 80))
        if isinstance(m, Dynamic_Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def sample_width(self, mode='random'):
        if mode == 'random':
            sampled_width = [random.choice(self.width_choice) for _ in range(22)]
            sampled_width.append(1.0)
        elif mode == 'max':
            sampled_width = [self.width_choice[-1] for _ in range(22)]
            sampled_width.append(1.0)
        elif mode == 'min':
            sampled_width = [self.width_choice[0] for _ in range(22)]
            sampled_width.append(1.0)
        elif mode == 'mid':
            sampled_width = [self.width_choice[1] for _ in range(22)]
            sampled_width.append(1.0)            
        return sampled_width

    def sample_depth(self, mode='random'):
        if mode == 'random':
            sampled_depth = [random.choice(self.depth_choice) for _ in range(8)]
        elif mode == 'max':
            sampled_depth = [self.depth_choice[-1] for _ in range(8)]
        elif mode == 'min':
            sampled_depth = [self.depth_choice[0] for _ in range(8)]
        return sampled_depth

    def set_active_subnet(self, width=None, depth=None):
        depth_i = 0
        for m in self.model:
            if m.i > 1:
                index = m.i
                m.apply(lambda m: setattr(m, 'out_width_mult', width[index-2]))
                if isinstance(m, Dynamic_BottleneckCSP):
                    m.sample_depth = depth[depth_i]
                    depth_i += 1

    def get_active_subnet(self, width=None, depth=None):
        y, in_group = [], []
        in_groups = []
        depth_i = 0
        subnet_layers = []
        
        is_training = self.training
        if is_training:
            self.eval()
        x = torch.zeros(1, 3, 640, 640).type_as(next(self.parameters()))
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                in_group = [in_groups[j] for j in m.f]
            if m.i > 1 and (width is not None or depth is not None):
                if width is not None:
                    index = m.i
                    m.apply(lambda m: setattr(m, 'out_width_mult', width[index-2]))
                if (depth is not None) and isinstance(m, Dynamic_BottleneckCSP):
                    m.sample_depth = depth[depth_i]
                    depth_i += 1
            subnet_layers.append(m.get_active_subnet(in_group, width[m.i-2], depth[depth_i-1]))
            x, in_group = m(x, in_group)  # run

            in_groups.append(in_group)
            y.append(x if m.i in self.save else None)  # save output
        if is_training:
            self.train()
        return subnet_layers

    def get_active_yaml(self, width=None, depth=None):
        _yaml = self.yaml.copy()
        _yaml['depth_multiple'] = 1
        _yaml['width_multiple'] = 1
        _detector = []
        depth_i = 0
        for m in self.model:
            if m.i > 1:
                if isinstance(m, Dynamic_BottleneckCSP):
                    _detector.append(m.get_active_yaml(width[m.i-2], depth[depth_i]))
                    depth_i += 1
                else:
                    _detector.append(m.get_active_yaml(width[m.i-2]))
            else:
                _detector.append(m.get_active_yaml())
        _yaml['backbone'] = _detector[:10]
        _yaml['head'] = _detector[10:]
        return _yaml

    def forward(self,
                x,
                augment=False,
                profile=False,
                width=None,
                depth=None):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile, width=width, depth=depth)  # single-scale inference, train

    def forward_once(self, x, profile=False, width=None, depth=None):
        y, dt, in_group = [], [], []  # outputs
        in_groups = []
        depth_i = 0
#        for m in self.model:
        for layer_id, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                in_group = [in_groups[j] for j in m.f]
            if profile:
                try:
                    import thop
                    import copy
                    m.thop = True
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x, in_group)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            if m.i > 1 and (width is not None or depth is not None):
                if width is not None:
                    index = m.i
                    m.apply(lambda m: setattr(m, 'out_width_mult', width[index-2]))
                if (depth is not None) and isinstance(m, Dynamic_BottleneckCSP):
                    m.sample_depth = depth[depth_i]
                    depth_i += 1
            x, in_group = m(x, in_group)  # run

            in_groups.append(in_group)
            y.append(x if m.i in self.save else None)  # save output

            # SHAOHUA: debug info
#            if layer_id > 1 and layer_id < 24:
#                print('layer {} {} {}'.format(layer_id, m.out_width_mult, x.shape[1]))
        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Dynamic_Conv, Dynamic_Bottleneck, Dynamic_SPP, Focus, Dynamic_BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [Dynamic_BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is DYBatchNorm2d:
            args = [ch[f]]
        elif m is Dynamic_Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Dynamic_Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='dynamic_yolov5x.yaml', help='model.yaml')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    img = torch.rand(8, 3, 640, 640).to(device)
    y = model(img, profile=False)