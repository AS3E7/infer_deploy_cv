# This file contains modules common to various models
from io import BufferedRandom
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DYConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(DYConv2d, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias)
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.out_width_mult = 1.
        self.groups = groups

    def forward(self, input, in_group):
        if len(in_group) <= 2:
            self.in_channels_ = input.shape[1]
            self.out_channels_ = int(self.out_channels_max *
                                     self.out_width_mult)
            weight = self.weight[:self.out_channels_, :self.in_channels_, :, :]
            if self.bias is not None:
                bias = self.bias[:self.out_channels_]
            else:
                bias = self.bias
        else:
            max_channels = in_group[::2]
            recent_channels = in_group[1::2]
            start_index = [
                0 if i == 0 else sum(max_channels[:i])
                for i in range(len(max_channels))
            ]
            end_index = [
                start_index[i] + recent_channels[i]
                for i in range(len(max_channels))
            ]
            weight_tmp = []
            self.out_channels_ = int(self.out_channels_max *
                                     self.out_width_mult)
            for start, end in zip(start_index, end_index):
                weight_tmp.append(self.weight[:, start:end, :, :])
            weight = torch.cat(weight_tmp, dim=1)
            weight = weight[:self.out_channels_, :, :, :]
            if self.bias is not None:
                bias = self.bias[:self.out_channels_]
            else:
                bias = self.bias

        y = nn.functional.conv2d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)
        return y, [self.out_channels_max, y.shape[1]]


class DYBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, out_channels, eps=0.0001, momentum=0.1):
        super(DYBatchNorm2d, self).__init__(out_channels,
                                            affine=True,
                                            eps=eps,
                                            momentum=momentum,
                                            track_running_stats=True)
        self.out_channels_max = out_channels
        self.out_width_mult = 1.

    def forward(self, input, in_group):
        self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
        assert self.out_channels_ == input.shape[1]
        if len(in_group) <= 2:
            weight = self.weight[:self.out_channels_]
            bias = self.bias[:self.out_channels_]

        else:
            max_channels = in_group[::2]
            recent_channels = in_group[1::2]
            start_index = [
                0 if i == 0 else sum(max_channels[:i])
                for i in range(len(max_channels))
            ]
            end_index = [
                start_index[i] + recent_channels[i]
                for i in range(len(max_channels))
            ]
            weight_tmp = []
            bias_tmp = []
            bn_mean_tmp = []
            bn_var_tmp = []
            self.out_channels_ = int(self.out_channels_max *
                                     self.out_width_mult)
            for max_start, max_end in zip(start_index, end_index):
                weight_tmp_i = self.weight[max_start:max_end]
                weight_tmp.append(weight_tmp_i)
                bias_tmp_i = self.bias[max_start:max_end]
                bias_tmp.append(bias_tmp_i)
                bn_mean_tmp_i = self.running_mean[max_start:max_end]
                bn_mean_tmp.append(bn_mean_tmp_i)
                bn_var_tmp_i = self.running_var[max_start:max_end]
                bn_var_tmp.append(bn_var_tmp_i)

            weight = torch.cat(weight_tmp)
            bias = torch.cat(bias_tmp)

        if self.out_channels_ == self.out_channels_max:
            y = nn.functional.batch_norm(input, self.running_mean,
                                         self.running_var, weight, bias,
                                         self.training, self.momentum,
                                         self.eps)
        else:
            y = nn.functional.batch_norm(input,
                                         self.running_mean[:weight.shape[0]],
                                         self.running_var[:weight.shape[0]],
                                         weight, bias, self.training,
                                         self.momentum, self.eps)
        return y, in_group


class Dynamic_Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Dynamic_Conv, self).__init__()
        self.conv = DYConv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = DYBatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.c2 = c2

        self._c1 = c1
        self._c2 = c2
        self._k = k
        self._s = s
        self._p = p
        self._g = g
        self._act = act

    def forward(self, x, in_group):
        conv_output, in_group = self.conv(x, in_group)
        bn_output, in_group = self.bn(conv_output, in_group)
        in_group = [self.c2, bn_output.shape[1]]
        return self.act(bn_output), in_group

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import Conv as _Conv
        in_group = in_group[1::2]
        _layer = _Conv(sum(in_group), int(self._c2*width), self._k, self._s, self._p, self._g, self._act)
        _layer.i, _layer.f = self.i, self.f
        # load weights
        _layer.conv.weight.data.copy_(self.conv.weight[:int(self._c2*width), :sum(in_group), :, :].data)
        _layer.bn.weight.data.copy_(self.bn.weight[:int(self._c2*width)].data)
        _layer.bn.bias.data.copy_(self.bn.bias[:int(self._c2*width)].data)
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import Conv as _Conv
        return [self.f, 1, _Conv.__name__, [int(self._c2*width), self._k, self._s]]


class Upsample(nn.Upsample):
    def forward(self, x, in_group):
        return F.interpolate(x, self.size, self.scale_factor, self.mode,
                             self.align_corners), in_group
    
    def get_active_subnet(self, in_group=None, width=None, depth=None):
        _layer = nn.Upsample(None, 2, 'nearest')
        _layer.i, _layer.f = self.i, self.f
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        return [self.f, 1, 'nn.Upsample', [None, 2, 'nearest']]


class Dynamic_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Dynamic_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Dynamic_Conv(c1, c_, 1, 1)
        self.cv2 = Dynamic_Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

        self._c1 = c1
        self._c2 = c2
        self._shortcut = shortcut
        self._g = g
        self._e = e

    def forward(self, x, in_group):
        cv1_output, in_group = self.cv1(x, in_group)
        cv2_output, in_group = self.cv2(cv1_output, in_group)
        return x + cv2_output if self.add else cv2_output, in_group


class Dynamic_BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(Dynamic_BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.c1 = c1
        self.c2 = c2
        self.cv1 = Dynamic_Conv(c1, c_, 1, 1)
        self.cv2 = DYConv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = DYConv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Dynamic_Conv(2 * c_, c2, 1, 1)
        self.bn = DYBatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.ReLU(inplace=True)
        self.m = nn.ModuleList()
        for _ in range(n):
            self.m.append(Dynamic_Bottleneck(c_, c_, shortcut, g, e=1.0))
        self.max_depth = n
        self.depth = n
        self.sample_depth = 1.0

        self._c1 = c1
        self._c2 = c2
        self._n = n
        self._shortcut = shortcut
        self._g = g
        self._e = e

    def forward(self, x, in_group):
        # if self.i == 13:
        #     import pdb; pdb.set_trace()
        start_group = in_group.copy()
        tmp, in_group = self.cv1(x, start_group)
        for i in range(int(self.depth * self.sample_depth)):
            tmp, in_group = self.m[i](tmp, in_group)
        y1, in_group_1 = self.cv3(tmp, in_group)
        y2, in_group_2 = self.cv2(x, start_group)
        in_group = []
        in_group.extend(in_group_1)
        in_group.extend(in_group_2)
        bn_output, in_group = self.bn(torch.cat((y1, y2), dim=1), in_group)
        output, in_group = self.cv4(self.act(bn_output), in_group)
        in_group = [self.c2, output.shape[1]]
        return output, in_group

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import BottleneckCSP as _BottleneckCSP
        in_group = in_group[1::2] if in_group else [self._c1]
        _branch = len(in_group)
        _layer = _BottleneckCSP(sum(in_group), int(self._c2*width), int(self._n*depth), self._shortcut, self._g, self._e)
        _layer.i, _layer.f = self.i, self.f
        # load weights
        _c = int(int(self._c2 * width) * self._e)
        ## load c
        if _branch == 1:
            _layer.cv1.conv.weight.data.copy_(self.cv1.conv.weight[:_c, :sum(in_group) ,: ,:].data)
        else:
            _cv1_w = torch.cat([self.cv1.conv.weight[:_c, i*self._c1//_branch:i*self._c1//_branch+in_group[i]].data for i in range(_branch)], 1)
            _layer.cv1.conv.weight.data.copy_(_cv1_w)
        _layer.cv1.bn.weight.data.copy_(self.cv1.bn.weight[:_c].data)
        _layer.cv1.bn.bias.data.copy_(self.cv1.bn.bias[:_c].data)
            
        ## load c2
        if _branch == 1:
            _layer.cv2.weight.data.copy_(self.cv2.weight[:_c, :sum(in_group), :, :].data)
        else:
            _cv2_w = torch.cat([self.cv2.weight[:_c, i*self._c1//_branch:i*self._c1//_branch+in_group[i]].data for i in range(_branch)], 1)
            _layer.cv2.weight.data.copy_(_cv2_w)            
        ## load c3
        _layer.cv3.weight.data.copy_(self.cv3.weight[:_c, :_c, :, :].data)
        ## load c4
        _w4 = torch.cat([self.cv4.conv.weight[:int(self._c2*width), :_c, :, :].data,
                         self.cv4.conv.weight[:int(self._c2*width), int(self._c2*self._e):int(self._c2*self._e)+_c, :, :].data],
                         1)
        _layer.cv4.conv.weight.data.copy_(_w4)
        _layer.cv4.bn.weight.data.copy_(self.cv4.bn.weight[:int(self._c2*width)].data)
        _layer.cv4.bn.bias.data.copy_(self.cv4.bn.bias[:int(self._c2*width)].data)
        ## load bn
        _bn_weight = torch.cat([self.bn.weight[:_c].data, self.bn.weight[int(self._c2*self._e):int(self._c2*self._e)+_c].data], 0)
        _bn_bias= torch.cat([self.bn.bias[:_c].data, self.bn.bias[int(self._c2*self._e):int(self._c2*self._e)+_c].data], 0)
        _layer.bn.weight.data.copy_(_bn_weight)
        _layer.bn.bias.data.copy_(_bn_bias)
        ## load m
        def load_bottleneck(m, n, width):
            # copy n to m
            nonlocal _c
            m.cv1.conv.weight.data.copy_(n.cv1.conv.weight[:_c, :_c, :, :].data)
            m.cv1.bn.weight.data.copy_(n.cv1.bn.weight[:_c].data)
            m.cv1.bn.bias.data.copy_(n.cv1.bn.bias[:_c].data)
            m.cv2.conv.weight.data.copy_(n.cv2.conv.weight[:_c, :_c, :, :].data)
            m.cv2.bn.weight.data.copy_(n.cv2.bn.weight[:_c].data)
            m.cv2.bn.bias.data.copy_(n.cv2.bn.bias[:_c].data)
            return
        for i in range(int(len(_layer.m))):
            load_bottleneck(_layer.m[i], self.m[i], width)
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import BottleneckCSP as _BottleneckCSP
        if self._shortcut:
            return [self.f, int(self._n*depth), _BottleneckCSP.__name__, [int(self.c2*width),]]
        else:
            return [self.f, int(self._n*depth), _BottleneckCSP.__name__, [int(self.c2*width), False]]


class Dynamic_SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(Dynamic_SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Dynamic_Conv(c1, c_, 1, 1)
        self.cv2 = Dynamic_Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

        self._c1 = c1
        self._c2 = c2
        self._k = k

    def forward(self, x, in_group):
        x, in_group = self.cv1(x, in_group)
        before_cat = [x]
        for m in self.m:
            tmp = x
            tmp = m(tmp)
            before_cat.append(tmp)
            in_group.extend([in_group[0], in_group[1]])
        after_cat = torch.cat(before_cat, 1)
        x, in_group = self.cv2(after_cat, in_group)
        return x, in_group

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import SPP as _SPP
        # from models.common import Conv as _Conv
        in_group = in_group[1::2]
        _layer = _SPP(sum(in_group), int(self._c2*width), self._k)
        # _layer.cv1 = _Conv(sum(in_group), int(self._c1//2*width), 1, 1)
        # _layer.cv2 = _Conv(int(self._c1//2*width)*(len(self._k)+1), int(self._c2*width), 1, 1)
        _layer.i, _layer.f = self.i, self.f
        # load weights
        _layer.cv1.conv.weight.data.copy_(self.cv1.conv.weight[:sum(in_group)//2, :sum(in_group), :, :].data)
        _layer.cv1.bn.weight.data.copy_(self.cv1.bn.weight[:sum(in_group)//2].data)
        _layer.cv1.bn.bias.data.copy_(self.cv1.bn.bias[:sum(in_group)//2].data)
        _w2 = torch.cat([self.cv2.conv.weight[:int(self._c2*width), i*self._c1//2:i*self._c1//2+sum(in_group)//2].data
            for i in range(len(self._k)+1)], 1)
        _layer.cv2.conv.weight.data.copy_(_w2)
        _layer.cv2.bn.weight.data.copy_(self.cv2.bn.weight[:int(self._c2*width)].data)
        _layer.cv2.bn.bias.data.copy_(self.cv2.bn.bias[:int(self._c2*width)].data)
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import SPP as _SPP
        return [self.f, 1, _SPP.__name__, [int(self._c2*width), self._k]]


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              autopad(k, p),
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self._c1 = c1
        self._c2 = c2
        self._k = k
        self._s = s
        self._p = p
        self._g = g
        self._act = act

    def forward(self, x, in_group):
        return self.act(self.bn(self.conv(x))), in_group

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import Conv as _Conv
        _layer = _Conv(self._c1, self._c2, self._k, self._s, self._p, self._g, self._act)
        _layer.i, _layer.f = self.i, self.f
        _layer.load_state_dict(self.state_dict())
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import Conv as _Conv
        return [self.f, 1, _Conv.__name__, [self._c2, self._k, self._s]]


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        self._c1 = c1
        self._c2 = c2
        self._k = k
        self._s = s
        self._p = p
        self._g = g
        self._act = act

    def forward(self, x, in_group):
        x, in_group = self.conv(
            torch.cat([
                x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], 1), in_group)
        return x, in_group

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import Focus as _Focus
        _layer = _Focus(self._c1, self._c2, self._k, self._s, self._p, self._g, self._act)
        _layer.i, _layer.f = self.i, self.f
        _layer.load_state_dict(self.state_dict())
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import Focus as _Focus
        return [self.f, 1, _Focus.__name__, [self._c2, self._k]]


class Dynamic_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Dynamic_Concat, self).__init__()
        self.d = dimension

    def forward(self, x, in_group):
        in_group_ = []
        for j in in_group:
            in_group_.extend(j)
        return torch.cat(x, self.d), in_group_

    def get_active_subnet(self, in_group=None, width=None, depth=None):
        from models.common import Concat as _Concat
        _layer = _Concat(self.d)
        _layer.i, _layer.f = self.i, self.f
        return _layer

    def get_active_yaml(self, width=None, depth=None):
        from models.common import Concat as _Concat
        return [self.f, 1, _Concat.__name__, [1]]


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              autopad(k, p),
                              groups=g,
                              bias=False)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))  # flatten to x(b,c2)
