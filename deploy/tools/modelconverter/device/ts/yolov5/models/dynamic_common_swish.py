# This file contains modules common to various models
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(DYConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.out_width_mult = None
        self.groups = groups
        self.in_group = []
    def forward(self, input, in_group):
        if len(self.in_group) <= 2:
            self.in_channels_ = input.shape[1]
            self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
            weight = self.weight[:self.out_channels_, :self.in_channels_,:,:]
            if self.bias is not None:
                bias = self.bias[:self.out_channels_]
            else:
                bias = self.bias
        else:
            max_channels = in_group[::2]
            recent_channels = in_group[1::2]
            start_index = [0 if i==0 else sum(max_channels[:i]) for i in range(len(max_channels))]
            end_index = [start_index[i] + recent_channels[i] for i in range(len(max_channels))]
            weight_tmp = []
            self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
            for start, end in zip(start_index, end_index):
                weight_tmp.append(self.weight[:,start:end,:,:])
            weight = torch.cat(weight_tmp, dim=1)
            weight = weight[:self.out_channels_, :,:,:]
            if self.bias is not None:
                bias = self.bias[:self.out_channels_]
            else:
                bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y, [self.out_channels_max,y.shape[1]]


# Version Zeyang Dou, dynamic batch normalization with max and min model
#class DYBatchNorm2d(nn.BatchNorm2d):
#    def __init__(self, out_channels, eps = 0.0001, momentum=0.1):
#        super(DYBatchNorm2d, self).__init__(
#            out_channels, affine=True, eps=eps, momentum=momentum, track_running_stats=True)
#        self.out_channels_max = out_channels
#        self.out_width_mult = None
#        self.in_group = []
#        self.bn = nn.ModuleList([
#            nn.BatchNorm2d(i, affine=False) for i in [self.out_channels_max, int(self.out_channels_max*32/80)]])
#        self.mode = None
#    def forward(self, input, in_group):
#        self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
#        assert self.out_channels_ == input.shape[1]
#        if len(in_group)<=2:
#            weight = self.weight[:self.out_channels_]
#            bias = self.bias[:self.out_channels_]
#
#        else:
#            max_channels = in_group[::2]
#            recent_channels = in_group[1::2]
#            start_index = [0 if i==0 else sum(max_channels[:i]) for i in range(len(max_channels))]
#            end_index = [start_index[i] + recent_channels[i] for i in range(len(max_channels))]
#            weight_tmp = []
#            bias_tmp = []
#            bn_mean_tmp = []
#            bn_var_tmp = []
#            self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
#            for max_start, max_end in zip(start_index, end_index):
#                weight_tmp_i = self.weight[max_start:max_end]
#                weight_tmp.append(weight_tmp_i)
#                bias_tmp_i = self.bias[max_start:max_end]
#                bias_tmp.append(bias_tmp_i)
#                bn_mean_tmp_i = self.running_mean[max_start:max_end]
#                bn_mean_tmp.append(bn_mean_tmp_i)
#                bn_var_tmp_i = self.running_var[max_start:max_end]
#                bn_var_tmp.append(bn_var_tmp_i)
#
#            weight = torch.cat(weight_tmp)
#            bias = torch.cat(bias_tmp)
#
#
#        if self.out_channels_ == self.out_channels_max:
#            y = nn.functional.batch_norm(
#                input,
#                self.bn[0].running_mean,
#                self.bn[0].running_var,
#                weight,
#                bias,
#                self.training,
#                self.momentum,
#                self.eps)
#        elif self.out_channels_ == int(self.out_channels_max*32/80):
#            y = nn.functional.batch_norm(
#                input,
#                self.bn[1].running_mean,
#                self.bn[1].running_var,
#                weight,
#                bias,
#                self.training,
#                self.momentum,
#                self.eps)
#        else:
#            y = nn.functional.batch_norm(
#                input,
#                self.running_mean[:weight.shape[0]],
#                self.running_var[:weight.shape[0]],
#                weight,
#                bias,
#                self.training,
#                self.momentum,
#                self.eps)
#        return y, in_group

# Version Shaohua Li, dynamic batch normalization with only max model
class DYBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, out_channels, eps = 0.0001, momentum=0.1):
        super(DYBatchNorm2d, self).__init__(
            out_channels, affine=True, eps=eps, momentum=momentum, track_running_stats=True)
        self.out_channels_max = out_channels
        self.out_width_mult = None
        self.in_group = []
        self.mode = None
    def forward(self, input, in_group):
        self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
        assert self.out_channels_ == input.shape[1]
        if len(in_group)<=2:
            weight = self.weight[:self.out_channels_]
            bias = self.bias[:self.out_channels_]

        else:
            max_channels = in_group[::2]
            recent_channels = in_group[1::2]
            start_index = [0 if i==0 else sum(max_channels[:i]) for i in range(len(max_channels))]
            end_index = [start_index[i] + recent_channels[i] for i in range(len(max_channels))]
            weight_tmp = []
            bias_tmp = []
            bn_mean_tmp = []
            bn_var_tmp = []
            self.out_channels_ = int(self.out_channels_max * self.out_width_mult)
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
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight,
                bias,
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean[:weight.shape[0]],
                self.running_var[:weight.shape[0]],
                weight,
                bias,
                self.training,
                self.momentum,
                self.eps)
        return y, in_group

class Dynamic_Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Dynamic_Conv, self).__init__()
        self.conv = DYConv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = DYBatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.c2 = c2
    def forward(self, x, in_group):
        setattr(self.conv, 'in_group', in_group)
        conv_output, in_group = self.conv(x, in_group)
        bn_output, in_group = self.bn(conv_output, in_group)
        in_group = [self.c2, bn_output.shape[1]]
        return self.act(bn_output), in_group

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Upsample):
    def forward(self, x, in_group):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners), in_group

class Dynamic_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Dynamic_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Dynamic_Conv(c1, c_, 1, 1)
        self.cv2 = Dynamic_Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.in_group = []
    def forward(self, x, in_group):
        cv1_output, in_group = self.cv1(x, in_group)
        cv2_output, in_group = self.cv2(cv1_output, in_group)
        return x + cv2_output if self.add else cv2_output, in_group


class Dynamic_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.ModuleList()
        for _ in range(n):
            self.m.append(Dynamic_Bottleneck(c_, c_, shortcut, g, e=1.0))
        self.max_dpeth = n
        self.depth = n
        self.sample_depth=1.0
    def forward(self, x, in_group):
        setattr(self.cv1, 'in_group', in_group)
        setattr(self.cv2, 'in_group', in_group)
        start_group = in_group.copy()
        tmp, in_group = self.cv1(x, start_group)
        for i in range(int(self.depth*self.sample_depth)):
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


class Dynamic_SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(Dynamic_SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Dynamic_Conv(c1, c_, 1, 1)
        self.cv2 = Dynamic_Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x, in_group):
        x, in_group = self.cv1(x, in_group)
        before_cat = [x]
        for m in self.m:
            tmp = x
            tmp = m(tmp)
            before_cat.append(tmp)
            in_group.extend([in_group[0],in_group[1]])
        after_cat = torch.cat(before_cat, 1)
        x, in_group = self.cv2(after_cat, in_group)
        # x, in_group = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1), in_group)
        return x, in_group


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x, in_group):
        return self.act(self.bn(self.conv(x))), in_group


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
    def forward(self, x, in_group):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x, in_group = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1), in_group)
        return x, in_group


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


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
