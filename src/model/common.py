import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
        
        
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class ResBlockCA(nn.Module):
    def __init__(
        self, conv, n_feats, shrink_size, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockCA, self).__init__()
        m = []
        
        m.append(conv(n_feats, shrink_size, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        
        m.append(conv(shrink_size, n_feats, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.CA = CALayer(n_feats)
    
    def forward(self, x):
        res = self.CA(self.body(x).mul(self.res_scale))
        res += x

        return res


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, shrink_size, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        
        m.append(conv(n_feats, shrink_size, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        
        m.append(conv(shrink_size, n_feats, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        
#        for i in range(2):
#            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if i == 0:
#                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
    
class MBlock(nn.Module):
    def __init__(
        self, conv, n_feats, shrink_size, kernel_size,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MBlock, self).__init__()
        m = []
        
        m.append(nn.Conv2d(n_feats, n_feats*6, 1, 1, 0, bias=False))
        if bn:
            m.append(nn.BatchNorm2d(n_feats*6))
        m.append(act)
        
        m.append(nn.Conv2d(n_feats*6, n_feats*6, 3, 1, 1, groups=n_feats*6, bias=False))
        if bn:
            m.append(nn.BatchNorm2d(n_feats*6))
        m.append(act)
        
        m.append(nn.Conv2d(n_feats*6, n_feats, 1, 1, 0, bias=False))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
    
    
    
    
    
class MSBlock(nn.Module):
    def __init__(self, conv, n_feats, shrink_size, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, baseWidth=16, scale = 4):
        
        super(MSBlock, self).__init__()

        width = int(math.floor(n_feats * (baseWidth/64.0)))
        self.scale = scale
        convs = []
        for i in range(self.scale):
            convs.append(nn.Conv2d(width*(i+1), width*(i+1), kernel_size=3, stride = 1, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.relu = nn.ReLU(inplace=True)
        self.width  = width
        
        self.conv2 = conv(shrink_size, n_feats, kernel_size, bias=bias)

    def forward(self, x):
        
        spx = torch.split(x, self.width, 1)
        
        sp = self.relu(self.convs[0](spx[0]))
        sp = torch.cat((sp, spx[1]), 1)
        sp = self.relu(self.convs[1](sp))
        sp = torch.cat((sp, spx[2]), 1)
        sp = self.relu(self.convs[2](sp))
        sp = torch.cat((sp, spx[3]), 1)
        sp = self.relu(self.convs[3](sp))
        
        res = self.conv2(sp)
        res += x

        return res
      
#class Upsampler(nn.Sequential):
#    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
#
#        m = []
#        if scale == 2:
#            m.append(conv(n_feats, 256, 3, bias))
#            m.append(nn.PixelShuffle(2))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if act == 'relu':
#                m.append(nn.ReLU(True))
#            elif act == 'prelu':
#                m.append(nn.PReLU(n_feats))
#        elif scale == 4:
#            m.append(conv(n_feats, 256, 3, bias))
#            m.append(nn.PixelShuffle(2))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if act == 'relu':
#                m.append(nn.ReLU(True))
#            elif act == 'prelu':
#                m.append(nn.PReLU(n_feats))
#            m.append(conv(64, 256, 3, bias))
#            m.append(nn.PixelShuffle(2))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if act == 'relu':
#                m.append(nn.ReLU(True))
#            elif act == 'prelu':
#                m.append(nn.PReLU(n_feats))
#        elif scale == 3:
#            m.append(conv(n_feats, 576, 3, bias))
#            m.append(nn.PixelShuffle(3))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if act == 'relu':
#                m.append(nn.ReLU(True))
#            elif act == 'prelu':
#                m.append(nn.PReLU(n_feats))
#        else:
#            raise NotImplementedError
#
#        super(Upsampler, self).__init__(*m)
         
#class Interpolator(nn.Sequential):
#    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
#
#        m = []
#        if scale == 2 or scale == 3:
#            m.append(F.interpolate(res, scale_factor=scale, mode='nearest'))
#            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
#            m.append(nn.ReLU(True))
#        elif scale == 4:
#            m.append(F.interpolate(res, scale_factor=2, mode='nearest'))
#            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
#            m.append(nn.ReLU(True))
#            m.append(F.interpolate(res, scale_factor=2, mode='nearest'))
#            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
#            m.append(nn.ReLU(True))
#        else:
#            raise NotImplementedError
#
#        super(Interpolator, self).__init__(*m)
        
        
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        
        
class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(64, 64, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=64)
        self.conv_a = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 64, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        '''
        who is guidance? who is filtering input?
        x_lr: guidance, deep_feature
        y_lr: filtering input, shallow feature
        x_hr: upscaled deep_feature
        '''
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 64, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x
        

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b
    
def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
    
class EdgeGuidedFilter(nn.Module):
    def __init__(self, r=4, eps=1e-3):
        super(EdgeGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


class ConvEdgeGuidedFilter(nn.Module):
    def __init__(self, radius=1):
        super(ConvEdgeGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(64, 64, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=64)
        self.conv_a = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 64, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0
        
    def forward(self, x, y):
        '''
        x: guidance
        y: filtering input
        '''
        
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        
        N = self.box_filter(x.data.new().resize_((1, 64, h_x, w_x)).fill_(1.0))
        
        mean_x = self.box_filter(x)/N
        mean_y = self.box_filter(y)/N
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        var_x = self.box_filter(x * x) / N - mean_x * mean_x
        
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y-A*mean_x
        
        mean_A = self.box_filter(A)/N
        mean_b = self.box_filter(b)/N
        
        return mean_A*x+mean_b
    
class EdgeResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, shrink_size, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(EdgeResBlock, self).__init__()
        m = []
        
        m.append(conv(n_feats, shrink_size, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        
        m.append(conv(shrink_size, n_feats, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.conv_edge_guided_filter = ConvEdgeGuidedFilter()
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        edge = self.conv_edge_guided_filter(x,x)
        output = x + edge

        return output