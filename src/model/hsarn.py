
from model import common
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def make_model(args, parent=False):
    return HSARN(args)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.final_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//2, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = phi.view(-1, ch//2, phi.shape[2]*phi.shape[3])
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.W(attn_g)
        # Out
        attn_g = self.final_conv(torch.cat((x, attn_g),1))
        out = x + attn_g
        return out

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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.SA = Self_Attn(n_feat)

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.SA(res)

## Residual Channel Attention Network (RCAN)
class HSARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HSARN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))









#from model import common
#import math
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#def make_model(args, parent=False):
#    return HSARN(args)
#
#class Self_Attn(nn.Module):
#    """ Self attention Layer"""
#
#    def __init__(self, in_channels):
#        super(Self_Attn, self).__init__()
#        self.in_channels = in_channels
#        self.snconv1x1_theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
#        self.snconv1x1_phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
#        self.snconv1x1_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
#        self.snconv1x1_attn = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
#        self.softmax  = nn.Softmax(dim=-1)
#        self.sigma = nn.Parameter(torch.zeros(1))
#
#    def forward(self, x):
#        """
#            inputs :
#                x : input feature maps(B X C X W X H)
#            returns :
#                out : self attention value + input feature 
#                attention: B X N X N (N is Width*Height)
#        """
#        _, ch, h, w = x.size()
#        # Theta path
#        theta = self.snconv1x1_theta(x)
#        theta = theta.view(-1, ch//8, h*w)
#        # Phi path
#        phi = self.snconv1x1_phi(x)
#        phi = self.maxpool(phi)
#        phi = phi.view(-1, ch//8, phi.shape[2]*phi.shape[3])
#        # Attn map
#        attn = torch.bmm(theta.permute(0, 2, 1), phi)
#        attn = self.softmax(attn)
#        # g path
#        g = self.snconv1x1_g(x)
#        g = self.maxpool(g)
#        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
#        # Attn_g
#        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
#        attn_g = attn_g.view(-1, ch//2, h, w)
#        attn_g = self.snconv1x1_attn(attn_g)
#        # Out
#        out = x + self.sigma*attn_g
#        return out
#
### Channel Attention (CA) Layer
#class CALayer(nn.Module):
#    def __init__(self, channel, reduction=16):
#        super(CALayer, self).__init__()
#        # global average pooling: feature --> point
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        # feature channel downscale and upscale --> channel weight
#        self.conv_du = nn.Sequential(
#                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        y = self.avg_pool(x)
#        y = self.conv_du(y)
#        return x * y
#    
### Hierarchical Multi-Scale Self-Attention Residual Block (HSARB)
#class HSARB(nn.Module):
#    def __init__(self, n_feats, baseWidth=26, scale = 4):
#        
#        super(HSARB, self).__init__()
#
#        width = int(math.floor(n_feats * (baseWidth/64.0)))
#        self.width  = width
#        self.scale = scale
#        
#        self.conv1 = nn.Conv2d(n_feats, width*scale, kernel_size=1, bias=True)
#        
#        convs = []
#        for i in range(self.scale):
#            convs.append(nn.Conv2d(width, width, kernel_size=3, stride = 1, padding=1, bias=True))
#        self.convs = nn.ModuleList(convs)
#
#        self.conv3 = nn.Conv2d(width*scale, n_feats, kernel_size=3, stride = 1, padding=1, bias=True)
#        
#        self.relu = nn.ReLU(True)
#        
#        self.CA = CALayer(n_feats)
#
#    def forward(self, x):
#        
#        out = self.relu(self.conv1(x))
#        
#        spx = torch.split(out, self.width, 1)
#        
#        sp = self.relu(self.convs[0](spx[0]))
#        out = sp
#        sp = sp+spx[1]
#        sp = self.relu(self.convs[1](sp))
#        out = torch.cat((out, sp), 1)
#        sp = sp+spx[2]
#        sp = self.relu(self.convs[2](sp))
#        out = torch.cat((out, sp), 1)
#        sp = sp+spx[3]
#        sp = self.relu(self.convs[3](sp))
#        out = torch.cat((out, sp), 1)
#        
#        return self.CA(self.conv3(out))+x
#
### Residual Group (RG)
#class ResidualGroup(nn.Module):
#    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#        super(ResidualGroup, self).__init__()
#        modules_body = []
#        modules_body = [HSARB(n_feats=n_feat) for _ in range(n_resblocks)]
#        modules_body.append(conv(n_feat, n_feat, kernel_size))
##        modules_body.append(Self_Attn(n_feat))
#        self.body = nn.Sequential(*modules_body)
#        self.SA = Self_Attn(n_feat)
#
#    def forward(self, x):
#        res = self.body(x)
#        res += x
#        return self.SA(res)
#    
### Hierarchical Multi-Scale Self-Attention Residual Network (HSARN)  
#class HSARN(nn.Module):
#    def __init__(self, args, conv=common.default_conv):
#        super(HSARN, self).__init__()
#        
#        n_resgroups = args.n_resgroups
#        n_resblocks = args.n_resblocks
#        n_feats = args.n_feats
#        kernel_size = 3
#        reduction = args.reduction 
#        scale = args.scale[0]
#        act = nn.ReLU(True)
#        self.act = act
#        self.scale = scale
#        
#        # RGB mean for DIV2K
#        rgb_mean = (0.4488, 0.4371, 0.4040)
#        rgb_std = (1.0, 1.0, 1.0)
#        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
#        
#        # define head module
#        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
#
#        # define body module
#        modules_body = [
#            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
#        modules_body.append(conv(n_feats, n_feats, kernel_size))
#
#        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
#
#        self.head = nn.Sequential(*modules_head)
#        self.body = nn.Sequential(*modules_body)
#        
#        # define tail module
#        self.tail_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True)
#        self.tail_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True)
#        self.final_conv = conv(n_feats, args.n_colors, kernel_size)
#
#    def forward(self, x):
#        x = self.sub_mean(x)
#        x = self.head(x)
#        
#        res = self.body(x)
#        res += x
#
#        if self.scale == 2 or self.scale == 3:
#            x = self.act(self.tail_conv1(F.interpolate(res, scale_factor=self.scale, mode='nearest')))
#            x = self.final_conv(x)
#        elif self.scale == 4:
#            x = self.act(self.tail_conv1(F.interpolate(res, scale_factor=2, mode='nearest')))
#            x = self.act(self.tail_conv2(F.interpolate(x, scale_factor=2, mode='nearest')))
#            x = self.final_conv(x)
#        x = self.add_mean(x)
#
#        return x 
#
#    def load_state_dict(self, state_dict, strict=False):
#        own_state = self.state_dict()
#        for name, param in state_dict.items():
#            if name in own_state:
#                if isinstance(param, nn.Parameter):
#                    param = param.data
#                try:
#                    own_state[name].copy_(param)
#                except Exception:
#                    if name.find('tail') >= 0:
#                        print('Replace pre-trained upsampler to new one...')
#                    else:
#                        raise RuntimeError('While copying the parameter named {}, '
#                                           'whose dimensions in the model are {} and '
#                                           'whose dimensions in the checkpoint are {}.'
#                                           .format(name, own_state[name].size(), param.size()))
#            elif strict:
#                if name.find('tail') == -1:
#                    raise KeyError('unexpected key "{}" in state_dict'
#                                   .format(name))
#
#        if strict:
#            missing = set(own_state.keys()) - set(state_dict.keys())
#            if len(missing) > 0:
#                raise KeyError('missing keys in state_dict: "{}"'.format(missing)) 
