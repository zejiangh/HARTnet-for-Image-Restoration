from model import common
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DDHRNMEMB(args)

## Spatial-Channel-Attention (SCA) Layer
class SCALayer(nn.Module):
    def __init__(self, channel, reduction=16, dilation=4):
        super(SCALayer, self).__init__()
        # channel attention 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_channel = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        # spatial attention
        self.dilation = dilation
        self.conv_spatial = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=self.dilation, dilation=self.dilation, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=self.dilation, dilation=self.dilation, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.dual = nn.Conv2d(channel*2, channel, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        # channel attention feature
        y = self.avg_pool(x)
        y = self.conv_channel(y)
        c = x*y
        # spatial attention feature
        z = self.conv_spatial(x)
        z = z.view(x.size(0), 1, x.size(2), x.size(3))
        s = x*z
        # aggregate
        return self.dual(torch.cat((c,s),1))

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
    
## Hierarchical Multi-Scale Residual Block (HMSRB)
class HMSRB(nn.Module):
    def __init__(self, index, n_feats, baseWidth=32, scale = 4):
        
        super(HMSRB, self).__init__()

        width = int(math.floor(n_feats * (baseWidth/64.0)))
        self.width  = width
        self.scale = scale
        self.index = index
        
        self.conv1 = nn.Conv2d(n_feats, width*scale, kernel_size=1, bias=True)
        
        convs = []
        for i in range(self.scale):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride = 1, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(width*scale, n_feats, kernel_size=3, stride = 1, padding=1, bias=True)
        
        self.relu = nn.ReLU(True)
        
        self.SCA = SCALayer(n_feats)

    def forward(self, x):
        out = self.conv1(x)
        
        spx = torch.split(out, self.width, 1)
        
        sp = self.relu(self.convs[0](spx[0]))
        out = sp
        for k in range (self.scale-1):
            sp = sp+spx[k+1]
            sp = self.relu(self.convs[k+1](sp))
            out = torch.cat((out, sp), 1)
        
        return self.SCA(self.conv3(out))+x
    
## Residual Group with densely connected block
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, cardinality, lowd, idx):
        super(ResidualGroup, self).__init__()
        
        modules_body = [HMSRB(index=index+1, n_feats=n_feat, baseWidth=lowd, scale=cardinality) for index in range(n_resblocks)]
        self.body = nn.ModuleList(modules_body)
        self.gate = nn.Conv2d(n_feat*(idx+n_resblocks), n_feat, kernel_size=1, bias=True)
        
        self.idx = idx
        self.n_resblocks = n_resblocks

    def forward(self, x, ys):
        x = self.body[0](x)
        out = x
        for i in range (self.n_resblocks-1):
            x = self.body[i+1](x)
            out = torch.cat((out,x), 1)
        out = torch.cat((out, ys), 1)
        return self.gate(out)
    
## Hierarchical Multi-Scale Residual Network with densely connected group 
class DDHRNMEMB(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DDHRNMEMB, self).__init__()  
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.act = act
        self.scale = scale
        self.cardinality = args.cardinality
        self.lowd = args.lowd
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size)

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, cardinality=self.cardinality, lowd=self.lowd, idx=k+1) for k in range(n_resgroups)]
        self.body = nn.ModuleList(modules_body)
        self.conv = conv(n_feats, n_feats, kernel_size)
        
        # define tail module
        self.upsampler = common.Upsampler(conv, scale, n_feats, act=False)
        self.tail = conv(n_feats, args.n_colors, kernel_size)
        
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x0 = x
        ys = x
        for k in range (self.n_resgroups):
            x = self.body[k](x, ys)
            ys = torch.cat((ys, x), 1)
        res = x0+self.conv(x)
        
        x = self.tail(self.upsampler(res))
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
