from model import common
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DDHRNBAM(args)

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
        
        self.header = nn.Conv2d(n_feats*index, n_feats, kernel_size=1, bias=True)
        
        self.conv1 = nn.Conv2d(n_feats, width*scale, kernel_size=1, bias=True)
        
        convs = []
        for i in range(self.scale):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride = 1, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(width*scale, n_feats, kernel_size=3, stride = 1, padding=1, bias=True)
        
        self.relu = nn.ReLU(True)
        
        self.SCA = SCALayer(n_feats)

    def forward(self, x):
        x0 = x
        if self.index>1:
            x = self.header(x)
        
        out = self.conv1(x)
        
        spx = torch.split(out, self.width, 1)
        
        sp = self.relu(self.convs[0](spx[0]))
        out = sp
        sp = sp+spx[1]
        sp = self.relu(self.convs[1](sp))
        out = torch.cat((out, sp), 1)
        sp = sp+spx[2]
        sp = self.relu(self.convs[2](sp))
        out = torch.cat((out, sp), 1)
        sp = sp+spx[3]
        sp = self.relu(self.convs[3](sp))
        out = torch.cat((out, sp), 1)
        
        ms = self.conv3(out)+x
        
        return torch.cat((self.SCA(ms)+ms,x0),1)
    
## Residual Group with densely connected block
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, idx):
        super(ResidualGroup, self).__init__()
        self.header = nn.Conv2d(n_feat*idx, n_feat, kernel_size=1, bias=True)
        modules_body = []
        modules_body = [HMSRB(index=index+1, n_feats=n_feat, baseWidth=32, scale=4) for index in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat*(n_resblocks+1), n_feat, kernel_size=1, bias=True))
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        
        self.idx = idx

    def forward(self, x):
        x0 = x
        if self.idx > 1:
            x = self.header(x)
        res = self.body(x)
        res += x
        return torch.cat((x0, res), 1)
    
## Hierarchical Multi-Scale Residual Network with densely connected group 
class DDHRNBAM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DDHRNBAM, self).__init__()  
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.act = act
        self.scale = scale
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, idx=k+1) for k in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats*(n_resgroups+1), n_feats, kernel_size=1, bias=True))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),conv(n_feats, args.n_colors, kernel_size)]
        
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
