from model import common
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return HRN(args)

## Self-attnetion module
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

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
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, phi.shape[2]*phi.shape[3])
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, g.shape[2]*g.shape[3])
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
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
        

## Hierarchical Multi-Scale Residual Block (HMSRB)             
class HSARB(nn.Module):
    def __init__(self, n_feats, baseWidth=32, scale = 4):
        
        super(HSARB, self).__init__()

        width = int(math.floor(n_feats * (baseWidth/64.0)))
        self.width  = width
        self.scale = scale
        
        self.conv1 = nn.Conv2d(n_feats, width*scale, kernel_size=1, bias=True)
        
        convs = []
        for i in range(self.scale):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride = 1, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(width*scale, n_feats, kernel_size=3, stride = 1, padding=1, bias=True)
        
        self.relu = nn.ReLU(True)
        
#        self.SCA = SCALayer(n_feats)

    def forward(self, x):
        out = self.conv1(x)
        
        spx = torch.split(out, self.width, 1)
        
        sp = self.relu(self.convs[0](spx[0]))
        out = sp
        for k in range (self.scale-1):
                sp = sp+spx[k+1]
                sp = self.relu(self.convs[k+1](sp))
                out = torch.cat((out,sp),1)
        
        return self.conv3(out).mul(0.1)+x
    
## Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, cardinality, lowd):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [HSARB(n_feats=n_feat, baseWidth=lowd, scale=cardinality) for _ in range(n_resblocks)]
#        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x).mul(0.1)
        res += x
        return res
    
## Hierarchical Multi-Scale Residual Network (HMSRN)  
class HRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HRN, self).__init__()
        
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
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, cardinality=self.cardinality, lowd=self.lowd) for _ in range(n_resgroups)]
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