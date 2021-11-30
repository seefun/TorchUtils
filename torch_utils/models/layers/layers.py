import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable


def conv3x3(in_channel, out_channel):  # not change resolusion
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=1, padding=1, dilation=1, bias=False)


def conv1x1(in_channel, out_channel):  # not change resolution
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

# Attention


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()
        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x
        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x
        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SCSE, self).__init__()
        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        x = cSE + sSE
        return x


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel // reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel // reduction, in_channel).apply(init_weight)
        )

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = torch.sigmoid(x1 + x2)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2, 1).apply(init_weight)

    def forward(self, inputs):
        x1, _ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = torch.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x


class CoordAttention(nn.Module):
    '''Coordinate Attention for Efficient Mobile Network Design'''

    def __init__(self, in_channels, out_channels, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = nn.SiLU(True)

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


# TODO:
# add GloRe(GCN attention)
# https://github.com/facebookresearch/GloRe/blob/master/network/global_reasoning_unit.py
# add CCAtention
# https://github.com/speedinghzl/CCNet/blob/master/networks/ccnet.py#L99


def gem(x, p=1, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, flatten=True, eps=1e-6, requires_grad=True):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps
        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = False

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            return self.flatten(x)
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeM_cw(nn.Module):
    """ channel-wise GeM Pooling """

    def __init__(self, num_channel, p=1, flatten=True, eps=1e-6):
        super(GeM_cw, self).__init__()
        self.p = Parameter(torch.ones(num_channel) * p)
        self.eps = eps
        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = False

    def forward(self, x):
        p = self.p.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = gem(x, p=p, eps=self.eps)
        if self.flatten:
            return self.flatten(x)
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=True):  # flatten == True : pool + flatten
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class FastGlobalConcatPool2d(nn.Module):
    def __init__(self, flatten=True):  # flatten == True : pool + flatten
        super(FastGlobalConcatPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            x = x.view((in_size[0], in_size[1], -1))
            return torch.cat([x.mean(dim=2), x.max(dim=2).values], 1)
        else:
            x = x.view(x.size(0), x.size(1), -1)
            return torch.cat([x.mean(-1), x.max(-1).values], 1).view(x.size(0), 2 * x.size(1), 1, 1)


class MultiSampleDropoutFC(nn.Module):
    def __init__(self, in_ch, out_ch, num_sample=5, dropout=0.5):
        super(MultiSampleDropoutFC, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_sample)])
        self.fc = nn.Linear(in_ch, out_ch, bias=True)

    def forward(self, x):
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        if self.training:
            epsilon = torch.randn(x.size()) * self.alpha + 1
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
        else:
            return x


class UOut(nn.Module):
    def __init__(self, beta=0.1):
        super(UOut, self).__init__()
        self.beta = torch.Tensor([beta])

    def forward(self, x):
        if self.training:
            epsilon = torch.rand(x.size()) * self.beta * 2 - self.beta + 1
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
        else:
            return x

###### activation ######


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class FReLU(nn.Module):
    """
    FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv_frelu(x)
        y = self.bn_frelu(y)
        x = torch.max(x, y)
        return x


###### example ######

def get_simple_fc(in_ch, num_classes, flatten=False):
    if flatten:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 1280),
            Swish(inplace=True),
            nn.Dropout(),
            nn.Linear(1280, num_classes),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_ch, 1280),
            Swish(),
            nn.Dropout(),
            nn.Linear(1280, num_classes),
        )


def get_attention_fc(in_ch, num_classes, flatten=False):
    if flatten:
        return nn.Sequential(
            nn.Flatten(),
            SEBlock(in_ch),
            MultiSampleDropoutFC(in_ch, num_classes),
        )
    else:
        return nn.Sequential(
            SEBlock(in_ch),
            MultiSampleDropoutFC(in_ch, num_classes),
        )


#########################
# DepthToSpace == pixel shuffle
# Official: torch.nn.PixelShuffle(upscale_factor)
# SpaceToDepth == inverted pixel shuffle
# Official: torch.nn.PixelUnshuffle(downscale_factor)

def pixelshuffle(x, factor_hw):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC // (pH * pW), iH * pH, iW * pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixelshuffle_invert(x, factor_hw):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


# ASPP

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)

        return x
