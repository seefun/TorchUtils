import torch
import torch.nn as nn
import torch.nn.functional as F


class Anti_Alias_Filter(nn.Module):
    """ adaptive low pass filter (anti-alias) used before downsampling (pooling or stride>1 conv)
        idea from Delving-Deeper-Into-Anti-Aliasing-in-ConvNets (BMVC2020 best paper) https://arxiv.org/pdf/2008.09604.pdf
        code modified from: https://github.com/MaureenZOU/Adaptive-anti-Aliasing/blob/master/models_lpf/layers/pasa.py
    """

    def __init__(self, in_channels, kernel_size=3, groups=8):
        super(Anti_Alias_Filter, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.kernel_size = kernel_size
        self.groups = groups
        assert in_channels % groups == 0

        self.conv = nn.Conv2d(in_channels, groups * kernel_size * kernel_size, kernel_size=kernel_size, stride=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(groups * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n, c, h, w = sigma.shape

        sigma = sigma.reshape(n, 1, c, h * w)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n, c, self.kernel_size * self.kernel_size, h * w))

        n, c1, p, q = x.shape
        x = x.permute(1, 0, 2, 3).reshape(self.groups, c1 // self.groups, n, p, q).permute(2, 0, 1, 3, 4)

        n, c2, p, q = sigma.shape
        sigma = sigma.permute(2, 0, 1, 3).reshape((p // (self.kernel_size * self.kernel_size),
                                                   self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0, 3, 1, 4)

        x = torch.sum(x * sigma, dim=3).reshape(n, c1, h, w)

        return x


class AntiAliasDownsampleLayer(nn.Module):
    """ Fixed Blur filter with stride 2 (BlurPool)
        From: Making convolutional networks shift-invariant again. In ICML, 2019. | From: T-ResNet.
    """

    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_aa_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels

        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)

        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


BlurPool = AntiAliasDownsampleLayer
AAFilter = Anti_Alias_Filter
