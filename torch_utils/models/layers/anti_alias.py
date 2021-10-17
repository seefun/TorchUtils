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
