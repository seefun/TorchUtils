import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()
        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x
        x = x.view(*(x.shape[:-2]),-1).mean(-1)
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

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
  

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False): # flatten == True : pool + flatten
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class FastGlobalConcatPool2d(nn.Module):
    def __init__(self, flatten=False): # flatten == True : pool + flatten
        super(FastGlobalConcatPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            x = x.view((in_size[0], in_size[1], -1))
            return torch.cat([x.mean(dim=2), x.max(dim=2).values], 1)
        else:
            x = x.view(x.size(0), x.size(1), -1)
            return torch.cat([x.mean(-1), x.max(-1).values], 1).view(x.size(0), 2*x.size(1), 1, 1)


class MultiSampleDropoutFC(nn.Module):
    def __init__(self, in_ch, out_ch, num_sample = 5, dropout = 0.5):
        super(MultiSampleDropoutFC, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_sample)])
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self,x):
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out

###### activation ######

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


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


###### example ###### 

def get_simple_fc(in_ch, num_classes, flatten=False):
    if flatten:
        return nn.Sequential(
            Flatten(),
            nn.Linear(in_ch, 512),
            Swish(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_ch, 512),
            Swish(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )


def get_attention_fc(in_ch, num_classes, flatten=False):
    if flatten:
        return nn.Sequential(
            Flatten(),
            SEBlock(in_ch),
            MultiSampleDropoutFC(in_ch,num_classes),
        )
    else:
        return nn.Sequential(
            SEBlock(in_ch),
            MultiSampleDropoutFC(in_ch,num_classes),
        )


#########################
# DepthToSpace == pixel shuffle
# SpaceToDepth == inverted pixel shuffle
# Official: torch.nn.PixelShuffle(upscale_factor)

def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
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
    def __call__(self, x: torch.Tensor):
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
