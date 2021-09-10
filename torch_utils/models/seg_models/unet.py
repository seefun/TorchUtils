import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from torch_utils.layers import *
from torch_utils.models import create_timm_model


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample, attention=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        if attention == 'scse':
            self.attention = SCSE(out_channel, r=8)
        elif attention == 'cbam':
            self.attention = CBAM(out_channel, reduction=16)
        else:
            self.attention = nn.Identity()
        self.conv1x1 = conv1x1(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.relu(self.bn2(x)))
        x = self.attention(x)
        x += self.conv1x1(self.upsample(inputs))  # shortcut
        return x


def get_encoder_info(name='resnest50d'):
    model = create_timm_model(name, pretrained=False).eval()
    features = model(torch.rand(1, 3, 224, 224))
    encoder_channels = []
    for i, feat in enumerate(features):
        print('Feature %d' % i, feat.shape)
        encoder_channels.append(feat.shape[1])
    return encoder_channels


class UNet_neck(nn.Module):
    '''Input feature list and output feature list'''

    def __init__(self,
                 encoder_channels=[64, 256, 512, 1024, 2048],
                 center_channel=512,
                 decoder_channels=[64, 64, 64, 64, 64],
                 attention='cbam',
                 drop_first=True):
        super(UNet).__init__()
        self.drop_first = drop_first

        # center
        self.center = CenterBlock(
            encoder_channels[-1], decoder_channels[0])  # ->(*,512,h/32,w/32)

        # decoder
        self.decoder4 = DecodeBlock(
            center_channel + encoder_channels[4], decoder_channels[0],
            upsample=True, attention=attention)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(
            decoder_channels[0] + encoder_channels[3], decoder_channels[1],
            upsample=True, attention=attention)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(
            decoder_channels[1] + encoder_channels[2], decoder_channels[2],
            upsample=True, attention=attention)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(
            decoder_channels[2] + encoder_channels[1], decoder_channels[3],
            upsample=True, attention=attention)  # ->(*,64,h/2,w/2)
        if drop_first:
            self.decoder0 = DecodeBlock(
                decoder_channels[3], decoder_channels[4], upsample=True,
                attention=attention)  # ->(*,64,h,w)
        else:
            self.decoder0 = DecodeBlock(
                decoder_channels[3] + encoder_channels[0], decoder_channels[4], upsample=True,
                attention=attention)  # ->(*,64,h,w)

        # upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    @autocast()
    def forward(self, inputs):
        # encoder
        x0, x1, x2, x3, x4 = inputs

        # center
        y5 = self.center(x4)  # ->(*,320,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        if self.drop_first:
            y0 = self.decoder0(y1)  # ->(*,64,h,w)
        else:
            y0 = self.decoder0(torch.cat([x0, y1]))  # ->(*,64,h,w)
        
        return [y4, y3, y2, y1, y0]

class UNet(nn.Module):
    def __init__(self,
                 backbone='resnest50d',
                 pretrained=True,
                 neck='unet',  # unet or none
                 drop_first=True,
                 encoder_channels=[64, 256, 512, 1024, 2048],
                 center_channel=512,
                 decoder_channels=[64, 64, 64, 64, 64],
                 out_channel=1,
                 attention='cbam',
                 hypercolumns=True):
        super(UNet).__init__()
        self.backbone = create_timm_model(backbone, pretrained)
        self.attention = attention
        self.hypercolumns = hypercolumns
        self.out_channel = out_channel

        if neck == 'unet':
            self.neck = UNet_neck(encoder_channels=encoder_channels,
                                  center_channel=center_channel,
                                  decoder_channels=decoder_channels,
                                  attention=attention,
                                  drop_first=drop_first)
        else:
            self.neck = None

        # final conv
        if hypercolumns:
            final_channel = sum(decoder_channels)
        else:
            final_channel = decoder_channels[-1]
        
        self.final_conv = nn.Sequential(
            conv3x3(final_channel, decoder_channels[4]).apply(init_weight),
            nn.SiLU(True),
            conv1x1(decoder_channels[4], self.out_channel).apply(init_weight)
        )

    @ autocast()
    def forward(self, inputs):
        # encoder
        y4, y3, y2, y1, y0 = self.backbone(inputs)
        if self.neck:
            y4, y3, y2, y1, y0 = self.neck([y4, y3, y2, y1, y0])

        if self.hypercolumns:
        # hypercolumns
            y4 = self.upsample4(y4)  # ->(*,64,h,w)
            y3 = self.upsample3(y3)  # ->(*,64,h,w)
            y2 = self.upsample2(y2)  # ->(*,64,h,w)
            y1 = self.upsample1(y1)  # ->(*,64,h,w)
            hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)
        else:
            hypercol = y4

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        return logits
