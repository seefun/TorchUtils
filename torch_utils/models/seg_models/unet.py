import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from torch_utils.models.layers import *
from torch_utils.models import create_timm_model


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample, attention=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3x3_2 = conv3x3(in_channel, out_channel)
        if attention == 'scse':
            self.attention = SCSE(out_channel, r=8)
        elif attention == 'cbam':
            self.attention = CBAM(out_channel, reduction=16)
        else:
            self.attention = nn.Identity()
        self.conv1x1 = conv1x1(in_channel, out_channel)

    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.relu(self.bn2(x)))
        x = self.attention(x)
        x += self.conv1x1(self.upsample(inputs))  # shortcut
        return x


def get_encoder_info(name='resnest50d', verbose=True):
    model = create_timm_model(name, pretrained=False).eval()
    features = model(torch.rand(1, 3, 224, 224))
    encoder_channels = []
    for i, feat in enumerate(features):
        if verbose:
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
        super().__init__()
        self.drop_first = drop_first

        # center
        self.center = CenterBlock(
            encoder_channels[-1], center_channel)  # ->(*,512,h/32,w/32)

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
                 hypercolumns=True,
                 deepsupervision=False,
                 clshead=False):
        super().__init__()
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
        if neck:
            if hypercolumns:
                final_channel = sum(decoder_channels)
            else:
                final_channel = decoder_channels[-1]

            self.final_conv = nn.Sequential(
                conv3x3(final_channel, decoder_channels[4]),
                nn.SiLU(True),
                conv1x1(decoder_channels[4], self.out_channel)
            )
        else:
            if hypercolumns:
                final_channel = sum(encoder_channels)
            else:
                final_channel = encoder_channels[0]

            self.final_conv = nn.Sequential(
                conv1x1(final_channel, self.out_channel * 4),
                nn.SiLU(True),
                conv3x3(self.out_channel * 4, self.out_channel)
            )

        # upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # deepsupervision
        self.deepsupervision = deepsupervision
        if self.deepsupervision:
            self.deep4 = conv1x1(decoder_channels[0], 1)
            self.deep3 = conv1x1(decoder_channels[1], 1)
            self.deep2 = conv1x1(decoder_channels[2], 1)
            self.deep1 = conv1x1(decoder_channels[3], 1)

        # classification head
        self.clshead = clshead
        if self.clshead:
            self.clshead = nn.Sequential(
                FastGlobalConcatPool2d(),
                get_simple_fc(2 * encoder_channels[-1], out_channel)
            )

    @ autocast()
    def forward(self, inputs):
        # encoder
        y0, y1, y2, y3, y4 = self.backbone(inputs)
        # cls head
        if self.clshead:
            logits_clf = self.clshead(y4)

        if self.neck:
            y4, y3, y2, y1, y0 = self.neck([y0, y1, y2, y3, y4])
        else:
            y4 = self.upsample1(y4)  # ->(*,64,h//16,w//16)
            y3 = self.upsample1(y3)  # ->(*,64,h//8,w//8)
            y2 = self.upsample1(y2)  # ->(*,64,h//4,w//4)
            y1 = self.upsample1(y1)  # ->(*,64,h//2,w//2)
            y0 = self.upsample1(y0)  # ->(*,64,h,w)

        if self.hypercolumns:
            # hypercolumns
            y4 = self.upsample4(y4)  # ->(*,64,h,w)
            y3 = self.upsample3(y3)  # ->(*,64,h,w)
            y2 = self.upsample2(y2)  # ->(*,64,h,w)
            y1 = self.upsample1(y1)  # ->(*,64,h,w)
            hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)
        else:
            hypercol = y0

        if self.deepsupervision:
            s4 = self.deep4(y4)
            s3 = self.deep3(y3)
            s2 = self.deep2(y2)
            s1 = self.deep1(y1)
            logits_deeps = [s4, s3, s2, s1]

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        if self.deepsupervision and self.clshead:
            return logits, logits_deeps, logits_clf

        if self.clshead:
            return logits, logits_clf

        if self.deepsupervision:
            return logits, logits_deeps

        return logits


def get_hrnet(name, out_channel, pretrained=True):
    encoder_channels = get_encoder_info(name, False)
    model = UNet(backbone=name,
                 pretrained=pretrained,
                 neck=None,
                 encoder_channels=encoder_channels,
                 out_channel=out_channel,
                 hypercolumns=True)
    return model


def get_unet(name, out_channel, pretrained=True):
    encoder_channels = get_encoder_info(name, False)
    model = UNet(backbone=name,
                 pretrained=pretrained,
                 neck='unet',
                 drop_first=True,
                 encoder_channels=encoder_channels,
                 center_channel=512,
                 decoder_channels=[64, 64, 64, 64, 64],
                 out_channel=out_channel,
                 attention='cbam',
                 hypercolumns=True,
                 deepsupervision=False,
                 clshead=False)
    return model

# # Loss Example:
# # BCE + Lovasz Hinge with deepsupervision

# criterion_bce = nn.BCEWithLogitsLoss().to(device)
# criterion_lovasz = tu.BinaryLovaszLoss().to(device)
# criterion_clf = nn.BCEWithLogitsLoss().to(device)

# def criterion_lovasz_hinge_non_empty(criterion, logits_deep, y):
#     batch,c,h,w = y.size()
#     y2 = y.view(batch*c,-1)
#     logits_deep2 = logits_deep.view(batch*c,-1)

#     y_sum = torch.sum(y2, dim=1)
#     non_empty_idx = (y_sum!=0)

#     if non_empty_idx.sum()==0:
#         return torch.tensor(0)
#     else:
#         loss  = criterion(logits_deep2[non_empty_idx], 
#                           y2[non_empty_idx])
#         loss += criterion_lovasz(logits_deep2[non_empty_idx].view(-1,h,w), 
#                                  y2[non_empty_idx].view(-1,h,w))
#         return loss

# loss = criterion_bce(logits, y_true)                 
# loss += criterion_lovasz(logits.view(-1,h,w), y_true.view(-1,h,w))
# if deepsupervision:
#     for logits_deep in logits_deeps:
#          loss += 0.1 * criterion_lovasz_hinge_non_empty(criterion_bce, logits_deep, y_true)
# if clshead:                        
#     loss += criterion_clf(logits_clf.squeeze(-1),y_clf)
