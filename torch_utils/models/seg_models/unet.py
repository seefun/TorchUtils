import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from torch_utils.models.layers import conv1x1, conv3x3, SCSE, CBAM, CoordAttention, \
    ASPP, FastGlobalConcatPool2d, get_simple_fc
from torch_utils.models import create_timm_model


class PixelShuffleUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor=2):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel * scale_factor * scale_factor)
        self.ps = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.ps(feature)
        return feature


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel, aspp=False, dilations=[1, 6, 12, 18]):
        super().__init__()
        if aspp:
            self.conv = ASPP(inplanes=in_channel,
                             mid_c=out_channel,
                             dilations=dilations)
        else:
            self.conv = conv3x3(in_channel, out_channel)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample, attention=None, dropout=0.125, upsample_shape=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.dropout = nn.Dropout2d(dropout)
        self.upsample = nn.Sequential()
        self._upsample_shape = upsample_shape
        if upsample:
            if self._upsample_shape:
                self.upsample.add_module('upsample', nn.Upsample(size=self._upsample_shape, mode='bilinear', align_corners=True))
            else:
                self.upsample.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv3x3_1 = conv3x3(in_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3x3_2 = conv3x3(out_channel, out_channel)
        if attention == 'scse':
            self.attention = SCSE(out_channel, r=16)
        elif attention == 'cbam':
            self.attention = CBAM(out_channel, reduction=16)
        elif attention == 'coord':
            self.attention = CoordAttention(out_channel, out_channel, reduction=16)
        else:
            self.attention = nn.Identity()
        self.conv1x1 = conv1x1(in_channel, out_channel)

    @property
    def upsample_shape(self):
        return self._upsample_shape

    @upsample_shape.setter
    def upsample_shape(self, value):
        self._upsample_shape = value
        self.upsample.upsample = nn.Upsample(size=self._upsample_shape, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        x = self.dropout(F.relu(self.bn1(inputs)))
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


def get_deepsupervision_head(in_channel, out_channel, relu=False):
    if relu:
        return nn.Sequential(
                    nn.BatchNorm2d(in_channel),
                    nn.SiLU(True),
                    nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
                )
    else:
        return nn.Sequential(
                    nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
                )


class UNet_neck(nn.Module):
    '''
    UNet neck
    Input feature list and output feature list
    '''

    def __init__(self,
                 encoder_channels=[64, 256, 512, 1024, 2048],
                 center_channel=512,
                 decoder_channels=[64, 64, 64, 64, 64],
                 attention='cbam',
                 drop_first=True,
                 aspp=False,
                 dilations=[1, 6, 12, 18],
                 dropout=0.125):
        super().__init__()
        self.drop_first = drop_first

        # center
        self.center = CenterBlock(
            encoder_channels[-1],
            center_channel,
            aspp=aspp,
            dilations=dilations)  # ->(*,512,h/32,w/32)

        # decoder
        self.decoder4 = DecodeBlock(
            center_channel + encoder_channels[4], decoder_channels[0],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(
            decoder_channels[0] + encoder_channels[3], decoder_channels[1],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(
            decoder_channels[1] + encoder_channels[2], decoder_channels[2],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(
            decoder_channels[2] + encoder_channels[1], decoder_channels[3],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/2,w/2)
        if drop_first:
            self.decoder0 = DecodeBlock(
                decoder_channels[3], decoder_channels[4], upsample=True,
                attention=attention, dropout=dropout)  # ->(*,64,h,w)
        else:
            self.decoder0 = DecodeBlock(
                decoder_channels[3] + encoder_channels[0], decoder_channels[4], upsample=True,
                attention=attention, dropout=dropout)  # ->(*,64,h,w)

    def forward(self, inputs):
        # encoder
        x0, x1, x2, x3, x4 = inputs

        # center
        y5 = self.center(x4)  # ->(*,512,h/32,w/32)

        # decoder
        h = x0.shape[2] // 2 * 4 + x0.shape[2] % 2
        w = x0.shape[3] // 2 * 4 + x0.shape[3] % 2
        self.decoder0.upsample_shape = (h, w)
        self.decoder1.upsample_shape = (x0.shape[2], x0.shape[3])
        self.decoder2.upsample_shape = (x1.shape[2], x1.shape[3])
        self.decoder3.upsample_shape = (x2.shape[2], x2.shape[3])
        self.decoder4.upsample_shape = (x3.shape[2], x3.shape[3])

        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        if self.drop_first:
            y0 = self.decoder0(y1)  # ->(*,64,h,w)
        else:
            y0 = self.decoder0(torch.cat([x0, y1], dim=1))  # ->(*,64,h,w)

        return [y4, y3, y2, y1, y0]


class unet_ps_neck(nn.Module):
    '''
    UNet with pixelshuffle neck
    Input feature list and output feature list
    '''

    def __init__(self,
                 encoder_channels=[64, 256, 512, 1024, 2048],
                 center_channel=512,
                 decoder_channels=[128, 128, 64, 64, 64],
                 attention='cbam',
                 drop_first=True,
                 aspp=False,
                 dilations=[1, 6, 12, 18],
                 dropout=0.125):
        super().__init__()
        self.drop_first = drop_first

        # center
        self.center = CenterBlock(
            encoder_channels[-1],
            center_channel,
            aspp=aspp,
            dilations=dilations)  # ->(*,512,h/32,w/32)

        self.ps4 = PixelShuffleUpSample(encoder_channels[4], decoder_channels[0])
        self.ps3 = PixelShuffleUpSample(encoder_channels[3], decoder_channels[1])
        self.ps2 = PixelShuffleUpSample(encoder_channels[2], decoder_channels[2])
        self.ps1 = PixelShuffleUpSample(encoder_channels[1], decoder_channels[3])

        # decoder
        self.decoder4 = DecodeBlock(
            center_channel, decoder_channels[0],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(
            decoder_channels[0] * 2, decoder_channels[1],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(
            decoder_channels[1] * 2, decoder_channels[2],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(
            decoder_channels[2] * 2, decoder_channels[3],
            upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h/2,w/2)
        if self.drop_first:
            self.decoder0 = DecodeBlock(
                decoder_channels[3] * 2, decoder_channels[4],
                upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h,w)
        else:
            self.decoder0 = DecodeBlock(
                decoder_channels[3] * 2 + encoder_channels[0], decoder_channels[4],
                upsample=True, attention=attention, dropout=dropout)  # ->(*,64,h,w)

    def forward(self, inputs):
        # encoder
        x0, x1, x2, x3, x4 = inputs

        # center
        y5 = self.center(x4)  # ->(*,512,h/32,w/32)

        x4 = self.ps4(x4)
        x3 = self.ps3(x3)
        x2 = self.ps2(x2)
        x1 = self.ps1(x1)

        # decoder
        y4 = self.decoder4(y5)  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x4, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x3, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x2, y2], dim=1))  # ->(*,64,h/2,w/2)
        if self.drop_first:
            y0 = self.decoder0(torch.cat([x1, y1], dim=1))  # ->(*,64,h,w)
        else:
            y0 = self.decoder0(torch.cat([x0, x1, y1], dim=1))  # ->(*,64,h,w)

        return [y4, y3, y2, y1, y0]


class UNet(nn.Module):
    def __init__(self,
                 backbone='resnest50d',
                 pretrained=True,
                 in_channel=3,
                 neck='unet',  # unet or none
                 drop_first=True,
                 encoder_channels=[64, 256, 512, 1024, 2048],
                 center_channel=512,
                 decoder_channels=[256, 128, 64, 32, 16],
                 dropout=0.125,
                 out_channel=1,
                 attention='cbam',
                 aspp=False,
                 dilations=[1, 6, 12, 18],
                 hypercolumns=True,
                 deepsupervision=False,
                 clshead=False):
        super().__init__()
        self.backbone = create_timm_model(backbone, pretrained, in_channel=in_channel)
        self.attention = attention
        self.hypercolumns = hypercolumns
        self.out_channel = out_channel

        if neck == 'unet':
            self.neck = UNet_neck(encoder_channels=encoder_channels,
                                  center_channel=center_channel,
                                  decoder_channels=decoder_channels,
                                  attention=attention,
                                  drop_first=drop_first,
                                  aspp=aspp,
                                  dilations=dilations,
                                  dropout=dropout)
        elif neck == 'unet_ps':
            self.neck = unet_ps_neck(encoder_channels=encoder_channels,
                                     center_channel=center_channel,
                                     decoder_channels=decoder_channels,
                                     attention=attention,
                                     drop_first=drop_first,
                                     aspp=aspp,
                                     dilations=dilations,
                                     dropout=dropout)
        else:
            self.neck = None

        # final conv
        if neck:
            if hypercolumns:
                final_channel = sum(decoder_channels)
            else:
                final_channel = decoder_channels[-1]

            self.head = nn.Sequential(
                conv3x3(final_channel, decoder_channels[4]),
                nn.BatchNorm2d(decoder_channels[4]),
                nn.ReLU(True),
                nn.Conv2d(decoder_channels[4], self.out_channel,
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
            )
        else:
            if hypercolumns:
                final_channel = sum(encoder_channels)
            else:
                final_channel = encoder_channels[0]

            head_hidden_dim = min(self.out_channel * 16, final_channel)

            self.head = nn.Sequential(
                conv1x1(final_channel, head_hidden_dim),
                nn.BatchNorm2d(head_hidden_dim),
                nn.ReLU(True),
                nn.Conv2d(head_hidden_dim, self.out_channel,
                          kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
            )

        # deepsupervision
        self.deepsupervision = deepsupervision
        if self.deepsupervision:
            if neck:
                self.deep4 = get_deepsupervision_head(decoder_channels[0], self.out_channel, relu=True)
                self.deep3 = get_deepsupervision_head(decoder_channels[1], self.out_channel, relu=True)
                self.deep2 = get_deepsupervision_head(decoder_channels[2], self.out_channel, relu=True)
                self.deep1 = get_deepsupervision_head(decoder_channels[3], self.out_channel, relu=True)
            else:
                self.deep4 = get_deepsupervision_head(encoder_channels[-1], self.out_channel)
                self.deep3 = get_deepsupervision_head(encoder_channels[-2], self.out_channel)
                self.deep2 = get_deepsupervision_head(encoder_channels[-3], self.out_channel)
                self.deep1 = get_deepsupervision_head(encoder_channels[-4], self.out_channel)

        # classification head
        self.clshead = clshead
        if self.clshead:
            self.clshead = nn.Sequential(
                FastGlobalConcatPool2d(),
                get_simple_fc(2 * encoder_channels[-1], out_channel)
            )

    @ autocast()
    def forward(self, inputs):
        # shape
        h, w = inputs.shape[2], inputs.shape[3]
        # encoder
        y0, y1, y2, y3, y4 = self.backbone(inputs)
        # cls head
        if self.clshead:
            logits_clf = self.clshead(y4)

        if self.neck:
            y4, y3, y2, y1, y0 = self.neck([y0, y1, y2, y3, y4])

        y0 = F.interpolate(y0, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
        all_upsample_to_original = False

        if self.hypercolumns:
            # hypercolumns
            y4 = F.interpolate(y4, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
            y3 = F.interpolate(y3, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
            y2 = F.interpolate(y2, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
            y1 = F.interpolate(y1, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
            all_upsample_to_original = True
            hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)
        else:
            hypercol = y0

        if self.deepsupervision:
            if not all_upsample_to_original:
                y4 = F.interpolate(y4, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
                y3 = F.interpolate(y3, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
                y2 = F.interpolate(y2, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
                y1 = F.interpolate(y1, size=(h, w), mode='bilinear', align_corners=True)  # ->(*,64,h,w)
                all_upsample_to_original = True
            s4 = self.deep4(y4)
            s3 = self.deep3(y3)
            s2 = self.deep2(y2)
            s1 = self.deep1(y1)
            logits_deeps = [s4, s3, s2, s1]

        # final conv
        logits = self.head(hypercol)  # ->(*,1,h,w)

        if self.deepsupervision and self.clshead:
            return logits, logits_deeps, logits_clf

        if self.clshead:
            return logits, logits_clf

        if self.deepsupervision:
            return logits, logits_deeps

        return logits


def get_hrnet(name, out_channel, in_channel=3, pretrained=True):
    encoder_channels = get_encoder_info(name, False)
    model = UNet(backbone=name,
                 pretrained=pretrained,
                 in_channel=in_channel,
                 neck=None,
                 encoder_channels=encoder_channels,
                 out_channel=out_channel,
                 hypercolumns=True)
    return model


def get_unet(name, out_channel, in_channel=3, attention='cbam', aspp=False, pretrained=True):
    encoder_channels = get_encoder_info(name, False)
    model = UNet(backbone=name,
                 pretrained=pretrained,
                 in_channel=in_channel,
                 neck='unet',
                 drop_first=True,
                 encoder_channels=encoder_channels,
                 center_channel=512,
                 decoder_channels=[64, 64, 64, 64, 64],
                 dropout=0.0,
                 out_channel=out_channel,
                 attention=attention,
                 aspp=aspp,
                 hypercolumns=True,
                 deepsupervision=False,
                 clshead=False)
    return model


def get_unet_ps(name, out_channel, in_channel=3, attention='scse', aspp=False, pretrained=True):
    encoder_channels = get_encoder_info(name, False)
    model = UNet(backbone=name,
                 pretrained=pretrained,
                 in_channel=in_channel,
                 neck='unet_ps',
                 drop_first=False,
                 encoder_channels=encoder_channels,
                 center_channel=320,
                 decoder_channels=[128, 128, 64, 64, 64],
                 dropout=0.125,
                 out_channel=out_channel,
                 attention=attention,
                 aspp=aspp,
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
