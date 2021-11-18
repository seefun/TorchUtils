import torch
from torch import nn
from torch.cuda.amp import autocast

from torch_utils.models.layers import FastGlobalConcatPool2d, FastGlobalAvgPool2d, GeM_cw, MultiSampleDropoutFC, SEBlock
from torch_utils.models import create_timm_model


class ImageModel(nn.Module):

    def __init__(self,
                 name='seresnext50_32x4d',
                 pretrained=True,
                 pooling='concat',
                 fc='multi-dropout',
                 num_feature=2048,
                 classes=1,
                 in_channel=3):
        super(ImageModel, self).__init__()
        self.model = create_timm_model(name, pretrained, in_channel=in_channel)

        if pooling == 'concat':
            self.pooling = FastGlobalConcatPool2d()
            num_feature *= 2
        elif pooling == 'gem':
            self.pooling = GeM_cw(num_feature)
        else:
            self.pooling = FastGlobalAvgPool2d()

        if fc == 'multi-dropout':
            self.fc = nn.Sequential(
                        MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes))

        if fc == 'attention':
            self.fc = nn.Sequential(
                        SEBlock(num_feature),
                        MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes))

        elif fc == 'dropout':
            self.fc = nn.Sequential(
                        nn.Dropout(0.25),
                        nn.Linear(num_feature, classes, bias=True))

        elif fc == '2layers':
            self.fc = nn.Sequential(
                        nn.Linear(num_feature, 1280, bias=False),
                        nn.BatchNorm1d(1280),
                        nn.SiLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(1280, classes, bias=True))

        else:
            self.fc = nn.Linear(in_features=num_feature, out_features=classes, bias=True)

    @autocast()
    def forward(self, x):
        feature_map = self.model(x)[-1]
        embedding = self.pooling(feature_map)
        logits = self.fc(embedding)
        return logits, embedding


def get_encoder_last_channel(name='seresnext50_32x4d', verbose=True):
    model = create_timm_model(name, pretrained=False).eval()
    features = model(torch.rand(1, 3, 224, 224))
    if verbose:
        for i, feat in enumerate(features):
            print('Feature [%d], channel num: %d' % (i, feat.shape[1]))
    return features[-1].shape[1]


def get_conv_model(name='seresnext50_32x4d',
                   pretrained=True,
                   pooling='avg',
                   fc='multi-dropout',
                   classes=1,
                   in_channel=3):
    encoder_last_channel = get_encoder_last_channel(name, verbose=False)
    conv_model = ImageModel(name=name, 
                            pretrained=pretrained,
                            pooling=pooling,
                            fc=fc,
                            num_feature=encoder_last_channel,
                            classes=classes,
                            in_channel=in_channel)
    return conv_model
