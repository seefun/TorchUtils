from torch import nn
from torch.cuda.amp import autocast

from torch_utils.models.layers import FastGlobalConcatPool2d, FastGlobalAvgPool2d, GeM_cw, MultiSampleDropoutFC, SEBlock
from torch_utils.models import create_timm_model


class ImageModel(nn.Module):

    def __init__(self,
                 name='resnest50d',
                 pretrained=True,
                 pooling='concat',
                 fc='multi-dropout',
                 num_feature=2048,
                 classes=1):
        super(ImageModel, self).__init__()
        self.model = create_timm_model(name, pretrained)

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

        elif fc == '2layers':
            self.fc = nn.Sequential(
                        nn.Linear(num_feature, 512, bias=False),
                        nn.BatchNorm1d(512),
                        nn.SiLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(512, classes, bias=True))
        else:
            self.fc = nn.Linear(in_features=num_feature, out_features=classes, bias=True)

    @autocast()
    def forward(self, x):
        feature_map = self.model(x)[-1]
        embedding = self.pooling(feature_map)
        logits = self.fc(embedding)
        return logits, embedding
