import timm
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from torch.cuda.amp import autocast

from torch_utils.layers import *


def create_model(name, pretrained, pool=True):
    if pool: # pool with flatten (bs, c)
        model = timm.create_model(name, features_only=True, pretrained=pretrained)
    else:  # (bs, c, f_h, f_w)
        try:
            model = timm.create_model(name, features_only=True, pretrained=pretrained,
                                      global_pool='', num_classes = 0)
        except:
            model = timm.create_model(name, features_only=True, pretrained=pretrained,
                                      global_pool='')
    return model
        
def timm_create_model(name, pretrained, num_classes):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        
conv_models = {
    'resnest50d': 2048,
    'resnetv2_50x1_bitm': 2048,
    'swsl_resnext50_32x4d': 2048,
    'densenet121': 1024,
    'seresnext50_32x4d': 2048,
}

class ImageModel(nn.Module):
    
    def __init__(self, name='resnest50d', pretrained=True, pooling='concat', fc='multi-dropout', num_feature=2048, classes=1):
        super(ImageModel, self).__init__()
        self.model = create_model(name, pretrained, pool=False)
        
        if pooling == 'concat':
            self.pooling = FastGlobalConcatPool2d()
            num_feature *= 2
        elif pooling == 'gem':
            self.pooling = GeM_cw(num_feature)
        else:
            self.pooling = FastGlobalAvgPool2d()
        
        if fc == 'multi-dropout':
            self.fc = nn.Sequential(
                        MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes)
                        )
            
        if fc == 'attention':
            self.fc = nn.Sequential(
                        SEBlock(num_feature),
                        MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes)
                        )
            
        elif fc== '2layers':
            self.fc = nn.Sequential(
                        nn.Linear(num_feature, 512,  bias=False),
                        nn.BatchNorm1d(512),
                        torch.nn.SiLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(512, classes, bias=True),
                        )
        else:
            self.fc = nn.Linear(in_features=num_feature, out_features=classes, bias=True)
        
    @autocast()
    def forward(self, x):
        feature_map = self.model(x)[-1]
        embedding = self.pooling(feature_map)
        logits = self.fc(embedding)
        return logits, embedding
    