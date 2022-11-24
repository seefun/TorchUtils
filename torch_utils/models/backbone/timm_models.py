import timm
import torch.nn as nn
from torch_utils.models.layers import LayerNorm2d


popular_models = {
    'resnest50d': 2048,
    'resnetv2_50x1_bitm': 2048,
    'swsl_resnext50_32x4d': 2048,
    'densenet121': 1024,
    'seresnext50_32x4d': 2048,
}


def create_timm_model(name, pretrained=True, num_classes=0, in_channel=3):
    # when in_channel==1, we suggest to manually modify the weight of the first layer by sum func
    # timm implementation uses strategy of circular copying RGB channel weight and rescale (not good for all cases)
    if num_classes:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channel)
    else:
        model = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=in_channel)
    return model


def convert_ln(model):
    for child_name, child in model.named_children():
        if isinstance(child, timm.models.layers.norm.LayerNorm2d):
            sd = child.state_dict()
            m = LayerNorm2d(sd['weight'].shape[0])
            m.load_state_dict(sd)
            setattr(model, child_name, m)
        else:
            convert_ln(child)


def create_timm_backbone(name, pretrained=True, in_channel=3, fast_norm=True, **args):
    """ 使用pytorch-image-models(timm库)的模型作为backbone;
        from: https://github.com/rwightman/pytorch-image-models
    """
    timm_backbone = timm.create_model(name,
                                      features_only=True,
                                      pretrained=pretrained,
                                      in_chans=in_channel,
                                      exportable=True,
                                      num_classes=0,
                                      **args)
    if fast_norm:
        convert_ln(timm_backbone)

    # avoid adding head in timm models
    timm_backbone.forward_head = nn.Identity()
    timm_backbone.global_pool = None
    timm_backbone.pool = None
    timm_backbone.pooling = None
    timm_backbone.incre_modules = None
    timm_backbone.head = None
    timm_backbone.classifier = None
    timm_backbone.fc = None
    return timm_backbone
