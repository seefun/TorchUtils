import timm

popular_models = {
    'resnest50d': 2048,
    'resnetv2_50x1_bitm': 2048,
    'swsl_resnext50_32x4d': 2048,
    'densenet121': 1024,
    'seresnext50_32x4d': 2048,
}


def create_timm_model(name, pretrained, num_classes=0, in_channel=3):
    # when in_channel==1, we suggest to manually modify the weight of the first layer by sum func
    # timm implementation uses strategy of circular copying RGB channel weight and rescale (not good for all cases)
    if num_classes:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channel)
    else:
        model = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=in_channel)
    return model
