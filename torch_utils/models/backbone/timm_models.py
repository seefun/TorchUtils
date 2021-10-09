import timm

popular_models = {
    'resnest50d': 2048,
    'resnetv2_50x1_bitm': 2048,
    'swsl_resnext50_32x4d': 2048,
    'densenet121': 1024,
    'seresnext50_32x4d': 2048,
}


def create_timm_model(name, pretrained, num_classes=0):
    if num_classes:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    else:
        model = timm.create_model(name, features_only=True, pretrained=pretrained)
    return model
