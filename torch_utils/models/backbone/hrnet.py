import timm


def create_hrnet_model(name, pretrained=True, num_classes=0, in_channel=3, seg_mode=True):
    if num_classes:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channel)
    else:
        model = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=in_channel)
        if seg_mode:
            model.incre_modules = None
            model.num_features = 256
    return model

# aux fcnhead 3x3(c->c) -> 1x1(c->out_c)

# pretrained inchannel changed: conv.weight = nn.Parameter(conv.weight.repeat(xxx) just like timm
