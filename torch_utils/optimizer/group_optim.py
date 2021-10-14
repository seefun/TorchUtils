from torch.nn.modules.batchnorm import _BatchNorm


def get_params(model, key):
    # conv weight
    if key == 'conv_weight':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    # bn weight
    if key == 'bn_weight':
        for m in model.named_modules():
            if isinstance(m[1], _BatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight
            if isinstance(m[1], nn.GroupNorm):
                if m[1].weight is not None:
                    yield m[1].weight
    # all bias
    if key == 'bias':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], _BatchNorm):
                if m[1].bias is not None:
                    yield m[1].bias
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):
                if m[1].bias is not None:
                    yield m[1].bias


"""
optimizer = torch.optim.SGD(
            params=[
                {
                    'params': get_params(model_conv, key='conv_weight'),
                    'lr': 1 * LR,
                    'weight_decay': 1 * WD,
                },
                {
                    'params': get_params(model_conv, key='bn_weight'),
                    'lr': 1 * LR,
                    'weight_decay':  0.1 * WD,
                },
                {
                    'params': get_params(model_conv, key='bias'),
                    'lr': 2 * LR,
                    'weight_decay': 0.0,
                }],
            momentum=0.9)
"""
