# this is a example:

"""
from torch.nn.modules.batchnorm import _BatchNorm

def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], _BatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight
            if isinstance(m[1], nn.GroupNorm):
                if m[1].weight is not None:
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], _BatchNorm):
                if m[1].bias is not None:
                    yield m[1].bias
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):
                if m[1].bias is not None:
                    yield m[1].bias

optimizer = torch.optim.SGD(
            params=[
                {
                    'params': get_params(model_conv, key='1x'),
                    'lr': 1 * LR,
                    'weight_decay': 1e-4,
                },
                {
                    'params': get_params(model_conv, key='1y'),
                    'lr': 1 * LR,
                    'weight_decay': 0.0,
                },
                {
                    'params': get_params(model_conv, key='2x'),
                    'lr': 2 * LR,
                    'weight_decay': 0.0,
                }],
            momentum=0.9)
"""