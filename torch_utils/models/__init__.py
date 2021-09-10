# https://github.com/TylerYep/torchinfo
try:
    from torchinfo import summary
except:
    print('[Warning] torchinfo not installed')

# https://github.com/davidtvs/pytorch-lr-finder
try:
    from torch_lr_finder import LRFinder
except:
    print('[Warning] torch_lr_finder not installed')

# TODO: https://github.com/Stonesjtu/pytorch_memlab

# https://github.com/Lyken17/pytorch-OpCounter
try:
    import thop
    import torch
    def profile(model, input_shape=(1, 3, 224, 224)):
        macs, params = thop.profile(model, inputs=(torch.randn(*input_shape), ))
        return {'macs': macs, 'params': params}
except:
    print('[Warning] thop not installed')

from . import layers
from .utils.freeze_bn import set_bn_eval
from .backbone import *
from .cls_models import *
from .seg_models import *
