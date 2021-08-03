# https://github.com/TylerYep/torchinfo
from torchinfo import summary

# https://github.com/davidtvs/pytorch-lr-finder
from torch_lr_finder import LRFinder

# TODO: https://github.com/Stonesjtu/pytorch_memlab

# https://github.com/Lyken17/pytorch-OpCounter
import thop
import torch
def profile(model, input_shape=(1, 3, 224, 224)):
    macs, params = thop.profile(model, inputs=(torch.randn(*input_shape), ))
    return {'macs': macs, 'params': params}

from .freeze_bn import set_bn_eval

from .timm_model import create_model, timm_create_model, ImageModel
