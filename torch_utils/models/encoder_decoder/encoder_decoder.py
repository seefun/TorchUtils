# Encoder-Decoder (Decoupled)
from abc import ABCMeta
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.models.backbone.timm_models import create_timm_model


class EncoderDecoder(nn.Module):
    """Encoder-Decoder
    Args:
        backbone: nn.Module returns list of image features (e.g. ResNet)
        head: nn.Module witch get list feature (e.g. FCNHead)
        neck: nn.Module witch get list feature and return list of featuers (e.g. UNetNeck)
    """

    def __init__(self, backbone, head, neck=None):
        super().__init__()
        # Backbone
        if isinstance(backbone, str):
            create_timm_model(backbone)
        else:
            self.backbone = backbone
        # Neck
        if neck is None:
            self.neck = nn.Identity()
        else:
            self.neck = neck
        # Head
        self.head = head

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)
        neck_out = self.neck(backbone_out)
        head_out = self.head(neck_out)
        return head_out


def transform_inputs(inputs,
                     in_index=None,
                     input_transform='resize_concat',
                     interpolate_mode='bilinear'):
    """
    inputs (list[Tensor]): list of feature map.
    in_index (list[int]): backbone/neck output index list.
    input_transform (str): 'resize_concat' or 'multiple_select'.
    interpolate_mode (str): 'nearest' or 'bilinear'.
    """
    if in_index is None:
        in_index = list(range(len(inputs)))
    if input_transform == 'resize_concat':
        inputs = [inputs[i] for i in in_index]
        upsampled_inputs = [
            F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode=interpolate_mode,
            ) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)

    elif input_transform == 'multiple_select':
        inputs = [inputs[i] for i in in_index]
    elif input_transform == 'single_select':
        inputs = inputs[in_index]

    return inputs
