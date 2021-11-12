# Hybrid Vision Transformer
import torch.nn as nn
import timm
from timm.models.vision_transformer_hybrid import HybridEmbed


class HybridModel(nn.Module):
    def __init__(self, vit='swin_base_patch4_window7_224', embedder='tf_efficientnet_b4_ns',
                 classes=1, input_size=448, pretrained=True):
        super(HybridModel, self).__init__()
        self.vit = timm.create_model(vit, pretrained=pretrained)
        self.embedder = timm.create_model(embedder, features_only=True, out_indices=[2], pretrained=pretrained)
        self.vit.patch_embed = HybridEmbed(self.embedder, img_size=input_size, embed_dim=128)
        self.n_features = self.vit.head.in_features
        self.vit.head = nn.Linear(self.n_features, classes)

    def forward(self, images):
        features = self.vit.forward_features(images)
        x = self.vit.head(features)
        return x


def get_hybrid_swin(swin_type='base', embedder='tf_efficientnet_b4_ns', classes=1, pretrained=True):
    # input size 448x448
    assert swin_type in ['large', 'base', 'small', 'tiny']
    swin_name = 'swin_' + swin_type + '_patch4_window7_224'
    model = HybridModel(vit=swin_name, embedder=embedder, classes=classes, input_size=448, pretrained=pretrained)
    return model
