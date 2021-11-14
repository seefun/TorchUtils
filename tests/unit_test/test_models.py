import torch
from torch_utils import models
from torch_utils import advanced


class TestModel:
    inputs = torch.rand(1, 3, 224, 224)

    def test_cls_model(self):
        pooling = ['gem', 'concat', 'avg']
        fc = ['multi-dropout', 'attention', '2layers', 'simple', None]

        for pool_i in pooling:
            model_conv = models.ImageModel(pooling=pool_i, pretrained=False).eval()
            model_conv(TestModel.inputs)
        for fc_i in fc:
            model_conv = models.ImageModel(fc=fc_i, pretrained=False).eval()
            model_conv(TestModel.inputs)

    def test_unet(self):
        model_conv = models.UNet(pretrained=False, aspp=True, hypercolumns=False, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

        model_conv = models.UNet(pretrained=False, aspp=True, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

    def test_unet_ps(self):
        model_conv = models.UNet(pretrained=False, neck='unet_ps', aspp=True, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

    def test_hrnet(self):
        model_conv = models.UNet(pretrained=False, neck=None, hypercolumns=False, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

        model_conv = models.UNet(pretrained=False, neck=None, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

    def test_DolgNet(self):
        model_conv = advanced.DolgNet('resnet101', False, 224, 3, 512, 512).eval()
        model_conv(TestModel.inputs)

    def test_hybrid(self):
        model_conv = models.HybridModel(vit='swin_base_patch4_window7_224',
                                        embedder='tf_efficientnet_b4_ns',
                                        classes=1, input_size=448, pretrained=False)
        model_conv(torch.rand(1, 3, 448, 448))
