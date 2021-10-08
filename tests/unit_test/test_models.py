import torch
from torch_utils import models


class TestModel:
    inputs = torch.rand(2, 3, 224, 224)

    def test_cls_model(self):
        pooling = ['gem', 'concat', 'avg']
        fc = ['multi-dropout', 'attention', '2layers', 'simple']

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

    def test_hrnet(self):
        model_conv = models.UNet(pretrained=False, neck=None, hypercolumns=False, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)

        model_conv = models.UNet(pretrained=False, neck=None, deepsupervision=True, clshead=True).eval()
        model_conv(TestModel.inputs)
