import torch
from torch_utils import layers
from torch_utils import set_bn_eval


class TestModel:
    feature = torch.rand(2, 32, 56, 56)

    def test_anti_alias(self):
        anti_alias = layers.Anti_Alias_Filter(32).eval()
        assert anti_alias(TestModel.feature).shape == (2, 32, 56, 56)
