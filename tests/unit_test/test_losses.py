import torch
from torch_utils import criterion


class TestBinaryLoss:
    y_pred = torch.tensor([0.88, -0.10, 0.55]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)

    def test_bitempered_loss(self):
        loss_fn = criterion.BiTemperedLogisticLoss()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_focal_loss(self):
        loss_fn = criterion.BinaryFocalLoss()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_lovasz_loss(self):
        loss_fn = criterion.BinaryLovaszLoss()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_smooth_bce(self):
        loss_fn = criterion.SmoothBCEwLogits()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_kld(self):
        loss_fn = criterion.KLDivLosswSoftmax()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_soft_target_ce(self):
        loss_fn = criterion.SoftTargetCrossEntropy()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)


class TestMultiLoss:
    y_pred = torch.tensor([[+1, -1, -1, -1],
                           [-1, +1, -1, -1],
                           [-1, -1, +1, -1],
                           [-1, -1, -1, +1]]).float()
    y_true = torch.tensor([0, 1, 2, 3]).long()

    def test_focal(self):
        loss_fn = criterion.FocalLoss()
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)

    def test_lovasz(self):
        loss_fn = criterion.LovaszLoss()
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)

    def test_smoothing(self):
        loss_fn = criterion.LabelSmoothingCrossEntropy()
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)
