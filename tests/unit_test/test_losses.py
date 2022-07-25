import torch
from torch_utils import criterion


class TestBinaryLoss:
    y_pred = torch.tensor([0.88, -0.10, 0.55]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1.0, 0.0, 1.0])).view(1, 1, 1, -1)

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

    def test_kld_softmax(self):
        loss_fn = criterion.KLDivLosswSoftmax()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_kld_sigmoid(self):
        loss_fn = criterion.KLDivLosswLogits()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_jsd_softmax(self):
        loss_fn = criterion.JSDivLosswSoftmax()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_jsd_sigmoid(self):
        loss_fn = criterion.JSDivLosswLogits()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_soft_target_ce(self):
        loss_fn = criterion.SoftTargetCrossEntropy()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_toolbelt_ce_binary(self):
        loss_fn = criterion.SoftBCEWithLogitsLoss()
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_rmi(self):
        loss_fn = criterion.RMILoss()
        loss = loss_fn(torch.rand(1, 2, 32, 32), torch.rand(1, 2, 32, 32))
        print(loss)

    def test_dice_binary(self):
        loss_fn = criterion.DiceLoss('binary')
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_tversky_binary(self):
        loss_fn = criterion.TverskyLoss('binary')
        loss = loss_fn(TestBinaryLoss.y_pred, TestBinaryLoss.y_true)
        print(loss)

    def test_wasserstein(self):
        loss_fn = criterion.SinkhornDistance()
        loss, P, C = loss_fn(TestBinaryLoss.y_pred.view(-1, 1),
                             TestBinaryLoss.y_true.view(-1, 1))
        print(loss)

    def test_polybceloss(self):
        loss_fn = criterion.PolyBCELoss()
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

    def test_toolbelt_ce_multiply(self):
        loss_fn = criterion.SoftCrossEntropyLoss()
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)

    def test_dice_multiply(self):
        loss_fn = criterion.DiceLoss('multiclass')
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)

    def test_tversky_multiply(self):
        loss_fn = criterion.TverskyLoss('multiclass')
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)

    def test_polyloss(self):
        loss_fn = criterion.PolyLoss()
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)
        # Compare with CE
        loss_fn = criterion.PolyLoss(0)
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)
        ce = torch.nn.CrossEntropyLoss()
        ce_loss = ce(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(ce_loss)
        assert abs(loss - ce_loss) < 0.000001

    def test_seesawloss(self):
        loss_fn = criterion.SeesawLoss(len(TestMultiLoss.y_true.unique()))
        loss = loss_fn(TestMultiLoss.y_pred, TestMultiLoss.y_true)
        print(loss)


class TestRegressionLoss:
    y_pred = torch.tensor([0.88, -0.10, 0.55]).view(-1, 1)
    y_true = torch.tensor(([1.0, 0.4, 1.1])).view(-1, 1)

    def test_bmc_loss(self):
        loss_fn = criterion.BMCLoss()
        loss = loss_fn(TestRegressionLoss.y_pred, TestRegressionLoss.y_true)
        print(loss)
