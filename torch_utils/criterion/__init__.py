from .cross_entropy import LabelSmoothingCrossEntropy, SmoothBCEwLogits
from .cross_entropy import SoftTargetCrossEntropy
from .cross_entropy import KLDivLosswSoftmax, KLDivLosswLogits, JSDivLosswSoftmax, JSDivLosswLogits
from .cross_entropy import topkLoss

from .metric_loss import CircleLoss, ArcFaceLoss, SupConLoss
from .metric_loss import InfoNCE
from .metric_loss import CrossBatchMemory
from .metric_loss import MoCo, SupConLoss_MoCo

# seg losses
from .cross_entropy import SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
from .lovasz import BinaryLovaszLoss, LovaszLoss
from .focal import BinaryFocalLoss, FocalLoss
from .bitempered_loss import BiTemperedLogisticLoss, BinaryBiTemperedLogisticLoss
from .dice import DiceLoss, TverskyLoss
from .rmi import RMILoss
