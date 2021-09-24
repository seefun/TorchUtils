from .cross_entropy import LabelSmoothingCrossEntropy, SmoothBCEwLogits
from .cross_entropy import SoftTargetCrossEntropy, KLDivLoss
from .cross_entropy import topkLoss

from .metric_loss import CircleLoss, ArcFaceLoss, SupConLoss
from .metric_loss import InfoNCE
from .metric_loss import CrossBatchMemory
from .metric_loss import MoCo, SupConLoss_MoCo

from .lovasz import BinaryLovaszLoss, LovaszLoss
