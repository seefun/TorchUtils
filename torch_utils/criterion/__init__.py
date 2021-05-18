from .CrossEntropy import LabelSmoothingCrossEntropy, SmoothBCEwLogits
from .CrossEntropy import SoftTargetCrossEntropy, KLDivLoss
from .CrossEntropy import topkLoss

from .MetricLoss import CircleLoss, ArcFaceLoss, SupConLoss
from .MetricLoss import InfoNCE
from .MetricLoss import CrossBatchMemory
from .MetricLoss import MoCo, SupConLoss_MoCo