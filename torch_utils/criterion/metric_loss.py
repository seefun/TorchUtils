# from https://kevinmusgrave.github.io/pytorch-metric-learning/losses/

from pytorch_metric_learning.losses import CircleLoss, ArcFaceLoss, SupConLoss
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

# logist, embeddings = model_conv(input)
# loss_func = ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64).to(torch.device('cuda'))
# loss_func = CircleLoss(m=0.4, gamma=80).to(torch.device('cuda'))
# loss_func = SupConLoss(temperature=0.1).to(torch.device('cuda'))
# metric_loss = loss_func(embeddings, labels) # in your training for-loop

CircleLoss = CircleLoss
ArcFaceLoss = ArcFaceLoss
SupConLoss = SupConLoss

InfoNCE = NTXentLoss
CrossBatchMemory = CrossBatchMemory
# CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None)


class MoCo(CrossBatchMemory):
    def __init__(self, embedding_size, memory_size):
        super(MoCo, self).__init__(NTXentLoss(temperature=0.1),
                                   embedding_size,
                                   memory_size)


class SupConLoss_MoCo(CrossBatchMemory):
    def __init__(self, embedding_size, memory_size):
        super(SupConLoss_MoCo, self).__init__(SupConLoss(temperature=0.1),
                                              embedding_size,
                                              memory_size)
