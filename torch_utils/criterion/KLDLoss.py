# with softmax

import torch
import torch.nn.functional as F
import torch.nn as nn

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss()

    def forward(self, model_output, target):
        log = F.log_softmax(model_output, dim=1)
        loss = self.loss(log, target)
        return loss
