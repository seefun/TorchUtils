# R-Drop: Regularized Dropout for Neural Networks
# A method of build-in Contrastive Learning

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDropFC(nn.Module):

    def __init__(self, in_chans, num_classes, drop_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_chans, num_classes, bias=True)

    def forward(self, x):
        logits1 = self.fc(self.dropout(x))
        logits2 = self.fc(self.dropout(x))
        return logits1, logits2


class RDropCELoss(nn.Module):
    def __init__(self, kl_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.kl_weight = kl_weight

    def forward(self, logits1, logits2, target):
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1))
        kl_2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1))
        loss = ce_loss + self.kl_weight * (kl_1 + kl_2) / 2
        return loss


if __name__ == '__main__':
    x = torch.randn(2, 32)
    fc = RDropFC(32, 2)
    logits1, logits2 = fc(x)
    target = torch.LongTensor([0, 1])
    print(logits1, logits2)

    criterion = RDropCELoss()
    loss = criterion(logits1, logits2, target)
    print(loss)
