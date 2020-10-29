# TODO: criterion - binary / multiclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

# class topk_BCEWithLogits(nn.Module):
#     def __init__(self, top_k=0.75):
#         super(topk_BCEWithLogits, self).__init__()
#         self.top_k = top_k
#         self.loss = nn.BCEWithLogits(reduction='none')

#     def forward(self, input, target):
#         loss = self.loss(input, target)
#         if self.top_k == 1:
#             return torch.mean(loss)
#         else:
#             valid_loss, idxs = torch.topk(loss, round(self.top_k * loss.size()[0]), dim=0)    
#             return torch.mean(valid_loss)

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.eps,self.reduction = eps,reduction
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.eps)*nll + self.eps*(loss/c) 
 

from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.seight = weight
        self.reduction = reduction
        
    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0<=smoothing<1
        with torch.no_grad():
            targets = targets * (1.0-smoothing) + 0.5*smoothing
        return targets
    
    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)
        
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.mean()
            
        return loss
