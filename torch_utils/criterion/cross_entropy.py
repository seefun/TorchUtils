import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
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


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class KLDivLoss(nn.Module):
    """KL-divergence with softmax"""

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss()

    def forward(self, model_output, target):
        log = F.log_softmax(model_output, dim=1)
        loss = self.loss(log, target)
        return loss


class topkLoss(nn.Module):
    """topkLoss: Online Hard Example Mining"""

    def __init__(self, loss, top_k=0.75):
        super(topkLoss, self).__init__()
        self.top_k = top_k
        self.loss = loss

    def forward(self, input, target):
        loss = self.loss(input, target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, round(self.top_k * loss.size()[0]), dim=0)
            return torch.mean(valid_loss)

# class JsdCrossEntropy(nn.Module):
#     """ Jensen-Shannon Divergence + Cross-Entropy Loss for AugMix
#     Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
#     From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
#     https://arxiv.org/abs/1912.02781
#     Hacked together by / Copyright 2020 Ross Wightman
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/jsd.py
#     require: AugMixDataset(split data) + split_data_collate + JsdCrossEntropy
#     optional: split bn for different strength of augmentations

#     Example:
#         >>> num_aug_splits = 3
#         >>> # from timm.models import convert_splitbn_model
#         >>> # model = convert_splitbn_model(model, max(num_aug_splits, 2))
#         >>> dataset = AugMixDataset(dataset_train, num_splits=num_aug_splits) # TODO: re-implement augmix dataset from timm
#         >>> dataloader = DataLoader(dataset, bs, collate_fn=split_data_collate)
#         >>> # split_data_collate: (s,s_a1,s_a2) to [s1,s2,s3, s1_a1,s2_a1,s3_a1, s1_a2,s2_a2,s3_a2]
#         >>> train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=0.1).cuda()
#     """
#     def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
#         super().__init__()
#         self.num_splits = num_splits
#         self.alpha = alpha
#         if smoothing is not None and smoothing > 0:
#             self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
#         else:
#             self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

#     def __call__(self, output, target):
#         split_size = output.shape[0] // self.num_splits
#         assert split_size * self.num_splits == output.shape[0]
#         logits_split = torch.split(output, split_size)

#         # Cross-entropy is only computed on clean images
#         loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
#         probs = [F.softmax(logits, dim=1) for logits in logits_split]

#         # Clamp mixture distribution to avoid exploding KL divergence
#         logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
#         loss += self.alpha * sum([F.kl_div(
#             logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
#         return loss
