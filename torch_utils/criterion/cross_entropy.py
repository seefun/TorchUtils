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
    """ CrossEntropy Loss for soft label """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class KLDivLosswSoftmax(nn.Module):
    """KL-divergence with softmax"""

    def __init__(self):
        super(KLDivLosswSoftmax, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_output, target):
        log = F.log_softmax(model_output, dim=-1)
        loss = self.loss(log, target)
        return loss


class KLDivLosswLogits(nn.Module):
    """KL-divergence with Logits"""

    def __init__(self):
        super(KLDivLosswLogits, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_output, target):
        log = model_output.softmax(-1).log()
        loss = self.loss(log, target.softmax(-1))
        return loss


class JSDivLosswSoftmax(nn.Module):
    """JS-divergence with softmax"""

    def __init__(self):
        super(JSDivLosswSoftmax, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_output, target):
        pred = model_output.softmax(-1)
        m = (pred + target) / 2.0
        loss = 0.5 * (self.loss(pred.log(), m) + self.loss(target.log(), m))
        return loss


class JSDivLosswLogits(nn.Module):
    """JS-divergence with Logits"""

    def __init__(self):
        super(JSDivLosswLogits, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_output, target):
        pred = model_output.softmax(-1)
        target = target.softmax(-1)
        m = (pred + target) / 2.0
        loss = 0.5 * (self.loss(pred.log(), m) + self.loss(target.log(), m))
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


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self, weight=None, ignore_index=None, reduction="mean", smooth_factor=None, pos_weight=None
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input, target):
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)

        loss = F.binary_cross_entropy_with_logits(
            input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            not_ignored_mask = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index=255, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

class DoubleDropoutLoss(nn.Module):
    """Loss function for double dropout"""

    def __init__(self, criterion=nn.CrossEntropyLoss(), contrast_weight=0.5):
        super(DoubleDropoutLoss, self).__init__()
        self.criterion = criterion
        self.contrast_weight = contrast_weight
        self.jsd = JSDivLosswLogits()

    def forward(self, model_output1, model_output2, target):
        loss1 =  self.criterion(model_output1, target)
        loss2 =  self.criterion(model_output2, target)
        jsd = self.jsd(model_output1, model_output2)
        loss = (loss1 + loss2) / 2.0 + self.contrast_weight * jsd
        return loss
