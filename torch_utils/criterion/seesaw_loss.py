# from https://github.com/jahongir7174/SeesawLoss/blob/master/seesawloss/seesawloss.py
import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


class SeesawLoss(torch.nn.Module):
    """
    Implementation of seesaw loss.
    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>
    Args:
        num_classes (int): The number of classes.
                Default to 1000 for the ImageNet dataset.
        p (float): The ``p`` in the mitigation factor.
                Defaults to 0.8.
        q (float): The ``q`` in the compensation factor.
                Defaults to 2.0.
        eps (float): The min divisor to smooth the computation of compensation factor.
                Default to 1e-2.
    """

    def __init__(self, num_classes=1000,
                 p=0.8, q=2.0, eps=1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps

        # cumulative samples for each category
        self.register_buffer('accumulated',
                             torch.zeros(self.num_classes, dtype=torch.float))

    def forward(self, outputs, targets):
        # accumulate the samples for each category
        for unique in targets.unique():
            self.accumulated[unique] += (targets == unique.item()).sum()

        onehot_targets = one_hot(targets, self.num_classes)
        seesaw_weights = outputs.new_ones(onehot_targets.size())

        # mitigation factor
        if self.p > 0:
            matrix = self.accumulated[None, :].clamp(min=1) / self.accumulated[:, None].clamp(min=1)
            index = (matrix < 1.0).float()
            sample_weights = matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[targets.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # compensation factor
        if self.q > 0:
            scores = softmax(outputs.detach(), dim=1)
            self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), targets.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        outputs = outputs + (seesaw_weights.log() * (1 - onehot_targets))

        return cross_entropy(outputs, targets)
