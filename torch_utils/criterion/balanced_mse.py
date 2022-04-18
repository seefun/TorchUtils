# adapted from https://github.com/jiawei-ren/BalancedMSE
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(pred.device))
    loss = loss * (2 * noise_var).detach()
    return loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        self.noise_var = self.noise_sigma ** 2

    def forward(self, pred, target):
        loss = bmc_loss(pred, target, self.noise_var)
        return loss
