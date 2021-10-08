import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def _poly_lr_scheduler(iter, max_iter, gamma=0.05, power=0.9):
    if iter >= max_iter:
        return gamma
    new_lr = ((1 - float(iter) / max_iter) ** power) * (1 - gamma) + gamma
    return new_lr


def _flat_anneal_schedule(iter, max_iter, warmup_iter=0, decay_start=0.5, anneal='cos', gamma=0.05):
    if iter >= max_iter:
        return gamma
    if warmup_iter and (iter < warmup_iter):
        # warmup from 0 to 1
        return iter / warmup_iter
    if iter <= max_iter * decay_start:
        return 1
    if anneal == 'cos':
        decay_status = (iter / max_iter - decay_start) / (1 - decay_start) * math.pi
        new_lr = (math.cos(decay_status) + 1.0) / 2.0
        new_lr = new_lr * (1 - gamma) + gamma
    else:
        decay_status = (iter / max_iter - decay_start) / (1 - decay_start)
        new_lr = (1 - decay_status) * (1 - gamma) + gamma
    return new_lr


def get_scheduler(optimizer, lmbda):
    # support multiple parameter groups
    num_param_groups = len(optimizer.param_groups)
    return LambdaLR(optimizer, lr_lambda=[lmbda] * num_param_groups)


def get_poly_scheduler(optimizer, max_iter, gamma=0.05, power=0.9):
    def decay_lambda(iter): 
        return _poly_lr_scheduler(iter, max_iter, gamma, power)
    return get_scheduler(optimizer, decay_lambda)


def get_flat_anneal_scheduler(optimizer, max_iter, warmup_iter=0, decay_start=0.5, anneal='cos', gamma=0.05):
    def decay_lambda(iter): 
        return _flat_anneal_schedule(iter, max_iter, warmup_iter, decay_start, anneal, gamma)
    return get_scheduler(optimizer, decay_lambda)
