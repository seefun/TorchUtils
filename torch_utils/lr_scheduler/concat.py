import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps, pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.last_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()

    def get_lr(self):
        if self.last_epoch <= self.step_start:
            return self.scheduler1.get_lr()
        else:
            return self.scheduler2.get_lr()

# Example: (from https://github.com/mgrankin/over9000/blob/master/train.py)
#
# from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR
#
# def d(x): 
#     return 1
#
# if sched_type == 'flat_and_anneal':
#     dummy = LambdaLR(optimizer, d)
#     cosine = CosineAnnealingLR(optimizer, total_steps*(1-ann_start))
#     scheduler = ConcatLR(optimizer, dummy, cosine, total_steps, ann_start)
# else:
#     scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3,
#                            div_factor=10, cycle_momentum=True)
