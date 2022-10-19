import torch


class FGM:
    def __init__(self, model, criterion, optimizer, adv_param="weight", epsilon=0.5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.epsilon = epsilon
        self.backup = {}

    def _attack_step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def attack_backward(self, inputs, labels):
        self._attack_step()
        y_preds = self.model(inputs)
        adv_loss = self.criterion(y_preds, labels)
        self.optimizer.zero_grad()
        return adv_loss

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    """
    Implements weighted adverserial perturbation
    Args:
        adv_param (str): layer name to be attacked
        adv_lr (float): common: 1.0 for embedding, 0.1 for all weight
        adv_eps (float): (0,+∞), usually(0,1)
    """

    def __init__(self, model, criterion, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        y_preds = self.model(inputs)

        adv_loss = self.criterion(y_preds, labels)
        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


"""
# Use Case:
adv_method = AWP(model, criterion, optimizer, xxxx)

for step, batch in enumerate(train_loader):
    inputs, labels = batch
    optimizer.zero_grad()
    # regular training
    predicts = model(inputs)
    loss = loss_fn(predicts, labels) 
    loss.backward()
    # adv training
    if adv_start >= epoch:
        loss = adv_method.attack_backward(inputs, labels)
        loss.backward()
        adv_method.restore()
    optimizer.step()
"""
