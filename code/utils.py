import numpy as np
import torch
from sklearn import datasets
import os
import torchvision
from torchvision import transforms


def tonp(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def batch_eval(f, loader):
    res = []
    for x in loader:
        res.append(f(x))
    return res


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-(self.last_epoch + 1) * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
