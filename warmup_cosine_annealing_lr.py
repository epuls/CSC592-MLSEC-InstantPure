import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        T_max=95,
        eta_min=1e-6,
        start_factor=0.1,
        warmup_iters=5,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.start_factor = start_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        # No warmup case
        if self.warmup_iters <= 0:
            return self._get_cosine_lrs(epoch)

        # Warmup phase
        if epoch < self.warmup_iters:
            # epoch=0 -> start_factor
            # epoch=warmup_iters -> 1.0 (handled by cosine branch below)
            progress = epoch / self.warmup_iters
            factor = self.start_factor + (1.0 - self.start_factor) * progress
            return [base_lr * factor for base_lr in self.base_lrs]

        # Cosine phase
        cosine_epoch = epoch - self.warmup_iters
        return self._get_cosine_lrs(cosine_epoch)

    def _get_cosine_lrs(self, cosine_epoch):
        t = min(cosine_epoch, self.T_max)
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]