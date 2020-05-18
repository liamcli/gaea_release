from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class TriangleScheduler(_LRScheduler):
    """
    Simple class to linearly increase lr until loss stops decreasing
    with a certain grace period, then, linearly decrease lr.
    """

    def __init__(
        self,
        optimizer,
        slope,
        max_lr,
        min_lr,
        patience,
        max_epoch,
        backoff_scheduler,
        last_epoch=-1,
    ):
        self.slope = slope
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.max_epoch = max_epoch
        self.backoff_scheduler = backoff_scheduler

        # Initialize stateful vars
        self.counter = 0
        self.min_loss = 100
        self.increase = True

        super(TriangleScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if not self.increase and self.backoff_scheduler is not None:
        #    self.backoff_scheduler.last_epoch = self.last_epoch
        #    return self.backoff_scheduler.get_lr()

        if self.last_epoch == 0:
            return self.base_lrs

        lrs = []
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            if self.increase:
                lr += self.slope
            else:
                lr -= self.slope
            lr = max(min(lr, self.max_lr), self.min_lr)
            lrs.append(lr)
        return lrs

    def update_lr_state(self, loss):
        """
        Toggles loss to decreasing if grace period before loss decrease
        is used up.
        """
        if loss < self.min_loss - 0.01:
            self.min_loss = loss
        else:
            self.counter += 1
        if not self.increase and loss > self.min_loss:
            self.slope += self.slope

        if self.counter > self.patience:
            self.increase = False
        if (self.max_lr - self.min_lr) / (
            self.max_epoch - self.last_epoch
        ) > self.slope:
            self.increase = False


class CosinePowerAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer,
        power,
        cycles,
        min_lr,
        max_epoch,
        cycle_decay=0.5,
        last_epoch=-1,
    ):
        self.power = power
        self.cycles = cycles
        self.min_lr = min_lr
        self.cycle_decay = cycle_decay
        self.max_epoch = max_epoch
        self.epochs_per_cycle = int(self.max_epoch / self.cycles)
        super(CosinePowerAnnealing, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = int(self.last_epoch / self.epochs_per_cycle)
        lr_decay = self.cycle_decay ** cycle
        if self.power == 1:
            numerator = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch % self.epochs_per_cycle)
                    / self.epochs_per_cycle
                )
            )
            denominator = 1
        else:
            numerator = (
                self.power
                ** (
                    0.5
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (self.last_epoch % self.epochs_per_cycle)
                            / self.epochs_per_cycle
                        )
                    )
                    + 1
                )
                - self.power
            )
            denominator = self.power ** 2 - self.power

        return [
            self.min_lr + (lr_decay * base_lr - self.min_lr) * numerator / denominator
            for base_lr in self.base_lrs
        ]
