import numpy as np


class AdaptiveLR(object):
    def __init__(self, base_lr, min_lr, max_lr, fields=("alphas")):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.fields = fields
        self.alpha_grad_norms = {"normal": 0, "reduce": 0}
        self.edge_grad_norms = {"normal": 0, "reduce": 0}

    def update_norm_get_lr(self, field, ct, value):
        assert field in self.fields
        if field == "alphas":
            self.alpha_grad_norms[ct] += value ** 2
            lr = self.base_lr / np.sqrt(max(1, self.alpha_grad_norms[ct]))
        else:
            self.edge_grad_norms[ct] += value ** 2
            lr = self.base_lr / np.sqrt(max(1, self.edge_grad_norms[ct]))
        return max(self.min_lr, min(lr, self.max_lr))
