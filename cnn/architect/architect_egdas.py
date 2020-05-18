import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from .architect import Architect
from .history import History
from .expgrad_lr import AdaptiveLR


def normalize(x, dim, min_v=1e-5):
    normed = x / x.sum(dim=dim, keepdim=True)
    normed = torch.clamp(normed, min=min_v)
    return normed


class ArchitectEGDAS(Architect):
    def __init__(self, model, args, writer):
        self.lr = args.search.arch_learning_rate
        self.max_tau = args.search.max_tau
        self.min_tau = args.search.min_tau
        self.max_epochs = args.run.epochs - 1
        self.warmup_epochs = args.search.warmup_epochs
        super(ArchitectEGDAS, self).__init__(model, args, writer)
        self.history = History(
            model, self, to_save=("alphas", "l2_norm", "l2_norm_from_init")
        )
        self.grad_clip = args.search.arch_grad_clip
        self.adaptive_lr = AdaptiveLR(self.lr, 0.001, 0.3)
        self.weight_decay = args.search.arch_weight_decay

    def set_tau(self):
        self.tau = (
            self.max_tau
            - (self.max_tau - self.min_tau)
            * max(self.epochs - self.warmup_epochs, 0)
            / self.max_epochs
        )

    def initialize_alphas(self):
        self.set_tau()
        k = self.n_edges
        num_ops = self.model._num_ops
        for ct in self.cell_types:
            self.alphas[ct] = Variable(
                normalize(torch.ones(k, num_ops).cuda(), dim=-1), requires_grad=True
            )

    def initialize_edge_weights(self):
        for ct in self.cell_types:
            self.edges[ct] = Variable(
                torch.ones(self.n_edges).cuda(), requires_grad=False
            )

    def discretize_alpha(self):
        alpha_discrete = {}
        for ct in self.cell_types:
            probs = self.alphas[ct]
            index = probs.max(-1, keepdim=True)[1]
            one_ht = torch.zeros_like(probs).scatter_(-1, index, 1.0)
            alpha_discrete[ct] = one_ht
        return alpha_discrete

    def get_alphas(self):
        alpha_sample = {}
        for ct in self.cell_types:
            # print('discretizing alphas for {} cells'.format(ct))
            while True:
                gumbels = -torch.empty_like(self.alphas[ct]).exponential_().log()
                logits = (self.alphas[ct].log() + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                # print(probs)
                hardwts = one_h - probs.detach() + probs
                if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break
            alpha_sample[ct] = hardwts
        # print(alpha_sample)

        return alpha_sample

    def get_edge_weights(self):
        return self.edges

    def get_weights(self):
        discrete_alphas = self.discretize_alpha()
        self.model.set_alphas(discrete_alphas)
        alphas = self.alphas
        edges = self.edges
        weights = {}
        for ct in self.cell_types:
            weights[ct] = (alphas[ct].data * edges[ct].data.unsqueeze(1)).cpu().numpy()
        return weights

    def step(self, input_train, target_train, input_valid, target_valid, eta, **kwargs):
        if self.epochs >= self.warmup_epochs:
            self.set_tau()
            self.model.zero_grad()
            self.zero_arch_var_grad()
            self.set_model_alphas()
            self.set_model_edge_weights()

            # Perform exponentiated gradient manually
            self._backward_step(input_valid, target_valid)
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm(
                    [self.alphas[ct] for ct in self.cell_types], self.grad_clip
                )

            lr = self.lr

            for ct in self.cell_types:
                # print('performing exponentiated update for {} cells'.format(ct))
                p = self.alphas[ct]
                norm_inf = max(torch.norm(p.grad.data, p=float("inf"), dim=-1))
                self.writer.add_scalar(
                    "{}_alphas_grad_max".format(ct), norm_inf, self.steps
                )
                # lr = self.adaptive_lr.update_norm_get_lr('alphas', ct, norm_inf.item())
                # print('alpha ({}) lr: {}'.format(ct, lr))
                p.data.mul_(torch.exp(-lr * p.grad.data))
                p.data = normalize(p.data, -1)
                p.grad.detach_()
                p.grad.zero_()
                # lr *= 3
            self.steps += 1

    def _backward_step(self, input_valid, target_valid):
        entropic_reg = 0
        for ct in self.cell_types:
            entropic_reg += torch.sum(
                self.alphas[ct] * torch.log(self.alphas[ct] / (1 / self.n_ops))
            )
        loss = (
            self.model._loss(input_valid, target_valid, discrete=True)
            + self.weight_decay * entropic_reg
        )
        loss.backward()
