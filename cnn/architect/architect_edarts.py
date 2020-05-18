import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from .architect import Architect
from .history import History
from .expgrad_lr import AdaptiveLR


def normalize(x, dim, min_v=1e-5):
    x = torch.clamp(x, min=min_v)
    normed = x / x.sum(dim=dim, keepdim=True)
    return normed


class ArchitectEDARTS(Architect):
    def __init__(self, model, args, writer):
        self.learn_edges = args.search.learn_edges
        self.trace_norm = args.search.trace_norm
        super(ArchitectEDARTS, self).__init__(model, args, writer)
        self.lr = args.search.arch_learning_rate
        self.edge_lr = args.search.edge_learning_rate
        to_save = ["alphas", "l2_norm", "l2_norm_from_init"]
        if self.learn_edges:
            to_save.append("edges")

        self.history = History(model, self, to_save=to_save)
        self.history.dict["grads"] = {}
        for v in ["alphas", "edges"]:
            self.history.dict["grads"][v] = {}
            for ct in self.cell_types:
                self.history.dict["grads"][v][ct] = []
        self.grad_clip = args.search.arch_grad_clip
        self.adapt_lr = args.search.adapt_lr
        self.adaptive_lr = AdaptiveLR(self.lr, 0.001, 0.3)
        self.weight_decay = args.search.arch_weight_decay
        self.gd = args.search.gd

    def initialize_alphas(self):
        k = self.n_edges
        num_ops = self.model._num_ops
        for ct in self.cell_types:
            self.alphas[ct] = Variable(
                normalize(torch.ones(k, num_ops).cuda(), dim=-1), requires_grad=True
            )

        self._arch_parameters = [self.alphas[ct] for ct in self.cell_types]

    def get_edge_scaling(self):
        i = 0
        n_inputs = self.n_inputs
        scale = np.zeros(self.n_edges)

        for n in range(self.n_nodes):
            scale[i : i + n_inputs] = 1 / n_inputs
            i += n_inputs
            n_inputs += 1

        scale = torch.Tensor(scale).cuda()
        return scale

    def initialize_edge_weights(self):
        requires_grad = False
        scale = 1
        if self.learn_edges:
            requires_grad = True
            scale = self.get_edge_scaling()
        for ct in self.cell_types:
            self.edges[ct] = Variable(
                torch.ones(self.n_edges).cuda(), requires_grad=requires_grad
            )
            self.edges[ct].data = self.edges[ct].data * scale

    def get_alphas(self):
        return self.alphas

    def get_edge_weights(self):
        return self.edges

    def step(self, input_train, target_train, input_valid, target_valid, eta, **kwargs):
        self.model.zero_grad()
        self.zero_arch_var_grad()
        self.set_model_alphas()
        self.set_model_edge_weights()

        # Perform exponentiated gradient manually
        self._backward_step(input_valid, target_valid)

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm(self._arch_parameters, self.grad_clip)

        update_set = [("alphas", self.alphas)]
        if self.learn_edges:
            update_set.append(("edges", self.edges))

        for var_name, params in update_set:
            for ct in self.cell_types:
                p = params[ct]
                norm_inf = torch.norm(p.grad.data, p=float("inf")).item()
                norm2 = torch.norm(p.grad.data, p=2).item()
                self.writer.add_scalar(
                    "{}_{}_grad_2".format(var_name, ct), norm2, self.steps
                )
                self.writer.add_scalar(
                    "{}_{}_grad_max".format(var_name, ct), norm_inf, self.steps
                )
                self.history.dict["grads"][var_name][ct].append(
                    p.grad.data.cpu().numpy()
                )
                lr = self.lr if var_name == "alphas" else self.edge_lr
                if self.adapt_lr:
                    # lr = self.adaptive_lr.update_norm_get_lr(var_name, ct, norm_inf.item())
                    lr = lr / norm_inf
                    print("{} ({}) lr: {}".format(var_name, ct, lr))
                if self.gd:
                    p.data.sub_(lr * p.grad.data)
                else:
                    p.data.mul_(torch.exp(-lr * p.grad.data))
                if var_name == "alphas":
                    p.data = normalize(p.data, -1)
                else:
                    ## If edges, we normalize by node to 2
                    # p.data = torch.clamp(p.data, min=1e-5)
                    # p.data = self.trace_norm / torch.sum(p.data) * p.data
                    node_weights = torch.zeros([self.n_edges]).cuda()
                    offset = 0
                    n_inputs = self.n_inputs
                    for i in range(self.n_nodes):
                        node_weights[offset : offset + n_inputs] = sum(
                            p.data[offset : offset + n_inputs]
                        )
                        offset += n_inputs
                        n_inputs += 1
                    p.data = p.data / node_weights

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
            self.model._loss(input_valid, target_valid)
            + self.weight_decay * entropic_reg
        )
        loss.backward()
