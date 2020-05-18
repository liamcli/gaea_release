import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .architect import Architect
from .history import History


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


# TODO: modify to include learnable edge weights.
class ArchitectDARTS(Architect):
    def __init__(self, model, args, writer):
        super(ArchitectDARTS, self).__init__(model, args, writer)

        self.network_momentum = args.train.momentum
        self.network_weight_decay = args.train.weight_decay
        if args.search.gd:
            self.optimizer = torch.optim.SGD(
                self._arch_parameters,
                lr=args.search.arch_learning_rate,
                weight_decay=args.search.arch_weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self._arch_parameters,
                lr=args.search.arch_learning_rate,
                betas=(0.5, 0.999),
                weight_decay=args.search.arch_weight_decay,
            )
        self.history = History(
            model, self, to_save=("alphas", "l2_norm", "l2_norm_from_init")
        )
        self.history.dict["grads"] = {}
        for v in ["alphas", "edges"]:
            self.history.dict["grads"][v] = {}
            for ct in self.cell_types:
                self.history.dict["grads"][v][ct] = []

    def initialize_alphas(self):
        k = self.n_edges
        num_ops = self.model._num_ops
        for ct in self.cell_types:
            self.alphas[ct] = Variable(
                1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
            )
        self._arch_parameters = [self.alphas[ct] for ct in self.cell_types]

    def initialize_edge_weights(self):
        for ct in self.cell_types:
            self.edges[ct] = Variable(
                torch.ones(self.n_edges).cuda(), requires_grad=False
            )

    def get_alphas(self):
        return {ct: F.softmax(self.alphas[ct], dim=-1) for ct in self.cell_types}

    def get_edge_weights(self):
        return self.edges

    def step(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        unrolled,
        **kwargs
    ):
        self.optimizer.zero_grad()
        self.zero_arch_var_grad()
        self.set_model_alphas()
        self.set_model_edge_weights()

        if unrolled:
            self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        self.steps += 1

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        for ct in self.cell_types:
            self.history.dict["grads"]["alphas"][ct].append(
                self.alphas[ct].grad.data.cpu().numpy()
            )

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params = {}
        offset = 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        self.initialize_new_model_arch_params(model_new)

        return model_new.cuda()

    def copy_architecture_params(self):
        new_alphas = {
            ct: Variable(self.alphas[ct].data.clone().cuda(), requires_grad=True)
            for ct in self.cell_types
        }
        new_edges = {
            ct: Variable(self.edges[ct].data.clone().cuda(), requires_grad=False)
            for ct in self.cell_types
        }
        return new_alphas, new_edges

    def initialize_new_model_arch_params(self, new_model):
        self._new_alphas, self._new_edges = self.copy_architecture_params()
        alphas = {ct: F.softmax(self._new_alphas[ct], dim=-1) for ct in self.cell_types}
        new_model.set_alphas(alphas)
        new_model.set_edge_weights(self._new_edges)
        new_model.drop_path_prob = self.model.drop_path_prob

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, self.model.parameters())).data
            + self.network_weight_decay * theta
        )
        self.set_model_alphas()
        self.set_model_edge_weights()
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self._arch_parameters)
        self.set_model_alphas()
        self.set_model_edge_weights()

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self._arch_parameters)
        self.set_model_alphas()
        self.set_model_edge_weights()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _backward_step_unrolled(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
    ):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [self._new_alphas[ct].grad for ct in self.cell_types]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self._arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
