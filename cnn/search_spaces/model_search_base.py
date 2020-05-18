from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SuperNetwork(nn.Module):
    def __init__(self, C, num_classes, nodes, layers, criterion):
        super(SuperNetwork, self).__init__()
        self._C = C
        self._nodes = nodes
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion

    @abstractmethod
    def forward(self, input, discrete):
        pass

    @abstractmethod
    def new(self):
        pass

    @abstractmethod
    def genotype(self, weights):
        pass

    def store_init_weights(self):
        self.init_parameters = {}
        for name, w in self.named_parameters():
            self.init_parameters[name] = torch.Tensor(w.data).cuda()

    def get_save_states(self):
        return {
            "state_dict": self.state_dict(),
            "init_parameters": self.init_parameters,
        }

    def load_states(self, save_states):
        self.load_state_dict(save_states["state_dict"])
        self.init_parameters = save_states["init_parameters"]

    def compute_norm(self, from_init=False):
        norms = {}
        for name, w in self.named_parameters():
            if from_init:
                norms[name] = torch.norm(self.init_parameters[name] - w.data, 2)
            else:
                norms[name] = torch.norm(w.data, 2)
        return norms

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_edge_weights(self, edges):
        self.edges = edges

    def _loss(self, input, target, discrete=False):
        logits, _ = self(input, discrete=discrete)
        return self._criterion(logits, target)
