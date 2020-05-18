from abc import ABC, abstractmethod
import torch


class Architect(ABC):
    """Base class for architecture optimizers."""

    def __init__(self, model, args, writer):
        self.model = model
        self.writer = writer
        self.cell_types = ["normal"]
        if model.search_reduce_cell:
            self.cell_types.append("reduce")
        self.args = args
        self.op_names = model.op_names
        self.n_nodes = model._nodes
        self.n_ops = model._num_ops
        self.n_inputs = model.n_inputs
        self.n_edges = sum(
            1 for i in range(self.n_nodes) for n in range(self.n_inputs + i)
        )
        self.optimizer = None
        self.epochs = 0
        self.steps = 0

        # Architecture variables
        self.history = None
        self.alphas = {}
        self.edges = {}

        # Initialize architecture variables
        self.initialize_alphas()
        self.set_model_alphas()
        self.initialize_edge_weights()
        self.set_model_edge_weights()

    @abstractmethod
    def initialize_alphas(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_edge_weights(self):
        raise NotImplementedError

    @abstractmethod
    def get_alphas(self):
        raise NotImplementedError

    @abstractmethod
    def get_edge_weights(self):
        raise NotImplementedError

    def set_model_alphas(self):
        alphas = self.get_alphas()
        self.model.set_alphas(alphas)

    def set_model_edge_weights(self):
        edge_weights = self.get_edge_weights()
        self.model.set_edge_weights(edge_weights)

    @abstractmethod
    def step(self, **kwawrgs):
        raise NotImplementedError

    def zero_arch_var_grad(self):
        for ct in self.cell_types:
            for p in [
                self.alphas[ct],
                self.model.alphas[ct],
                self.edges[ct],
                self.model.edges[ct],
            ]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_weights(self):
        alphas = self.get_alphas()
        edges = self.get_edge_weights()
        weights = {}
        for ct in self.cell_types:
            weights[ct] = (alphas[ct].data * edges[ct].data.unsqueeze(1)).cpu().numpy()
        return weights

    def genotype(self, weights=None):
        if weights is None:
            weights = self.get_weights()

        return self.model.genotype(weights)

    def get_save_states(self):
        save_dict = {}
        save_dict["alphas"] = self.alphas
        save_dict["edges"] = self.edges
        save_dict["steps"] = self.steps

        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
        return save_dict

    def load_states(self, saved_states):
        for ct in self.cell_types:
            self.alphas[ct].data = saved_states["alphas"][ct].data
            self.edges[ct].data = saved_states["edges"][ct].data

        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved_states["optimizer"])

    # Assume history is a History object
    # that is initialized by each subclass
    def get_history(self):
        return self.history.dict

    def load_history(self, history):
        self.epochs = history["epochs"]
        self.history.dict = history

    def update_history(self):
        self.epochs += 1
        self.history.update_history(self.epochs)

    def log_vars(self, epoch, writer):
        self.history.log_vars(epoch, writer)

    # TODO: This belongs in stochastic architect.
    # def get_alphas_from_genotype(self, arch):
    #    """
    #    This function returns alphas assuming that edge weights are
    #    uniformly 1.
    #    """
    #    k = self.n_edges
    #    num_ops = self.n_ops
    #    n_nodes = self.n_nodes
    #    op_to_ind = dict(zip(self.op_names, range(num_ops)))

    #    alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    #    alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    #    offset = 0
    #    for i in range(n_nodes):
    #        normal1 = arch.normal[2*i]
    #        normal2 = arch.normal[2*i+1]
    #        reduce1 = arch.reduce[2*i]
    #        reduce2 = arch.reduce[2*i+1]
    #        alphas_normal[offset+normal1[1], op_to_ind[normal1[0]]] = 1
    #        alphas_normal[offset+normal2[1], op_to_ind[normal2[0]]] = 1
    #        alphas_reduce[offset+reduce1[1], op_to_ind[reduce1[0]]] = 1
    #        alphas_reduce[offset+reduce2[1], op_to_ind[reduce2[0]]] = 1
    #        offset += (i+2)

    #    arch_parameters = [
    #      alphas_normal,
    #      alphas_reduce,
    #    ]
    #    return arch_parameters
