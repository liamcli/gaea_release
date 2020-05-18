import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from torch.autograd import Variable
from .genotypes import Genotype, DARTS_OPS, DARTS_NOZERO, SMALL, SMALL_NOZERO
from train_utils import drop_path
from ..model_search_base import SuperNetwork


def set_grad(module, input_grad, output_grad):
    module.output_grad_value = output_grad


class MixedOp(nn.Module):
    def __init__(self, C, stride, op_names):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in op_names:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def drop_path_op(self, op, x, drop_path_prob):
        if not isinstance(op, Identity):
            return drop_path(op(x), drop_path_prob)
        return op(x)

    def forward(self, x, weights, drop_path_prob=0, discrete=False):
        # ind = torch.nonzero(weights)
        # if discrete:
        #    assert len(ind) == 1
        # if len(ind) == 1:
        #    return self.drop_path_op(self._ops[ind[0][0]], x, drop_path_prob)
        self._fs = [op(x) for op in self._ops]
        # print(weights)
        return sum(w * op for w, op in zip(weights, self._fs) if w > 0)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Cell(nn.Module):
    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        op_names,
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, op_names)
                # op.register_backward_hook(set_grad)
                self._ops.append(op)

    def forward(
        self, s0, s1, weights, edge_weights=None, discrete=False, drop_path_prob=0
    ):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # TODO: test edge parameterization
        for i in range(self._steps):
            if discrete:
                s = sum(
                    self._ops[offset + j](
                        h, weights[offset + j], drop_path_prob, discrete
                    )
                    for j, h in enumerate(states)
                    if len(torch.nonzero(weights[offset + j])) > 0
                )
            else:
                if edge_weights is not None:
                    s = sum(
                        edge_weights[offset + j]
                        * self._ops[offset + j](h, weights[offset + j])
                        for j, h in enumerate(states)
                    )
                else:
                    s = sum(
                        self._ops[offset + j](h, weights[offset + j])
                        for j, h in enumerate(states)
                    )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class DARTSNetwork(SuperNetwork):
    def __init__(
        self,
        C,
        num_classes,
        nodes,
        layers,
        criterion,
        search_space_name,
        exclude_zero=False,
        multiplier=4,
        stem_multiplier=3,
        auxiliary=False,
        **kwargs
    ):
        super(DARTSNetwork, self).__init__(C, num_classes, nodes, layers, criterion)
        # Whether search space learns a separate structure for reduction cell.
        self.search_reduce_cell = True

        # Get operations list by search space name.
        self.search_space = search_space_name
        if search_space_name == "darts":
            self.op_names = DARTS_OPS
        elif search_space_name == "darts_nozero":
            self.op_names = DARTS_NOZERO
        elif search_space_name == "darts_small":
            self.op_names = SMALL
        elif search_space_name == "darts_small_nozero":
            self.op_names = SMALL_NOZERO
        else:
            raise Exception("Unknown search space")
        if exclude_zero:
            if "none" in self.op_names:
                self.op_names.remove("none")

        # Number of input nodes.
        self.n_inputs = 2
        self.add_output_node = False

        self._num_ops = len(self.op_names)
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._auxiliary = auxiliary

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                self._nodes,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                self.op_names,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        self._auxiliary = auxiliary

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Store init parameters for norm computation.
        self.store_init_weights()

    def new(self):
        model_new = DARTSNetwork(
            self._C,
            self._num_classes,
            self._nodes,
            self._layers,
            self._criterion,
            self.search_space,
            multiplier=self._multiplier,
            stem_multiplier=self._stem_multiplier,
            auxiliary=self._auxiliary,
        ).cuda()
        return model_new

    def forward(self, input, discrete=False):
        # TODO: remove discrete arg
        s0 = s1 = self.stem(input)
        logits_aux = None
        num_ops = self._num_ops
        for i, cell in enumerate(self.cells):
            # TODO: port this to architect.py
            if cell.reduction:
                weights = self.alphas["reduce"]
                edge_weights = self.edges["reduce"]
            else:
                weights = self.alphas["normal"]
                edge_weights = self.edges["normal"]
            s0, s1 = (
                s1,
                cell(s0, s1, weights, edge_weights, discrete, self.drop_path_prob),
            )
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    def _parse(self, weights):
        gene = []
        n = 2
        start = 0
        for i in range(self._nodes):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(
                range(i + 2),
                key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if self.op_names[k] != "none"
                ),
            )[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if self.op_names[k] != "none":
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((self.op_names[k_best], j))
            start = end
            n += 1
        return gene

    def genotype(self, weights):
        normal_weights = weights["normal"]
        reduce_weights = weights["reduce"]
        gene_normal = self._parse(normal_weights)
        gene_reduce = self._parse(reduce_weights)

        concat = range(2 + self._nodes - self._multiplier, self._nodes + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
