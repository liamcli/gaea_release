from lib.models import get_search_spaces, get_cifar_models, get_imagenet_models
from lib.config_utils import load_config, dict2config, configure2str
from lib.models.cell_searchs.search_cells import NAS201SearchCell as SearchCell
from lib.models.cell_operations import ResNetBasicblock
from lib.models.cell_searchs.genotypes import Structure
import torch
import torch.nn as nn
from copy import deepcopy
from ..model_search_base import SuperNetwork


class NASBENCH201Network(SuperNetwork):
    def __init__(
        self,
        C,
        num_classes,
        nodes,
        layers,
        criterion,
        search_space_name="nasbench-201",
        exclude_zero=False,
        affine=False,
        track_running_stats=True,
        **kwargs
    ):

        super(NASBENCH201Network, self).__init__(
            C, num_classes, nodes, layers, criterion
        )
        # Architect needs these attributes from model:
        #   op_names, _num_ops, n_inputs, search_reduce_cell

        self.search_reduce_cell = False
        if exclude_zero:
            search_space_name = "nas-bench-201-nozero"
        self.search_space_name = search_space_name
        search_space = get_search_spaces("cell", search_space_name)
        self.op_names = deepcopy(search_space)
        self._num_ops = len(self.op_names)

        # In contrast to DARTS, this search space includes the first input node
        # as a node.
        self.max_nodes = nodes + 1
        self.n_inputs = 1

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        N = layers

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    self.max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Store init parameters for norm computation.
        self.store_init_weights()

    def forward(self, inputs, discrete=False):
        # TODO: add discrete forward step to speed up probabilistic methods.
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                if discrete:
                    weights = self.alphas["normal"] * self.edges["normal"].unsqueeze(1)
                    indices = weights.max(-1, keepdim=True)[1]
                    feature = cell.forward_gdas(feature, weights, indices)
                else:
                    feature = cell.forward_edge_weights(
                        feature, self.alphas["normal"], self.edges["normal"]
                    )
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits, None

    def new(self):
        model_new = NASBENCH201Network(
            self._C,
            self._num_classes,
            self._nodes,
            self._layers,
            self._criterion,
            self.search_space_name,
        ).cuda()
        return model_new

    def genotype(self, weights):
        normal_weights = weights["normal"]
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = normal_weights[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)
