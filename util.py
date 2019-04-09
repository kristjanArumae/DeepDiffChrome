from torch import nn
import copy


def create_n_layers(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
