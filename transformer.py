from torch import nn
import torch

from util import create_n_layers


class EncoderLayer(nn.Module):
    def __init__(self, n_dim, attn, ff, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = attn
        self.ff = ff
        self.layer_ls = create_n_layers(Residual(n_dim, dropout), 2)
        self.n_dim = n_dim

    def forward(self, inpt, mask):
        inpt = self.layer_ls[0](inpt, lambda inpt: self.attn(inpt, inpt, inpt, mask))
        return self.layer_ls[1](inpt, self.ff)


class TransfromerEncoder(nn.Module):
    """
    This encapsulates stacked transformers
    """
    def __init__(self, num_layers, layer):
        super(TransfromerEncoder, self).__init__()

        self.layer_ls = create_n_layers(layer, num_layers)
        self.norm = Norm(layer.size)

    def forward(self, inpt, mask):
        for l in self.layer_ls:
            inpt = l(inpt, mask)

        return self.norm(inpt)


class MultiAttn(nn.Module):
    def __init__(self, num_heads, n_dim, dropout=0.5):
        super(MultiAttn, self).__init__()

        self.d_k = n_dim // num_heads
        self.num_heads = num_heads
        self.linear_ls = create_n_layers(nn.Linear(n_dim, n_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_b = query.size(0)

        query, key, value = [l(a).view(n_b, -1, self.num_heads, self.d_k).transpose(1, 2) for l, a in
                             zip(self.linear_ls, (query, key, value))]

        x, self.attn = attn(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(n_b, -1, self.num_heads * self.d_k)

        return self.linear_ls[3](x)


class Norm(nn.Module):
    """
    Normalization
    """
    def __init__(self, n_f):
        super(Norm, self).__init__()

        self.alpha = nn.Parameter(torch.ones(n_f))
        self.beta = nn.Parameter(torch.zeros(n_f))

    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        stdev = input.std(-1, keepdim=True)

        return self.alpha * (input - mean) / (stdev + 1e-6) + self.beta


class Residual(nn.Module):
    """
    Traditional Residual connection + normalization and dropout
    """
    def __init__(self, n_dim, dropout=0.5):
        super(Residual, self).__init__()

        self.norm = Norm(n_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inpt, l):
        return inpt + self.dropout(l(self.norm(inpt)))
