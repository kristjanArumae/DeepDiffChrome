import torch
from torch import nn
import math


class Position(nn.Module):
    def __init__(self, n_dim, dropout, max_len):
        super(Position, self).__init__()

        positional_encoding = torch.zeros(max_len, n_dim)
        posit = torch.arange(0, max_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, n_dim, 2) * -(math.log(1e5) / n_dim))

        positional_encoding[:, 0::2] = torch.sin(posit * denominator)
        positional_encoding[:, 1::2] = torch.cos(posit * denominator)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + torch.autograd.Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False))


class FeedForward(nn.Module):
    def __init__(self, n_dim, features, dropout=0.5):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(n_dim, features)
        self.w2 = nn.Linear(features, n_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inpt):
        return self.w2(self.dropout(torch.nn.functional.relu(self.w1(inpt))))


def attn(q, k, v, mask, dropout):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    a = torch.nn.functional.softmax(scores, dim=-1)
    a = dropout(a)

    return torch.matmul(a, v), a

