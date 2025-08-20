import torch.nn as nn
from inits import uniform
import torch.nn.functional as F
import math
import torch
from torch.nn import Parameter

class DenseGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True):
        assert aggr in ['add', 'mean', 'max']
        super(DenseGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()


    def forward(self, x, adj, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)

        if self.aggr == 'mean':
            out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        elif self.aggr == 'max':
            out = out.max(dim=-1)[0]

        out = out + self.lin(x)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)