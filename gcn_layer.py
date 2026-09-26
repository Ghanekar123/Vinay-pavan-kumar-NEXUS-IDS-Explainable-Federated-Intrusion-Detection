import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Simple Graph Convolutional Layer"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return F.relu(output)