import torch
import torch.nn as nn

from gcn_layer import GraphConvLayer


class DynamicSpatioTemporalGraphNetwork(nn.Module):
    """
    Implementation of Dynamic Spatio-Temporal Graph Networks.
    - Spatial: Graph Convolution over feature correlations.
    - Temporal: GRU over time-window sequences.
    """

    def __init__(self, num_nodes, node_features, hidden_dim, num_classes, seq_len=5):
        super(DynamicSpatioTemporalGraphNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.seq_len = seq_len

        # Spatial Layer (GCN)
        self.gcn = GraphConvLayer(node_features, hidden_dim)

        # Temporal Layer (GRU)
        self.gru = nn.GRU(input_size=num_nodes * hidden_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)

        # Classification Head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        # x shape: (batch, seq_len, num_nodes, node_features)
        batch_size, seq_len, num_nodes, node_features = x.shape

        x_reshaped = x.view(batch_size * seq_len, num_nodes, node_features)

        # Apply Graph Convolution (Spatial)
        gcn_out = self.gcn(x_reshaped, adj)

        # Flatten nodes for GRU input
        gcn_out_flat = gcn_out.view(batch_size, seq_len, -1)

        # Apply GRU (Temporal)
        gru_out, _ = self.gru(gcn_out_flat)

        # Take the output of the last time step
        final_out = gru_out[:, -1, :]

        # Classification
        logits = self.fc(final_out)
        return logits