import numpy as np
import torch


def create_graph_and_sequences(X, seq_len=5):
    """
    Converts flat tabular data into Graph Sequences for DSTGN.
    1. Creates Adjacency Matrix based on Feature Correlation.
    2. Reshapes data into temporal sequences.
    """
    print("[*] Constructing Dynamic Spatio-Temporal Graph...")

    # 1. Build Adjacency Matrix (Spatial)
    corr_matrix = np.corrcoef(X, rowvar=False)
    adj_matrix = np.where(np.abs(corr_matrix) > 0.3, 1.0, 0.0)
    np.fill_diagonal(adj_matrix, 1.0)
    adj_tensor = torch.FloatTensor(adj_matrix)

    num_nodes = X.shape[1]

    # 2. Build Temporal Sequences
    num_samples = len(X) // seq_len
    X_trimmed = X[:num_samples * seq_len]

    X_sequences = X_trimmed.reshape(num_samples, seq_len, num_nodes)

    return X_sequences, adj_tensor, num_nodes