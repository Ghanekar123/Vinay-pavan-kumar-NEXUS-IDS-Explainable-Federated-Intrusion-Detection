import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dstgn import DynamicSpatioTemporalGraphNetwork


def federated_learning_simulation(global_model, X_train, y_train, adj_matrix, num_clients=3):
    print(f"\n[*] Initializing Federated Learning with {num_clients} Clients...")
    client_data = np.array_split(np.arange(len(X_train)), num_clients)
    global_weights = global_model.state_dict()

    for round_num in range(1, 4):
        print(f"    -> Federated Round {round_num}")
        local_weights_list = []

        for i, indices in enumerate(client_data):
            local_model = DynamicSpatioTemporalGraphNetwork(
                num_nodes=global_model.num_nodes,
                node_features=global_model.node_features,
                hidden_dim=64,
                num_classes=5,
                seq_len=global_model.seq_len
            )
            local_model.load_state_dict(global_weights)

            X_local = torch.FloatTensor(X_train[indices])
            y_local = torch.LongTensor(y_train[indices])
            local_loader = DataLoader(TensorDataset(X_local, y_local), batch_size=32, shuffle=True)

            optimizer = optim.SGD(local_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            for _ in range(2):
                for inputs, labels in local_loader:
                    optimizer.zero_grad()
                    outputs = local_model(inputs, adj_matrix)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            local_weights_list.append(local_model.state_dict())

        # FedAvg
        avg_weights = {}
        for key in global_weights.keys():
            avg_weights[key] = torch.stack([local_weights_list[i][key] for i in range(num_clients)], 0).mean(0)
        global_weights = avg_weights
        print(f"       Global model updated via FedAvg.")

    global_model.load_state_dict(global_weights)
    return global_model