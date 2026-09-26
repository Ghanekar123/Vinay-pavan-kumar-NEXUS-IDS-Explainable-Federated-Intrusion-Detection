import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from config import DEFAULT_SEQ_LEN, DEFAULT_HIDDEN_DIM, DEFAULT_EPOCHS
from data_pipeline import DataPipeline
from graph_builder import create_graph_and_sequences
from dstgn import DynamicSpatioTemporalGraphNetwork
from nsga3 import NSGA3Optimizer
from federated import federated_learning_simulation
from trainer import train_model
from nsar import nsar_inference

warnings.filterwarnings('ignore')


def main():
    # 1. Load Data
    try:
        pipeline = DataPipeline('data/CIC-IDS2018.csv')
        X, y = pipeline.load_and_preprocess()
    except FileNotFoundError:
        print("[!] CSV file not found. Generating synthetic data...")
        n_samples = 5000
        n_features = 40
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)
        pipeline = DataPipeline(None)

    # 2. Balance Data
    X_bal, y_bal = pipeline.balance_data(X, y)

    # 3. Create Graph Sequences (DSTGN Preparation)
    SEQ_LEN = DEFAULT_SEQ_LEN
    X_seq, adj_matrix, num_nodes = create_graph_and_sequences(X_bal, seq_len=SEQ_LEN)

    # Align labels with sequences (take the label of the last flow in each sequence)
    num_sequences = X_seq.shape[0]
    y_seq = y_bal[SEQ_LEN - 1: num_sequences * SEQ_LEN: SEQ_LEN]

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

    # 5. NSGA-III Optimization
    optimizer = NSGA3Optimizer(num_nodes, 5)
    best_params = optimizer.optimize(X_train, y_train)

    # 6. Initialize DSTGN Model
    model = DynamicSpatioTemporalGraphNetwork(
        num_nodes=num_nodes,
        node_features=1,
        hidden_dim=best_params['hidden_dim'],
        num_classes=5,
        seq_len=SEQ_LEN
    )

    # 7. Federated Learning Simulation
    fl_model = federated_learning_simulation(model, X_train, y_train, adj_matrix, num_clients=3)

    # 8. Final Training
    print("\n[*] Fine-tuning Global DSTGN Model...")
    trained_model, acc = train_model(fl_model, train_loader, val_loader, adj_matrix, epochs=DEFAULT_EPOCHS)

    # 9. NSAR Inference
    predictions = nsar_inference(trained_model, X_test, y_test, adj_matrix)

    print("\n[SUCCESS] Pipeline Execution Complete.")


if __name__ == "__main__":
    main()