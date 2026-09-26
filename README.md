# Vinay-pavan-kumar-NEXUS-IDS-Explainable-Federated-Intrusion-Detection
NEXUS-IDS: A Neural-Evolutionary Framework for Explainable Intrusion Detection with Multi-Objective Architecture Search and Federated Learning 
# DSTGN-IDS

Dynamic Spatio-Temporal Graph Network for Intrusion Detection System (CIC-IDS2018)

## Pipeline
1. **Data Preprocessing** — Cleaning, correlation-based feature reduction
2. **SMOTE Balancing** — Class balancing
3. **Graph Construction** — Feature-correlation adjacency matrix + temporal sequences
4. **DSTGN Model** — GCN (spatial) + GRU (temporal)
5. **NSGA-III Optimization** — Multi-objective hyperparameter tuning
6. **Federated Learning** — FedAvg across 3 simulated clients
7. **NSAR Inference** — Neuro-Symbolic Attack Reasoning

## Setup
```bash
pip install -r requirements.txt
