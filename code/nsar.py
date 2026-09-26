import torch
from sklearn.metrics import classification_report

from config import CLASS_NAMES


def nsar_inference(model, X_test, y_test, adj_matrix):
    print("\n[*] Running Neuro-Symbolic Attack Reasoning (NSAR)...")
    model.eval()
    X_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        outputs = model(X_tensor, adj_matrix)
        _, predicted = torch.max(outputs, 1)

    print("\n=== Final Classification Report ===")
    print(classification_report(y_test, predicted.numpy(), target_names=list(CLASS_NAMES.values())))
    return predicted.numpy()