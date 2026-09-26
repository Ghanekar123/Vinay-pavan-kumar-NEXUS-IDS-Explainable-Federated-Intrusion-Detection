import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, adj_matrix, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, adj_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs, adj_matrix)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"    Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

    return model, best_acc