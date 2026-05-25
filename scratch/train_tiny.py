import torch
import torch.nn as nn
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_data_loaders
import os

def train_tiny():
    device = torch.device("cpu")
    train_loader, _ = get_data_loaders(batch_size=8)
    model = CharacterCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    print("Starting tiny training...")
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Step {i+1} complete, loss: {loss.item():.4f}")
        if i >= 5: break
    
    print("Tiny training complete.")

if __name__ == "__main__":
    train_tiny()
