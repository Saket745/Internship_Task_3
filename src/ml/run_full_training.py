import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_data_loaders
import os
import time

def run_full():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Increase batch size for speed on CPU if memory allows, but 128 is safe
    train_loader, test_loader = get_data_loaders(batch_size=128)
    
    model = CharacterCNN(num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started...")
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    # Validation
    print("Validating...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    
    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/character_cnn.pth")
    
    # ONNX Export
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model, dummy_input, "models/character_cnn.onnx",
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Models saved and exported to ONNX.")

if __name__ == "__main__":
    run_full()
