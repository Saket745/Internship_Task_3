import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_data_loaders
import os
import time

def train_robust(epochs=5, batch_size=256, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Initialize model
    model = CharacterCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} Summary: Test Acc: {test_acc:.2f}%, Duration: {duration:.2f}s")
        
        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("models", "character_cnn.pth"))

    print("Training complete. Exporting to ONNX...")
    export_onnx(model, device)

def export_onnx(model, device):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = os.path.join("models", "character_cnn.onnx")
    
    # Try to export without dynamic axes first if it fails
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model exported to {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Retrying with simplified export...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11
        )
        print(f"Simplified ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    train_robust()
