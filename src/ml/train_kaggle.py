import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.model import CharacterCNN
from src.ml.kaggle_loader import get_kaggle_loaders
import os
import time

def train_model(num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Batch size 256 for speed
    train_loader, test_loader = get_kaggle_loaders(batch_size=256, data_dir='./data/kaggle2')
    
    model = CharacterCNN(num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
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
        
        scheduler.step()
        
        # Validation after each epoch
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
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} complete. Accuracy: {accuracy:.2f}%. Time: {duration:.2f}s")

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/character_cnn.pth")
    
    # ONNX Export
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model, dummy_input, "models/character_cnn.onnx",
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Models saved and exported.")

if __name__ == "__main__":
    train_model(num_epochs=10)
