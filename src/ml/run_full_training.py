import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_data_loaders
import os
import time
import mlflow
import mlflow.pytorch

def validate(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0.0
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
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

def train_model(num_epochs=5, batch_size=256, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # MLflow setup
    mlflow.set_experiment("EMNIST_ResNet_Training")
    
    print("Loading data loaders...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Data loaders ready. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    print("Initializing model...")
    model = CharacterCNN(num_classes=62).to(device)
    print("Model initialized and moved to device.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    
    with mlflow.start_run() as run:
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("device", str(device))
        mlflow.log_param("model_architecture", "ResNet-18")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} started...")
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 50 == 49:
                    current_loss = running_loss / 50
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {current_loss:.4f}")
                    mlflow.log_metric("batch_loss", current_loss, step=epoch * len(train_loader) + i)
                    running_loss = 0.0

            # Per-epoch validation
            print(f"End of Epoch {epoch+1}, validating...")
            accuracy, val_loss = validate(model, test_loader, device, criterion)
            print(f"Validation - Accuracy: {accuracy:.2f}%, Avg Loss: {val_loss:.4f}")
            
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch_duration", time.time() - epoch_start_time, step=epoch)

            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/character_cnn_best.pth")
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
                mlflow.log_metric("best_accuracy", best_acc, step=epoch)

        # Final Save
        torch.save(model.state_dict(), "models/character_cnn_final.pth")
        
        # ONNX Export
        print("Exporting to ONNX...")
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        onnx_path = "models/character_cnn.onnx"
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        mlflow.log_artifact(onnx_path)
        print(f"Models saved and exported to ONNX. Final Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model(num_epochs=5)
