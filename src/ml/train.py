import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_data_loaders
import mlflow
import argparse
import os

def train_model(epochs=10, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # MLflow setup
    mlflow.set_experiment("EMNIST_Character_Recognition")
    
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = CharacterCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("model_type", "ResNet18")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
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

            avg_test_loss = test_loss / len(test_loader)
            test_acc = 100. * correct / total
            print(f"Epoch {epoch+1} Summary: Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "character_cnn.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)
