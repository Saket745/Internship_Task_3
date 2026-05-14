import torch
from torchvision import datasets, transforms
from src.ml.model import CharacterCNN
from src.ml.data_loader import get_transforms
import os
from torch.utils.data import DataLoader

def micro_train():
    device = torch.device('cpu')
    model = CharacterCNN(num_classes=47).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Loading data...")
    train_dataset = datasets.EMNIST(
        root='./data', 
        split='balanced', 
        train=True, 
        transform=get_transforms(augment=True)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print("Starting micro-train (50 steps)...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print_every = 5
    
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
        
        if i % print_every == print_every - 1:
            print(f"Step {i+1}/441 | Loss: {running_loss/print_every:.4f} | Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0
            
        if i >= 50:
            break
            
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), 'models/character_cnn.pth')
    print("Micro-train complete. Model saved to models/character_cnn.pth")

if __name__ == "__main__":
    micro_train()
