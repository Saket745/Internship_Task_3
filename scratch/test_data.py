import torch
from torchvision import datasets, transforms
import time

print("Starting data test...")
start = time.time()
train_dataset = datasets.EMNIST(
    root='./data', 
    split='byclass', 
    train=True, 
    download=False
)
print(f"Data loaded in {time.time() - start:.2f}s")
print(f"Dataset size: {len(train_dataset)}")
