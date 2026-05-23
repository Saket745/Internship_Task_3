import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class KaggleEMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        print(f"Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].values
        label = int(row[0])
        image = row[1:].astype(np.uint8).reshape(28, 28)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def fix_orientation(img):
    return img.transpose(Image.Transpose.TRANSPOSE)

def get_kaggle_loaders(batch_size=128, data_dir='./data/kaggle2'):
    transform = transforms.Compose([
        transforms.Lambda(fix_orientation),
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])
    
    train_transform = transforms.Compose([
        transforms.Lambda(fix_orientation),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])

    train_ds = KaggleEMNISTDataset(f"{data_dir}/emnist-balanced-train.csv", transform=train_transform)
    test_ds = KaggleEMNISTDataset(f"{data_dir}/emnist-balanced-test.csv", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
