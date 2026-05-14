import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

def get_transforms(augment=False):
    """
    Returns transforms for EMNIST. 
    Fixes the default transposition quirk (EMNIST is natively WxH).
    """
    # Base transforms: Fix orientation, convert to tensor, and normalize
    # EMNIST ByClass approximate mean/std: 0.1736, 0.3317
    transform_list = [
        # Correct EMNIST orientation (swap X and Y)
        transforms.Lambda(lambda img: TF.hflip(TF.rotate(img, -90, interpolation=TF.InterpolationMode.BILINEAR))),
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ]
    
    if augment:
        # Add random rotation and affine transforms for training robustness
        augmentations = [
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        ]
        # Insert augmentations before ToTensor
        transform_list = augmentations + transform_list

    return transforms.Compose(transform_list)

def get_data_loaders(batch_size=64, data_dir='./data'):
    """
    Downloads EMNIST (Balanced split) and returns train/test DataLoaders.
    The 'Balanced' split contains 131,600 samples across 47 classes.
    """
    
    train_dataset = datasets.EMNIST(
        root=data_dir, 
        split='balanced', 
        train=True, 
        download=True, 
        transform=get_transforms(augment=True)
    )

    test_dataset = datasets.EMNIST(
        root=data_dir, 
        split='balanced', 
        train=False, 
        download=True, 
        transform=get_transforms(augment=False)
    )

    # num_workers=0 is safer for initial testing on Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def visualize_batch(loader, classes=None):
    """Utility to visualize a batch of images from the loader."""
    images, labels = next(iter(loader))
    plt.figure(figsize=(10, 5))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i+1)
        img = images[i].numpy().squeeze()
        # Un-normalize for visualization
        img = img * 0.3317 + 0.1736
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing DataLoader setup...")
    train_loader, _ = get_data_loaders(batch_size=8)
    print(f"Dataset Size: {len(train_loader.dataset)} samples")
    visualize_batch(train_loader)
