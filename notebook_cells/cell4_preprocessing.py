# CELL 4 — Preprocessing Pipeline
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST
from PIL import Image
import numpy as np
from collections import Counter
from constants import IMG_SIZE, SEED, BATCH_SIZE, NUM_WORKERS

# --- Custom Dataset with orientation fix + edge case handling ---
class EMNISTByClassDataset(Dataset):
    """Wraps torchvision EMNIST with orientation fix and augmentation."""

    def __init__(self, split='train', transform=None):
        self.data = EMNIST(root='./data', split='byclass', train=(split != 'test'), download=False)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # Edge case: Fix EMNIST orientation (images are transposed in source)
        img = img.transpose(Image.Transpose.TRANSPOSE)

        # Edge case: Ensure grayscale (handle RGBA/RGB if present)
        if img.mode != 'L':
            img = img.convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Transforms ---
# Use PIXEL_MEAN and PIXEL_STD from Cell 3 (typical: ~0.1736, ~0.3317)
PIXEL_MEAN = 0.1736
PIXEL_STD = 0.3317

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ElasticTransform(alpha=50.0, sigma=5.0),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize([PIXEL_MEAN], [PIXEL_STD]),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([PIXEL_MEAN], [PIXEL_STD]),
])

# --- Create Datasets ---
full_train = EMNISTByClassDataset(split='train', transform=train_transform)
test_dataset = EMNISTByClassDataset(split='test', transform=val_transform)

# 80/10/10 split from training data (test set is already separate)
total = len(full_train)
val_size = int(0.1 * total)
train_size = total - val_size

train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
# Override val transforms (random_split keeps parent transforms)
val_dataset_clean = EMNISTByClassDataset(split='train', transform=val_transform)
val_indices = val_dataset.indices
val_dataset = torch.utils.data.Subset(val_dataset_clean, val_indices)

print(f'Train: {train_size:,} | Val: {val_size:,} | Test: {len(test_dataset):,}')

# --- Edge Case: Class Imbalance → WeightedRandomSampler ---
train_labels = [full_train.data.targets[i].item() for i in train_dataset.indices]
class_counts = Counter(train_labels)
class_weights = {c: 1.0 / count for c, count in class_counts.items()}
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f'Batches per epoch: {len(train_loader):,}')
print(f'Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}')

# Quick sanity check
batch_imgs, batch_labels = next(iter(train_loader))
print(f'Batch shape: {batch_imgs.shape} | Labels shape: {batch_labels.shape}')
print(f'Pixel range: [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]')
