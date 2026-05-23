# CELL 3 — Data Exploration (EDA)
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.datasets import EMNIST
from PIL import Image
from constants import CLASS_LABELS, NUM_CLASSES

train_raw = EMNIST(root='./data', split='byclass', train=True, download=False)

# --- Class Distribution ---
labels = train_raw.targets.numpy()
counter = Counter(labels)

fig, ax = plt.subplots(figsize=(18, 5))
classes_sorted = sorted(counter.keys())
counts = [counter[c] for c in classes_sorted]
colors = ['#2196F3' if c < 10 else '#4CAF50' if c < 36 else '#FF9800' for c in classes_sorted]
ax.bar(range(len(classes_sorted)), counts, color=colors)
ax.set_xticks(range(len(classes_sorted)))
ax.set_xticklabels(CLASS_LABELS, fontsize=7)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('EMNIST ByClass Distribution (Blue=Digits, Green=Upper, Orange=Lower)')
plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png', dpi=150)
plt.show()

print(f'Min samples/class: {min(counts):,} (class {CLASS_LABELS[classes_sorted[np.argmin(counts)]]})')
print(f'Max samples/class: {max(counts):,} (class {CLASS_LABELS[classes_sorted[np.argmax(counts)]]})')
print(f'Imbalance ratio:   {max(counts)/min(counts):.1f}x')

# --- Sample Grid (5x5 per category) ---
fig, axes = plt.subplots(3, 10, figsize=(20, 7))
fig.suptitle('Sample Images (Row1=Digits, Row2=Upper, Row3=Lower)', fontsize=14)
for row, start_class in enumerate([0, 10, 36]):
    for col in range(10):
        cls = start_class + col
        if cls >= NUM_CLASSES:
            axes[row, col].axis('off')
            continue
        idx = (labels == cls).nonzero()[0][0]
        img = train_raw[idx][0]
        # Fix EMNIST orientation: transpose
        img = img.transpose(Image.Transpose.TRANSPOSE)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(CLASS_LABELS[cls], fontsize=10)
        axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/sample_grid.png', dpi=150)
plt.show()

# --- Pixel Statistics (for normalization) ---
all_pixels = train_raw.data.float() / 255.0
PIXEL_MEAN = all_pixels.mean().item()
PIXEL_STD = all_pixels.std().item()
print(f'\nPixel Mean: {PIXEL_MEAN:.4f}')
print(f'Pixel Std:  {PIXEL_STD:.4f}')
print('(Use these for transforms.Normalize)')

# --- Check for anomalies ---
zero_images = (train_raw.data.sum(dim=(1, 2)) == 0).sum().item()
print(f'\nBlank/corrupt images: {zero_images}')
