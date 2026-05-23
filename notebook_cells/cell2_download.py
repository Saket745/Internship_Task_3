# CELL 2 — Dataset Download & Verification
# Option A: Kaggle CLI (run these as shell commands with ! prefix in notebook)
# !pip install kaggle -q
# !kaggle datasets download -d crawford/emnist -p ./data --force
# !unzip -q ./data/emnist.zip -d ./data/emnist_extracted

# Option B: torchvision (recommended - handles everything)
from torchvision.datasets import EMNIST
import os

DATA_ROOT = './data'
os.makedirs(DATA_ROOT, exist_ok=True)

print('Downloading EMNIST ByClass via torchvision...')
train_raw = EMNIST(root=DATA_ROOT, split='byclass', train=True, download=True)
test_raw = EMNIST(root=DATA_ROOT, split='byclass', train=False, download=True)

print(f'\nTraining samples: {len(train_raw):,}')
print(f'Test samples:     {len(test_raw):,}')
print(f'Total:            {len(train_raw) + len(test_raw):,}')
print(f'Classes:          {len(train_raw.classes)}')
print(f'Image shape:      {train_raw[0][0].size} (PIL)')

# Verify folder structure
for root, dirs, files in os.walk(DATA_ROOT):
    level = root.replace(DATA_ROOT, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:
        for f in files[:5]:
            print(f'{indent}  {f}')
        if len(files) > 5:
            print(f'{indent}  ... and {len(files)-5} more files')
