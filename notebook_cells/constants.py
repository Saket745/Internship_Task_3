"""
Shared project constants for all notebook cells.
Defined once here so each cell can import cleanly without
re-running the entire environment setup.
"""
import os
import random
import torch
import numpy as np

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# ── Model / Data Hyperparameters ─────────────────────────────
NUM_CLASSES = 62
IMG_SIZE = 128
BATCH_SIZE = 128          # T4-optimized; fall back to 64 on OOM
NUM_EPOCHS = 25
LR = 1e-3
PATIENCE = 5
NUM_WORKERS = 2
CHECKPOINT_PATH = '/kaggle/working/best_model.pth'

# ── Class Labels (ByClass: 0-9, A-Z, a-z) ────────────────────
CLASS_LABELS: list[str] = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    [chr(i) for i in range(ord('a'), ord('z') + 1)]
)
assert len(CLASS_LABELS) == NUM_CLASSES, (
    f"CLASS_LABELS length {len(CLASS_LABELS)} != NUM_CLASSES {NUM_CLASSES}"
)

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Pixel Stats (EMNIST ByClass, grayscale) ───────────────────
PIXEL_MEAN = (0.1307,)
PIXEL_STD  = (0.3081,)
