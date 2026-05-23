# CELL 1 — Environment Check & GPU Verification
import torch, sys, random, os
import numpy as np

print("=" * 60)
print("CELL 1: ENVIRONMENT CHECK & GPU VERIFICATION")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB')
    print(f'CUDA: {torch.version.cuda}')
else:
    print('WARNING: No GPU! Enable in Runtime > Change runtime type > GPU')

print(f'PyTorch: {torch.__version__} | Python: {sys.version.split()[0]} | Device: {device}')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# Project Constants
NUM_CLASSES = 62
IMG_SIZE = 128
BATCH_SIZE = 128  # T4-optimized; fallback to 64 if OOM
NUM_EPOCHS = 25
LR = 1e-3
PATIENCE = 5
NUM_WORKERS = 2
CHECKPOINT_PATH = '/kaggle/working/best_model.pth'

CLASS_LABELS = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z')+1)] +
    [chr(i) for i in range(ord('a'), ord('z')+1)]
)
assert len(CLASS_LABELS) == NUM_CLASSES

print(f'\nConfig: {NUM_CLASSES} classes | {IMG_SIZE}x{IMG_SIZE} | batch={BATCH_SIZE} | lr={LR}')
print(f'Seed: {SEED} | Checkpoint: {CHECKPOINT_PATH}')
print("=" * 60)
