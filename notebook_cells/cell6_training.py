# CELL 6 — Production Training Loop
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from constants import (
    LR, NUM_EPOCHS, PATIENCE, device,
    CLASS_LABELS, NUM_CLASSES, IMG_SIZE,
    PIXEL_MEAN, PIXEL_STD, CHECKPOINT_PATH
)

# --- Optimizer & Scheduler ---
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler('cuda')

# --- OOM Fallback ---
GRAD_ACCUM_STEPS = 1  # Increase to 2 if OOM with batch_size=128
MAX_GRAD_NORM = 1.0

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False  # not improved

early_stop = EarlyStopping(patience=PATIENCE)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# --- Training Loop ---
print(f'Training on {device} | Epochs: {NUM_EPOCHS} | Patience: {PATIENCE}')
print(f'Mixed Precision: ON | Grad Clipping: {MAX_GRAD_NORM} | Accum Steps: {GRAD_ACCUM_STEPS}')
print('=' * 70)

for epoch in range(NUM_EPOCHS):
    t0 = time.time()

    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        try:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'OOM at batch {batch_idx}! Try reducing BATCH_SIZE to 64 with GRAD_ACCUM_STEPS=2')
                torch.cuda.empty_cache()
                continue
            raise e

        train_loss += loss.item() * GRAD_ACCUM_STEPS * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # Log
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    elapsed = time.time() - t0
    improved = early_stop(val_acc)

    marker = ' *** BEST ***' if improved else ''
    print(f'Epoch {epoch+1:02d}/{NUM_EPOCHS} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
          f'{elapsed:.0f}s{marker}')

    # Save best checkpoint
    if improved:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'class_labels': CLASS_LABELS,
            'config': {'num_classes': NUM_CLASSES, 'img_size': IMG_SIZE,
                       'pixel_mean': PIXEL_MEAN, 'pixel_std': PIXEL_STD}
        }, CHECKPOINT_PATH)

    if early_stop.should_stop:
        print(f'\nEarly stopping at epoch {epoch+1}. Best val_acc: {early_stop.best_score:.4f}')
        break

print('=' * 70)
print(f'Training complete. Best val_acc: {early_stop.best_score:.4f}')

# --- Plot Training Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.legend(); ax1.set_xlabel('Epoch')
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.legend(); ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()
