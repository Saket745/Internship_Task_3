# CELL 7 — Evaluation
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from constants import CLASS_LABELS, NUM_CLASSES, device, CHECKPOINT_PATH

# Load best checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded best model from epoch {checkpoint["epoch"]+1} (val_acc={checkpoint["val_acc"]:.4f})')

# --- Run on test set ---
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = (all_preds == all_labels).mean()
print(f'\nTest Accuracy: {test_acc:.4f} ({(all_preds == all_labels).sum():,}/{len(all_labels):,})')

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix (Test Acc: {test_acc:.4f})', fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

# --- Per-Class Report ---
report = classification_report(all_labels, all_preds, target_names=CLASS_LABELS, output_dict=True)
print('\n--- Per-Class Precision / Recall / F1 ---')
print(classification_report(all_labels, all_preds, target_names=CLASS_LABELS))

# --- Top 5 Most Confused Pairs ---
np.fill_diagonal(cm, 0)
confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((cm[i, j], CLASS_LABELS[i], CLASS_LABELS[j]))
confused_pairs.sort(reverse=True)

print('\n--- Top 5 Most Confused Pairs ---')
for count, true_cls, pred_cls in confused_pairs[:5]:
    print(f'  {true_cls} → {pred_cls}: {count:,} misclassifications')

# --- Visual Grid of Misclassified Samples ---
misclassified_idx = np.where(all_preds != all_labels)[0]
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('Misclassified Samples (True → Predicted)', fontsize=14)
for i, ax in enumerate(axes.flat):
    if i >= len(misclassified_idx):
        ax.axis('off')
        continue
    idx = misclassified_idx[i]
    img, _ = test_dataset[idx]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'{CLASS_LABELS[all_labels[idx]]}→{CLASS_LABELS[all_preds[idx]]}',
                 color='red', fontsize=11)
    ax.axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/misclassified.png', dpi=150)
plt.show()
