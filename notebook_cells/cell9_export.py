# CELL 9 — Export & Save
import torch
import os

# --- Save as .pth ---
print('1. Saving model as .pth ...')
final_save_path = '/kaggle/working/emnist_efficientnet_b0_62cls.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'class_labels': CLASS_LABELS,
    'num_classes': NUM_CLASSES,
    'architecture': 'efficientnet_b0',
    'img_size': IMG_SIZE,
    'pixel_mean': PIXEL_MEAN,
    'pixel_std': PIXEL_STD,
    'best_val_acc': early_stop.best_score,
}, final_save_path)
print(f'   Saved: {final_save_path} ({os.path.getsize(final_save_path)/1e6:.1f} MB)')

# --- Export to ONNX ---
print('\n2. Exporting to ONNX ...')
onnx_path = '/kaggle/working/emnist_efficientnet_b0_62cls.onnx'
model.eval()
model_cpu = model.cpu()
dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model_cpu, dummy_input, onnx_path,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17
)
print(f'   Saved: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)')

# Move model back to GPU
model = model.to(device)

# --- Verify ONNX ---
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('   ONNX model verified OK')
except ImportError:
    print('   onnx package not installed; skip verification (pip install onnx)')

# --- Upload to Kaggle Models (optional) ---
print('\n3. To upload to Kaggle Models, run:')
print('   !kaggle models instances versions create \\')
print('     --model <your-username>/emnist-efficientnet-b0 \\')
print('     --instance-slug default \\')
print('     --version-notes "EfficientNet-B0 on EMNIST ByClass 62 classes" \\')
print('     --dir-or-file /kaggle/working/')

# --- Final Summary ---
print('\n' + '=' * 60)
print('FINAL OUTPUT FILES:')
print('=' * 60)
for f in ['best_model.pth', 'emnist_efficientnet_b0_62cls.pth',
          'emnist_efficientnet_b0_62cls.onnx',
          'class_distribution.png', 'sample_grid.png',
          'training_curves.png', 'confusion_matrix.png', 'misclassified.png']:
    fpath = f'/kaggle/working/{f}'
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / 1e6
        print(f'  [OK] {f} ({size:.1f} MB)')
    else:
        print(f'  [--] {f} (not found)')

print('\nDone! Notebook complete.')
