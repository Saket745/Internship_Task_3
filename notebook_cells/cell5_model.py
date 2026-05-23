# CELL 5 — Model Setup (EfficientNet-B0 Fine-Tuning)
import torch
import torch.nn as nn
import torchvision.models as models
from constants import NUM_CLASSES, IMG_SIZE, device

# --- Decision Logic ---
DATASET_SIZE = len(full_train)
USE_PRETRAINED = True  # Fine-tune even with >200k images; transfer learning still helps

print(f'Dataset size: {DATASET_SIZE:,} images')
if DATASET_SIZE > 200_000:
    print('Dataset > 200k: Could train from scratch, but fine-tuning is STILL faster + better.')
print(f'Strategy: {"Fine-tune EfficientNet-B0 (ImageNet)" if USE_PRETRAINED else "Train from scratch"}')

# --- Option B: Fine-tune EfficientNet-B0 (WINNER) ---
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Modify input: 3ch RGB → 1ch grayscale
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )
    # Initialize from pretrained: average RGB weights into single channel
    if pretrained:
        with torch.no_grad():
            model.features[0][0].weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    # Modify output: 1000 ImageNet classes → 62 EMNIST classes
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    # --- Freeze/Unfreeze Strategy ---
    # Freeze all backbone layers first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last 2 conv blocks (features[7] and features[8]) for fine-tuning
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.features[8].parameters():
        param.requires_grad = True

    # Classifier is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

# --- Option A: Train from scratch (alternative) ---
class CharCNN(nn.Module):
    """Lightweight CNN for training from scratch if needed."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

# --- Build chosen model ---
model = build_model(num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED)
model = model.to(device)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f'\nModel: EfficientNet-B0')
print(f'Total params:     {total_params:,}')
print(f'Trainable params: {trainable_params:,}')
print(f'Frozen params:    {frozen_params:,}')
print(f'Input:  1x{IMG_SIZE}x{IMG_SIZE} grayscale')
print(f'Output: {NUM_CLASSES} classes')
