# CELL 8 — Inference Pipeline
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model_for_inference(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = build_model(num_classes=config['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([config['pixel_mean']], [config['pixel_std']]),
    ])
    return model, preprocess, checkpoint['class_labels']

def predict_single_image(img_path_or_pil, model, preprocess, class_labels, device='cuda'):
    """
    Predict a single image. Accepts file path or PIL Image.
    Returns: predicted class, confidence, top-3 predictions.
    """
    # Load image
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil)
    else:
        img = img_path_or_pil

    # Edge case: convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Edge case: aspect-ratio-preserving resize with padding
    w, h = img.size
    max_dim = max(w, h)
    padded = Image.new('L', (max_dim, max_dim), 0)
    padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    img = padded

    # Preprocess and predict
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast('cuda'):
            logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().squeeze()

    # Top-3 predictions
    top3_probs, top3_idx = probs.topk(3)
    top3 = [(class_labels[idx], prob.item()) for idx, prob in zip(top3_idx, top3_probs)]

    predicted_class = top3[0][0]
    confidence = top3[0][1]

    return predicted_class, confidence, top3

# --- Demo ---
inf_model, inf_preprocess, inf_labels = load_model_for_inference(CHECKPOINT_PATH, device)

# Test on a random test image
demo_img, demo_label = test_dataset[42]
demo_pil = transforms.ToPILImage()(demo_img)

pred_class, conf, top3 = predict_single_image(demo_pil, inf_model, inf_preprocess, inf_labels, device)

print(f'True Label:  {CLASS_LABELS[demo_label]}')
print(f'Predicted:   {pred_class} (confidence: {conf:.4f})')
print(f'Top-3:')
for cls, prob in top3:
    print(f'  {cls}: {prob:.4f}')
