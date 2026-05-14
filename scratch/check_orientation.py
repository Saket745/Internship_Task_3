import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

def test_orientation():
    # Load one image from EMNIST without transform
    ds_raw = datasets.EMNIST(root='./data', split='byclass', train=True, download=True)
    img, label = ds_raw[0]
    
    # Original image
    img_orig = np.array(img)
    
    # Applied transform from data_loader.py
    # transforms.Lambda(lambda img: TF.hflip(TF.rotate(img, -90, interpolation=TF.InterpolationMode.BILINEAR)))
    img_fixed = TF.rotate(img, -90, interpolation=TF.InterpolationMode.BILINEAR)
    img_fixed = TF.hflip(img_fixed)
    img_fixed = np.array(img_fixed)
    
    print(f"Original shape: {img_orig.shape}")
    print(f"Fixed shape: {img_fixed.shape}")
    
    # If the fixed image looks like a character and the original is tilted, then it's correct.
    # But we can't see it here easily.
    # However, let's check the label.
    print(f"Label: {label}")

if __name__ == "__main__":
    test_orientation()
