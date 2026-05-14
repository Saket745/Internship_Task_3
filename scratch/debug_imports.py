import time
print("Importing torch...")
import torch
print("Importing torch.nn...")
import torch.nn as nn
print("Importing torch.optim...")
import torch.optim as optim
print("Importing torchvision...")
import torchvision
from torchvision import datasets, transforms
print("Importing matplotlib...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("Importing others...")
import os
import numpy as np
from PIL import Image
print("All imports successful!")
print(f"CUDA: {torch.cuda.is_available()}")
