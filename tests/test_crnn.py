import torch
import pytest
from src.ml.crnn import CRNNModel

def test_crnn_output_shape():
    batch_size = 4
    num_classes = 37
    model = CRNNModel(num_classes=num_classes)
    
    # Input shape: (batch, channels, height, width)
    # Task 5.1 specifies 32x128 images
    x = torch.randn(batch_size, 1, 32, 128)
    
    output = model(x)
    
    # Expected output shape: (batch, sequence_length, num_classes)
    # Sequence length is 32 because of the CNN reductions:
    # 128 -> 64 -> 32 -> 32 -> 32 -> 32
    
    assert output.shape == (batch_size, 32, num_classes), f"Expected shape {(batch_size, 32, num_classes)}, got {output.shape}"

def test_crnn_summary(capsys):
    model = CRNNModel(num_classes=37)
    model.get_model_summary()
    captured = capsys.readouterr()
    assert "Total parameters:" in captured.out
    assert "Trainable parameters:" in captured.out
