import torch
from src.ml.model import CharacterCNN
import os

def export_to_onnx(model_path=None, onnx_path=None):
    """
    Exports the trained PyTorch model to ONNX format.
    """
    # Robust path discovery
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if model_path is None:
        model_path = os.path.join(project_root, "models", "character_cnn.pth")
    if onnx_path is None:
        onnx_path = os.path.join(project_root, "models", "character_cnn.onnx")

    model = CharacterCNN()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: No weights found at {model_path}. Exporting untrained model for structural verification.")

    model.eval()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    
    # Dummy input matching the model's input shape
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported successfully to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
