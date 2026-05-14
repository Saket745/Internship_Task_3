import onnxruntime as ort
import numpy as np
import os

class CharacterInference:
    def __init__(self, model_path="models/character_cnn.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Mapping for EMNIST Balanced (47 classes)
        # 0-9: 0-9
        # 10-35: A-Z
        # 36-46: a,b,d,e,f,g,h,n,q,r,t
        self.mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

    def predict(self, image_tensor):
        """
        image_tensor: numpy array of shape (1, 1, 28, 28)
        """
        outputs = self.session.run(None, {self.input_name: image_tensor})
        logits = outputs[0]
        
        # Softmax for confidence
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        predicted_idx = np.argmax(probs)
        confidence = float(probs[0, predicted_idx])
        
        return self.mapping[predicted_idx], confidence

if __name__ == "__main__":
    # Test inference with robust path discovery
    # Assuming this script is in src/api/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, "models", "character_cnn.onnx")
    
    if os.path.exists(model_path):
        inf = CharacterInference(model_path)
        dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
        char, conf = inf.predict(dummy_input)
        print(f"Prediction: {char}, Confidence: {conf:.4f}")
    else:
        print(f"Model not found at {model_path}. Run training and export first.")
