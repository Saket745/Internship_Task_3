import unittest
import torch
from src.ml.data_loader import get_transforms

class TestDataLoader(unittest.TestCase):
    def test_transforms_shape(self):
        """Verify that transforms produce correct shapes and types."""
        from PIL import Image
        import numpy as np
        
        # Create a dummy grayscale image (28x28)
        dummy_img = Image.fromarray(np.uint8(np.random.rand(28, 28) * 255))
        
        # Get transforms (no augmentation)
        transform = get_transforms(augment=False)
        
        # Apply transform
        tensor_img = transform(dummy_img)
        
        # Check shape: (Channel, Height, Width) -> (1, 28, 28)
        self.assertEqual(tensor_img.shape, (1, 28, 28))
        self.assertIsInstance(tensor_img, torch.Tensor)

    def test_normalization(self):
        """Verify that normalization is applied."""
        from PIL import Image
        import numpy as np
        
        # Create a white image
        white_img = Image.fromarray(np.uint8(np.ones((28, 28)) * 255))
        
        transform = get_transforms(augment=False)
        tensor_img = transform(white_img)
        
        # Normalized value for 1.0 (white) with mean 0.1736 and std 0.3317:
        # (1.0 - 0.1736) / 0.3317 = 2.4914
        expected_val = (1.0 - 0.1736) / 0.3317
        self.assertAlmostEqual(tensor_img.max().item(), expected_val, places=3)

if __name__ == '__main__':
    unittest.main()
