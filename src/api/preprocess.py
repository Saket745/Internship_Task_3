import cv2
import numpy as np
import os

def preprocess_image(
    image_path: str,
    target_size: int = 28,
    min_contour_area: int = 50,
    padding_ratio: float = 0.1
) -> list[np.ndarray]:
    """
    Args:
        image_path: Path to image file
        target_size: Output size (default 28x28)
        min_contour_area: Minimum contour area in pixels²
        padding_ratio: Padding ratio around character
    
    Returns:
        List of 28×28 normalized float32 arrays
    
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If image cannot be read or is invalid
        RuntimeError: If preprocessing fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image unreadable: {image_path}")
        
    # Convert to grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
            
    # Filter overlapping bboxes (keep larger ones)
    filtered_bboxes = []
    for i, bbox1 in enumerate(bboxes):
        x1, y1, w1, h1 = bbox1
        keep = True
        for j, bbox2 in enumerate(bboxes):
            if i == j:
                continue
            x2, y2, w2, h2 = bbox2
            # Check if bbox1 is inside bbox2
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                if w1 * h1 < w2 * h2:
                    keep = False
                    break
        if keep:
            filtered_bboxes.append(bbox1)
            
    results = []
    for bbox in filtered_bboxes:
        x, y, w, h = bbox
        roi = binary[y:y+h, x:x+w]
        
        if roi.size == 0:
            continue
            
        # Add padding (10% margin)
        pad_h = int(h * padding_ratio)
        pad_w = int(w * padding_ratio)
        
        # Ensure at least 1 pixel padding if ratio > 0
        if padding_ratio > 0:
            pad_h = max(pad_h, 1)
            pad_w = max(pad_w, 1)
            
        padded = cv2.copyMakeBorder(
            roi, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Resize
        try:
            resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            raise RuntimeError(f"Resize failed for bbox {bbox}: {e}")
            
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Expand dims to (1, 28, 28)
        tensor = np.expand_dims(normalized, axis=0)
        
        # Assertions
        assert resized.shape == (target_size, target_size)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
        
        results.append(tensor)
        
    return results

if __name__ == "__main__":
    print("Testing preprocess_image...")
    import tempfile
    
    # Create a dummy image with a white square on black background
    # Wait, the function expects dark text on light background and inverts it.
    # So we should create dark text (0) on light background (255).
    dummy_img = np.ones((100, 100), dtype=np.uint8) * 255
    # Draw a black square (dark text)
    dummy_img[30:70, 30:70] = 0
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        dummy_path = f.name
        cv2.imwrite(dummy_path, dummy_img)
        
    try:
        results = preprocess_image(dummy_path)
        print(f"Found {len(results)} characters.")
        if len(results) > 0:
            print(f"Tensor shape: {results[0].shape}")
            assert results[0].shape == (1, 28, 28)
            print("Preprocess tests passed!")
        else:
            print("Failed to find character.")
    finally:
        os.remove(dummy_path)
