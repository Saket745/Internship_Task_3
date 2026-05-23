import pytest
import cv2
import numpy as np
import os
from src.api.preprocess import preprocess_image

def test_preprocess_single_contour(tmp_path):
    """Test with a single contour image (dark character on light background)"""
    img_path = tmp_path / "single.png"
    # Create light background image with a dark square
    img = np.full((100, 100), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 0, -1)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 1
    assert results[0].shape == (1, 28, 28)
    assert results[0].dtype == np.float32
    assert results[0].min() >= 0.0
    assert results[0].max() <= 1.0

def test_preprocess_multiple_contours(tmp_path):
    """Test with multiple contours"""
    img_path = tmp_path / "multiple.png"
    img = np.full((100, 100), 255, dtype=np.uint8)
    # Two separate dark squares
    cv2.rectangle(img, (10, 10), (30, 30), 0, -1)
    cv2.rectangle(img, (60, 60), (80, 80), 0, -1)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 2

def test_preprocess_no_contours(tmp_path):
    """Test with no contours (empty result or everything matches bg)"""
    img_path = tmp_path / "empty.png"
    img = np.full((100, 100), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 0

def test_preprocess_color_image(tmp_path):
    """Test with RGB/BGR color spaces"""
    img_path = tmp_path / "color.png"
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 1

def test_preprocess_small_image(tmp_path):
    """Test with very small image (< 50x50)"""
    img_path = tmp_path / "small.png"
    img = np.full((30, 30), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (25, 25), 0, -1)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 1

def test_preprocess_large_image(tmp_path):
    """Test with very large image (> 2000x2000)"""
    img_path = tmp_path / "large.png"
    img = np.full((2100, 2100), 255, dtype=np.uint8)
    cv2.rectangle(img, (500, 500), (1500, 1500), 0, -1)
    cv2.imwrite(str(img_path), img)
    
    results = preprocess_image(str(img_path))
    assert len(results) == 1

def test_preprocess_file_not_found():
    """Verify FileNotFoundError is raised"""
    with pytest.raises(FileNotFoundError):
        preprocess_image("non_existent_file.png")

def test_preprocess_invalid_image(tmp_path):
    """Verify ValueError is raised for invalid images"""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("not an image")
    
    with pytest.raises(ValueError):
        preprocess_image(str(txt_path))
