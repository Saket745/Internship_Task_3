# Hallucination-Proof Implementation Plan
## Tasks: 4.3, 4.4, 5.1, 5.4

---

## TASK 4.3: OpenCV Preprocessing Pipeline

### Objective
Build a robust character detection and bounding box extraction pipeline that converts raw handwritten digit/character images to normalized 28×28 tensors.

### Input Specifications
- **Image format**: PNG, JPG (any color space)
- **Image size**: Variable (will be resized)
- **Expected content**: Handwritten digits or single characters on light background
- **Quality**: May contain noise, varying contrast

### Processing Steps (in order)

#### Step 1: Load & Validate Input
```
Input: File path
Output: numpy array (height, width, 3) OR (height, width)
Validation:
  - Check file exists
  - Check readable by cv2.imread()
  - Assert shape is not None
  - Assert shape has 2 or 3 dimensions
  - Raise ValueError if fails
```

#### Step 2: Convert to Grayscale
```
Input: Image array (any channels)
Output: Grayscale array (height, width)
Logic:
  - If already grayscale (2D): pass through
  - If color (3D): use cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Validation:
  - Assert output shape is 2D
```

#### Step 3: Apply Thresholding
```
Input: Grayscale image
Output: Binary image (0 or 255 values only)
Method: cv2.THRESH_BINARY_INV (inverted for dark text on light bg)
Threshold value: 127 (standard midpoint)
Logic:
  ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
Validation:
  - Assert output only contains 0 and 255
  - Assert output is 2D
```

#### Step 4: Find Contours
```
Input: Binary image
Output: List of contours
Logic:
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Validation:
  - Assert contours is list/tuple
  - If len(contours) == 0: return empty list with warning
  - Filter contours by area (minimum 50 pixels²)
```

#### Step 5: Extract Bounding Boxes
```
Input: List of contours
Output: List of bounding boxes (x, y, w, h)
Logic:
  for contour in contours:
    area = cv2.contourArea(contour)
    if area >= 50:
      x, y, w, h = cv2.boundingRect(contour)
      bboxes.append((x, y, w, h))
Validation:
  - Assert all values are positive integers
  - Assert w > 0 and h > 0
  - Filter overlapping bboxes (keep larger ones)
```

#### Step 6: Extract & Resize to 28×28
```
Input: Original image, bounding boxes
Output: List of 28×28 normalized tensors

For each bbox (x, y, w, h):
  1. Extract region: roi = image[y:y+h, x:x+w]
  2. Validate region exists: assert roi.size > 0
  3. Add padding (10% margin): 
     - padded = add_padding(roi, pad_ratio=0.1)
  4. Resize to 28×28:
     - resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_LINEAR)
  5. Normalize to [0, 1]:
     - normalized = resized / 255.0
  6. Convert to tensor:
     - tensor = np.expand_dims(normalized, axis=0)  # Shape: (1, 28, 28)

Validation:
  - Assert resized.shape == (28, 28)
  - Assert normalized.min() >= 0.0
  - Assert normalized.max() <= 1.0
  - Assert normalized.dtype == np.float32
```

### Function Signature
```python
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
```

### Error Handling
- **File not found**: Raise FileNotFoundError with path
- **Cannot read image**: Raise ValueError("Image unreadable: {reason}")
- **No contours found**: Log warning, return empty list
- **Contour too small**: Skip with debug log
- **Resize fails**: Raise RuntimeError("Resize failed for bbox {bbox}")

### Testing Checklist
- [ ] Test with 8-bit, 16-bit grayscale images
- [ ] Test with RGB and BGR color spaces
- [ ] Test with single contour image
- [ ] Test with multiple contours
- [ ] Test with no contours (empty result)
- [ ] Test with very small image (< 50×50)
- [ ] Test with very large image (> 2000×2000)
- [ ] Verify output shape is always (1, 28, 28)
- [ ] Verify output values in [0, 1] range
- [ ] Verify output dtype is float32

---

## TASK 4.4: API Integration Tests (pytest & FastAPI)

### Objective
Write comprehensive integration tests for the FastAPI digit recognition API without mocking the actual preprocessing/model pipeline.

### Test Structure
```
tests/
├── test_api.py          (integration tests)
├── conftest.py          (fixtures and setup)
├── test_data/
│   ├── valid_digit.png  (28×28, single digit)
│   ├── invalid.txt      (non-image file)
│   └── empty.png        (empty/noise image)
```

### Setup Fixtures (conftest.py)

```python
@pytest.fixture
def client():
    """FastAPI TestClient instance"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)

@pytest.fixture
def sample_image_path():
    """Path to valid test image"""
    return "tests/test_data/valid_digit.png"

@pytest.fixture
def invalid_image_path():
    """Path to invalid test image"""
    return "tests/test_data/invalid.txt"
```

### Test Cases

#### Test Group 1: Health & Basic Endpoints

```python
def test_api_health_check(client):
    """
    Verify API is alive
    Expected: 200 OK
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_api_root_endpoint(client):
    """
    Verify root endpoint returns API info
    Expected: 200 OK with version/description
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data or "message" in data
```

#### Test Group 2: Image Upload - Valid Cases

```python
def test_upload_valid_png(client, sample_image_path):
    """
    Upload valid PNG image
    Expected: 200 OK, predictions array
    Validation:
      - Response contains 'predictions' key
      - predictions is list or array
      - All values are floats between 0-1
    """
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (sample_image_path, f, "image/png")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], (list, dict))
    
    # Validate prediction values
    preds = data["predictions"]
    if isinstance(preds, dict):
        preds = list(preds.values())
    for pred in preds:
        assert isinstance(pred, (int, float))
        assert 0 <= pred <= 1

def test_upload_valid_jpg(client):
    """
    Upload valid JPG image
    Expected: 200 OK, same as PNG
    """
    # Similar to PNG test but with JPG format
    pass

def test_predictions_are_deterministic(client, sample_image_path):
    """
    Same image should produce same predictions
    Expected: Two uploads of same image return identical predictions
    """
    with open(sample_image_path, "rb") as f:
        resp1 = client.post(
            "/predict",
            files={"file": (sample_image_path, f, "image/png")}
        )
    
    with open(sample_image_path, "rb") as f:
        resp2 = client.post(
            "/predict",
            files={"file": (sample_image_path, f, "image/png")}
        )
    
    assert resp1.json()["predictions"] == resp2.json()["predictions"]
```

#### Test Group 3: Image Upload - Invalid Cases

```python
def test_upload_nonexistent_file(client):
    """
    Upload non-existent file
    Expected: 400 or 422 Bad Request
    """
    with pytest.raises(FileNotFoundError):
        with open("nonexistent.png", "rb") as f:
            client.post("/predict", files={"file": f})

def test_upload_non_image_file(client, invalid_image_path):
    """
    Upload .txt file (not an image)
    Expected: 400 Bad Request
    Validation:
      - Response status is 400
      - Response contains error message
    """
    with open(invalid_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (invalid_image_path, f, "text/plain")}
        )
    
    assert response.status_code == 400
    assert "error" in response.json() or "message" in response.json()

def test_upload_no_file(client):
    """
    Send POST to /predict without file
    Expected: 422 Unprocessable Entity
    """
    response = client.post("/predict")
    assert response.status_code in [400, 422]

def test_upload_corrupted_image(client):
    """
    Upload corrupted/truncated image file
    Expected: 400 Bad Request
    """
    with open("tests/test_data/corrupted.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("corrupted.png", f, "image/png")}
        )
    
    assert response.status_code == 400
```

#### Test Group 4: Response Format

```python
def test_response_json_structure(client, sample_image_path):
    """
    Verify JSON response structure is consistent
    Expected response format:
    {
        "predictions": [0.1, 0.9, ...],  or {"0": 0.1, "1": 0.9, ...}
        "confidence": float,
        "class": int or str,
        "timestamp": str (ISO format)
    }
    """
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (sample_image_path, f, "image/png")}
        )
    
    data = response.json()
    
    # Required fields
    assert "predictions" in data
    
    # Optional but recommended
    if "confidence" in data:
        assert isinstance(data["confidence"], (int, float))
        assert 0 <= data["confidence"] <= 1
    
    if "class" in data:
        assert isinstance(data["class"], (int, str))
    
    if "timestamp" in data:
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"])  # Should not raise

def test_response_content_type(client, sample_image_path):
    """
    Verify response is JSON
    """
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (sample_image_path, f, "image/png")}
        )
    
    assert response.headers["content-type"] == "application/json"
```

#### Test Group 5: Batch Operations (if applicable)

```python
def test_batch_predict(client, sample_image_path):
    """
    Upload multiple images at once
    Expected: 200 OK with array of predictions
    """
    files = []
    with open(sample_image_path, "rb") as f:
        files.append(("files", (sample_image_path, f, "image/png")))
    
    response = client.post("/predict_batch", files=files)
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == len(files)
```

#### Test Group 6: Performance & Limits

```python
def test_upload_large_image(client):
    """
    Upload very large image (>10MB)
    Expected: Either process or reject with 413
    """
    # Create or use pre-made large image
    with open("tests/test_data/large_image.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("large_image.png", f, "image/png")}
        )
    
    assert response.status_code in [200, 413]  # 413 = Payload Too Large

def test_concurrent_requests(client, sample_image_path):
    """
    Send 10 concurrent requests
    Expected: All succeed without race conditions
    """
    import concurrent.futures
    
    def upload():
        with open(sample_image_path, "rb") as f:
            return client.post(
                "/predict",
                files={"file": (sample_image_path, f, "image/png")}
            )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(upload) for _ in range(10)]
        responses = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in responses)
```

### Test Execution Command
```bash
pytest tests/test_api.py -v --tb=short --color=yes
pytest tests/test_api.py::test_upload_valid_png -v  # Single test
pytest tests/test_api.py -k "valid" -v              # Filter by keyword
```

### Coverage Target
- Minimum 80% code coverage for API endpoints
- Check with: `pytest --cov=main tests/test_api.py`

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: API Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install pytest fastapi opencv-python
      - run: pytest tests/test_api.py -v
```

---

## TASK 5.1: Implement CRNN Model

### Objective
Implement a CRNN (CNN + BiLSTM + Dense) architecture that accepts 32×128 images and outputs sequence predictions.

### Architecture Specification

#### Input
- **Shape**: (batch_size, 1, 32, 128) — grayscale images
- **Format**: float32, values in [0, 1]
- **Interpretation**: 32 height × 128 width (landscape orientation for text)

#### Output
- **Shape**: (batch_size, sequence_length, num_classes)
- **Format**: float32, logits (not softmax)
- **Interpretation**: Per-timestep class probabilities (for CTC loss)

### Network Components (in order)

#### Component 1: CNN Feature Extractor

```python
class CNNFeatureExtractor(nn.Module):
    """
    Extract spatial features from image
    Output shape: (batch, 512, height_reduced, width_reduced)
    
    Architecture:
    - Conv2D filters follow pattern: 32 → 64 → 128 → 256 → 512
    - All kernels: 3×3, padding=1 (preserve spatial dims)
    - Activation: ReLU after each conv
    - MaxPool: 2×2 after specific layers to reduce spatial dims
    - Batch normalization: After each conv layer
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2, W/2
        
        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4, W/4
        
        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))  # H/8, W/4 (vertical pool only)
        
        # Block 4: 128 → 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # H/16, W/4
        
        # Block 5: 256 → 512 channels
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 128)
        Returns:
            features: (batch, 512, 2, 32)
        """
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))     # (B, 32, 16, 64)
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))     # (B, 64, 8, 32)
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))     # (B, 128, 4, 32)
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))     # (B, 256, 2, 32)
        x = self.bn5(F.relu(self.conv5(x)))                 # (B, 512, 2, 32)
        
        # Validation
        assert x.shape[1] == 512, f"Expected 512 channels, got {x.shape[1]}"
        
        return x
```

#### Component 2: BiLSTM Sequence Encoder

```python
class BiLSTMSequenceEncoder(nn.Module):
    """
    Encode spatial features into sequence representation
    
    Process:
    1. Reshape (batch, 512, 2, 32) → (batch, 32, 2*512)
       - Collapse height dimension by stacking: 2 × 512 = 1024
       - Sequence length = 32 (width)
    2. BiLSTM: hidden_size = 256, 2 layers
    3. Output: (batch, 32, 512) [256 forward + 256 backward]
    
    Why this reshape:
    - LSTM expects (batch, sequence_length, features)
    - We interpret width as sequence (left-to-right text)
    - Height is collapsed into features
    """
    
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        
        # Input size = height (2) × feature_channels (512)
        self.input_size = 2 * 512  # 1024
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
    
    def forward(self, features):
        """
        Args:
            features: (batch, 512, 2, 32) from CNN
        Returns:
            lstm_out: (batch, 32, 512) [256*2 for bidirectional]
        """
        batch_size = features.shape[0]
        
        # Reshape: collapse height dimension
        # (B, 512, 2, 32) → (B, 32, 2*512)
        features_reshaped = features.permute(0, 3, 1, 2)  # (B, 32, 512, 2)
        features_reshaped = features_reshaped.contiguous().view(
            batch_size, features.shape[3], -1
        )  # (B, 32, 1024)
        
        # Validation
        assert features_reshaped.shape == (batch_size, 32, 1024), \
            f"Expected (B, 32, 1024), got {features_reshaped.shape}"
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(features_reshaped)
        
        # Validation
        assert lstm_out.shape == (batch_size, 32, 2 * self.hidden_size), \
            f"Expected (B, 32, 512), got {lstm_out.shape}"
        
        return lstm_out
```

#### Component 3: Classification Head

```python
class ClassificationHead(nn.Module):
    """
    Dense layers to predict character class per timestep
    
    Input: (batch, sequence_length, 512) from BiLSTM
    Output: (batch, sequence_length, num_classes) logits
    
    No softmax here (softmax applied in CTC loss)
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Dense layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, lstm_out):
        """
        Args:
            lstm_out: (batch, sequence_length, 512)
        Returns:
            logits: (batch, sequence_length, num_classes)
        """
        # Apply dense layers per timestep
        x = self.dropout(F.relu(self.fc1(lstm_out)))  # (B, T, 256)
        logits = self.fc2(x)                          # (B, T, num_classes)
        
        # Validation
        expected_shape = (lstm_out.shape[0], lstm_out.shape[1], self.num_classes)
        assert logits.shape == expected_shape, \
            f"Expected {expected_shape}, got {logits.shape}"
        
        return logits
```

#### Full CRNN Model

```python
class CRNNModel(nn.Module):
    """
    Complete CRNN: CNN + BiLSTM + Dense
    
    Forward pass flow:
    (B, 1, 32, 128) → CNN → (B, 512, 2, 32) 
                          → BiLSTM → (B, 32, 512)
                          → Dense → (B, 32, num_classes)
    """
    
    def __init__(self, num_classes=37):
        """
        Args:
            num_classes: Number of output classes (default 37 for MNIST + CTC blank)
        """
        super().__init__()
        self.num_classes = num_classes
        
        self.cnn = CNNFeatureExtractor()
        self.lstm = BiLSTMSequenceEncoder(hidden_size=256, num_layers=2)
        self.head = ClassificationHead(num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 128) - grayscale images
        Returns:
            logits: (batch, 32, num_classes) - raw predictions
        """
        features = self.cnn(x)              # (B, 512, 2, 32)
        lstm_out = self.lstm(features)      # (B, 32, 512)
        logits = self.head(lstm_out)        # (B, 32, num_classes)
        
        return logits
    
    def get_model_summary(self):
        """Print model architecture and parameter count"""
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
```

### Model Instantiation & Validation

```python
# Create model
model = CRNNModel(num_classes=37)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Test forward pass with dummy data
dummy_input = torch.randn(4, 1, 32, 128)  # Batch of 4 images
output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: torch.Size([4, 32, 37])")
assert output.shape == (4, 32, 37), f"Shape mismatch: {output.shape}"
```

### Important Constraints
- **NO ImageNet pretraining**: Train from scratch for handwriting
- **Batch normalization**: Essential for convergence with BiLSTM
- **Dropout**: 0.5 after dense layers, not after LSTM (LSTM has built-in dropout)
- **Sequence length**: Fixed at 32 timesteps (from width reduction)
- **Device handling**: Must work on CPU and GPU

### Error Prevention Checklist
- [ ] All forward passes checked with `assert` statements
- [ ] Shape assertions after each major component
- [ ] ReLU applied consistently (not before BN)
- [ ] Dropout only during training (use `model.train()` / `model.eval()`)
- [ ] No softmax in model (applied in CTC loss)
- [ ] LSTM input shape is (batch, seq_len, features)
- [ ] Output logits shape matches num_classes
- [ ] Model can be loaded/saved with `torch.save(model.state_dict())`

---

## TASK 5.4: Prepare IAM/CVL Dataset Loader

### Objective
Build a robust data loader for handwritten text dataset with preprocessing, batching, and augmentation.

### Dataset Specifications

#### Option A: IAM Handwriting Database
- **Size**: ~115k word images
- **Format**: PNG images + XML annotations
- **Text**: English words
- **Access**: Requires registration (free)
- **URL**: http://www.fki.inf.unibe.ch/databases/iam-handwriting-db

#### Option B: CVL Database
- **Size**: ~20k word images
- **Format**: PNG images + text files
- **Text**: English sentences and words
- **Access**: Open access
- **URL**: http://cvc.uab.es/databases/cvl/

### Implementation Steps

#### Step 1: Data Directory Structure

```
data/
├── iam/
│   ├── words/           (downloaded PNG images)
│   │   ├── a01/
│   │   │   ├── a01-000/
│   │   │   │   ├── a01-000-00.png
│   │   │   │   ├── a01-000-01.png
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── words.txt        (annotations file)
│   └── download_iam.sh  (helper script)
│
├── cvl/
│   ├── images/          (downloaded PNG images)
│   │   ├── trainImages/
│   │   ├── testImages/
│   │   └── validationImages/
│   ├── train.txt        (file list + labels)
│   ├── test.txt
│   └── validation.txt
│
└── vocab.txt            (character vocabulary)
```

#### Step 2: Character Vocabulary

```python
class VocabularyBuilder:
    """
    Build character vocabulary from dataset
    
    Includes:
    - All alphanumeric characters (a-z, A-Z, 0-9)
    - Common punctuation (., comma, !, ?, etc.)
    - CTC blank token (index 0)
    """
    
    def __init__(self):
        # CTC blank always at index 0
        self.chars = ['<blank>']
        self.char_to_idx = {'<blank>': 0}
        self.idx_to_char = {0: '<blank>'}
    
    def build_from_dataset(self, text_labels):
        """
        Args:
            text_labels: List of strings from annotations
        
        Process:
            1. Collect all unique characters
            2. Sort alphabetically
            3. Assign indices (blank is always 0)
        """
        unique_chars = set()
        for text in text_labels:
            unique_chars.update(text)
        
        for idx, char in enumerate(sorted(unique_chars), start=1):
            self.chars.append(char)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'w') as f:
            for char, idx in sorted(self.char_to_idx.items()):
                f.write(f"{idx}\t{char}\n")
    
    def load(self, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            for line in f:
                idx, char = line.strip().split('\t')
                self.chars.append(char)
                self.char_to_idx[char] = int(idx)
                self.idx_to_char[int(idx)] = char
```

#### Step 3: IAM Data Loader

```python
class IAMDataLoader:
    """
    Load IAM Handwriting Database
    
    Format of words.txt:
    status  file_id  part  gray  num_components  bounding_box  grammar_tag  sentence_tag  word_tag  word_value
    ok      a01-000  0     ???   1               408 363 27 48  ???          ???            ???        the
    
    Lines starting with '#' or 'err' should be skipped
    """
    
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Path to IAM dataset root
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        self.images = []
        self.labels = []
        self.image_paths = []
    
    def _parse_words_file(self, words_file):
        """
        Parse words.txt to extract image paths and labels
        
        Returns:
            List of tuples: (image_path, word_text, status)
        """
        data = []
        
        with open(words_file, 'r') as f:
            for line in f:
                # Skip comments and error entries
                if line.startswith('#') or line.startswith('err'):
                    continue
                
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                status = parts[0]
                if status != 'ok':
                    continue
                
                file_id = parts[1]  # e.g., 'a01-000'
                word_text = parts[-1]
                
                # Reconstruct image path
                # From a01-000 → a01/a01-000/a01-000-00.png
                parts_split = file_id.split('-')
                folder1 = parts_split[0] + parts_split[1]  # e.g., 'a01000'
                folder2 = file_id
                
                image_path = os.path.join(
                    self.data_dir,
                    'words',
                    folder1[:3],
                    folder2,
                    f"{file_id}-00.png"
                )
                
                data.append((image_path, word_text, status))
        
        return data
    
    def load(self, words_file):
        """Load all data from words.txt"""
        data = self._parse_words_file(words_file)
        
        # Filter by split
        split_indices = self._get_split_indices(len(data))
        
        for idx in split_indices:
            path, label, status = data[idx]
            
            # Validate file exists
            if not os.path.exists(path):
                print(f"Warning: Image not found: {path}")
                continue
            
            # Validate image is readable
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    print(f"Warning: Cannot read image: {path}")
                    continue
            except Exception as e:
                print(f"Warning: Error reading {path}: {e}")
                continue
            
            self.image_paths.append(path)
            self.labels.append(label)
        
        print(f"Loaded {len(self.images)} samples for split '{self.split}'")
    
    def _get_split_indices(self, total):
        """
        Deterministic split: train 60%, val 20%, test 20%
        Seed=42 for reproducibility
        """
        import random
        random.seed(42)
        
        indices = list(range(total))
        random.shuffle(indices)
        
        train_size = int(0.6 * total)
        val_size = int(0.2 * total)
        
        if self.split == 'train':
            return indices[:train_size]
        elif self.split == 'val':
            return indices[train_size:train_size + val_size]
        elif self.split == 'test':
            return indices[train_size + val_size:]
```

#### Step 4: PyTorch Dataset Class

```python
class HandwritingDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for handwritten text
    
    Returns:
        - image: (1, 32, 128) float32 tensor, normalized to [0, 1]
        - label_indices: LongTensor of character indices
        - text: Original text string (for reference)
    """
    
    def __init__(self, image_paths, labels, vocab, target_height=32, target_width=128):
        """
        Args:
            image_paths: List of image file paths
            labels: List of text labels
            vocab: VocabularyBuilder instance
            target_height: Resize height
            target_width: Resize width
        """
        self.image_paths = image_paths
        self.labels = labels
        self.vocab = vocab
        self.target_height = target_height
        self.target_width = target_width
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (1, 32, 128) float32 tensor
            label_indices: LongTensor of shape (seq_len,)
            text: str (original text)
        """
        image_path = self.image_paths[idx]
        text = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or image.size == 0:
            raise RuntimeError(f"Cannot load image: {image_path}")
        
        # Resize to target dimensions
        image = cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension: (32, 128) → (1, 32, 128)
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image)
        
        # Encode text to indices
        label_indices = torch.LongTensor(self.vocab.encode(text))
        
        # Validation
        assert image_tensor.shape == (1, 32, 128), \
            f"Expected (1, 32, 128), got {image_tensor.shape}"
        assert image_tensor.dtype == torch.float32
        assert image_tensor.min() >= 0 and image_tensor.max() <= 1
        
        return {
            'image': image_tensor,
            'label_indices': label_indices,
            'text': text,
            'image_path': image_path
        }
```

#### Step 5: DataLoader with Collate Function

```python
def collate_sequence_batch(batch):
    """
    Custom collate function for variable-length sequences
    
    Problem: Labels have variable length (3-char word vs 10-char word)
    Solution: Pad to max length in batch, track actual lengths
    
    Returns:
        images: (batch_size, 1, 32, 128)
        label_indices: (batch_size, max_label_length) padded with -1
        label_lengths: (batch_size,) actual lengths for CTC
        texts: List of strings
    """
    images = []
    label_indices = []
    label_lengths = []
    texts = []
    
    for item in batch:
        images.append(item['image'])
        label_indices.append(item['label_indices'])
        label_lengths.append(len(item['label_indices']))
        texts.append(item['text'])
    
    # Stack images (all same size)
    images = torch.stack(images, dim=0)  # (B, 1, 32, 128)
    
    # Pad labels to max length
    max_length = max(label_lengths)
    padded_labels = []
    for labels in label_indices:
        padded = torch.full((max_length,), -1, dtype=torch.long)
        padded[:len(labels)] = labels
        padded_labels.append(padded)
    
    label_indices = torch.stack(padded_labels, dim=0)  # (B, max_length)
    label_lengths = torch.LongTensor(label_lengths)     # (B,)
    
    return {
        'images': images,
        'label_indices': label_indices,
        'label_lengths': label_lengths,
        'texts': texts
    }

# Create PyTorch DataLoader
def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create DataLoader with custom collate function
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequence_batch,
        num_workers=num_workers,
        pin_memory=True  # Speeds up GPU transfer
    )
```

#### Step 6: Data Loading Pipeline

```python
# Initialize
vocab = VocabularyBuilder()
vocab.build_from_dataset(all_labels)  # Pass all labels
vocab.save('data/vocab.txt')

# Create datasets for train/val/test
iam_loader = IAMDataLoader('data/iam')
iam_loader.load('data/iam/words.txt')

train_dataset = HandwritingDataset(
    image_paths=iam_loader.image_paths[:int(0.6*len(iam_loader))],
    labels=iam_loader.labels[:int(0.6*len(iam_loader))],
    vocab=vocab,
    target_height=32,
    target_width=128
)

val_dataset = HandwritingDataset(
    image_paths=iam_loader.image_paths[int(0.6*len(iam_loader)):int(0.8*len(iam_loader))],
    labels=iam_loader.labels[int(0.6*len(iam_loader)):int(0.8*len(iam_loader))],
    vocab=vocab
)

# Create dataloaders
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# Iterate example
for batch in train_loader:
    images = batch['images']        # (32, 1, 32, 128)
    label_indices = batch['label_indices']  # (32, max_length)
    label_lengths = batch['label_lengths']  # (32,)
    texts = batch['texts']          # List of 32 strings
    
    print(f"Images: {images.shape}, dtype: {images.dtype}")
    print(f"Labels: {label_indices.shape}")
    print(f"Lengths: {label_lengths}")
    print(f"Sample texts: {texts[:3]}")
    
    # Forward through model
    outputs = model(images)  # (32, 32, num_classes)
    
    break  # Just show one batch
```

#### Step 7: Validation Checks

```python
def validate_dataloader(dataloader, num_batches=3):
    """
    Verify dataloader output shapes and values
    """
    print("Validating dataloader...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        images = batch['images']
        labels = batch['label_indices']
        lengths = batch['label_lengths']
        
        # Shape checks
        assert images.shape[0] > 0, "Empty batch"
        assert images.shape[1:] == (1, 32, 128), f"Image shape: {images.shape}"
        assert labels.shape[0] == images.shape[0], "Batch size mismatch"
        assert len(lengths) == images.shape[0], "Length count mismatch"
        
        # Value checks
        assert images.dtype == torch.float32, f"Image dtype: {images.dtype}"
        assert images.min() >= 0 and images.max() <= 1, "Image values out of range"
        assert labels.dtype == torch.long, f"Label dtype: {labels.dtype}"
        assert all(l > 0 for l in lengths), f"Zero-length label: {lengths}"
        
        print(f"✓ Batch {batch_idx}: {images.shape}, "
              f"max_label_len={labels.shape[1]}, "
              f"unique_texts={len(set(batch['texts']))}")
    
    print("✓ All validation checks passed!")
```

### Error Prevention Checklist
- [ ] All image paths validated before loading
- [ ] Image files checked for readability (imread returns valid array)
- [ ] Vocabulary includes CTC blank at index 0
- [ ] Image tensors are float32 in [0, 1] range
- [ ] Label padding done with consistent token (-1)
- [ ] Sequence lengths tracked for CTC loss
- [ ] DataLoader tested with multiple batches
- [ ] Variable-length sequences handled correctly
- [ ] Train/val/test splits are deterministic (seeded random)
- [ ] No data leakage between splits

### Dataset Statistics Template
```
Dataset Statistics
==================
Total samples: XXXX
Train samples: XXXX (60%)
Val samples: XXXX (20%)
Test samples: XXXX (20%)

Vocabulary size: XXX characters
Blank token: <blank> (index 0)

Image dimensions: 1 × 32 × 128 (grayscale)
Label lengths: min=X, max=Y, mean=Z

Sample texts:
  - "the"
  - "quick"
  - "brown"
```

---

## Integration Points Between Tasks

### 4.3 → 5.4
- Preprocessed 28×28 images feed into data augmentation pipeline
- Bounding box extraction can be used for word-level cropping

### 5.4 → 5.1
- Dataset loader outputs (32, 128) images for CRNN input
- Vocabulary size determines num_classes in CRNN

### 5.1 → 4.4
- CRNN model is served via FastAPI `/predict` endpoint
- API tests validate CRNN inference latency & accuracy

### 4.4 → Integration Testing
- API tests should use data from 5.4 dataset as test fixtures
- End-to-end test: real image → preprocessing → CRNN → prediction

---

## Implementation Order (Recommended)

1. **Start with 5.4** (Dataset Loader)
   - Why: No dependencies, unblocks everything else
   - Time: 2-3 days
   - Validation: Run dataloader, print batch shapes

2. **Then 4.3** (OpenCV Preprocessing)
   - Why: Uses outputs from 5.4
   - Time: 1-2 days
   - Validation: Compare preprocessed images before/after

3. **Then 5.1** (CRNN Model)
   - Why: Needs fixed input size from 5.4
   - Time: 1 day
   - Validation: Test forward pass with dummy data

4. **Finally 4.4** (API Tests)
   - Why: Tests integrate all above
   - Time: 1-2 days
   - Validation: `pytest tests/test_api.py -v`

---

## Hallucination Prevention Strategies

### For Task 4.3 (Preprocessing)
✓ Explicit shape assertions after every operation
✓ Hardcoded target size (28×28) with no variation
✓ Minimum contour area threshold prevents false detections
✓ Normalize always to [0, 1] with explicit formula
✓ Error messages include actual vs. expected values

### For Task 4.4 (API Tests)
✓ All tests use real test image files, not mocks
✓ Status code validation against specific codes (200, 400, 422)
✓ Response structure validated with concrete keys
✓ Concurrent tests use proper threading primitives
✓ No assumptions about API behavior—only test observable outputs

### For Task 5.1 (CRNN Model)
✓ Explicit forward pass through each component separately
✓ Shape validation at EVERY layer
✓ CNN output hardcoded to (batch, 512, 2, 32)
✓ LSTM input shape explicitly reshaped with assertion
✓ Output shape must exactly match (batch, 32, num_classes)

### For Task 5.4 (Dataset Loader)
✓ IAM path reconstruction shown with exact example
✓ File existence validated before loading
✓ Image readability checked with try-catch
✓ Vocabulary always includes blank at index 0
✓ Padding token explicitly set to -1
✓ Train/val/test split uses seeded random (42)
✓ Collate function shows exact tensor operations

---

## Testing & Validation Checklist

### Before Running Gemini 3 Flash
- [ ] Copy this plan into prompt
- [ ] Ask Gemini to generate code file-by-file
- [ ] For each file, ask: "Does this handle all error cases in the plan?"
- [ ] Cross-reference generated shapes with assertions

### During Implementation
- [ ] Run generated code immediately (don't wait for all tasks)
- [ ] Check shapes match assertions (print them)
- [ ] Verify error messages include actual values
- [ ] Test with smallest possible datasets first

### After Implementation
- [ ] Run all validation checklists above
- [ ] Test with real data (IAM/CVL)
- [ ] Compare preprocessed images visually
- [ ] Profile memory usage (especially dataloader)

---

## Output File Structure

```
project/
├── src/
│   ├── preprocessing.py      (Task 4.3)
│   ├── models/
│   │   └── crnn.py           (Task 5.1)
│   ├── data/
│   │   └── dataset.py        (Task 5.4)
│   └── main.py               (FastAPI app)
│
├── tests/
│   ├── test_api.py           (Task 4.4)
│   ├── conftest.py
│   └── test_data/
│       ├── valid_digit.png
│       ├── invalid.txt
│       └── corrupted.png
│
├── data/
│   ├── iam/
│   ├── cvl/
│   └── vocab.txt
│
└── requirements.txt
```

---

## Success Criteria

| Task | Success Metric |
|------|----------------|
| **4.3** | Preprocessing outputs 28×28 tensors; all values in [0, 1] |
| **4.4** | 80%+ API test coverage; all tests pass locally & in CI |
| **5.1** | CRNN forward pass produces (B, 32, num_classes) shape |
| **5.4** | DataLoader batches verified; train/val/test splits non-overlapping |

---
