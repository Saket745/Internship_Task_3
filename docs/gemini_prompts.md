# Gemini 3 Flash Prompts & Validation Guide
## For Tasks 4.3, 4.4, 5.1, 5.4

---

## TASK 4.3: OpenCV Preprocessing Pipeline

### Prompt Template

```
You are implementing a production-grade OpenCV preprocessing pipeline for handwritten digit/character recognition. 

STRICT REQUIREMENTS:
1. Every operation must include shape assertions
2. Output MUST be exactly (1, 28, 28) float32 array
3. Output values MUST be in range [0.0, 1.0]
4. Error messages MUST show actual vs expected values
5. Return type MUST be list[np.ndarray] (list of arrays, not single array)

SPECIFICATIONS FROM PLAN:
- Input: Variable-size PNG/JPG images
- Step 1: Load & validate (check file exists, readable)
- Step 2: Convert to grayscale
- Step 3: Apply THRESH_BINARY_INV thresholding at value 127
- Step 4: Find contours with cv2.RETR_EXTERNAL and CHAIN_APPROX_SIMPLE
- Step 5: Filter contours with minimum area 50 pixels²
- Step 6: Extract bounding boxes, add 10% padding, resize to 28×28, normalize to [0, 1]

EXACT FUNCTION SIGNATURE:
def preprocess_image(
    image_path: str,
    target_size: int = 28,
    min_contour_area: int = 50,
    padding_ratio: float = 0.1
) -> list[np.ndarray]:

IMPLEMENTATION CHECKLIST:
□ Validate image_path exists → raise FileNotFoundError if not
□ Load with cv2.imread() → raise ValueError if None
□ Check shape is 2D or 3D → raise ValueError otherwise
□ Convert to grayscale with cv2.cvtColor() if needed
□ Apply threshold with ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
□ Assert binary contains only 0 and 255
□ Find contours: contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
□ Filter by area: area = cv2.contourArea(contour), keep if area >= 50
□ For each bbox:
    - Extract ROI: roi = image[y:y+h, x:x+w]
    - Assert roi.size > 0
    - Add padding (10%): Calculate pad_top, pad_bottom, pad_left, pad_right
    - Apply padding with cv2.copyMakeBorder()
    - Resize to 28×28: cv2.resize(padded, (28, 28), cv2.INTER_LINEAR)
    - Normalize: resized / 255.0 → convert to float32
    - Assert shape == (28, 28)
    - Assert values in [0, 1]
    - Add channel dimension: np.expand_dims(normalized, axis=0) → (1, 28, 28)
    - Append to output list

ERROR HANDLING:
- FileNotFoundError: Path doesn't exist
- ValueError: Image unreadable, invalid format
- RuntimeError: Processing error with details

RETURN:
- Empty list if no contours found (with warning log)
- List of (1, 28, 28) float32 arrays

Generate ONLY the function implementation. Include:
1. Docstring with Args, Returns, Raises sections
2. Inline assertions with descriptive messages
3. Comments explaining each step
4. No external dependencies except cv2 and numpy
```

### After Receiving Code

**Validation Steps:**
1. Copy the code into a Python file
2. Create test images: simple hand-drawn digits, MNIST sample, complex background
3. Run:
   ```python
   result = preprocess_image("test_digit.png")
   print(f"Length: {len(result)}, Shape: {result[0].shape if result else 'empty'}")
   print(f"Dtype: {result[0].dtype}, Min: {result[0].min():.4f}, Max: {result[0].max():.4f}")
   assert all(r.shape == (1, 28, 28) for r in result), "Shape mismatch!"
   assert all(r.dtype == np.float32 for r in result), "Dtype mismatch!"
   assert all(0 <= r.min() and r.max() <= 1 for r in result), "Range error!"
   print("✓ All assertions passed!")
   ```

4. **Report back to Gemini if fails:**
   - Actual vs expected shape
   - Actual vs expected dtype
   - Actual min/max values
   - Which assertion failed

---

## TASK 4.4: API Integration Tests

### Prompt Template

```
You are implementing comprehensive integration tests for a FastAPI digit recognition API.

CONTEXT:
- FastAPI app is in main.py
- Model endpoint is POST /predict that accepts 'file' form parameter
- Response is JSON with 'predictions' key (can be list or dict)
- Tests use FastAPI.TestClient (NOT requests library)

STRICT REQUIREMENTS:
1. Tests MUST use real test image files from tests/test_data/, NOT mocks
2. Tests MUST check actual HTTP status codes (200, 400, 422, 413)
3. Tests MUST validate actual response JSON structure
4. Tests MUST include error handling for edge cases
5. No pytest fixtures that don't exist—create conftest.py

SPECIFICATIONS FROM PLAN:
File structure:
  tests/
  ├── conftest.py (pytest fixtures)
  ├── test_api.py (all tests)
  └── test_data/
      ├── valid_digit.png (28×28 valid image)
      ├── invalid.txt (text file, not image)
      ├── corrupted.png (truncated/invalid PNG)

TEST GROUPS (in order):
1. Health Checks (2 tests):
   - GET /health → 200 OK, response has "status" key
   - GET / → 200 OK, response has "api_version" or "message" key

2. Valid Image Upload (3 tests):
   - POST /predict with valid PNG → 200 OK, "predictions" in response
   - POST /predict with valid JPG → 200 OK, "predictions" in response
   - Same image twice → Same predictions (deterministic)

3. Invalid Image Upload (3 tests):
   - POST /predict with .txt file → 400 Bad Request
   - POST /predict with no file → 400 or 422
   - POST /predict with corrupted PNG → 400 Bad Request

4. Response Format (2 tests):
   - Check response JSON structure (predictions, confidence, class, timestamp)
   - Check response Content-Type is application/json

5. Performance (2 tests):
   - Upload image > 10MB → 200 or 413 (Payload Too Large)
   - Send 10 concurrent requests → All 200 OK (no race conditions)

EXACT IMPORTS:
import pytest
from fastapi.testclient import TestClient
from main import app

FIXTURE REQUIREMENTS in conftest.py:
@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image_path():
    return "tests/test_data/valid_digit.png"

TEST FILE STRUCTURE in test_api.py:
def test_api_health_check(client):
    # Your code
    
def test_upload_valid_png(client, sample_image_path):
    # Your code
    
# ... continue for all 12 tests

RESPONSE VALIDATION:
- Check status_code explicitly: assert response.status_code == 200
- Parse JSON: data = response.json()
- Validate keys: assert "predictions" in data
- Validate types: assert isinstance(data["predictions"], (list, dict))

ERROR CASES:
- Missing file → 422 Unprocessable Entity
- Wrong file type → 400 Bad Request
- Corrupted image → 400 Bad Request
- Payload too large → 413 Payload Too Large

Generate ONLY the test code split into TWO files:
1. conftest.py (with fixtures)
2. test_api.py (with all 12+ tests)

Include:
- Full test function implementations
- Docstrings explaining what each test validates
- Assertions with clear failure messages
- Comments for complex logic
```

### After Receiving Code

**Validation Steps:**
1. Create test images directory:
   ```bash
   mkdir -p tests/test_data
   # Create valid_digit.png (28×28 grayscale PNG of digit)
   # Create invalid.txt (text file)
   # Create corrupted.png (truncated PNG file)
   ```

2. Place conftest.py and test_api.py in tests/ directory

3. Run tests:
   ```bash
   cd project_root
   pytest tests/test_api.py -v
   ```

4. **Expected output:**
   ```
   test_api.py::test_api_health_check PASSED
   test_api.py::test_upload_valid_png PASSED
   test_api.py::test_upload_valid_jpg PASSED
   test_api.py::test_predictions_are_deterministic PASSED
   test_api.py::test_upload_non_image_file PASSED
   test_api.py::test_upload_no_file PASSED
   test_api.py::test_response_json_structure PASSED
   
   ============ 7 passed in 0.45s ==============
   ```

5. **If tests fail, report to Gemini:**
   - Which test failed
   - Actual vs expected status code
   - Actual vs expected response structure
   - Exact error message

---

## TASK 5.1: CRNN Model

### Prompt Template

```
You are implementing a CRNN (CNN + BiLSTM + Dense) model for sequence prediction.

STRICT SPECIFICATIONS:
Input shape: (batch_size, 1, 32, 128) — grayscale images
Output shape: (batch_size, 32, num_classes) — sequence logits
No softmax in model (applied in CTC loss)

ARCHITECTURE BREAKDOWN:

CNN Feature Extractor:
- Conv1: 1→32 channels, kernel=3×3, padding=1, then ReLU, BN, MaxPool(2,2) → (B,32,16,64)
- Conv2: 32→64 channels, kernel=3×3, padding=1, then ReLU, BN, MaxPool(2,2) → (B,64,8,32)
- Conv3: 64→128 channels, kernel=3×3, padding=1, then ReLU, BN, MaxPool((2,1)) → (B,128,4,32)
- Conv4: 128→256 channels, kernel=3×3, padding=1, then ReLU, BN, MaxPool((2,1)) → (B,256,2,32)
- Conv5: 256→512 channels, kernel=3×3, padding=1, then ReLU, BN → (B,512,2,32)

BiLSTM Sequence Encoder:
- Input: (B,512,2,32) reshape to (B,32,1024) [collapse height: 2*512=1024, width=32 is seq length]
- BiLSTM: input_size=1024, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True
- Output: (B,32,512) [256*2 for bidirectional]

Classification Head:
- FC1: 512→256, ReLU, Dropout(0.5)
- FC2: 256→num_classes
- Output: (B,32,num_classes)

CLASS STRUCTURE:

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        # 5 Conv blocks with BN and MaxPool
    
    def forward(self, x):
        # x: (B, 1, 32, 128)
        # return: (B, 512, 2, 32)
        # ASSERT shape before each return

class BiLSTMSequenceEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        # BiLSTM with input_size=1024
    
    def forward(self, features):
        # features: (B, 512, 2, 32)
        # Reshape to (B, 32, 1024) [permute then view]
        # ASSERT reshaped shape is (B, 32, 1024)
        # Run LSTM
        # ASSERT output shape is (B, 32, 512)
        # return lstm_out

class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        # 2 FC layers with ReLU, BN, Dropout
    
    def forward(self, lstm_out):
        # lstm_out: (B, 32, 512)
        # return: (B, 32, num_classes)

class CRNNModel(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = BiLSTMSequenceEncoder(hidden_size=256, num_layers=2)
        self.head = ClassificationHead(num_classes)
    
    def forward(self, x):
        # x: (B, 1, 32, 128)
        features = self.cnn(x)           # (B, 512, 2, 32)
        lstm_out = self.lstm(features)   # (B, 32, 512)
        logits = self.head(lstm_out)     # (B, 32, num_classes)
        return logits

ASSERTION REQUIREMENTS:
- After CNN: assert features.shape == (batch, 512, 2, 32)
- After reshape in BiLSTM: assert reshaped.shape == (batch, 32, 1024)
- After LSTM: assert lstm_out.shape == (batch, 32, 512)
- Final output: assert logits.shape == (batch, 32, num_classes)

KEY CONSTRAINTS:
✓ NO ImageNet pretraining
✓ Batch normalization AFTER conv, BEFORE activation
✓ Dropout only in dense layers (not LSTM)
✓ LSTM input: (batch, seq_len, features) NOT (batch, features, seq_len)
✓ No softmax in forward pass
✓ Must work on CPU and GPU

Generate ONLY the model code (can be in one file or multiple classes).

Include:
1. All 4 classes with complete __init__ and forward methods
2. Detailed comments explaining shapes
3. Assertions at every major point
4. Proper initialization of weights
5. Module docstrings

After implementation, test with:
dummy_input = torch.randn(4, 1, 32, 128)
model = CRNNModel(num_classes=37)
output = model(dummy_input)
assert output.shape == (4, 32, 37), f"Got {output.shape}"
print("✓ Model forward pass successful!")
```

### After Receiving Code

**Validation Steps:**
1. Save code to src/models/crnn.py

2. Test forward pass:
   ```python
   import torch
   from src.models.crnn import CRNNModel
   
   model = CRNNModel(num_classes=37)
   dummy_input = torch.randn(4, 1, 32, 128)
   output = model(dummy_input)
   
   print(f"Input shape: {dummy_input.shape}")
   print(f"Output shape: {output.shape}")
   print(f"Output dtype: {output.dtype}")
   
   assert output.shape == (4, 32, 37), f"Expected (4, 32, 37), got {output.shape}"
   assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
   print("✓ Forward pass successful!")
   ```

3. **Test memory usage:**
   ```python
   import torch
   
   model = CRNNModel(num_classes=37).cuda()
   batch_input = torch.randn(32, 1, 32, 128).cuda()
   output = model(batch_input)
   
   total_params = sum(p.numel() for p in model.parameters())
   print(f"Total parameters: {total_params:,}")
   print(f"Output shape: {output.shape}")
   ```

4. **If forward pass fails, report:**
   - Actual output shape
   - Which assertion failed
   - Which module (CNN, LSTM, Head)
   - Error traceback

---

## TASK 5.4: IAM/CVL Dataset Loader

### Prompt Template

```
You are implementing a production-grade dataset loader for handwritten text recognition.

DATASET CONTEXT:
- IAM Handwriting Database: 115k word images
- CVL Database: 20k word images
- Both contain PNG images + text annotations

STRICT SPECIFICATIONS:

1. VOCABULARY BUILDER:
   - CTC blank token ALWAYS at index 0
   - Unique characters extracted from dataset
   - Deterministic ordering (alphabetical)
   - Can save/load to file

2. IAM DATA LOADER:
   - Parse words.txt annotation file
   - Skip lines starting with '#' or status='err'
   - Extract: file_id, word_text, status
   - Validate image file exists
   - Validate image is readable (cv2.imread() returns non-None)
   - Implement train/val/test split (60/20/20)
   - Use seed=42 for reproducible split

3. PYTORCH DATASET CLASS:
   - Input: image_path, label_text, vocab, target_height=32, target_width=128
   - Output per item:
     {
       'image': (1, 32, 128) float32 tensor in [0, 1],
       'label_indices': LongTensor of encoded characters,
       'text': original text string,
       'image_path': file path
     }
   - Validate image shape and dtype
   - Validate label_indices type

4. COLLATE FUNCTION:
   - Handle variable-length labels
   - Pad to max length in batch with value -1
   - Stack images (all same size)
   - Track actual label lengths for CTC loss
   - Return dict with keys: images, label_indices, label_lengths, texts

5. DATALOADER CREATION:
   - Use torch.utils.data.DataLoader
   - Custom collate_fn=collate_sequence_batch
   - pin_memory=True for GPU
   - num_workers can be 0 (single process safe)

CLASS STRUCTURE:

class VocabularyBuilder:
    def __init__(self):
        # Initialize with blank token
    
    def build_from_dataset(self, text_labels):
        # Extract unique chars, sort, assign indices
    
    def encode(self, text: str) -> list[int]:
        # Convert text to character indices
    
    def decode(self, indices: list[int]) -> str:
        # Convert indices back to text
    
    def save(self, filepath: str):
        # Save vocab to file: idx\tchar\n
    
    def load(self, filepath: str):
        # Load vocab from file

class IAMDataLoader:
    def __init__(self, data_dir: str, split: str = 'train'):
        # split: 'train', 'val', or 'test'
    
    def _parse_words_file(self, words_file: str) -> list[tuple]:
        # Parse words.txt, return list of (image_path, word_text, status)
        # Skip lines with '#' or status != 'ok'
    
    def load(self, words_file: str):
        # Load all data, filter by split
        # Validate each image path exists
        # Try cv2.imread, skip if fails
        # Store image_paths and labels

class HandwritingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, labels: list, vocab: VocabularyBuilder, 
                 target_height: int = 32, target_width: int = 128):
        # Store paths, labels, vocab
    
    def __len__(self):
        # Return dataset size
    
    def __getitem__(self, idx: int) -> dict:
        # Load image from image_paths[idx]
        # Resize to target_height × target_width with cv2.resize
        # Normalize to [0, 1]: image / 255.0
        # Add channel: (32, 128) → (1, 32, 128)
        # Convert to float32 tensor
        # Encode label with vocab
        # Return dict with keys: image, label_indices, text, image_path
        # ASSERT image.shape == (1, 32, 128)
        # ASSERT image.dtype == torch.float32
        # ASSERT 0 <= image.min() and image.max() <= 1

def collate_sequence_batch(batch: list[dict]) -> dict:
    # Input: list of dicts from __getitem__
    # Stack images: (B, 1, 32, 128)
    # Pad labels to max length with -1
    # Track actual lengths
    # Return dict with keys: images, label_indices, label_lengths, texts

def create_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    # Create DataLoader with collate_sequence_batch
    # Return loader

VALIDATION FUNCTION:
def validate_dataloader(dataloader, num_batches: int = 3):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        images = batch['images']
        labels = batch['label_indices']
        lengths = batch['label_lengths']
        
        # Shape checks
        assert images.shape[0] > 0, "Empty batch"
        assert images.shape[1:] == (1, 32, 128)
        assert labels.shape[0] == images.shape[0]
        assert len(lengths) == images.shape[0]
        
        # Value checks
        assert images.dtype == torch.float32
        assert images.min() >= 0 and images.max() <= 1
        assert labels.dtype == torch.long
        assert all(l > 0 for l in lengths)

REQUIRED DATA DIRECTORY STRUCTURE:
data/
├── iam/
│   ├── words/
│   │   ├── a01/
│   │   │   ├── a01-000/
│   │   │   │   ├── a01-000-00.png
│   │   │   │   └── ...
│   │   └── ...
│   └── words.txt
├── cvl/
│   ├── images/
│   │   ├── trainImages/
│   │   ├── testImages/
│   │   └── validationImages/
│   ├── train.txt
│   └── test.txt
└── vocab.txt

INTEGRATION TEST:
# After implementation:
vocab = VocabularyBuilder()
loader = IAMDataLoader('data/iam')
loader.load('data/iam/words.txt')
dataset = HandwritingDataset(loader.image_paths, loader.labels, vocab)
dataloader = create_dataloader(dataset, batch_size=32)
validate_dataloader(dataloader)
print("✓ Dataset loader validated!")

Generate ONLY the dataset loader code.

Include:
1. All 5 classes/functions with complete implementations
2. Detailed docstrings
3. Error handling with try-catch for file I/O
4. Shape and dtype assertions
5. Comments for complex logic (especially collate function)
```

### After Receiving Code

**Validation Steps:**
1. Save code to src/data/dataset.py

2. Create test data structure:
   ```bash
   mkdir -p data/{iam/words/a01,cvl/images}
   # Copy or symlink actual IAM data here
   # Or create dummy test images
   ```

3. Test vocabulary:
   ```python
   from src.data.dataset import VocabularyBuilder
   
   vocab = VocabularyBuilder()
   test_texts = ["hello", "world", "123"]
   vocab.build_from_dataset(test_texts)
   
   # Test encode/decode
   encoded = vocab.encode("hello")
   decoded = vocab.decode(encoded)
   print(f"Original: hello, Encoded: {encoded}, Decoded: {decoded}")
   
   assert vocab.char_to_idx['<blank>'] == 0, "Blank token must be at index 0"
   print("✓ Vocabulary works!")
   ```

4. Test dataset loading:
   ```python
   from src.data.dataset import HandwritingDataset, IAMDataLoader, create_dataloader
   import torch
   
   # Load IAM data
   loader = IAMDataLoader('data/iam', split='train')
   loader.load('data/iam/words.txt')
   
   # Create dataset
   dataset = HandwritingDataset(
       loader.image_paths[:100],  # Use first 100
       loader.labels[:100],
       vocab,
       target_height=32,
       target_width=128
   )
   
   print(f"Dataset size: {len(dataset)}")
   
   # Test single item
   item = dataset[0]
   print(f"Image shape: {item['image'].shape}, dtype: {item['image'].dtype}")
   print(f"Label shape: {item['label_indices'].shape}")
   print(f"Text: {item['text']}")
   
   assert item['image'].shape == (1, 32, 128)
   assert item['image'].dtype == torch.float32
   assert 0 <= item['image'].min() and item['image'].max() <= 1
   print("✓ Single item works!")
   ```

5. Test dataloader with collate:
   ```python
   dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)
   
   batch = next(iter(dataloader))
   print(f"Batch images shape: {batch['images'].shape}")
   print(f"Batch labels shape: {batch['label_indices'].shape}")
   print(f"Batch lengths: {batch['label_lengths']}")
   print(f"Batch texts: {batch['texts'][:3]}")
   
   assert batch['images'].shape[0] == 16, "Batch size mismatch"
   assert batch['images'].shape[1:] == (1, 32, 128)
   assert len(batch['label_lengths']) == 16
   print("✓ Dataloader works!")
   ```

6. **If dataset loading fails, report:**
   - Missing files
   - Shape mismatches
   - Vocabulary encoding errors
   - Dataloader collation errors

---

## Gemini Prompt Checklist

Before sending each prompt:

- [ ] Copy **ENTIRE** specification section from plan
- [ ] Include function signatures EXACTLY as shown
- [ ] List all validation/assertion requirements
- [ ] Specify error handling cases
- [ ] Provide example input/output shapes
- [ ] Ask only for function implementations, not explanations
- [ ] Request complete docstrings

After receiving code:

- [ ] Paste into file immediately
- [ ] Run validation steps
- [ ] Note any failures with actual vs expected values
- [ ] Report back to Gemini with exact error messages (NOT interpretations)
- [ ] Ask Gemini to fix specific failed assertions

---

## Order of Implementation

```
DAY 1: Task 5.4 (Dataset Loader)
  - No dependencies on other tasks
  - Longest to implement, but unblocks everything
  - Test with dummy/small data first

DAY 2: Task 4.3 (Preprocessing)
  - Depends: None (but can use output in later tasks)
  - Short to implement
  - Test with various image types

DAY 3: Task 5.1 (CRNN Model)
  - Depends: Task 5.4 (for input shape info)
  - Medium to implement
  - Test with dummy data before real data

DAY 4: Task 4.4 (API Tests)
  - Depends: Tasks 4.3, 5.1 (for API to exist)
  - Medium to implement
  - Test with FastAPI development server running
```

---

## Common Hallucinations to Watch For

### Task 4.3 (Preprocessing)
❌ **Hallucination**: Returns single 28×28 array instead of list
✓ **Fix**: Explicitly ask for `return list[np.ndarray]`

❌ **Hallucination**: Normalize to wrong range (e.g., [-1, 1])
✓ **Fix**: Specify formula: `normalized = image / 255.0`

❌ **Hallucination**: Skips shape assertions
✓ **Fix**: Ask for assertions after EVERY operation

### Task 4.4 (Tests)
❌ **Hallucination**: Uses `requests` library instead of TestClient
✓ **Fix**: Explicitly require `from fastapi.testclient import TestClient`

❌ **Hallucination**: Mocks the image upload
✓ **Fix**: Specify "use real test image files from tests/test_data/"

❌ **Hallucination**: Assumes specific API response format
✓ **Fix**: Ask to check for "predictions" key, not assuming structure

### Task 5.1 (CRNN)
❌ **Hallucination**: CNN output shape is wrong
✓ **Fix**: Explicitly specify output after each block: (B, C, H, W)

❌ **Hallucination**: LSTM input shape is (B, C, H, W)
✓ **Fix**: Show reshape example: permute(0, 3, 1, 2).view(B, W, C*H)

❌ **Hallucination**: Output includes softmax
✓ **Fix**: Specify "No softmax in model (applied in CTC loss)"

### Task 5.4 (Dataset)
❌ **Hallucination**: Variable-length padding done wrong
✓ **Fix**: Show exact collate code structure

❌ **Hallucination**: Vocabulary doesn't include blank at index 0
✓ **Fix**: Specify "CTC blank token ALWAYS at index 0"

❌ **Hallucination**: IAM path reconstruction incorrect
✓ **Fix**: Show exact path format with example

---

## Success Indicators

✓ Code runs without errors on first try
✓ All assertions pass
✓ Output shapes match specifications exactly
✓ No warnings or deprecation messages
✓ Works on both CPU and GPU (if applicable)
✓ Handles edge cases (empty inputs, corrupted files, etc.)

---
