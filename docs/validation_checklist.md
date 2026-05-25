# Implementation Validation Checklist
## Quick Reference for Each Task

---

## TASK 4.3: OpenCV Preprocessing Pipeline

### Code Validation Checklist
- [ ] Function signature: `def preprocess_image(image_path: str, target_size: int = 28, ...) -> list[np.ndarray]`
- [ ] Returns list of arrays, NOT single array
- [ ] File existence check raises `FileNotFoundError`
- [ ] Image readability check raises `ValueError`
- [ ] Output shape: `(1, 28, 28)`
- [ ] Output dtype: `float32`
- [ ] Output values: ALL in range `[0.0, 1.0]`

### Runtime Validation (After Implementation)
```python
from src.preprocessing import preprocess_image
import numpy as np

# Test 1: Valid single-digit image
result = preprocess_image("tests/test_data/digit_3.png")
assert isinstance(result, list), f"Expected list, got {type(result)}"
assert len(result) > 0, "Expected non-empty result"
assert result[0].shape == (1, 28, 28), f"Shape mismatch: {result[0].shape}"
assert result[0].dtype == np.float32, f"Dtype mismatch: {result[0].dtype}"
assert result[0].min() >= 0.0 and result[0].max() <= 1.0, \
    f"Range error: [{result[0].min()}, {result[0].max()}]"

# Test 2: Multiple characters
result = preprocess_image("tests/test_data/text_hello.png")
assert len(result) >= 4, f"Expected >=4 characters, got {len(result)}"
for arr in result:
    assert arr.shape == (1, 28, 28)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0

# Test 3: Error handling
try:
    preprocess_image("nonexistent.png")
    assert False, "Should raise FileNotFoundError"
except FileNotFoundError:
    pass

# Test 4: Invalid format
try:
    preprocess_image("tests/test_data/invalid.txt")
    assert False, "Should raise ValueError"
except ValueError:
    pass

print("✓ ALL TESTS PASSED")
```

### Common Issues & Fixes
| Issue | Expected | Actual | Fix |
|-------|----------|--------|-----|
| Returns single array | `list[np.ndarray]` | `np.ndarray` | Wrap in list: `return [output_array]` |
| Wrong dtype | `float32` | `uint8` or `float64` | Add `.astype(np.float32)` |
| Wrong range | `[0, 1]` | `[0, 255]` | Divide by 255: `normalized / 255.0` |
| Wrong shape | `(1, 28, 28)` | `(28, 28)` | Add dimension: `np.expand_dims(arr, axis=0)` |
| No error checking | FileNotFoundError | No error | Add: `if not os.path.exists(path): raise FileNotFoundError()` |

---

## TASK 4.4: API Integration Tests

### Code Validation Checklist
- [ ] File 1: `tests/conftest.py` with `client` fixture
- [ ] File 2: `tests/test_api.py` with 12+ test functions
- [ ] Test data files exist: `tests/test_data/{valid_digit.png, invalid.txt, corrupted.png}`
- [ ] Uses `TestClient` from `fastapi.testclient`
- [ ] Tests use real image files (NOT mocks)
- [ ] All status codes are actual (200, 400, 422, 413)
- [ ] Response validation checks actual JSON keys

### Runtime Validation (After Implementation)
```bash
# Run all tests
pytest tests/test_api.py -v

# Expected output:
# test_api.py::test_api_health_check PASSED
# test_api.py::test_upload_valid_png PASSED
# test_api.py::test_predictions_are_deterministic PASSED
# ...
# ============ 12 passed in 1.23s ==============

# Run with coverage
pytest tests/test_api.py --cov=main --cov-report=term-missing
# Should show 80%+ coverage
```

### Test Structure Validation
```python
# Check conftest.py
from tests.conftest import client, sample_image_path
# Should not raise ImportError

# Check client fixture
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
response = client.get("/health")
print(f"Status: {response.status_code}")  # Should be 200
```

### Common Issues & Fixes
| Issue | Expected | Actual | Fix |
|-------|----------|--------|-----|
| Uses `requests` lib | `TestClient` | requests.post() | Change to `client.post()` |
| Mocks images | Real files | Mock objects | Load: `with open(path, 'rb') as f: ...` |
| Wrong status code check | 200, 400, 422 | Assumes 200 always | Add explicit: `assert response.status_code == 200` |
| Missing JSON parsing | `response.json()` | `response.text` | Change to `data = response.json()` |
| Missing error tests | Tests both valid & invalid | Only valid cases | Add tests for 400, 422 cases |

---

## TASK 5.1: CRNN Model

### Code Validation Checklist
- [ ] Class 1: `CNNFeatureExtractor(nn.Module)`
  - [ ] Output shape: `(batch, 512, 2, 32)`
  - [ ] All conv blocks have BN
  - [ ] MaxPool layers reduce spatial dims correctly

- [ ] Class 2: `BiLSTMSequenceEncoder(nn.Module)`
  - [ ] Input shape: `(batch, 512, 2, 32)`
  - [ ] Reshape to: `(batch, 32, 1024)`
  - [ ] Output shape: `(batch, 32, 512)` (bidirectional 256*2)
  - [ ] All assertions present

- [ ] Class 3: `ClassificationHead(nn.Module)`
  - [ ] Input: `(batch, 32, 512)`
  - [ ] Output: `(batch, 32, num_classes)`
  - [ ] NO softmax
  - [ ] Dropout only in dense layers

- [ ] Class 4: `CRNNModel(nn.Module)`
  - [ ] Combines all 3 components
  - [ ] Forward pass returns logits

### Runtime Validation (After Implementation)
```python
import torch
from src.models.crnn import CRNNModel

# Create model
model = CRNNModel(num_classes=37)

# Test forward pass
batch_size = 4
dummy_input = torch.randn(batch_size, 1, 32, 128)
output = model(dummy_input)

# Validate output
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Output min: {output.min():.4f}, max: {output.max():.4f}")

assert output.shape == (4, 32, 37), f"Shape mismatch: {output.shape}"
assert output.dtype == torch.float32, f"Dtype: {output.dtype}"

# Test with different batch size
different_batch = torch.randn(8, 1, 32, 128)
different_output = model(different_batch)
assert different_output.shape == (8, 32, 37)

# Test on GPU (if available)
if torch.cuda.is_available():
    model = model.cuda()
    gpu_input = dummy_input.cuda()
    gpu_output = model(gpu_input)
    assert gpu_output.shape == (batch_size, 32, 37)
    print("✓ GPU mode works")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("✓ ALL TESTS PASSED")
```

### Shape Tracking Validation
```python
# Track shapes through each component
batch = 4
x = torch.randn(batch, 1, 32, 128)
print(f"Input: {x.shape}")

# CNN
cnn = model.cnn
features = cnn(x)
assert features.shape == (batch, 512, 2, 32), f"CNN output: {features.shape}"
print(f"CNN output: {features.shape}")

# BiLSTM
lstm = model.lstm
lstm_out = lstm(features)
assert lstm_out.shape == (batch, 32, 512), f"LSTM output: {lstm_out.shape}"
print(f"LSTM output: {lstm_out.shape}")

# Head
head = model.head
logits = head(lstm_out)
assert logits.shape == (batch, 32, 37), f"Head output: {logits.shape}"
print(f"Head output: {logits.shape}")

print("✓ Shape tracking validation passed")
```

### Common Issues & Fixes
| Issue | Expected | Actual | Fix |
|-------|----------|--------|-----|
| CNN output shape wrong | (B, 512, 2, 32) | (B, 256, 4, 32) | Check MaxPool layers, verify final conv |
| LSTM reshape wrong | (B, 32, 1024) | (B, 1024, 32) | Use: `permute(0, 3, 1, 2).view(B, 32, -1)` |
| LSTM output shape | (B, 32, 512) | (B, 32, 256) | Check bidirectional=True, hidden_size=256 |
| Output has softmax | Logits (no softmax) | Probabilities [0,1] | Remove `F.softmax()` from head |
| Assertions missing | Multiple asserts | No asserts | Add after each major operation |

---

## TASK 5.4: Dataset Loader

### Code Validation Checklist
- [ ] Class 1: `VocabularyBuilder`
  - [ ] Blank token at index 0
  - [ ] `encode(text)` returns list of ints
  - [ ] `decode(indices)` returns string
  - [ ] `save()` and `load()` work

- [ ] Class 2: `IAMDataLoader`
  - [ ] `_parse_words_file()` correctly parses
  - [ ] Skips '#' and 'err' lines
  - [ ] Validates image files exist
  - [ ] Validates image is readable
  - [ ] Implements train/val/test split (60/20/20)
  - [ ] Uses seed=42 for reproducibility

- [ ] Class 3: `HandwritingDataset(torch.utils.data.Dataset)`
  - [ ] `__getitem__` returns dict with keys: image, label_indices, text, image_path
  - [ ] Image shape: `(1, 32, 128)`
  - [ ] Image dtype: `float32`
  - [ ] Image range: `[0, 1]`
  - [ ] Label indices: `LongTensor`

- [ ] Function 4: `collate_sequence_batch(batch)`
  - [ ] Handles variable-length sequences
  - [ ] Pads with -1
  - [ ] Returns dict with keys: images, label_indices, label_lengths, texts
  - [ ] Images shape: `(batch, 1, 32, 128)`
  - [ ] Labels shape: `(batch, max_length)`

- [ ] Function 5: `create_dataloader()`
  - [ ] Uses `collate_fn=collate_sequence_batch`
  - [ ] Accepts batch_size, shuffle, num_workers
  - [ ] Uses `pin_memory=True`

### Runtime Validation (After Implementation)
```python
from src.data.dataset import (
    VocabularyBuilder,
    IAMDataLoader,
    HandwritingDataset,
    create_dataloader,
    collate_sequence_batch
)
import torch

# Test 1: Vocabulary
print("Testing Vocabulary...")
vocab = VocabularyBuilder()
texts = ["hello", "world", "123", "test!"]
vocab.build_from_dataset(texts)

assert vocab.char_to_idx['<blank>'] == 0, "Blank must be at index 0"
assert vocab.decode(vocab.encode("hello")) == "hello"
assert len(vocab.chars) > 10, f"Too few chars: {len(vocab.chars)}"
print("✓ Vocabulary OK")

# Test 2: IAM Data Loader
print("Testing IAM Data Loader...")
iam = IAMDataLoader('data/iam', split='train')
# Don't actually call load() unless you have data
# iam.load('data/iam/words.txt')
print("✓ IAM Loader initialized")

# Test 3: Dataset
print("Testing HandwritingDataset...")
# Use dummy data for testing
dummy_paths = ['tests/test_data/valid_digit.png'] * 10
dummy_labels = ['5', '3', '8', '1', '9', '2', '4', '7', '6', '0']

try:
    dataset = HandwritingDataset(
        image_paths=dummy_paths,
        labels=dummy_labels,
        vocab=vocab,
        target_height=32,
        target_width=128
    )
    
    assert len(dataset) == 10
    item = dataset[0]
    
    assert 'image' in item
    assert 'label_indices' in item
    assert 'text' in item
    assert 'image_path' in item
    
    assert item['image'].shape == (1, 32, 128)
    assert item['image'].dtype == torch.float32
    assert 0 <= item['image'].min() and item['image'].max() <= 1
    assert item['label_indices'].dtype == torch.long
    assert item['text'] in dummy_labels
    
    print("✓ Dataset OK")
except Exception as e:
    print(f"Dataset error: {e}")

# Test 4: Collate Function
print("Testing Collate Function...")
sample_batch = [item for item in dataset][:4]  # 4 items
collated = collate_sequence_batch(sample_batch)

assert 'images' in collated
assert 'label_indices' in collated
assert 'label_lengths' in collated
assert 'texts' in collated

assert collated['images'].shape == (4, 1, 32, 128)
assert collated['label_indices'].shape[0] == 4
assert len(collated['label_lengths']) == 4
assert len(collated['texts']) == 4

assert collated['images'].dtype == torch.float32
assert collated['label_indices'].dtype == torch.long
assert collated['label_lengths'].dtype == torch.long

print("✓ Collate function OK")

# Test 5: DataLoader
print("Testing DataLoader...")
dataloader = create_dataloader(dataset, batch_size=4, shuffle=True)

for batch_idx, batch in enumerate(dataloader):
    if batch_idx == 0:
        print(f"Batch 0 shapes:")
        print(f"  images: {batch['images'].shape}")
        print(f"  label_indices: {batch['label_indices'].shape}")
        print(f"  label_lengths: {batch['label_lengths']}")
        print(f"  texts: {batch['texts']}")
        
        assert batch['images'].shape[0] <= 4
        assert batch['images'].shape[1:] == (1, 32, 128)
        assert batch['label_indices'].shape[0] == batch['images'].shape[0]
        assert len(batch['label_lengths']) == batch['images'].shape[0]
    break

print("✓ DataLoader OK")

print("\n✓ ALL DATASET TESTS PASSED")
```

### Batch Structure Validation
```python
# After getting first batch from dataloader:
batch = next(iter(dataloader))

print("Batch structure:")
print(f"  images: {batch['images'].shape} {batch['images'].dtype}")
print(f"  label_indices: {batch['label_indices'].shape} {batch['label_indices'].dtype}")
print(f"  label_lengths: {batch['label_lengths']} {batch['label_lengths'].dtype}")
print(f"  texts: {type(batch['texts'])} length {len(batch['texts'])}")

# Validate padding
max_label_len = batch['label_indices'].shape[1]
for length in batch['label_lengths']:
    assert length <= max_label_len

# Validate padding values
for row in batch['label_indices']:
    padded_part = row[batch['label_lengths'][0]:]
    assert all(v == -1 for v in padded_part if v != -1)

print("✓ Batch structure valid")
```

### Common Issues & Fixes
| Issue | Expected | Actual | Fix |
|-------|----------|--------|-----|
| Blank not at 0 | `char_to_idx['<blank>'] == 0` | `char_to_idx['<blank>'] == 1` | Initialize: `self.chars = ['<blank>']`, `self.char_to_idx = {'<blank>': 0}` |
| Image dtype wrong | `float32` | `uint8` | Add: `.astype(np.float32)` |
| Image not normalized | `[0, 1]` | `[0, 255]` | Divide: `image / 255.0` |
| Image missing channel | `(1, 32, 128)` | `(32, 128)` | Expand: `np.expand_dims(img, axis=0)` |
| Variable-length handling | Padded with -1 | Error on different lengths | Use collate_fn to pad |
| Label dtype | `LongTensor` | `FloatTensor` | Use: `torch.LongTensor()` |

---

## Quick Test Template

Copy this to quickly validate any task:

```python
# Task 4.3: Preprocessing
from src.preprocessing import preprocess_image
result = preprocess_image("test.png")
assert isinstance(result, list) and len(result) > 0
assert result[0].shape == (1, 28, 28) and result[0].dtype.name == 'float32'

# Task 4.4: API Tests
import subprocess
result = subprocess.run(["pytest", "tests/test_api.py", "-v"], capture_output=True)
assert result.returncode == 0, f"Tests failed:\n{result.stdout.decode()}"

# Task 5.1: CRNN Model
import torch
from src.models.crnn import CRNNModel
model = CRNNModel(num_classes=37)
output = model(torch.randn(4, 1, 32, 128))
assert output.shape == (4, 32, 37)

# Task 5.4: Dataset
from src.data.dataset import *
vocab = VocabularyBuilder()
vocab.build_from_dataset(["test"])
dataset = HandwritingDataset(["test.png"], ["t"], vocab)
loader = create_dataloader(dataset, batch_size=1)
batch = next(iter(loader))
assert batch['images'].shape == (1, 1, 32, 128)

print("✓ ALL QUICK TESTS PASSED")
```

---

## Decision Tree: What to Check First

```
Does the code run without errors?
├─ NO → Check imports, syntax, dependencies
└─ YES → Continue

Do shapes match specifications?
├─ NO → Check tensor/array operations
└─ YES → Continue

Are values in expected ranges?
├─ NO → Check normalization, casting
└─ YES → Continue

Do all assertions pass?
├─ NO → Check shape assertions, dtype assertions
└─ YES → ✓ TASK COMPLETE
```

---
