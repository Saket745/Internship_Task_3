# Master Implementation Specification: Handwritten Character Recognition

## 1. Project Overview
An end-to-end machine learning system designed to recognize handwritten characters from the EMNIST dataset. The project integrates a PyTorch-based CNN, ONNX optimization, a FastAPI backend, and an interactive React frontend.

## 2. Technical Architecture

### 2.1 Machine Learning Pipeline (PyTorch)
- **Dataset**: EMNIST (ByClass split for digits and letters).
- **Architecture**: Custom CNN with:
    - 2x Convolutional Layers (3x3 kernels, ReLU).
    - 2x Max Pooling Layers (2x2).
    - Batch Normalization & Dropout (0.25).
    - Fully Connected layers leading to output classes (62 classes for ByClass).
- **Optimization**: Export to ONNX format using `torch.onnx.export`.

### 2.2 Backend API (FastAPI)
- **Inference Engine**: `onnxruntime` for high-speed CPU inference.
- **Preprocessing**: 
    - Convert Base64 string to Grayscale.
    - Invert colors (if necessary to match EMNIST format).
    - Resize to 28x28 pixels.
    - Normalize pixel values to [0, 1].
- **Endpoints**:
    - `POST /predict`: Receives `{ "image": "base64_string" }`, returns `{ "prediction": "A", "confidence": 0.98 }`.

### 2.3 Frontend (React)
- **Component**: Interactive HTML5 Canvas using `react-canvas-draw` or native API.
- **Workflow**: 
    1. User draws character.
    2. "Predict" button captures canvas data.
    3. JSON POST request to FastAPI.
    4. Display result with animation.

## 3. SDLC & MLOps Workflow
- **Experiment Tracking**: MLflow logs every training run.
- **Containerization**: Docker multi-stage build for the FastAPI backend.
- **CI/CD**: GitHub Actions for automated linting (`flake8`) and testing (`pytest`).

## 4. Directory Structure
```text
Task 3/
├── src/
│   ├── ml/             # Training & ONNX export
│   ├── api/            # FastAPI & ONNX inference
│   └── frontend/       # React App
├── tests/              # Unit & Integration tests
├── docs/               # Specifications & Diagrams
├── requirements.txt
└── Dockerfile
```
