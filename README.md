# 🖋️ DeepHandwriting: Alphanumeric Recognition System

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3%2B-0194E2?logo=mlflow)](https://mlflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Inference-005ced?logo=onnx)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**DeepHandwriting** is a state-of-the-art, deep-learning powered handwriting recognition engine designed to identify 62 distinct character classes (0-9, A-Z, a-z). Leveraging a custom **ResNet-18** architecture and optimized via **ONNX**, it provides ultra-low latency inference for real-time applications.

---

## 🏗️ System Architecture

The architecture is designed for high-throughput and modularity, separating the compute-intensive training pipeline from the low-latency inference service.

```mermaid
graph TD
    subgraph "🎨 Frontend (Client)"
        UI["Drawing Canvas (React/HTML5)"]
        CLIENT_PRED["Prediction Display"]
    end

    subgraph "🌐 Backend (FastAPI)"
        MAIN["main.py (API Routes)"]
        INF["inference.py (Logic)"]
    end

    subgraph "🧠 Inference Engine"
        ORT["ONNX Runtime"]
        MODEL_ONNX["character_cnn.onnx"]
    end

    subgraph "🧪 Training Pipeline (PyTorch)"
        TRAIN["run_full_training.py"]
        ARCH["model.py (ResNet-18)"]
        DATA["data_loader.py (EMNIST)"]
        MLFLOW["MLflow Tracking Server"]
    end

    UI -- "POST /predict" --> MAIN
    MAIN --> INF
    INF --> ORT
    ORT --> MODEL_ONNX
    
    TRAIN --> ARCH
    TRAIN --> DATA
    TRAIN --> MLFLOW
    TRAIN -- "Export" --> MODEL_ONNX
    MODEL_ONNX -- "Load" --> ORT
```

### Core Technologies
-   **Model**: **ResNet-18** (Residual Networks) with skip connections to enable deeper learning without accuracy degradation.
-   **Experiment Tracking**: **MLflow** for robust lifecycle management, logging metrics, and artifact versioning.
-   **Inference Engine**: **ONNX Runtime** for hardware-agnostic execution, yielding sub-millisecond response times.
-   **API**: **FastAPI** with asynchronous request handling for scalable production deployments.

---

## 📊 Dataset & Reference

The model is trained on the **Extended MNIST (EMNIST)** dataset, specifically the **ByClass** split, which contains a balanced representation of digits and both case-sensitive letters.

-   **Total Samples**: ~814,255
-   **Classes**: 62 (0-9, A-Z, a-z)
-   **Primary Source**: [Kaggle - EMNIST Dataset by Crawford](https://www.kaggle.com/datasets/crawford/emnist)
-   **Characteristics**: $28 \times 28$ grayscale images, pre-processed with orientation correction and normalization.

---

## 🧠 Model Performance

Based on current training benchmarks on the EMNIST 'ByClass' dataset:

| Metric | Value |
| :--- | :--- |
| **Model Architecture** | ResNet-18 |
| **Accuracy (Top-1)** | **~86.87%** (Epoch 3) |
| **Inference Latency** | < 5ms (CPU/ONNX) |
| **Optimizer** | AdamW |
| **Schedulers** | OneCycleLR |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- (Optional) CUDA-compatible GPU for training acceleration.

### Installation
```bash
# Clone the repository
git clone https://github.com/Saket745/Internship_Task_3.git
cd Internship_Task_3

# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Usage
#### 1. Training & Tracking
Execute the full pipeline to train, validate, and export the model:
```bash
python -m src.ml.run_full_training
```
Launch the MLflow dashboard to monitor progress:
```bash
mlflow ui
```

#### 2. Local Deployment
Start the production-ready FastAPI server:
```bash
python -m src.api.main
```
The interactive drawing interface will be available at `http://localhost:8000`.

---

## 🛠️ Project Roadmap

- [x] **Core**: Implement ResNet-18 architecture.
- [x] **Pipeline**: Integrate MLflow and ONNX export.
- [x] **API**: Develop FastAPI inference service.
- [x] **Docs**: Professional README and Master Specification.
- [ ] **Frontend**: Migrate current static UI to a full **React** application.
- [ ] **Infrastructure**: Add **Dockerfile** for containerized deployment.
- [ ] **DevOps**: Setup **GitHub Actions** for automated CI/CD.

---

## 🤝 Contributing
Contributions are welcome! Please follow the existing code style and ensure all tests pass before submitting a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
Dataset license follows the original NIST/Kaggle terms.
