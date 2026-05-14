import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import base64
import io
from PIL import Image

client = TestClient(app)

def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"
        assert response.json()["model_loaded"] is True

def test_predict_real():
    # Create a dummy 28x28 black image
    img = Image.new('L', (28, 28), color=0)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    with TestClient(app) as client:
        response = client.post("/predict", json={"image": img_str})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["prediction"] != "?"
        assert len(data["prediction"]) == 1

def test_predict_invalid_base64():
    with TestClient(app) as client:
        response = client.post("/predict", json={"image": "not-a-base64-string"})
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid image data provided."
