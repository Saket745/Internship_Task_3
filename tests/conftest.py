import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import io
from PIL import Image

@pytest.fixture
def client():
    """FastAPI TestClient instance"""
    return TestClient(app)

@pytest.fixture
def sample_png():
    """Returns a valid in-memory PNG image bytes"""
    img = Image.new('L', (28, 28), color=0)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

@pytest.fixture
def sample_jpg():
    """Returns a valid in-memory JPG image bytes"""
    img = Image.new('L', (28, 28), color=0)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return buffered.getvalue()
