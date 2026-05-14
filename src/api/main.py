from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
import base64

app = FastAPI(title="Nebula Glass API")

# Path discovery
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "models", "character_cnn.onnx")

# EMNIST Balanced Mapping (47 classes)
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# Global session variable
ort_session = None

def get_ort_session():
    global ort_session
    if ort_session is None:
        if os.path.exists(MODEL_PATH):
            ort_session = ort.InferenceSession(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    return ort_session

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    session = get_ort_session()
    if session is None:
        # Mock prediction if model is not yet trained
        return {"prediction": "?", "confidence": 0.0, "status": "Model training in progress"}

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy and normalize
        # EMNIST mean/std: (0.1736, 0.3317)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - 0.1736) / 0.3317
        img_array = img_array.reshape(1, 1, 28, 28)
        
        # Run inference
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)
        
        # Post-process results
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        pred_idx = np.argmax(probs)
        confidence = float(probs[0][pred_idx])
        
        return {
            "prediction": CLASSES[pred_idx],
            "confidence": round(confidence * 100, 2),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-base64")
async def predict_base64(data: dict):
    session = get_ort_session()
    if session is None:
        return {"prediction": "?", "confidence": 0.0, "status": "Model training in progress"}

    try:
        header, encoded = data['image'].split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Important: EMNIST is trained on inverted images (white on black)
        # and transposed. But if the user draws white on black already, we just need to ensure consistency.
        image = image.resize((28, 28))
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - 0.1736) / 0.3317
        img_array = img_array.reshape(1, 1, 28, 28)
        
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)
        
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        pred_idx = np.argmax(probs)
        confidence = float(probs[0][pred_idx])
        
        return {
            "prediction": CLASSES[pred_idx],
            "confidence": round(confidence * 100, 2),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
