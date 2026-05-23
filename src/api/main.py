from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort
import base64

app = FastAPI(title="Nebula Glass API")

# Path discovery
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "models", "character_cnn.onnx")

# EMNIST ByClass Mapping (62 classes)
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

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

def preprocess_pil_image(image: Image.Image, target_size: int = 28, min_contour_area: int = 20, padding_ratio: float = 0.15) -> np.ndarray:
    """
    Preprocesses a drawing canvas image (black strokes on white background):
    1. Converts PIL image to OpenCV grayscale array.
    2. Thresholds and inverts the image to get white strokes on black background.
    3. Finds the character bounding box using contour detection to crop it.
    4. Pads the cropped region to avoid edge cutting.
    5. Resizes to 28x28.
    6. Normalizes with EMNIST statistics.
    """
    gray = np.array(image)
    
    # Thresholding & Inversion (cv2.THRESH_BINARY_INV turns dark strokes to white, white bg to black)
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
            
    # If no contours found, default to direct resize of binary
    if not filtered_bboxes:
        resized = cv2.resize(binary, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    else:
        # Get bounding box surrounding all filtered boxes
        xs = [b[0] for b in filtered_bboxes]
        ys = [b[1] for b in filtered_bboxes]
        ws = [b[2] for b in filtered_bboxes]
        hs = [b[3] for b in filtered_bboxes]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max([x + w for x, w in zip(xs, ws)])
        max_y = max([y + h for y, h in zip(ys, hs)])
        
        w = max_x - min_x
        h = max_y - min_y
        roi = binary[min_y:max_y, min_x:max_x]
        
        if roi.size == 0:
            resized = cv2.resize(binary, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            # Add padding
            pad_h = int(h * padding_ratio)
            pad_w = int(w * padding_ratio)
            if padding_ratio > 0:
                pad_h = max(pad_h, 1)
                pad_w = max(pad_w, 1)
                
            padded = cv2.copyMakeBorder(
                roi, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_CONSTANT, value=0
            )
            resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    
    # EMNIST ByClass statistics: mean=0.1736, std=0.3317
    normalized = (normalized - 0.1736) / 0.3317
    
    # Reshape to shape expected by the CNN: (1, 1, 28, 28)
    return normalized.reshape(1, 1, target_size, target_size)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

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
        
        img_array = preprocess_pil_image(image)
        
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
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
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
        
        img_array = preprocess_pil_image(image)
        
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
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
