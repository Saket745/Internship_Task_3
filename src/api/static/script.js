const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionResult = document.getElementById('predictionResult');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const brushSize = document.getElementById('brushSize');
const undoBtn = document.getElementById('undoBtn');

let drawing = false;
let lastX = 0;
let lastY = 0;
let history = [];

// Initialize Canvas: Black background, white ink (standard for MNIST/EMNIST)
function initCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    saveState();
}

function saveState() {
    if (history.length > 20) history.shift();
    history.push(canvas.toDataURL());
}

function undo() {
    if (history.length > 1) {
        history.pop();
        const img = new Image();
        img.src = history[history.length - 1];
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
        };
    }
}

// Drawing Logic
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => {
    if (drawing) {
        drawing = false;
        saveState();
    }
});
canvas.addEventListener('mouseout', () => drawing = false);

// Touch Support
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    [lastX, lastY] = [touch.clientX - rect.left, touch.clientY - rect.top];
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!drawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.lineWidth = brushSize.value;
    ctx.stroke();
    [lastX, lastY] = [x, y];
});

canvas.addEventListener('touchend', () => {
    if (drawing) {
        drawing = false;
        saveState();
    }
});

function draw(e) {
    if (!drawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.lineWidth = brushSize.value;
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

clearBtn.addEventListener('click', () => {
    initCanvas();
    predictionResult.innerText = '-';
    confidenceValue.innerText = '0%';
    confidenceBar.style.width = '0%';
});

undoBtn.addEventListener('click', undo);

// Prediction Integration
async function predict() {
    predictBtn.disabled = true;
    predictBtn.innerText = 'ANALYZING...';
    
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict-base64', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.status === "Success") {
            predictionResult.innerText = result.prediction;
            confidenceValue.innerText = `${result.confidence}%`;
            confidenceBar.style.width = `${result.confidence}%`;
            
            // Add punch animation to prediction
            predictionResult.style.animation = 'none';
            predictionResult.offsetHeight; // trigger reflow
            predictionResult.style.animation = 'glowPulse 1s ease-out';
        } else {
            predictionResult.innerText = '?';
            alert("Model is still training. Please wait a few moments.");
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        alert('Communication error with Neural Engine');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = 'RUN INFERENCE';
    }
}

predictBtn.addEventListener('click', predict);

// Init on load
initCanvas();
