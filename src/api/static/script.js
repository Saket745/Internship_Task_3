const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionResult = document.getElementById('predictionResult');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const brushSize = document.getElementById('brushSize');
const undoBtn = document.getElementById('undoBtn');
const predictionBox = document.querySelector('.prediction-box');

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

// Drawing Logic — Mouse
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

// Clear Canvas
clearBtn.addEventListener('click', () => {
    initCanvas();
    resetPrediction();
});

undoBtn.addEventListener('click', undo);

// UI State Helpers
function resetPrediction() {
    predictionResult.textContent = '-';
    confidenceValue.textContent = '0%';
    confidenceBar.style.width = '0%';
    predictionBox.classList.remove('success-glow');
    predictionResult.style.animation = 'none';
}

function showLoading() {
    predictionBox.classList.remove('success-glow');
    predictionResult.style.animation = 'none';
    
    // Create spinner + text
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    
    const loadText = document.createElement('p');
    loadText.className = 'loading-text';
    loadText.textContent = 'Running neural inference…';
    
    predictionResult.textContent = '';
    predictionResult.style.cssText = 'font-size: inherit; background: none; -webkit-text-fill-color: inherit; filter: none;';
    predictionResult.appendChild(spinner);
    predictionResult.appendChild(loadText);
    
    confidenceValue.textContent = '…';
    confidenceBar.style.width = '0%';
}

function showError(message) {
    predictionBox.classList.remove('success-glow');
    
    // Restore styling
    predictionResult.style.cssText = '';
    predictionResult.textContent = '';
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-state';
    
    const icon = document.createElement('div');
    icon.className = 'error-icon';
    icon.textContent = '⚠️';
    
    const msg = document.createElement('p');
    msg.className = 'error-msg';
    msg.textContent = message;
    
    const hint = document.createElement('p');
    hint.className = 'error-hint';
    hint.textContent = 'Ensure the API server is running on localhost:8000';
    
    errorDiv.appendChild(icon);
    errorDiv.appendChild(msg);
    errorDiv.appendChild(hint);
    
    predictionResult.style.cssText = 'font-size: inherit; background: none; -webkit-text-fill-color: inherit; filter: none;';
    predictionResult.appendChild(errorDiv);
    
    confidenceValue.textContent = '—';
    confidenceBar.style.width = '0%';
}

function showSuccess(prediction, confidence) {
    // Restore normal prediction styling
    predictionResult.style.cssText = '';
    predictionResult.textContent = prediction;
    
    confidenceValue.textContent = confidence + '%';
    confidenceBar.style.width = confidence + '%';
    
    // Add glow effect
    predictionBox.classList.add('success-glow');
    
    // Trigger punch animation
    predictionResult.style.animation = 'none';
    predictionResult.offsetHeight; // trigger reflow
    predictionResult.style.animation = 'glowPulse 1.2s ease-out';
}

// Prediction Integration
async function predict() {
    predictBtn.disabled = true;
    const originalContent = predictBtn.innerHTML;
    predictBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation: spin 0.8s linear infinite;"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>
        ANALYZING…
    `;
    
    showLoading();
    
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict-base64', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Server returned ' + response.status);
        }
        
        const result = await response.json();
        
        if (result.status === "Success") {
            showSuccess(result.prediction, result.confidence);
        } else if (result.status === "Model training in progress") {
            showError('Model is loading. Please wait a moment and try again.');
        } else {
            showError('Unexpected response from the engine.');
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        showError(error.message || 'Communication error with Neural Engine');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalContent;
    }
}

predictBtn.addEventListener('click', predict);

// Keyboard shortcut: Enter to predict, Escape to clear
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !predictBtn.disabled) {
        predict();
    } else if (e.key === 'Escape') {
        initCanvas();
        resetPrediction();
    } else if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undo();
    }
});

// Init on load
initCanvas();
