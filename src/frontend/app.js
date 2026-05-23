document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');

    let isDrawing = false;
    const API_URL = "http://localhost:8000";

    // Initialize Canvas Background
    function initCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = 15; // Thick lines
        ctx.strokeStyle = "black";
    }

    initCanvas();

    // Drawing Logic
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }

    function draw(e) {
        if (!isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        // Support both mouse and touch events
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    // Event Listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch support
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Clear Canvas
    clearBtn.addEventListener('click', () => {
        initCanvas();
        showEmpty();
    });

    // Prediction Logic
    predictBtn.addEventListener('click', async () => {
        const dataUrl = canvas.toDataURL("image/png");
        showLoading();

        try {
            const response = await fetch(`${API_URL}/predict-base64`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image: dataUrl }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            showSuccess(data.prediction, data.confidence);
        } catch (error) {
            console.error(error);
            showError(error.message || "Failed to reach the API.");
        }
    });

    // UI State Helpers
    function showEmpty() {
        resultContainer.className = "result-container empty";
        resultContainer.innerHTML = `<p id="resultText">Draw a character above and click predict!</p>`;
    }

    function showLoading() {
        resultContainer.className = "result-container loading";
        resultContainer.innerHTML = `<p>Analyzing drawing...</p>`;
    }

    function showError(msg) {
        resultContainer.className = "result-container error";
        resultContainer.innerHTML = `<p>Error: ${msg}</p>`;
    }

    function showSuccess(prediction, confidence) {
        resultContainer.className = "result-container success";
        const confPercent = confidence ? (confidence * 100).toFixed(1) : "0.0";
        resultContainer.innerHTML = `
            <h2>Prediction: <span class="predicted-char">${prediction}</span></h2>
            <p class="confidence">Confidence: ${confPercent}%</p>
        `;
    }
});
