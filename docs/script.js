// ── Network registry ───────────────────────────────────────────────────────
// Add more networks here as objects with their info.json path
const NETWORKS = {
  'grokking-mnist': 'networks/grokking-mnist/info.json'
};

// ── State ──────────────────────────────────────────────────────────────────
let currentWeights = null;
let isDrawing = false;
let lastX = 0, lastY = 0;

// ── DOM refs ───────────────────────────────────────────────────────────────
const canvas     = document.getElementById('draw-canvas');
const ctx        = canvas.getContext('2d');
const clearBtn   = document.getElementById('clear-btn');
const guessBtn   = document.getElementById('guess-btn');
const netSelect  = document.getElementById('network-select');
const loadStatus = document.getElementById('load-status');
const barChart   = document.getElementById('bar-chart');
const predDisplay = document.getElementById('prediction-display');

// ── Canvas setup ───────────────────────────────────────────────────────────
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth   = 18;
ctx.lineCap     = 'round';
ctx.lineJoin    = 'round';
ctx.strokeStyle = '#fff';

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY
  };
}

canvas.addEventListener('mousedown',  e => { isDrawing = true;  const p = getPos(e); lastX = p.x; lastY = p.y; });
canvas.addEventListener('mousemove',  e => { if (!isDrawing) return; draw(e); });
canvas.addEventListener('mouseup',    () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!isDrawing) return; draw(e); });
canvas.addEventListener('touchend',   () => { isDrawing = false; ctx.beginPath(); });

function draw(e) {
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x;
  lastY = p.y;
}

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predDisplay.innerHTML = '<span class="prediction-label">Draw something and press Guess</span>';
  barChart.innerHTML = '';
});

// ── Network loading ────────────────────────────────────────────────────────
async function loadNetwork(key) {
  loadStatus.textContent = 'Loading…';
  guessBtn.disabled = true;
  currentWeights = null;

  try {
    // Load info
    const infoRes = await fetch(NETWORKS[key]);
    const info    = await infoRes.json();

    document.getElementById('info-arch').textContent    = info.architecture;
    document.getElementById('info-tech').textContent    = info.techniques;
    document.getElementById('info-results').textContent = info.results;

    // Load weights
    const wRes    = await fetch(info.weights);
    currentWeights = await wRes.json();

    loadStatus.textContent = 'Ready';
    guessBtn.disabled = false;
  } catch (err) {
    loadStatus.textContent = 'Failed to load';
    console.error(err);
  }
}

netSelect.addEventListener('change', () => loadNetwork(netSelect.value));

// Load default network on page load
loadNetwork(netSelect.value);

// ── Forward pass (NumPy-equivalent in JS) ─────────────────────────────────

// Vector × matrix: vec[n] @ mat[n][m] → result[m]
function dotVecMat(vec, mat) {
  const n = vec.length;
  const m = mat[0].length;
  const out = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += vec[i] * mat[i][j];
    out[j] = s;
  }
  return out;
}

function tanh(vec) {
  return vec.map(v => Math.tanh(v));
}

function softmax(vec) {
  const max = Math.max(...vec);
  const exp = vec.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

function predict(pixels) {
  // pixels: flat Float32Array of 784 values, normalised 0–1
  const layer_1 = tanh(dotVecMat(pixels, currentWeights.weights_0_1));
  const layer_2 = softmax(dotVecMat(layer_1, currentWeights.weights_1_2));
  return layer_2;
}

// ── Image preprocessing ────────────────────────────────────────────────────
function getPixels() {
  // Shrink the 280×280 canvas down to 28×28
  const small = document.createElement('canvas');
  small.width = small.height = 28;
  const sCtx  = small.getContext('2d');
  sCtx.drawImage(canvas, 0, 0, 28, 28);
  const imgData = sCtx.getImageData(0, 0, 28, 28).data; // RGBA

  // Use the red channel (canvas is grayscale so R=G=B) and normalise to 0–1
  const pixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = imgData[i * 4] / 255.0;
  }
  return pixels;
}

// ── Guess ──────────────────────────────────────────────────────────────────
guessBtn.addEventListener('click', () => {
  if (!currentWeights) return;

  const pixels = getPixels();
  const probs  = predict(pixels);
  const pred   = probs.indexOf(Math.max(...probs));
  const conf   = (probs[pred] * 100).toFixed(1);

  // Show prediction
  predDisplay.innerHTML = `
    <div class="prediction-number">${pred}</div>
    <div class="prediction-conf">${conf}% confidence</div>
  `;

  // Show bar chart
  barChart.innerHTML = '';
  const sorted = [...probs].sort((a, b) => b - a);
  const topVal = sorted[0];

  probs.forEach((p, digit) => {
    const pct  = (p * 100).toFixed(1);
    const w    = (p / topVal * 100).toFixed(1);
    const isTop = digit === pred;

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-digit">${digit}</span>
      <div class="bar-track">
        <div class="bar-fill ${isTop ? 'top' : ''}" style="width:${w}%"></div>
      </div>
      <span class="bar-pct">${pct}%</span>
    `;
    barChart.appendChild(row);
  });
});
