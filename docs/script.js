// ── Bootstrap ─────────────────────────────────────────────────────────────
(async function () {
  const params = new URLSearchParams(window.location.search);
  const networkId = params.get('id');
  if (!networkId) return;  // landing page — nothing to do

  // Fetch network info
  const registry = await (await fetch('networks/registry.json')).json();
  const entry = registry.find(e => e.id === networkId);
  if (!entry) {
    document.getElementById('net-name').textContent = 'Network not found';
    return;
  }
  const info = await (await fetch(entry.path)).json();

  // Page title
  document.title = info.name;

  // Header meta
  document.getElementById('net-name').textContent = info.name;
  document.getElementById('net-arch').textContent = info.architecture;
  document.getElementById('net-results').textContent = info.results.summary || '';

  // Load weights + set up try-it widget.
  // Two backends:
  //   - 'mnist'      → hand-rolled forward pass over JSON weight matrices (Network 1)
  //   - 'mnist-cnn'  → TensorFlow.js LayersModel loaded from tfjs_model/model.json
  if (info.input_type === 'mnist') {
    const weights = await (await fetch(info.weights)).json();
    initTryIt(info, { kind: 'dense', weights });
  } else if (info.input_type === 'mnist-cnn') {
    const tfjsModel = await tf.loadLayersModel(info.tfjs_model);
    initTryIt(info, { kind: 'cnn', model: tfjsModel });
  }

  // Load notes
  if (info.notes) {
    try {
      const md = await (await fetch(info.notes)).text();
      document.getElementById('notes-content').innerHTML = marked.parse(md);
      document.getElementById('notes-section').style.display = '';
    } catch (_) {}
  }

  // Load notebook
  if (info.notebook) {
    try {
      const nb = await (await fetch(info.notebook)).json();
      renderNotebook(nb, info.name);
      document.getElementById('nb-container').style.display = '';
    } catch (e) {
      console.error('Failed to load notebook:', e);
    }
  }
})();

// ── Try-it: Canvas + Prediction ───────────────────────────────────────────
// `backend` is one of:
//   { kind: 'dense', weights }  — hand-rolled MLP forward pass
//   { kind: 'cnn',   model   }  — TensorFlow.js LayersModel
function initTryIt(info, backend) {
  const section = document.getElementById('try-it-section');
  section.style.display = '';

  const canvas = document.getElementById('draw-canvas');
  const ctx = canvas.getContext('2d');
  const clearBtn = document.getElementById('clear-btn');
  const guessBtn = document.getElementById('guess-btn');
  const predDisplay = document.getElementById('prediction-display');
  const barChart = document.getElementById('bar-chart');

  // Canvas setup
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';

  let isDrawing = false, lastX = 0, lastY = 0;

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * (canvas.width / rect.width),
      y: (src.clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  canvas.addEventListener('mousedown', e => { isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('mousemove', e => { if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
  canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });
  canvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('touchmove', e => { e.preventDefault(); if (!isDrawing) return; const p = getPos(e); ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke(); lastX = p.x; lastY = p.y; });
  canvas.addEventListener('touchend', () => { isDrawing = false; ctx.beginPath(); });

  clearBtn.addEventListener('click', () => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predDisplay.innerHTML = '<span class="prediction-label">Draw something and press Guess</span>';
    barChart.innerHTML = '';
  });

  guessBtn.disabled = false;
  guessBtn.addEventListener('click', () => {
    const pixels = getPixels(canvas);
    let probs;
    if (backend.kind === 'dense') {
      probs = forwardPass(pixels, backend.weights, info.layers);
    } else {
      // CNN: reshape the flat 784-length Float32Array into a (1,28,28,1) tensor,
      // run model.predict, pull the result back into a plain JS array.
      const input = tf.tensor4d(pixels, [1, 28, 28, 1]);
      const out = backend.model.predict(input);
      probs = Array.from(out.dataSync());
      input.dispose();
      out.dispose();
    }
    const pred = probs.indexOf(Math.max(...probs));

    predDisplay.innerHTML =
      `<div class="prediction-number">${pred}</div>` +
      `<div class="prediction-conf">${(probs[pred] * 100).toFixed(1)}% confidence</div>`;

    barChart.innerHTML = '';
    const top = Math.max(...probs);
    probs.forEach((p, d) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        `<span class="bar-digit">${d}</span>` +
        `<div class="bar-track"><div class="bar-fill ${d === pred ? 'top' : ''}" style="width:${(p / top * 100).toFixed(1)}%"></div></div>` +
        `<span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
      barChart.appendChild(row);
    });
  });
}

// ── Generic forward pass ──────────────────────────────────────────────────
const activations = {
  tanh: v => v.map(x => Math.tanh(x)),
  sigmoid: v => v.map(x => 1 / (1 + Math.exp(-x))),
  relu: v => v.map(x => Math.max(0, x)),
  softmax: v => {
    const mx = Math.max(...v);
    const e = v.map(x => Math.exp(x - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  },
  linear: v => v
};

function dotVecMat(vec, mat) {
  const m = mat[0].length;
  const out = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < vec.length; i++) s += vec[i] * mat[i][j];
    out[j] = s;
  }
  return Array.from(out);
}

function addBias(vec, bias) {
  return vec.map((v, i) => v + bias[i]);
}

function forwardPass(input, weights, layers) {
  let current = Array.from(input);
  for (const layer of layers) {
    current = dotVecMat(current, weights[layer.weights]);
    if (layer.bias && weights[layer.bias]) {
      current = addBias(current, weights[layer.bias]);
    }
    const act = activations[layer.activation] || activations.linear;
    current = act(current);
  }
  return current;
}

// ── Get 28x28 pixels from canvas ──────────────────────────────────────────
function getPixels(canvas) {
  const s = document.createElement('canvas');
  s.width = s.height = 28;
  s.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
  const d = s.getContext('2d').getImageData(0, 0, 28, 28).data;
  const p = new Float32Array(784);
  for (let i = 0; i < 784; i++) p[i] = d[i * 4] / 255;
  return p;
}

// ── Pyodide kernel ────────────────────────────────────────────────────────
let pyodide = null;
let pyodideReady = false;
let pyodideBooting = false;
let runCounter = 0;

function setKernelStatus(status) {
  const dot = document.getElementById('kernel-dot');
  const label = document.getElementById('kernel-label');
  if (!dot || !label) return;
  dot.className = 'nb-kernel-dot ' + status;
  const labels = { idle: 'Python (idle)', busy: 'Python (busy)', error: 'Python (error)', loading: 'Python (loading…)' };
  label.textContent = labels[status] || status;
}

async function bootPyodide() {
  if (pyodideReady || pyodideBooting) return;
  pyodideBooting = true;
  setKernelStatus('loading');
  try {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(['numpy', 'matplotlib']);
    await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

def _show_capture():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    print('__IMG__' + base64.b64encode(buf.read()).decode() + '__ENDIMG__')
    plt.close('all')
plt.show = _show_capture
`);
    pyodideReady = true;
    setKernelStatus('idle');
  } catch (e) {
    setKernelStatus('error');
    console.error('Pyodide failed:', e);
  }
  pyodideBooting = false;
}

async function runCellCode(code, outputContent, promptEl) {
  if (!pyodideReady) await bootPyodide();
  if (!pyodideReady) {
    outputContent.innerHTML = '<pre class="nb-error-text">Could not start Python runtime.</pre>';
    return;
  }

  runCounter++;
  const thisRun = runCounter;
  promptEl.textContent = 'In [*]:';
  outputContent.innerHTML = '<div class="nb-running-text">Running…</div>';
  setKernelStatus('busy');

  pyodide.runPython(`
import sys, io as _io
_buf = _io.StringIO()
sys.stdout = _buf
sys.stderr = _buf
`);

  let rawOutput = '';
  let hasError = false;

  try {
    await pyodide.loadPackagesFromImports(code);
    await pyodide.runPythonAsync(code);
  } catch (err) {
    hasError = true;
    rawOutput += err.message + '\n';
  }

  try {
    rawOutput = pyodide.runPython('_buf.getvalue()') + rawOutput;
    pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__');
  } catch (_) {}

  promptEl.textContent = `In [${thisRun}]:`;
  renderCellOutput(rawOutput, outputContent, hasError);
  setKernelStatus('idle');
}

function renderCellOutput(text, container, isError) {
  container.innerHTML = '';
  if (!text.trim()) return;

  const parts = text.split(/(__IMG__[\s\S]*?__ENDIMG__)/);
  parts.forEach(part => {
    const imgMatch = part.match(/^__IMG__([\s\S]*?)__ENDIMG__$/);
    if (imgMatch) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${imgMatch[1].trim()}`;
      img.className = 'nb-plot-img';
      container.appendChild(img);
    } else if (part.trim()) {
      const pre = document.createElement('pre');
      pre.className = isError ? 'nb-error-text' : 'nb-stdout';
      pre.textContent = part;
      container.appendChild(pre);
    }
  });
}

// ── Notebook renderer ─────────────────────────────────────────────────────
function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = ta.scrollHeight + 'px';
}

function renderNotebook(nb, title) {
  const container = document.getElementById('notebook-render');
  container.innerHTML = '';
  document.getElementById('nb-title').textContent = title || 'Notebook';

  (nb.cells || []).forEach(cell => {
    const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');

    if (cell.cell_type === 'markdown') {
      const div = document.createElement('div');
      div.className = 'nb-md-cell';
      div.innerHTML = marked.parse(src);
      container.appendChild(div);

    } else if (cell.cell_type === 'code') {
      const wrapper = document.createElement('div');
      wrapper.className = 'nb-code-cell';

      // Input area
      const inputArea = document.createElement('div');
      inputArea.className = 'nb-input-area';

      const inPrompt = document.createElement('div');
      inPrompt.className = 'nb-in-prompt';
      inPrompt.textContent = 'In [ ]:';

      const textarea = document.createElement('textarea');
      textarea.className = 'nb-editor';
      textarea.value = src;
      textarea.spellcheck = false;
      textarea.addEventListener('input', () => autoResize(textarea));
      textarea.addEventListener('focus', () => wrapper.classList.add('selected'));
      textarea.addEventListener('blur', () => wrapper.classList.remove('selected'));
      textarea.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); runBtn.click(); return; }
        if (e.key === 'Tab') {
          e.preventDefault();
          const s = textarea.selectionStart, en = textarea.selectionEnd;
          textarea.value = textarea.value.slice(0, s) + '    ' + textarea.value.slice(en);
          textarea.selectionStart = textarea.selectionEnd = s + 4;
        }
      });

      const runBtn = document.createElement('button');
      runBtn.className = 'nb-run-btn';
      runBtn.innerHTML = '&#9654;';
      runBtn.title = 'Run cell (Shift+Enter)';

      inputArea.appendChild(inPrompt);
      inputArea.appendChild(textarea);
      inputArea.appendChild(runBtn);

      // Output area
      const outputArea = document.createElement('div');
      outputArea.className = 'nb-output-area';

      const outPrompt = document.createElement('div');
      outPrompt.className = 'nb-out-prompt';

      const outputContent = document.createElement('div');
      outputContent.className = 'nb-output-content';

      // Render saved outputs from the .ipynb
      (cell.outputs || []).forEach(out => {
        if (out.output_type === 'stream') {
          const text = Array.isArray(out.text) ? out.text.join('') : out.text;
          const pre = document.createElement('pre');
          pre.className = 'nb-stdout';
          pre.textContent = text;
          outputContent.appendChild(pre);
        } else if (out.output_type === 'display_data' || out.output_type === 'execute_result') {
          const data = out.data || {};
          if (data['image/png']) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data['image/png']}`;
            img.className = 'nb-plot-img';
            outputContent.appendChild(img);
          } else if (data['text/plain']) {
            const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
            const pre = document.createElement('pre');
            pre.className = 'nb-stdout';
            pre.textContent = text;
            outputContent.appendChild(pre);
          }
        }
      });

      outputArea.appendChild(outPrompt);
      outputArea.appendChild(outputContent);
      wrapper.appendChild(inputArea);
      wrapper.appendChild(outputArea);
      container.appendChild(wrapper);

      // Wire run button
      runBtn.addEventListener('click', async () => {
        wrapper.classList.add('running');
        outPrompt.textContent = '';
        await runCellCode(textarea.value, outputContent, inPrompt);
        wrapper.classList.remove('running');
      });

      requestAnimationFrame(() => autoResize(textarea));
    }
  });

  // Run all
  document.getElementById('run-all-btn').onclick = async () => {
    for (const cell of container.querySelectorAll('.nb-code-cell')) {
      const runBtn = cell.querySelector('.nb-run-btn');
      if (runBtn) runBtn.click();
      await new Promise(r => setTimeout(r, 50));
    }
  };

  // Restart kernel
  document.getElementById('restart-btn').onclick = async () => {
    pyodide = null;
    pyodideReady = false;
    pyodideBooting = false;
    setKernelStatus('loading');
    runCounter = 0;
    container.querySelectorAll('.nb-in-prompt').forEach(p => p.textContent = 'In [ ]:');
    container.querySelectorAll('.nb-output-content').forEach(el => el.innerHTML = '');
    await bootPyodide();
  };
}
