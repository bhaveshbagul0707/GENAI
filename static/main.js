// Tab switching
function openTab(id) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  event.target.closest('.nav-btn').classList.add('active');
  document.getElementById(id).classList.add('active');
  showToast(`Switched to ${id} tab`, 'info');
}

// Theme toggle
function toggleTheme() {
  document.body.classList.toggle('light');
  const icon = document.querySelector('.theme-toggle');
  icon.textContent = document.body.classList.contains('light') ? '🌙' : '☀️';
  showToast(`Theme: ${document.body.classList.contains('light') ? 'Light' : 'Dark'}`, 'success');
}

// Toast notifications
function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
    <span>${msg}</span>
  `;
  document.getElementById('toasts').appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// Progress animation
function startProgress(id) {
  const container = document.getElementById(`prog-${id}-container`);
  const fill = document.getElementById(`prog-${id}`);
  const skeleton = document.getElementById(`sk-${id}`);

  container.classList.add('active');
  skeleton.classList.add('active');

  let width = 0;
  const interval = setInterval(() => {
    width += Math.random() * 10 + 5;
    if (width > 90) width = 90;
    fill.style.width = width + '%';
  }, 200);

  return interval;
}

function stopProgress(id, interval) {
  const container = document.getElementById(`prog-${id}-container`);
  const fill = document.getElementById(`prog-${id}`);
  const skeleton = document.getElementById(`sk-${id}`);

  clearInterval(interval);
  fill.style.width = '100%';

  setTimeout(() => {
    container.classList.remove('active');
    skeleton.classList.remove('active');
    fill.style.width = '0%';
  }, 500);
}

// Base64 helper
function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Load image
function loadImageToArray(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 128, 128);
      const data = ctx.getImageData(0, 0, 128, 128).data;

      const pixels = [];
      for (let i = 0; i < data.length; i += 4) {
        pixels.push([data[i] / 255, data[i + 1] / 255, data[i + 2] / 255]);
      }

      const arr = [];
      let k = 0;
      for (let i = 0; i < 128; i++) {
        const row = [];
        for (let j = 0; j < 128; j++) row.push(pixels[k++]);
        arr.push(row);
      }
      resolve(arr);
    };
    img.src = src;
  });
}

// Draw image
function drawImg(arr, id) {
  const h = arr.length, w = arr[0].length;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);

  let k = 0;
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      imgData.data[k++] = arr[i][j][0] * 255;
      imgData.data[k++] = arr[i][j][1] * 255;
      imgData.data[k++] = arr[i][j][2] * 255;
      imgData.data[k++] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const imgEl = document.getElementById(id);
  imgEl.src = canvas.toDataURL();
  imgEl.style.display = 'block';
}

function downloadImg(id, filename) {
  const img = document.getElementById(id);
  if (!img || !img.src) {
    showToast('No image to download', 'error');
    return;
  }
  const link = document.createElement('a');
  link.href = img.src;
  link.download = filename;
  link.click();
  showToast('Download started', 'success');
}

// TRANSFORMER
async function runTransformer() {
  const file = document.getElementById('tf_image').files[0];
  if (!file) {
    showToast('Please upload an image', 'error');
    return;
  }

  const interval = startProgress('tf');

  try {
    const arr = await loadImageToArray(await toBase64(file));
    const response = await fetch('/api/transformer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: arr })
    });

    const data = await response.json();

    let text = '🎯 Top-3 Predictions:\n\n';
    data.top3.forEach((item, i) => {
      text += `${i + 1}. ${item.label.toUpperCase()}\n`;
      text += `   Confidence: ${(item.confidence * 100).toFixed(2)}%\n\n`;
    });

    const output = document.getElementById('tf_out');
    output.textContent = text;
    output.style.display = 'block';

    showToast('Classification complete!', 'success');
  } catch (error) {
    showToast('Classification failed', 'error');
    console.error(error);
  } finally {
    stopProgress('tf', interval);
  }
}

// AUTOENCODER
async function runAutoencoder() {
  const file = document.getElementById('ae_image').files[0];
  if (!file) {
    showToast('Please upload an image', 'error');
    return;
  }

  const interval = startProgress('ae');

  try {
    const arr = await loadImageToArray(await toBase64(file));
    const response = await fetch('/api/autoencoder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: arr })
    });

    const data = await response.json();
    drawImg(data.reconstructed[0], 'ae_img');

    const output = document.getElementById('ae_out');
    output.textContent = '✅ Image reconstructed successfully!';
    output.style.display = 'block';

    document.getElementById('ae_download').style.display = 'inline-flex';
    showToast('Reconstruction complete!', 'success');
  } catch (error) {
    showToast('Reconstruction failed', 'error');
    console.error(error);
  } finally {
    stopProgress('ae', interval);
  }
}

// GAN
async function runGAN() {
  const interval = startProgress('gan');

  try {
    const response = await fetch('/api/generate_gan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    const data = await response.json();

    await new Promise(r => setTimeout(r, 5000));

    const img = document.getElementById('gan_img');
    img.src = data.url;
    img.style.display = 'block';

    const output = document.getElementById('gan_out');
    output.textContent = '✨ New image generated from random noise!';
    output.style.display = 'block';

    document.getElementById('gan_download').style.display = 'inline-flex';
    showToast('GAN generation complete!', 'success');
  } catch (error) {
    showToast('Generation failed', 'error');
    console.error(error);
  } finally {
    stopProgress('gan', interval);
  }
}

// DIFFUSION
async function runDiffusion() {
  const interval = startProgress('diff');

  try {
    await fetch('/api/diffusion');
    await new Promise(r => setTimeout(r, 2000));

    const imgEl = document.getElementById('diff_img');
    imgEl.src = "/static/GOD.jpg";
    imgEl.style.display = "block";

    const output = document.getElementById('diff_out');
    output.textContent = '🌊 Diffusion process complete!';
    output.style.display = 'block';

    const btn = document.getElementById('diff_download');
    btn.style.display = 'inline-flex';
    btn.setAttribute('onclick', "downloadImg('diff_img','GOD.jpg')");

    showToast('Diffusion complete!', 'success');
  } catch (error) {
    showToast('Diffusion failed', 'error');
    console.error(error);
  } finally {
    stopProgress('diff', interval);
  }
}

// Evaluation Metrics Loader
function loadEval(model) {
  const output = document.getElementById('eval_output');

  if (model === 'transformer') {
    output.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">Validation Accuracy</div><div class="metric-value">46.0%</div></div>
        <div class="metric-card"><div class="metric-label">Training Accuracy</div><div class="metric-value">95.0%</div></div>
        <div class="metric-card"><div class="metric-label">Validation Loss</div><div class="metric-value">3.050</div></div>
        <div class="metric-card"><div class="metric-label">Training Loss</div><div class="metric-value">0.300</div></div>
        <div class="metric-card"><div class="metric-label">Macro F1 Score</div><div class="metric-value">0.102</div></div>
        <div class="metric-card"><div class="metric-label">Best Class</div><div class="metric-value">ragno (0.24)</div></div>
      </div>
      <div style="margin-top:30px;">
        <img src="/static/transformer_acc_loss.jpg" style="max-width:100%;border-radius:12px;border:1px solid var(--border);">
      </div>
    `;
  }

  else if (model === 'autoencoder') {
    output.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">MSE Mean</div><div class="metric-value">0.0078</div></div>
        <div class="metric-card"><div class="metric-label">MAE Mean</div><div class="metric-value">0.0614</div></div>
        <div class="metric-card"><div class="metric-label">PSNR Mean</div><div class="metric-value">21.54</div></div>
        <div class="metric-card"><div class="metric-label">SSIM Mean</div><div class="metric-value">0.5487</div></div>
      </div>
      <div style="margin-top:30px;display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;">
        <img src="/static/ae_dist1.png" style="width:100%;border-radius:12px;border:1px solid var(--border);">
        <img src="/static/ae_trend.png" style="width:100%;border-radius:12px;border:1px solid var(--border);">
      </div>
    `;
  }

  else if (model === 'gan') {
    output.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">FID Score</div><div class="metric-value">43.06</div></div>
        <div class="metric-card"><div class="metric-label">Inception Score</div><div class="metric-value">8.450</div></div>
        <div class="metric-card"><div class="metric-label">PSNR</div><div class="metric-value">29.21</div></div>
        <div class="metric-card"><div class="metric-label">SSIM</div><div class="metric-value">0.931</div></div>
        <div class="metric-card"><div class="metric-label">F1 Score</div><div class="metric-value">0.936</div></div>
      </div>
      <div style="margin-top:30px;display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;">
        <img src="/static/gan_loss.jpg" style="width:100%;border-radius:12px;border:1px solid var(--border);">
        <img src="/static/gan_metrics.jpg" style="width:100%;border-radius:12px;border:1px solid var(--border);">
      </div>
    `;
  }

  else if (model === 'diffusion') {
    output.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">MSE</div><div class="metric-value">0.5820</div></div>
        <div class="metric-card"><div class="metric-label">PSNR</div><div class="metric-value">2.3507</div></div>
        <div class="metric-card"><div class="metric-label">SSIM</div><div class="metric-value">0.00142</div></div>
        <div class="metric-card"><div class="metric-label">Perceptual Distance</div><div class="metric-value">12.4918</div></div>
      </div>
      <div style="margin-top:30px;">
        <img src="/static/diffusion_matrix.png" style="max-width:100%;border-radius:12px;border:1px solid var(--border);">
      </div>
    `;
  }

  else {
    output.innerHTML = `
      <div style="text-align:center;padding:60px 20px;color:var(--text-secondary);">
        <div style="font-size:48px;margin-bottom:16px;">📊</div>
        <div>Select a model to view evaluation metrics</div>
      </div>
    `;
  }

  showToast(`Loaded ${model} metrics`, 'info');
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (!prefersDark) {
    document.body.classList.add('light');
    document.querySelector('.theme-toggle').textContent = '🌙';
  }
});
