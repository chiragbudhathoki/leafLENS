from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms.v2 as v2
import os
from urllib.request import urlretrieve

from model import Net

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AUTO-DOWNLOAD MODEL WEIGHTS FROM GOOGLE DRIVE ---
MODEL_PATH = "leaflens.pth"

if not os.path.exists(MODEL_PATH):
    print("Model weights missing locally. Downloading from Google Drive...")
    file_id = "1oqZP8awWiTeuxvXZj2aAcDePMHyhOgg5"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    urlretrieve(url, MODEL_PATH)
    print("Download completed successfully!")

# Load architecture and weights
model = Net().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("LeafLENS Model Loaded Successfully!")
except Exception as e:
    print(f"Warning: Model not loaded. Error: {e}")

# Exact transform from your training pipeline
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]
@app.get('/')
def root():
    return RedirectResponse(url='/homepage')
    
@app.get('/homepage', response_class=HTMLResponse)
def index():
    return """
      <!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>leafLENS – Is this leaf healthy?</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='28' fill='none' stroke='%2314382c' stroke-width='6'/%3E%3Cpath d='M32 14c12 8 14 22 0 36C18 36 20 22 32 14z' fill='%23cfe86a' stroke='%2314382c' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0f2a22;
    --spruce: #14382c;
    --leaf: #2f7d4f;
    --sprout: #cfe86a;
    --mist: #e8f0ea;
    --paper: #f7faf6;
    --line: #c9d8cd;
    --muted: #4c675a;

    --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
    --body: "Figtree", "Segoe UI", system-ui, sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; }
  [hidden] { display: none !important; }

  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    background: var(--mist);
    color: var(--ink);
    font-family: var(--body);
    font-size: 1.0625rem;
    line-height: 1.6;
  }

  :focus-visible { outline: 3px solid var(--leaf); outline-offset: 3px; }

  .page {
    max-width: 1120px;
    min-height: 100vh;
    min-height: 100dvh;
    margin: 0 auto;
    padding: 0 clamp(1.25rem, 4vw, 2.5rem);
    display: flex;
    flex-direction: column;
  }

  /* ---------- Header ---------- */
  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0;
  }
  .brand { display: inline-flex; align-items: center; gap: .6rem; text-decoration: none; color: var(--spruce); }
  .brand svg { width: 34px; height: 34px; flex: none; }
  .wordmark { font-family: var(--display); font-size: 1.55rem; letter-spacing: -0.01em; line-height: 1; }
  .wordmark .wm-leaf { font-weight: 500; }
  .wordmark .wm-lens { font-weight: 800; letter-spacing: .03em; }
  .top a.docs {
    color: var(--muted);
    font-weight: 500;
    font-size: .95rem;
    text-decoration: none;
    border-bottom: 1.5px solid var(--line);
    padding-bottom: 1px;
  }
  .top a.docs:hover { color: var(--spruce); border-color: var(--spruce); }

  /* ---------- Main layout ---------- */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 480px);
    grid-template-areas:
      "intro stage"
      "slot  stage";
    column-gap: clamp(2rem, 6vw, 5rem);
    row-gap: 2rem;
    align-content: center;
    padding: 1.5rem 0 4rem;
  }
  .intro { grid-area: intro; align-self: end; }
  .slot  { grid-area: slot;  align-self: start; }
  .stage { grid-area: stage; }

  h1 {
    font-family: var(--display);
    font-weight: 800;
    font-size: clamp(2.7rem, 6.4vw, 4.8rem);
    line-height: 1;
    letter-spacing: -0.03em;
    margin: 0;
    color: var(--spruce);
  }

  /* ---------- Lens ---------- */
  .stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2.75rem;
    padding: 40px;
  }
  .lens-wrap { position: relative; width: 100%; max-width: 400px; aspect-ratio: 1; }

  .ticks {
    position: absolute;
    inset: -34px;
    border-radius: 50%;
    background: repeating-conic-gradient(from -.4deg, rgba(20, 56, 44, .45) 0 .8deg, transparent .8deg 5deg);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 9px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 9px));
    pointer-events: none;
  }

  .lens {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    padding: 0;
    border: 0;
    border-radius: 50%;
    overflow: hidden;
    cursor: pointer;
    font: inherit;
    color: var(--spruce);
    background: radial-gradient(circle at 35% 28%, #ffffff 0%, var(--paper) 55%, #dbe8de 100%);
    box-shadow: 0 0 0 8px var(--spruce), 0 24px 50px -20px rgba(15, 42, 34, .55);
    transition: box-shadow .2s ease, background-color .2s ease;
  }
  .lens:hover { box-shadow: 0 0 0 8px var(--leaf), 0 24px 50px -20px rgba(15, 42, 34, .55); }
  .lens.is-drag { box-shadow: 0 0 0 8px var(--leaf), 0 0 0 20px rgba(207, 232, 106, .7); }
  .lens:focus-visible { outline: 3px solid var(--leaf); outline-offset: 14px; }

  .lens::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: radial-gradient(ellipse 42% 24% at 30% 17%, rgba(255, 255, 255, .75), transparent 70%);
    opacity: .9;
    pointer-events: none;
  }
  .lens.has-image::after { opacity: .35; }

  .empty {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: .35rem;
    padding: 2rem;
    text-align: center;
  }
  .empty svg { width: 34%; height: auto; margin-bottom: .75rem; transform: rotate(-16deg); }
  .empty-title { font-family: var(--display); font-weight: 700; font-size: 1.45rem; letter-spacing: -0.01em; }
  .empty-sub { color: var(--muted); font-size: .98rem; }

  .preview {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .scan {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    overflow: hidden;
    pointer-events: none;
    opacity: 0;
    transition: opacity .2s ease;
    z-index: 2;
  }
  .stage.is-scanning .scan { opacity: 1; }
  .scan::before {
    content: "";
    position: absolute;
    left: 0; right: 0; top: 0;
    height: 38%;
    background: linear-gradient(to bottom, transparent, rgba(207, 232, 106, .5));
    border-bottom: 3px solid var(--sprout);
    box-shadow: 0 8px 28px rgba(207, 232, 106, .9);
    transform: translateY(-100%);
    animation: sweep 1.7s linear infinite;
  }
  @keyframes sweep {
    to { transform: translateY(263%); }
  }
  @media (prefers-reduced-motion: reduce) {
    .scan::before { animation: none; transform: none; height: 100%; background: rgba(207, 232, 106, .35); border-bottom: 0; box-shadow: none; }
    .bar span { transition: none !important; }
  }

  /* ---------- Buttons ---------- */
  .actions { display: flex; flex-wrap: wrap; justify-content: center; gap: .75rem; }
  .btn {
    font: inherit;
    font-weight: 600;
    border-radius: 999px;
    padding: .8rem 1.6rem;
    cursor: pointer;
    border: 2px solid var(--spruce);
    transition: background-color .15s ease, color .15s ease, opacity .15s ease;
  }
  .btn-primary { background: var(--spruce); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--leaf); border-color: var(--leaf); }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  .btn-ghost { background: transparent; color: var(--spruce); }
  .btn-ghost:hover:not(:disabled) { background: rgba(20, 56, 44, .08); }
  .btn:disabled { cursor: not-allowed; }

  /* ---------- Slot: status and result ---------- */
  .card {
    --tone: #1f6b40;
    --tint: #d9efe0;
    --bar: var(--leaf);
    max-width: 480px;
    background: var(--paper);
    border: 1.5px solid var(--line);
    border-radius: 22px;
    padding: 1.5rem 1.6rem 1.4rem;
  }
  .card[data-tone="issue"]   { --tone: #8a4d0b; --tint: #f7e5c8; --bar: #c27a1d; }
  .card[data-tone="unknown"] { --tone: #48545f; --tint: #e3e7ea; --bar: #8a95a0; }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    padding: .28rem .8rem;
    border-radius: 999px;
    background: var(--tint);
    color: var(--tone);
    font-weight: 600;
    font-size: .9rem;
  }
  .pill::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }
  .card.is-loading .pill::before { animation: blink 1s ease-in-out infinite; }
  @keyframes blink { 50% { opacity: .25; } }
  @media (prefers-reduced-motion: reduce) { .card.is-loading .pill::before { animation: none; } }

  .crop { margin: 1.1rem 0 0; color: var(--muted); font-weight: 500; }
  .verdict {
    font-family: var(--display);
    font-weight: 800;
    font-size: clamp(1.9rem, 4vw, 2.5rem);
    line-height: 1.08;
    letter-spacing: -0.02em;
    margin: 1rem 0 0;
    color: var(--spruce);
  }
  .crop + .verdict { margin-top: .2rem; }
  .card p.detail { margin: .75rem 0 0; color: var(--muted); }

  .conf { margin-top: 1.4rem; }
  .conf-row { display: flex; justify-space-between; align-items: baseline; margin-bottom: .45rem; font-weight: 500; color: var(--muted); }
  .conf-row strong { font-family: var(--display); font-size: 1.35rem; color: var(--tone); }
  .bar { height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; }
  .bar span { display: block; height: 100%; width: 0; border-radius: inherit; background: var(--bar); transition: width .8s cubic-bezier(.2, .8, .2, 1); }

  .note { margin: 1.25rem 0 0; padding-top: 1rem; border-top: 1.5px solid var(--line); font-size: .92rem; color: var(--muted); }

  @media (max-width: 900px) {
    main {
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas: "intro" "stage" "slot";
      row-gap: 1rem;
    }
    .stage { padding: 36px 38px; }
    .card { max-width: none; }
  }
</style>
</head>
<body>
<div class="page">

  <header class="top">
    <a class="brand" href="/homepage" aria-label="leafLENS home">
      <svg viewBox="0 0 64 64" aria-hidden="true">
        <circle cx="32" cy="32" r="27" fill="none" stroke="currentColor" stroke-width="6"/>
        <path d="M32 13c13 9 15 24 0 39C17 37 19 22 32 13z" fill="#cfe86a" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/>
        <path d="M32 20v30" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
      </svg>
      <span class="wordmark"><span class="wm-leaf">leaf</span><span class="wm-lens">LENS</span></span>
    </a>
    <a class="docs" href="/docs">API docs</a>
  </header>

  <main>
    <section class="intro">
      <h1>Is this leaf healthy?</h1>
    </section>

    <section class="stage" id="stage">
      <div class="lens-wrap">
        <div class="ticks" aria-hidden="true"></div>
        <button type="button" class="lens" id="lens" aria-label="Choose a leaf photo">
          <span class="empty" id="empty">
            <svg viewBox="0 0 120 120" fill="none" aria-hidden="true">
              <path d="M60 8C94 28 102 68 60 112 18 68 26 28 60 8Z" fill="#cfe86a" fill-opacity=".55" stroke="#14382c" stroke-width="3" stroke-linejoin="round"/>
              <path d="M60 22V112M60 46L42 35M60 46L78 35M60 68L38 53M60 68L82 53M60 89L43 75M60 89L77 75" stroke="#14382c" stroke-width="2.6" stroke-linecap="round"/>
            </svg>
            <span class="empty-title">Drop a leaf photo</span>
            <span class="empty-sub">or click to browse</span>
          </span>
          <img class="preview" id="preview" alt="Selected leaf photo" hidden>
          <span class="scan" aria-hidden="true"></span>
        </button>
        <input type="file" id="file" accept="image/*" hidden>
      </div>

      <div class="actions">
        <button type="button" class="btn btn-primary" id="analyze" disabled>Analyze leaf</button>
        <button type="button" class="btn btn-ghost" id="change" hidden>Change photo</button>
      </div>
    </section>

    <section class="slot" id="slot" aria-live="polite">
    </section>
  </main>

</div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), lens =$('lens'), input = $('file'), preview =$('preview'),
        empty = $('empty'), analyzeBtn =$('analyze'), changeBtn = $('change'), slot =$('slot');

  const LABELS = {
    'Pepper__bell___Bacterial_spot': ['Bell pepper', 'Bacterial spot'],
    'Pepper__bell___healthy': ['Bell pepper', 'Healthy leaf'],
    'Potato___Early_blight': ['Potato', 'Early blight'],
    'Potato___Late_blight': ['Potato', 'Late blight'],
    'Potato___healthy': ['Potato', 'Healthy leaf'],
    'Tomato_Bacterial_spot': ['Tomato', 'Bacterial spot'],
    'Tomato_Early_blight': ['Tomato', 'Early blight'],
    'Tomato_Late_blight': ['Tomato', 'Late blight'],
    'Tomato_Leaf_Mold': ['Tomato', 'Leaf mold'],
    'Tomato_Septoria_leaf_spot': ['Tomato', 'Septoria leaf spot'],
    'Tomato_Spider_mites_Two_spotted_spider_mite': ['Tomato', 'Two-spotted spider mites'],
    'Tomato__Target_Spot': ['Tomato', 'Target spot'],
    'Tomato__Tomato_YellowLeaf__Curl_Virus': ['Tomato', 'Yellow leaf curl virus'],
    'Tomato__Tomato_mosaic_virus': ['Tomato', 'Mosaic virus'],
    'Tomato_healthy': ['Tomato', 'Healthy leaf']
  };

  let currentFile = null, previewUrl = null, busy = false, analyzed = false;

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function describe(raw) {
    if (LABELS[raw]) {
      const [crop, cond] = LABELS[raw];
      return { crop, cond, healthy: cond === 'Healthy leaf' };
    }
    const clean = String(raw).replace(/_+/g, ' ').trim();
    return { crop: '', cond: clean, healthy: /healthy/i.test(clean) };
  }

  function updateButtons() {
    analyzeBtn.disabled = !currentFile || busy || analyzed;
    changeBtn.disabled = busy;
    changeBtn.hidden = !currentFile;
    lens.disabled = busy;
  }

  function setFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      showMessage('unknown', 'Not an image', "That file isn't an image", 'Choose a JPG or PNG photo of a leaf.');
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    currentFile = file;
    analyzed = false;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    empty.hidden = true;
    lens.classList.add('has-image');
    lens.setAttribute('aria-label', 'Change leaf photo');
    slot.innerHTML = '';
    updateButtons();
  }

  function showMessage(tone, pill, title, detail, extra = '') {
    slot.innerHTML =
      `<div class="card" data-tone="${tone}">
         <span class="pill">${esc(pill)}</span>
         <p class="verdict">${esc(title)}</p>
         <p class="detail">${esc(detail)}</p>${extra}
        </div>`;
  }

  function showLoading() {
    slot.innerHTML =
      `<div class="card is-loading" data-tone="unknown">
         <span class="pill">Analyzing</span>
         <p class="verdict">Reading the leaf&hellip;</p>
         <p class="detail">Checking it against 15 conditions.</p>
        </div>`;
  }

  function showResult(data) {
    const conf = Math.max(0, Math.min(100, Number(data.confidence) || 0));
    const confText = conf.toFixed(1) + '%';

    if (/^unrecognized/i.test(String(data.prediction))) {
      showMessage(
        'unknown', 'Not recognized', "Couldn't identify this leaf",
        `The model wasn't sure enough to answer (confidence ${confText}). Try a closer, sharper photo of a single tomato, potato or bell pepper leaf.`
      );
      return;
    }

    const d = describe(data.prediction);
    const tone = d.healthy ? 'healthy' : 'issue';
    slot.innerHTML =
      `<div class="card" data-tone="${tone}">
         <span class="pill">${d.healthy ? 'Healthy' : 'Issue detected'}</span>
         ${d.crop ? `<p class="crop">${esc(d.crop)}</p>` : ''}
         <p class="verdict">${esc(d.cond)}</p>
         <div class="conf">
           <div class="conf-row"><span>Confidence</span><strong>${confText}</strong></div>
           <div class="bar" role="img" aria-label="Confidence ${confText}"><span></span></div>
         </div>
         <p class="note">A model made this prediction, so it can be wrong. Check with a local agricultural expert before treating your plants.</p>
        </div>`;
    const fill = slot.querySelector('.bar span');
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = conf + '%'; }));
  }

  async function analyze() {
    if (!currentFile || busy) return;
    busy = true;
    stage.classList.add('is-scanning');
    updateButtons();
    showLoading();

    const body = new FormData();
    body.append('file', currentFile);

    try {
      const [res] = await Promise.all([fetch('/upload', { method: 'POST', body }), wait(1000)]);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      showResult(data);
      analyzed = true;
    } catch (err) {
      showMessage('unknown', 'Error', 'Something went wrong',
        "The server couldn't analyze this image. Check that it's a JPG or PNG and that the app is running, then try again.");
    } finally {
      busy = false;
      stage.classList.remove('is-scanning');
      updateButtons();
    }
  }

  lens.addEventListener('click', () => input.click());
  changeBtn.addEventListener('click', () => input.click());
  analyzeBtn.addEventListener('click', analyze);
  input.addEventListener('change', () => { setFile(input.files[0]); input.value = ''; });

  ['dragenter', 'dragover'].forEach((t) => lens.addEventListener(t, (e) => { e.preventDefault(); lens.classList.add('is-drag'); }));
  ['dragleave', 'drop'].forEach((t) => lens.addEventListener(t, (e) => { e.preventDefault(); lens.classList.remove('is-drag'); }));
  lens.addEventListener('drop', (e) => { if (!busy) setFile(e.dataTransfer.files[0]); });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  updateButtons();
})();
</script>
</body>
</html>
        """

@app.post('/upload')
async def upload_img(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    return {
        'prediction': predicted_class,
        'confidence': conf_score
    }
