async function postJson(path, payload) {
  return await api(path, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

async function redrawWithState(nextState, snap, forceSignature = false) {
  state = nextState || await api(stateUrl());
  pruneSelected();
  draw(false);
  if (forceSignature) lastRenderSignature = renderSignature();
  restoreScrollState(snap);
}

async function scrubShot(manifest, index, time) {
  const result = await postJson('/api/shot-scrub', { manifest, index, time });
  if (!result.ok) return alert(result.error || 'Could not update shot frame');

  state = await api(stateUrl());
  pruneSelected();
  refreshShotRows('references', [index]);
  updateRunLogs();
  lastRenderSignature = renderSignature();
}

async function refreshReferenceRowFromState(nextState, index) {
  state = nextState || await api(stateUrl());
  pruneSelected();
  refreshShotRows('references', [index]);
  updateRunLogs();
  lastRenderSignature = renderSignature();
}

async function saveShotPrompt(manifest, index, prompt) {
  const result = await postJson('/api/shot-prompt', { manifest, index, prompt });
  if (!result.ok) return alert(result.error || 'Could not save prompt');

  state = await api(stateUrl());
}

async function saveOutpaintChunk(index) {
  const snap = captureScrollState();
  const payload = outpaintChunkForm(index);
  const result = await postJson('/api/outpaint-chunk', payload);
  if (!result.ok) return alert(result.error || 'Could not save chunk');

  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
    restoreScrollState(snap);
    return;
  }

  await redrawWithState(result.state, snap);
}

function outpaintChunkForm(index) {
  const customCheckbox = document.getElementById(`chunkCustom_${index}`);
  return {
    index,
    seed: document.getElementById(`chunkSeed_${index}`).value,
    custom_length: !!(customCheckbox && customCheckbox.checked),
    custom_seconds: outpaintChunkCustomSeconds(index),
    offset_x: document.getElementById(`chunkOffset_x_${index}`)?.value || '0',
    offset_y: document.getElementById(`chunkOffset_y_${index}`)?.value || '0',
    prompt_suffix: document.getElementById(`chunkPrompt_${index}`).value,
    negative_suffix: document.getElementById(`chunkNegative_${index}`).value,
  };
}

function openImageModal(src, title) {
  let modal = document.getElementById('imageModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.className = 'image-modal hidden';
    modal.innerHTML = `
      <div class="image-modal-backdrop" onclick="closeImageModal()"></div>
      <div class="image-modal-panel">
        <div class="image-modal-heading">
          <strong id="imageModalTitle"></strong>
          <button type="button" onclick="closeImageModal()" aria-label="Close image preview">Close</button>
        </div>
        <img id="imageModalImg" alt="">
      </div>
    `;
    document.body.appendChild(modal);
  }

  document.getElementById('imageModalTitle').textContent = title || 'Image preview';
  document.getElementById('imageModalImg').src = src;
  modal.classList.remove('hidden');
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  if (!modal) return;
  modal.classList.add('hidden');
  const img = document.getElementById('imageModalImg');
  if (img) img.removeAttribute('src');
}

const referenceEditor = {
  mode: 'reference',
  manifest: '',
  index: -1,
  chunkIndex: -1,
  guideIndex: -1,
  row: null,
  guide: null,
  guideSourcePath: '',
  tool: 'brush-add',
  brushSize: 28,
  sampledColor: '',
  samPoints: [],
  drawing: false,
  preview: '',
  previewPollTimer: null,
  previewPollStartedAt: 0,
};

function openReferenceEditor(manifest, index) {
  const rows = (state.shot_views && state.shot_views.references) || [];
  const row = rows[index];
  if (!row || !row.color_reference) return;
  referenceEditor.mode = 'reference';
  referenceEditor.manifest = manifest;
  referenceEditor.index = index;
  referenceEditor.chunkIndex = -1;
  referenceEditor.guideIndex = -1;
  referenceEditor.row = row;
  referenceEditor.guide = null;
  referenceEditor.guideSourcePath = '';
  referenceEditor.preview = '';
  referenceEditor.samPoints = [];
  ensureReferenceEditorModal();
  document.getElementById('referenceEditTitle').textContent = `Shot ${index + 1} Reference Editor`;
  document.getElementById('referenceEditInstruction').value = row.prompt || '';
  document.getElementById('referenceEditSample').textContent = 'No colour sampled';
  document.getElementById('referenceEditPreview').innerHTML = missingImage('No preview yet');
  document.getElementById('referenceEditRecent').innerHTML = referenceRecentHtml(row);
  document.getElementById('referenceEditModal').classList.remove('hidden');
  setReferenceTool(referenceEditor.tool);
  loadReferenceEditorImage(media(row.color_reference) + '&t=' + (row.color_reference_mtime || Date.now()));
}

async function openGuideEditor(chunkIndex, guideIndex, fallbackPath = '') {
  const rows = (state.outpaint_chunks && state.outpaint_chunks.rows) || [];
  const row = rows.find(item => Number(item.index) === Number(chunkIndex));
  const guide = row && (row.guides || [])[guideIndex];
  if (!row || !guide) return;
  let srcPath = guide.image_exists ? guide.image : (guide.source_preview || fallbackPath);
  if (!srcPath) {
    const frameIdx = Number(guide.frame_idx || 0);
    const result = await api(`/api/outpaint-guide-preview?chunk_index=${chunkIndex}&frame_idx=${frameIdx}`);
    srcPath = (result && result.preview) || guide.image;
    if (result && result.preview) guide.source_preview = result.preview;
  }
  if (!srcPath) return alert('Guide preview is still loading.');
  referenceEditor.mode = 'guide';
  referenceEditor.manifest = '';
  referenceEditor.index = -1;
  referenceEditor.chunkIndex = chunkIndex;
  referenceEditor.guideIndex = guideIndex;
  referenceEditor.row = row;
  referenceEditor.guide = guide;
  referenceEditor.guideSourcePath = srcPath;
  referenceEditor.preview = '';
  referenceEditor.samPoints = [];
  ensureReferenceEditorModal();
  document.getElementById('referenceEditTitle').textContent = `Chunk ${chunkIndex + 1} Guide ${guideIndex + 1} Editor`;
  document.getElementById('referenceEditInstruction').value = DEFAULT_ANCHOR_PROMPT;
  document.getElementById('referenceEditSample').textContent = 'No colour sampled';
  document.getElementById('referenceEditPreview').innerHTML = missingImage('No preview yet');
  document.getElementById('referenceEditRecent').innerHTML = guideRecentHtml(row, guideIndex);
  document.getElementById('referenceEditModal').classList.remove('hidden');
  setReferenceTool(referenceEditor.tool);
  loadReferenceEditorImage(media(srcPath) + '&t=' + (guide.image_mtime || Date.now()));
}

function referenceIcon(name) {
  const common = 'viewBox="0 0 24 24" aria-hidden="true" focusable="false"';
  const plus = '<path d="M18 15v6M15 18h6"/>';
  const minus = '<path d="M15 18h6"/>';
  const icons = {
    'sam-add': `<svg ${common}><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/><circle cx="12" cy="12" r="5"/><path d="M9.5 12h5M12 9.5v5"/>${plus}</svg>`,
    'sam-subtract': `<svg ${common}><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/><circle cx="12" cy="12" r="5"/><path d="M9.5 12h5"/>${minus}</svg>`,
    'brush-add': `<svg ${common}><path d="M14 4l6 6-8.5 8.5c-1.2 1.2-3.1 1.2-4.2 0l-1.8-1.8c-1.2-1.2-1.2-3.1 0-4.2L14 4z"/><path d="M13 5l6 6"/><path d="M4 20c1.7.2 3-.2 4-1.2"/>${plus}</svg>`,
    'brush-subtract': `<svg ${common}><path d="M14 4l6 6-8.5 8.5c-1.2 1.2-3.1 1.2-4.2 0l-1.8-1.8c-1.2-1.2-1.2-3.1 0-4.2L14 4z"/><path d="M13 5l6 6"/><path d="M4 20c1.7.2 3-.2 4-1.2"/>${minus}</svg>`,
    wand: `<svg ${common}><path d="M4 20l10-10"/><path d="M13 5l1 3 3 1-3 1-1 3-1-3-3-1 3-1 1-3z"/><path d="M18 3v3M20 5h-4M6 5v2M7 6H5"/></svg>`,
    dropper: `<svg ${common}><path d="M14 5l5 5"/><path d="M11 8l5 5-7.5 7.5H4v-4.5L11 8z"/><path d="M13 6l2-2c.8-.8 2.2-.8 3 0l2 2c.8.8.8 2.2 0 3l-2 2"/></svg>`,
    clear: `<svg ${common}><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></svg>`,
    invert: `<svg ${common}><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z"/></svg>`,
  };
  return icons[name] || '';
}

function referenceToolButton(tool, label) {
  return `<button class="reference-tool-button" type="button" data-tool="${tool}" title="${label}" aria-label="${label}" onclick="setReferenceTool('${tool}')">${referenceIcon(tool)}<span class="sr-only">${label}</span></button>`;
}

function referenceMaskActionButton(action, icon, label) {
  return `<button class="reference-tool-button" type="button" title="${label}" aria-label="${label}" onclick="${action}">${referenceIcon(icon)}<span class="sr-only">${label}</span></button>`;
}

function ensureReferenceEditorModal() {
  if (document.getElementById('referenceEditModal')) return;
  const modal = document.createElement('div');
  modal.id = 'referenceEditModal';
  modal.className = 'image-modal hidden';
  modal.innerHTML = `
    <div class="image-modal-backdrop" onclick="closeReferenceEditor()"></div>
    <div class="reference-editor-panel">
      <div class="image-modal-heading">
        <strong id="referenceEditTitle"></strong>
        <button type="button" onclick="closeReferenceEditor()">Close</button>
      </div>
      <div class="reference-editor-layout">
        <div class="reference-canvas-wrap">
          <canvas id="referenceImageCanvas"></canvas>
          <canvas id="referenceMaskCanvas"></canvas>
        </div>
        <aside class="reference-editor-tools">
          <label>Instruction</label>
          <textarea id="referenceEditInstruction" placeholder="make the selected coat green"></textarea>
          <div class="reference-tool-grid">
            ${referenceToolButton('sam-add', 'SAM2 add to mask')}
            ${referenceToolButton('brush-add', 'Brush add to mask')}
            ${referenceToolButton('wand', 'Magic wand selection')}
            ${referenceToolButton('sam-subtract', 'SAM2 subtract from mask')}
            ${referenceToolButton('brush-subtract', 'Brush subtract from mask')}
            ${referenceToolButton('dropper', 'Sample colour')}
          </div>
          <label>Brush size</label>
          <input id="referenceBrushSize" type="range" min="4" max="120" value="28" oninput="referenceEditor.brushSize=Number(this.value)">
          <div class="reference-mask-actions">
            ${referenceMaskActionButton('clearReferenceMask()', 'clear', 'Clear mask')}
            ${referenceMaskActionButton('invertReferenceMask()', 'invert', 'Invert mask')}
          </div>
          <div class="sample-row"><span id="referenceSampleSwatch"></span><span id="referenceEditSample">No colour sampled</span></div>
          <label>Recent reference images</label>
          <div id="referenceEditRecent" class="recent-reference-strip"></div>
          <div class="actions">
            <button class="primary" type="button" onclick="submitReferenceEditPreview()">Apply with Qwen</button>
            <button type="button" onclick="acceptReferenceEditPreview()">Accept</button>
            <button type="button" onclick="revertReferenceEdit()">Revert</button>
          </div>
          <div id="referenceEditStatus" class="shot-empty"></div>
          <label>Preview</label>
          <div id="referenceEditPreview"></div>
        </aside>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  wireReferenceEditorCanvas();
}

function referenceRecentHtml(row) {
  const paths = row.recent_color_references || [];
  if (!paths.length) return '<p class="shot-empty">No nearby colour references yet.</p>';
  return paths.map(path => `<img src="${media(path)}" alt="" title="${esc(path)}" onclick="sampleReferenceImageColor(event,this)">`).join('');
}

function guideRecentHtml(row, guideIndex) {
  const guides = (row.guides || []).filter((guide, index) => index !== guideIndex);
  const paths = guides.map(guide => guide.image_exists ? guide.image : guide.source_preview).filter(Boolean);
  if (!paths.length) return '<p class="shot-empty">No nearby guide previews yet.</p>';
  return paths.map(path => `<img src="${media(path)}" alt="" title="${esc(path)}" onclick="sampleReferenceImageColor(event,this)">`).join('');
}

function closeReferenceEditor() {
  stopReferencePreviewPolling();
  const modal = document.getElementById('referenceEditModal');
  if (modal) modal.classList.add('hidden');
}

function stopReferencePreviewPolling() {
  if (referenceEditor.previewPollTimer) {
    clearTimeout(referenceEditor.previewPollTimer);
    referenceEditor.previewPollTimer = null;
  }
}

function loadReferenceEditorImage(src) {
  const img = new Image();
  img.onload = () => {
    const imageCanvas = document.getElementById('referenceImageCanvas');
    const maskCanvas = document.getElementById('referenceMaskCanvas');
    const maxSide = 1400;
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * scale));
    const h = Math.max(1, Math.round(img.naturalHeight * scale));
    for (const canvas of [imageCanvas, maskCanvas]) {
      canvas.width = w;
      canvas.height = h;
      canvas.style.aspectRatio = `${w}/${h}`;
    }
    imageCanvas.getContext('2d').drawImage(img, 0, 0, w, h);
    clearReferenceMask();
  };
  img.src = src;
}

function wireReferenceEditorCanvas() {
  const wrap = document.querySelector('.reference-canvas-wrap');
  wrap.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    referenceEditor.drawing = true;
    handleReferenceCanvasPoint(event);
  });
  wrap.addEventListener('pointermove', event => {
    if (!referenceEditor.drawing) return;
    if ((event.buttons & 1) === 0) return;
    if (!referenceEditor.tool.startsWith('brush')) return;
    handleReferenceCanvasPoint(event);
  });
  window.addEventListener('pointerup', () => { referenceEditor.drawing = false; });
}

function setReferenceTool(tool) {
  referenceEditor.tool = tool;
  document.querySelectorAll('.reference-tool-button[data-tool]').forEach(button => {
    button.classList.toggle('active', button.dataset.tool === tool);
  });
  const labels = {
    'sam-add': 'SAM2 add to mask',
    'sam-subtract': 'SAM2 subtract from mask',
    'brush-add': 'Brush add to mask',
    'brush-subtract': 'Brush subtract from mask',
    wand: 'Magic wand selection',
    dropper: 'Sample colour',
  };
  const status = document.getElementById('referenceEditStatus');
  if (status) status.textContent = `Tool: ${labels[tool] || tool}`;
}

function referenceCanvasPoint(event) {
  const canvas = document.getElementById('referenceMaskCanvas');
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)),
  };
}

function handleReferenceCanvasPoint(event) {
  const point = referenceCanvasPoint(event);
  if (referenceEditor.tool === 'dropper') return sampleActiveReferenceColor(point.x, point.y);
  if (referenceEditor.tool === 'wand') return wandReferenceMask(point.x, point.y, 34);
  if (referenceEditor.tool === 'sam-add' || referenceEditor.tool === 'sam-subtract') return requestReferenceSamMask(point);
  drawReferenceBrush(point.x, point.y, referenceEditor.tool === 'brush-subtract');
}

function drawReferenceBrush(x, y, subtract) {
  const ctx = document.getElementById('referenceMaskCanvas').getContext('2d');
  ctx.save();
  ctx.globalCompositeOperation = subtract ? 'destination-out' : 'source-over';
  ctx.fillStyle = 'rgba(45,143,125,.58)';
  ctx.beginPath();
  ctx.arc(x, y, referenceEditor.brushSize / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function clearReferenceMask() {
  const canvas = document.getElementById('referenceMaskCanvas');
  if (!canvas) return;
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function invertReferenceMask() {
  const canvas = document.getElementById('referenceMaskCanvas');
  const ctx = canvas.getContext('2d');
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < data.data.length; i += 4) {
    const on = data.data[i + 3] > 0;
    data.data[i] = 45;
    data.data[i + 1] = 143;
    data.data[i + 2] = 125;
    data.data[i + 3] = on ? 0 : 148;
  }
  ctx.putImageData(data, 0, 0);
}

function sampleActiveReferenceColor(x, y) {
  const canvas = document.getElementById('referenceImageCanvas');
  const data = canvas.getContext('2d').getImageData(Math.floor(x), Math.floor(y), 1, 1).data;
  setReferenceSample(rgbToHex(data[0], data[1], data[2]));
}

function sampleReferenceImageColor(event, img) {
  const rect = img.getBoundingClientRect();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');
  try {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const x = Math.floor((event.clientX - rect.left) * canvas.width / rect.width);
    const y = Math.floor((event.clientY - rect.top) * canvas.height / rect.height);
    const data = ctx.getImageData(x, y, 1, 1).data;
    setReferenceSample(rgbToHex(data[0], data[1], data[2]));
  } catch {
    document.getElementById('referenceEditStatus').textContent = 'Could not sample that image.';
  }
}

function setReferenceSample(hex) {
  referenceEditor.sampledColor = hex;
  document.getElementById('referenceEditSample').textContent = hex;
  document.getElementById('referenceSampleSwatch').style.background = hex;
  const instruction = document.getElementById('referenceEditInstruction');
  if (instruction && !instruction.value.includes(hex)) {
    instruction.value = `${instruction.value.trim()} use ${hex}`.trim();
  }
}

function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(v => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0')).join('');
}

function wandReferenceMask(x, y, tolerance = 34) {
  const image = document.getElementById('referenceImageCanvas');
  const mask = document.getElementById('referenceMaskCanvas');
  const imageCtx = image.getContext('2d');
  const maskCtx = mask.getContext('2d');
  const pixels = imageCtx.getImageData(0, 0, image.width, image.height);
  const out = maskCtx.getImageData(0, 0, mask.width, mask.height);
  const sx = Math.floor(x), sy = Math.floor(y);
  const idx = (sy * image.width + sx) * 4;
  const target = [pixels.data[idx], pixels.data[idx + 1], pixels.data[idx + 2]];
  const seen = new Uint8Array(image.width * image.height);
  const stack = [[sx, sy]];
  while (stack.length) {
    const [cx, cy] = stack.pop();
    if (cx < 0 || cy < 0 || cx >= image.width || cy >= image.height) continue;
    const pos = cy * image.width + cx;
    if (seen[pos]) continue;
    seen[pos] = 1;
    const p = pos * 4;
    const d = Math.abs(pixels.data[p] - target[0]) + Math.abs(pixels.data[p + 1] - target[1]) + Math.abs(pixels.data[p + 2] - target[2]);
    if (d > tolerance * 3) continue;
    out.data[p] = 45; out.data[p + 1] = 143; out.data[p + 2] = 125; out.data[p + 3] = 148;
    stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
  }
  maskCtx.putImageData(out, 0, 0);
}

async function requestReferenceSamMask(point) {
  referenceEditor.samPoints.push({ x: point.x, y: point.y, label: referenceEditor.tool === 'sam-subtract' ? 'subtract' : 'add' });
  const canvas = document.getElementById('referenceMaskCanvas');
  const payload = {
    points: referenceEditor.samPoints,
    width: canvas.width,
    height: canvas.height,
  };
  const result = referenceEditor.mode === 'guide'
    ? await postJson('/api/guide-frame-mask-sam', {
        ...payload,
        chunk_index: referenceEditor.chunkIndex,
        guide_index: referenceEditor.guideIndex,
        fallback_path: referenceEditor.guideSourcePath,
      })
    : await postJson('/api/reference-mask-sam', {
        ...payload,
        manifest: referenceEditor.manifest,
        index: referenceEditor.index,
      });
  if (!result.ok) return alert(result.error || 'Smart mask failed');
  const img = new Image();
  img.onload = () => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 0.58;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
  };
  img.src = result.mask;
  document.getElementById('referenceEditStatus').textContent = `Smart select: ${result.provider || 'local region'}`;
}

function referenceMaskHasPixels() {
  const canvas = document.getElementById('referenceMaskCanvas');
  const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return true;
  }
  return false;
}

function referenceMaskDataUrl() {
  if (!referenceMaskHasPixels()) return '';
  const src = document.getElementById('referenceMaskCanvas');
  const out = document.createElement('canvas');
  out.width = src.width;
  out.height = src.height;
  const srcData = src.getContext('2d').getImageData(0, 0, src.width, src.height);
  const outCtx = out.getContext('2d');
  const data = outCtx.createImageData(out.width, out.height);
  for (let i = 0; i < srcData.data.length; i += 4) {
    const value = srcData.data[i + 3] > 0 ? 255 : 0;
    data.data[i] = value;
    data.data[i + 1] = value;
    data.data[i + 2] = value;
    data.data[i + 3] = 255;
  }
  outCtx.putImageData(data, 0, 0);
  return out.toDataURL('image/png');
}

async function submitReferenceEditPreview() {
  const status = document.getElementById('referenceEditStatus');
  status.textContent = 'Starting Qwen edit preview...';
  stopReferencePreviewPolling();
  const mask = referenceMaskDataUrl();
  const payload = {
    instruction: document.getElementById('referenceEditInstruction')?.value || '',
    sampled_color: referenceEditor.sampledColor,
    mask,
  };
  const result = referenceEditor.mode === 'guide'
    ? await postJson('/api/guide-frame-edit-preview', {
        ...payload,
        chunk_index: referenceEditor.chunkIndex,
        guide_index: referenceEditor.guideIndex,
      })
    : await postJson('/api/reference-edit-preview', {
        ...payload,
        manifest: referenceEditor.manifest,
        index: referenceEditor.index,
      });
  if (!result.ok) {
    status.textContent = result.error || result.message || 'Could not start reference edit';
    return;
  }
  referenceEditor.preview = result.preview || '';
  status.textContent = result.message || 'Reference edit started.';
  if (referenceEditor.preview) {
    startReferencePreviewPolling(referenceEditor.preview);
  }
  setTimeout(() => refresh(true), 1000);
}

function startReferencePreviewPolling(path) {
  const preview = document.getElementById('referenceEditPreview');
  const status = document.getElementById('referenceEditStatus');
  if (preview) preview.innerHTML = '<p class="shot-empty">Qwen preview is rendering...</p>';
  if (status) status.textContent = 'Qwen preview is rendering...';
  referenceEditor.previewPollStartedAt = Date.now();
  pollReferencePreview(path);
}

async function pollReferencePreview(path) {
  stopReferencePreviewPolling();
  if (!path || path !== referenceEditor.preview) return;
  const status = document.getElementById('referenceEditStatus');
  const preview = document.getElementById('referenceEditPreview');
  let result = null;
  try {
    result = await api('/api/media-status?path=' + encodeURIComponent(path));
  } catch {
    result = { ok: false };
  }
  if (path !== referenceEditor.preview) return;
  if (result && result.ok && result.exists) {
    if (preview) preview.innerHTML = `<img src="${media(path)}&t=${Date.now()}" alt="">`;
    if (status) status.textContent = 'Qwen preview ready.';
    refresh(true);
    return;
  }
  const elapsed = Date.now() - referenceEditor.previewPollStartedAt;
  if (result && result.ok && !result.running && elapsed > 5000) {
    if (preview) preview.innerHTML = '<p class="shot-empty">Qwen finished, but ARP could not find the preview image. Check the run log for details.</p>';
    if (status) status.textContent = 'Preview image was not found after Qwen finished.';
    refresh(true);
    return;
  }
  if (elapsed > 20 * 60 * 1000) {
    if (preview) preview.innerHTML = '<p class="shot-empty">Still waiting for the preview image. Check ComfyUI and the run log.</p>';
    if (status) status.textContent = 'Still waiting for Qwen preview.';
    return;
  }
  if (status) status.textContent = result && result.running ? `Qwen preview is rendering: ${result.running_stage || 'running'}...` : 'Waiting for Qwen preview image...';
  referenceEditor.previewPollTimer = setTimeout(() => pollReferencePreview(path), 1500);
}

async function acceptReferenceEditPreview() {
  if (!referenceEditor.preview) return alert('Generate a preview first.');
  const result = referenceEditor.mode === 'guide'
    ? await postJson('/api/guide-frame-edit-accept', {
        chunk_index: referenceEditor.chunkIndex,
        guide_index: referenceEditor.guideIndex,
        preview: referenceEditor.preview,
      })
    : await postJson('/api/reference-edit-accept', {
        manifest: referenceEditor.manifest,
        index: referenceEditor.index,
        preview: referenceEditor.preview,
      });
  if (!result.ok) return alert(result.error || 'Could not accept edit');
  state = result.state || await api(stateUrl());
  closeReferenceEditor();
  draw(false);
  lastRenderSignature = renderSignature();
}

async function revertReferenceEdit() {
  const result = referenceEditor.mode === 'guide'
    ? await postJson('/api/guide-frame-edit-revert', {
        chunk_index: referenceEditor.chunkIndex,
        guide_index: referenceEditor.guideIndex,
      })
    : await postJson('/api/reference-edit-revert', {
        manifest: referenceEditor.manifest,
        index: referenceEditor.index,
      });
  if (!result.ok) return alert(result.error || 'Could not revert edit');
  state = result.state || await api(stateUrl());
  closeReferenceEditor();
  draw(false);
  lastRenderSignature = renderSignature();
}

function outpaintChunkCustomSeconds(index) {
  const checkbox = document.getElementById(`chunkCustom_${index}`);
  const slider = document.getElementById(`chunkFrames_${index}`);
  if (!(checkbox && checkbox.checked && slider)) return '';
  const fps = Math.max(1, Number(slider.dataset.fps || 24));
  return (Math.max(1, Number(slider.value) || 1) / fps).toFixed(6);
}

function releaseChunkMedia(index) {
  const card = document.querySelectorAll('.chunk-card')[index];
  if (!card) return;

  card.querySelectorAll('video').forEach(video => {
    try {
      video.pause();
      video.removeAttribute('src');
      video.load();
    } catch {}
  });
}

async function regenerateOutpaintChunk(index) {
  releaseChunkMedia(index);
  await saveStage('outpaint');

  const result = await postJson('/api/outpaint-chunk-regenerate', outpaintChunkForm(index));
  if (!result.ok) return alert(result.error || result.message || 'Could not regenerate chunk');

  state = result.state;
  draw(false);
  setTimeout(() => refresh(true), 500);
}

async function saveShotEnabled(manifest, index, enabled) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-enabled', { manifest, index, enabled });
  if (!result.ok) return alert(result.error || 'Could not save shot setting');

  await redrawWithState(null, snap);
}

async function mergeShot(manifest, index) {
  if (!confirm('Merge this shot with the next one and use the same reference?')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/shot-merge', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not merge shots');

  await redrawWithState(result.state, snap, true);
}

async function splitShot(manifest, index) {
  if (!confirm('Split this shot at its midpoint? Existing generated references for the two halves will be cleared.')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/shot-split', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not split shot');

  await redrawWithState(result.state, snap, true);
}

async function setShotBoundary(manifest, index, edge, time) {
  const result = await postJson('/api/shot-boundary', { manifest, index, edge, time });
  if (!result.ok) return alert(result.error || 'Could not update shot boundary');

  state = result.state || await api(stateUrl());
  pruneSelected();
  refreshShotRows('shots', [index, edge === 'start' ? index - 1 : index + 1]);
  updateRunLogs();
  lastRenderSignature = renderSignature();
}

async function saveShotFade(manifest, index, enabled, crossfade_seconds) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-fade', { manifest, index, enabled, crossfade_seconds });
  if (!result.ok) return alert(result.error || 'Could not update fade transition');

  await redrawWithState(result.state, snap, true);
}

function nudgeShotBoundary(manifest, index, edge, frames) {
  const rows = (state.shot_views && state.shot_views.shots) || [];
  const row = rows[index];
  if (!row) return;

  const frameCount = Number(row.end_frame) - Number(row.start_frame) + 1;
  const duration = Math.max(0.001, Number(row.end) - Number(row.start));
  const fps = Math.max(1, frameCount / duration);
  const base = edge === 'start' ? Number(row.start) : Number(row.end);
  setShotBoundary(manifest, index, edge, base + (Number(frames) || 0) / fps);
}

const previewTimers = {};

function updateShotPreview(manifest, index, time, imgId, labelId) {
  document.getElementById(labelId).textContent = formatSeconds(time);
  clearTimeout(previewTimers[imgId]);

  previewTimers[imgId] = setTimeout(async () => {
    const query = '?manifest=' + encodeURIComponent(manifest)
      + '&index=' + index
      + '&time=' + encodeURIComponent(time);
    const result = await api('/api/shot-preview' + query);
    const img = document.getElementById(imgId);
    if (result.ok && result.path && img) img.src = media(result.path);
  }, 180);
}

function updateShotBoundaryPreview(manifest, index, time, imgId, labelId, dataset) {
  const label = document.getElementById(labelId);
  const edge = dataset && dataset.edge === 'end' ? 'End' : 'Start';
  const fps = Math.max(1, Number((dataset && dataset.fps) || 24));
  const frame = Math.max(0, Math.round(Number(time || 0) * fps) - (edge === 'End' ? 1 : 0));
  if (label) label.textContent = `${edge} frame ${frame}`;

  clearTimeout(previewTimers[imgId]);
  previewTimers[imgId] = setTimeout(async () => {
    const previewTime = Math.max(0, Number(time || 0) + Number((dataset && dataset.previewOffset) || 0));
    const query = '?manifest=' + encodeURIComponent(manifest)
      + '&index=' + index
      + '&time=' + encodeURIComponent(previewTime);
    const result = await api('/api/shot-preview' + query);
    const img = document.getElementById(imgId);
    if (result.ok && result.path && img) img.src = media(result.path);
  }, 120);
}

async function regenerateReference(manifest, index) {
  const provider = settings('references').method || 'qwen';
  if (provider === 'openai' && !((settings('references').openai_api_key || '').trim())) {
    alert('Add your OpenAI API key in Settings before generating with OpenAI.');
    active = 'settings';
    drawTabs();
    draw();
    return;
  }
  const snap = captureScrollState();
  const result = await postJson('/api/reference-regenerate', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not regenerate reference');

  await refreshReferenceRowFromState(result.state, index);
  restoreScrollState(snap);
  setTimeout(refresh, 1000);
}

async function deleteReference(manifest, index) {
  if (!confirm('Delete this color reference? It will be regenerated next time you run Reference Generation.')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/reference-delete', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not delete reference');

  await refreshReferenceRowFromState(result.state, index);
  restoreScrollState(snap);
}

async function chooseCustomReference(manifest, index) {
  const snap = captureScrollState();
  const result = await postJson('/api/reference-custom', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not install custom reference');
  if (!result.selected) return;

  await refreshReferenceRowFromState(result.state, index);
  restoreScrollState(snap);
}

async function exportMedia(path) {
  const result = await postJson('/api/export-media', { path });
  if (!result.ok) return alert(result.error || 'Could not save media file');
  if (result.saved) alert('Saved:\n' + result.saved);
}

const DEFAULT_ANCHOR_PROMPT = 'Replace the black bars.';

// ── Guide frame list actions ─────────────────────────────────────────────────

async function autoSaveGuideFrame(chunkIndex, guideIndex) {
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  const strEl = document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`);
  if (!slider) return;
  const frame_idx = Number(slider.value);
  const strength = strEl ? Number(strEl.value) : 0.7;
  await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx, strength });
}

async function guideFrameRedraw(result, snap) {
  if (!result.ok) return;
  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
    restoreScrollState(snap);
  }
}

async function addGuideFrame(chunkIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-add', { chunk_index: chunkIndex });
  if (!result.ok) return alert(result.error || 'Could not add guide frame');
  await guideFrameRedraw(result, snap);
}

async function removeGuideFrame(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-remove', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not remove guide frame');
  await guideFrameRedraw(result, snap);
}

async function saveGuideFrameSettings(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  const strEl = document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`);
  const frame_idx = slider ? Number(slider.value) : 0;
  const strength = strEl ? Number(strEl.value) : 0.7;
  const result = await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx, strength });
  if (!result.ok) return alert(result.error || 'Could not save guide frame');
  await guideFrameRedraw(result, snap);
}

async function uploadGuideFrameImage(chunkIndex, guideIndex) {
  await autoSaveGuideFrame(chunkIndex, guideIndex);
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-upload', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not upload guide frame image');
  if (!result.selected) return;
  await guideFrameRedraw(result, snap);
}

async function clearGuideFrameImage(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-clear', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not clear guide frame image');
  await guideFrameRedraw(result, snap);
}

function openGuideFrameGenerateModal(chunkIndex, guideIndex, frameIdx) {
  let modal = document.getElementById('guideFrameGenerateModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'guideFrameGenerateModal';
    modal.className = 'image-modal hidden';
    modal.innerHTML = `
      <div class="image-modal-backdrop" onclick="closeGuideFrameGenerateModal()"></div>
      <div class="prompt-modal-panel">
        <div class="image-modal-heading">
          <strong>Generate Guide Frame Image</strong>
          <button type="button" onclick="closeGuideFrameGenerateModal()">Close</button>
        </div>
        <p class="shot-empty">Qwen will colorize the source frame at the selected position.</p>
        <label>Qwen edit prompt</label>
        <textarea id="guideFrameGeneratePrompt"></textarea>
        <div class="actions">
          <button class="primary" type="button" onclick="submitGuideFrameGenerate()">Generate</button>
          <button type="button" onclick="closeGuideFrameGenerateModal()">Cancel</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }
  modal.dataset.chunkIndex = String(chunkIndex);
  modal.dataset.guideIndex = String(guideIndex);
  modal.dataset.frameIdx = String(frameIdx);
  // Pick up current slider value in case the user changed it without saving
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  if (slider) modal.dataset.frameIdx = slider.value;
  document.getElementById('guideFrameGeneratePrompt').value = DEFAULT_ANCHOR_PROMPT;
  modal.classList.remove('hidden');
}

function closeGuideFrameGenerateModal() {
  const modal = document.getElementById('guideFrameGenerateModal');
  if (modal) modal.classList.add('hidden');
}

async function submitGuideFrameGenerate() {
  const modal = document.getElementById('guideFrameGenerateModal');
  if (!modal) return;
  const chunkIndex = Number(modal.dataset.chunkIndex || 0);
  const guideIndex = Number(modal.dataset.guideIndex || 0);
  const frameIdx = Number(modal.dataset.frameIdx || 0);
  const prompt = document.getElementById('guideFrameGeneratePrompt')?.value || DEFAULT_ANCHOR_PROMPT;
  closeGuideFrameGenerateModal();

  // Save current frame_idx and strength before generating.
  await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx: frameIdx, strength: Number(document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`)?.value || 0.7) });

  const result = await postJson('/api/guide-frame-generate', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx: frameIdx, prompt });
  if (!result.ok) return alert(result.error || result.message || 'Could not generate guide frame image');
  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
  }
}

async function saveStage(key, redraw = false) {
  const snap = captureScrollState();
  await postJson('/api/settings', { stage: key, values: formValues() });

  state = await api(stateUrl());
  pruneSelected();

  if (redraw) {
    draw(false);
    restoreScrollState(snap);
  }

  showCommand(key);
}

function formValues() {
  const values = {};
  document.querySelectorAll('[data-field]').forEach(el => {
    values[el.dataset.field] = el.type === 'checkbox' ? String(el.checked) : el.value;
  });
  return values;
}

async function saveGlobal() {
  await postJson('/api/settings', {
    stage: 'global',
    values: { source: document.getElementById('globalSource').value },
  });

  selected = {};
  state = await api(stateUrl());
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function saveGlobalPipelineOptions() {
  const snap = captureScrollState();
  await postJson('/api/settings', {
    stage: 'global',
    values: {
      expand_outpaint: String(document.getElementById('globalExpandOutpaint').checked),
      colorize: String(document.getElementById('globalColorize').checked),
      upscale: String(document.getElementById('globalUpscale').checked),
      add_soundtrack: String(document.getElementById('globalSoundtrack').checked),
    },
  });

  state = await api(stateUrl());
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  drawTabs();
  draw(false);
  restoreScrollState(snap);
}

async function saveGlobalSection() {
  await postJson('/api/settings', {
    stage: 'global',
    values: {
      section_start: document.getElementById('sectionStart')?.value || '0',
      section_end: document.getElementById('sectionEnd')?.value || '',
    },
  });
  state = await api(stateUrl());
  pruneSelected();
  updateOverviewDynamicStatus();
  lastRenderSignature = renderSignature();
}

async function autoCropOutpaint() {
  const slider = document.getElementById('aspectPreviewTime');
  const time = slider ? slider.value : '0';
  const result = await api('/api/outpaint-auto-crop?time=' + encodeURIComponent(time));
  if (!result.ok) return alert(result.error || 'Auto Crop failed');

  state = result.state || state;
  ['crop_left', 'crop_right', 'crop_top', 'crop_bottom'].forEach(key => {
    const el = document.querySelector(`[data-field="${key}"]`);
    const value = result[key] ?? settings('outpaint')[key] ?? '0';
    if (el) {
      el.value = value;
      const label = document.getElementById(`${key}Value`);
      if (label) label.textContent = value;
    }
  });

  const img = document.getElementById('aspectPreviewImg');
  if (result.preview && img) img.src = media(result.preview) + '&t=' + Date.now();
  showCommand('outpaint');
  lastRenderSignature = renderSignature();
  lastOutpaintVisualSignature = outpaintVisualSignature();
}

async function nudgeSectionBoundary(edge, frames) {
  const id = edge === 'start' ? 'sectionStart' : 'sectionEnd';
  const labelId = edge === 'start' ? 'sectionStartLabel' : 'sectionEndLabel';
  const slider = document.getElementById(id);
  if (!slider) return;
  const fps = Math.max(1, Number(slider.dataset.fps || 24));
  const step = 1 / fps;
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 0);
  const current = Number(slider.value || 0);
  slider.value = Math.min(max, Math.max(min, current + frames * step)).toFixed(6);
  const label = document.getElementById(labelId);
  if (label) label.textContent = formatSeconds(slider.value);
  await saveGlobalSection();
}

async function markSourceSection(edge) {
  const video = document.getElementById('sourceSectionVideo');
  const target = document.getElementById(edge === 'start' ? 'sectionStart' : 'sectionEnd');
  if (!video || !target) return;

  target.value = Math.max(0, video.currentTime || 0).toFixed(3);
  const label = document.getElementById(edge === 'start' ? 'sectionStartLabel' : 'sectionEndLabel');
  if (label) label.textContent = formatSeconds(target.value);

  await saveGlobalSection();
}

async function browseGlobalSource() {
  const el = document.getElementById('globalSource');
  const result = await postJson('/api/browse-global-source', { current: el.value });
  if (!result.ok) return alert(result.error || 'Browse failed');

  if (!result.path) return await refresh(true);

  selected = {};
  state = result.state;
  pruneSelected();
  draw();
  lastRenderSignature = renderSignature();
}

async function clearOverview() {
  if (!confirm('Clear the selected source material from the UI? Generated files are left on disk.')) return;

  selected = {};
  const result = await postJson('/api/overview-clear', {});
  if (!result.ok) return alert(result.error || 'Could not clear overview');

  state = result.state;
  pruneSelected();
  active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function saveProject() {
  const result = await postJson('/api/project-save', { save_as: false });
  if (!result.ok) return alert(result.error || 'Could not save project');
  if (result.path) alert('Saved ARP project:\n' + result.path);
  if (result.state) {
    state = result.state;
    lastRenderSignature = renderSignature();
  }
}

async function saveProjectAs() {
  const result = await postJson('/api/project-save', { save_as: true });
  if (!result.ok) return alert(result.error || 'Could not save project');
  if (result.path) alert('Saved ARP project:\n' + result.path);
  if (result.state) {
    state = result.state;
    lastRenderSignature = renderSignature();
  }
}

async function loadProject() {
  const result = await postJson('/api/project-load', {});
  if (!result.ok) return alert(result.error || 'Could not load project');
  if (!result.path) return;

  selected = {};
  state = result.state;
  pruneSelected();
  active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function browseField(stageKey, fieldKey, kind) {
  const el = document.querySelector(`[data-field="${fieldKey}"]`);
  const result = await postJson('/api/browse', { kind, current: el.value });
  if (!result.ok) return alert(result.error || 'Browse failed');

  if (result.path) {
    el.value = result.path;
    await saveStage(stageKey);
  }
}

async function showCommand(key) {
  const result = await api('/api/command?stage=' + encodeURIComponent(key));
  const el = document.getElementById('cmd');
  if (el) el.textContent = result.command.join(' ');
}

async function confirmOverwrite(key) {
  const force = settings(key).force === 'true';
  if (!force && key !== 'shots') return true;

  const result = await api('/api/existing-outputs?stage=' + encodeURIComponent(key));
  if (!result.paths || !result.paths.length) return true;

  const reason = force ? 'Regenerate is enabled' : 'Shot Detection rewrites its manifest';
  return confirm(reason + ' and these output paths already exist:\n\n' + result.paths.join('\n') + '\n\nOverwrite them?');
}

async function runStage(key) {
  if (key === 'recomp') releaseFinalOutputVideos();
  if (key === 'upscale') releaseFinalOutputVideos();
  await saveStage(key);
  if (key === 'references' && (settings('references').method || 'qwen') === 'openai' && !((settings('references').openai_api_key || '').trim())) {
    alert('Add your OpenAI API key in Settings before running OpenAI Reference Generation.');
    active = 'settings';
    drawTabs();
    draw();
    return;
  }
  if (!(await confirmOverwrite(key))) return;

  const result = await postJson('/api/run', { stage: key });
  if (!result.ok) alert(result.message);
  setTimeout(() => refresh(true), 500);
}

async function generateUpscalePreview() {
  await saveStage('upscale');
  const result = await postJson('/api/upscale-preview', {});
  if (!result.ok) alert(result.message);
  setTimeout(() => pollUpscalePreview(), 500);
}

async function pollUpscalePreview() {
  state = await api(stateUrl());
  updateRunLogs();
  if (state.running) {
    setTimeout(() => pollUpscalePreview(), 1500);
    return;
  }
  refresh(true);
}

function releaseFinalOutputVideos() {
  const output = ((state.expected_outputs && state.expected_outputs.output) || [])[0]
    || settings('recomp').output
    || '';
  if (!output) return;

  const encoded = encodeURIComponent(output);
  document.querySelectorAll('video').forEach(video => {
    const src = video.getAttribute('src') || '';
    if (!src.includes(encoded) && !src.includes(output)) return;
    try {
      video.pause();
      video.removeAttribute('src');
      video.load();
    } catch {}
  });
}

async function runAll() {
  for (const st of state.stages) {
    if (st.key === 'output') continue;
    if (!(await confirmOverwrite(st.key))) return;
  }

  const result = await postJson('/api/run', { all: true });
  if (!result.ok) alert(result.message);
  setTimeout(() => refresh(true), 500);
}

async function stopRun() {
  await postJson('/api/stop', {});
  refresh(true);
}

async function deleteCacheFile(path) {
  if (!confirm('Delete this cached file?\n\n' + path + '\n\nThis cannot be undone.')) return;

  const result = await postJson('/api/cache-delete', { path });
  if (!result.ok) return alert(result.error || 'Could not delete cached file');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}

async function clearCacheCategory(category, title) {
  const message = 'Clear every cached file in "' + title + '"?\n\n'
    + 'This removes generated intermediate files in that category and cannot be undone.';
  if (!confirm(message)) return;

  const result = await postJson('/api/cache-delete', { category });
  if (!result.ok) return alert(result.error || 'Could not clear cache category');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}

async function clearAllCache() {
  const message = 'Clear ALL ARP cached/intermediate files?\n\n'
    + 'This deletes generated previews, outpaint chunks, prepared videos, references, colorized intermediates, and manifests. '
    + 'Source videos, installed tools, and downloaded models are left alone.\n\n'
    + 'This cannot be undone.';
  if (!confirm(message)) return;

  const result = await postJson('/api/cache-delete', { all: true });
  if (!result.ok) return alert(result.error || 'Could not clear cache');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}
