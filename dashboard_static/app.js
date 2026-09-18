const state = { cameras: [], pairing: null, editingTripwire: null };

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function toggleButton(camera, key, value, label) {
  const on = value === true;
  return `<button class="toggle ${on ? 'on' : 'off'}" data-camera="${esc(camera)}" data-kind="${key}" data-value="${!on}">${label}<b>${on ? 'ON' : 'OFF'}</b></button>`;
}

async function loadState() {
  try {
    const res = await fetch('/api/state', {cache: 'no-store'});
    const data = await res.json();
    state.cameras = data.cameras || [];
    if (!state.editingTripwire) renderCameras(state.cameras);
    renderFace(data.face_recognition || {});
    renderLedger(data.ledger || {});
    const online = state.cameras.filter(c => c.enabled && c.status === 'ONLINE').length;
    const total = state.cameras.length;
    const dot = document.getElementById('systemDot');
    document.getElementById('systemText').textContent = `${online}/${total} CAMERAS ONLINE`;
    dot.className = `dot ${online ? 'ok' : 'bad'}`;
  } catch (err) {
    document.getElementById('systemText').textContent = 'DASHBOARD OFFLINE';
    console.error(err);
  }
}

function renderCameras(cameras) {
  const grid = document.getElementById('cameraGrid');
  grid.innerHTML = cameras.map(cam => {
    const f = cam.features || {};
    const ai = cam.ai || {};
    const ids = (ai.global_ids || []).join(', ') || '—';
    const runtime = ai.ai_enabled ? (ai.runtime || 'RUNNING') : 'STANDBY';
    const osnet = ai.osnet || 'NOT LOADED';
    const wires = cam.tripwires || [];
    const hasWire = wires.length > 0;
    return `<article class="camera-card" data-camera-card="${esc(cam.camera_id)}">
      <div class="camera-head">
        <div><div class="cam-id">${esc(cam.camera_id)}</div><h3>${esc(cam.name)}</h3></div>
        <span class="badge ${cam.status === 'ONLINE' ? 'good' : cam.status === 'DISABLED' ? '' : 'warn'}">${esc(cam.status)}</span>
      </div>
      <div class="feed-wrap"><img data-feed-image src="/api/cameras/${encodeURIComponent(cam.camera_id)}/stream" alt="${esc(cam.name)} live feed"><canvas class="tripwire-canvas" data-tripwire-canvas="${esc(cam.camera_id)}"></canvas></div>
      <div class="camera-foot">
        <label class="ai-switch"><input type="checkbox" data-ai="${esc(cam.camera_id)}" ${cam.ai_enabled ? 'checked' : ''}><span></span> AI PROCESSING</label>
        <div class="feature-row">
          ${toggleButton(cam.camera_id, 'geofence', f.geofence, 'Geofence')}
          ${toggleButton(cam.camera_id, 'tripwire', f.tripwire, 'Tripwire')}
          ${toggleButton(cam.camera_id, 'anpr', f.anpr, 'ANPR')}
          ${toggleButton(cam.camera_id, 'face_recognition', f.face_recognition, 'Face ID')}
          ${toggleButton(cam.camera_id, 'enhancement', f.enhancement, 'Enhance')}
        </div>
        <div class="tripwire-bar">
          <span class="tripwire-state ${hasWire ? 'set' : ''}">${hasWire ? 'LINE CONFIGURED' : 'NO LINE CONFIGURED'}</span>
          <div class="tripwire-actions">
            <button class="line-btn" data-tripwire-edit="${esc(cam.camera_id)}">${hasWire ? 'EDIT LINE' : 'DRAW LINE'}</button>
            ${hasWire ? `<button class="line-btn danger" data-tripwire-clear="${esc(cam.camera_id)}">CLEAR</button>` : ''}
          </div>
        </div>
        <div class="camera-runtime">
          <span>${esc(runtime)}</span>
          <span>${ai.ai_enabled ? `${ai.persons || 0} PERSONS · ${ai.vehicles || 0} VEHICLES` : 'LIVE VIEW ONLY'}</span>
          <span>OSNET: ${esc(osnet)}</span>
          <span>GID: ${esc(ids)}</span>
        </div>
      </div>
    </article>`;
  }).join('');

  grid.querySelectorAll('.toggle').forEach(btn => btn.addEventListener('click', async () => {
    const camera = btn.dataset.camera;
    const feature = btn.dataset.kind;
    const value = btn.dataset.value === 'true';
    await updateCamera(camera, {feature, value});
  }));
  grid.querySelectorAll('input[data-ai]').forEach(input => input.addEventListener('change', async e => {
    await updateCamera(e.target.dataset.ai, {ai_enabled: e.target.checked});
  }));
  grid.querySelectorAll('[data-tripwire-edit]').forEach(btn => btn.addEventListener('click', () => beginTripwireEdit(btn.dataset.tripwireEdit)));
  grid.querySelectorAll('[data-tripwire-clear]').forEach(btn => btn.addEventListener('click', () => clearTripwire(btn.dataset.tripwireClear)));
  cameras.forEach(cam => drawSavedTripwire(cam));
}


function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function imageGeometry(img, box) {
  const nw = img.naturalWidth || 16;
  const nh = img.naturalHeight || 9;
  const scale = Math.max(box.width / nw, box.height / nh);
  const dw = nw * scale;
  const dh = nh * scale;
  return {scale, nw, nh, ox: (box.width - dw) / 2, oy: (box.height - dh) / 2};
}

function resizeTripwireCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function drawTripwirePoints(canvas, points, editing=false) {
  const img = canvas.parentElement.querySelector('img');
  const {ctx, width, height} = resizeTripwireCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const list = (points || []).slice(0, 2);
  if (list.length < 1) return;
  const g = imageGeometry(img, {width, height});
  const toDisplay = p => ({
    x: g.ox + clamp01(p[0]) * g.nw * g.scale,
    y: g.oy + clamp01(p[1]) * g.nh * g.scale
  });
  const a = toDisplay(list[0]);
  ctx.lineWidth = editing ? 4 : 3;
  ctx.strokeStyle = editing ? '#21d4ff' : '#ffd34d';
  ctx.fillStyle = ctx.strokeStyle;
  ctx.shadowBlur = editing ? 10 : 6;
  ctx.shadowColor = ctx.strokeStyle;
  if (list.length === 2) {
    const b = toDisplay(list[1]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    for (const p of [a, b]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.font = '700 12px ui-monospace, monospace';
    ctx.fillText('TRIPWIRE', a.x + 8, Math.max(16, a.y - 8));
  } else {
    ctx.beginPath();
    ctx.arc(a.x, a.y, 7, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function drawSavedTripwire(cam) {
  const canvas = document.querySelector('[data-tripwire-canvas="' + CSS.escape(cam.camera_id) + '"]');
  if (!canvas) return;
  const wire = cam.tripwires && cam.tripwires[0];
  const points = wire ? [wire.pt1, wire.pt2] : [];
  drawTripwirePoints(canvas, points, false);
}

function displayToNormalized(img, clientX, clientY) {
  const rect = img.getBoundingClientRect();
  const g = imageGeometry(img, {width: rect.width, height: rect.height});
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  return [
    clamp01((x - g.ox) / (g.nw * g.scale)),
    clamp01((y - g.oy) / (g.nh * g.scale))
  ];
}

function beginTripwireEdit(cameraId) {
  const card = document.querySelector('[data-camera-card="' + CSS.escape(cameraId) + '"]');
  if (!card) return;
  state.editingTripwire = {cameraId, points: []};
  const existing = state.cameras.find(c => c.camera_id === cameraId)?.tripwires?.[0];
  if (existing) state.editingTripwire.points = [existing.pt1, existing.pt2];

  const canvas = card.querySelector('.tripwire-canvas');
  canvas.classList.add('editing');
  canvas.style.pointerEvents = 'auto';
  drawTripwirePoints(canvas, state.editingTripwire.points, true);

  canvas.onclick = e => {
    const point = displayToNormalized(card.querySelector('[data-feed-image]'), e.clientX, e.clientY);
    if (state.editingTripwire.points.length >= 2) state.editingTripwire.points = [];
    state.editingTripwire.points.push(point);
    drawTripwirePoints(canvas, state.editingTripwire.points, true);

    const bar = card.querySelector('.tripwire-state');
    if (state.editingTripwire.points.length === 1) bar.textContent = 'SELECT END POINT';
    if (state.editingTripwire.points.length === 2) bar.textContent = 'LINE READY — SAVE';
  };

  const bar = card.querySelector('.tripwire-state');
  bar.textContent = state.editingTripwire.points.length === 2 ? 'LINE READY — SAVE' : 'SELECT START POINT';

  const actions = card.querySelector('.tripwire-actions');
  actions.innerHTML = '<button class="line-btn save" data-tripwire-save>SAVE LINE</button><button class="line-btn" data-tripwire-cancel>CANCEL</button>';
  actions.querySelector('[data-tripwire-save]').addEventListener('click', saveTripwire);
  actions.querySelector('[data-tripwire-cancel]').addEventListener('click', cancelTripwireEdit);
}

function cancelTripwireEdit() {
  const cameraId = state.editingTripwire?.cameraId;
  state.editingTripwire = null;
  if (cameraId) renderCameras(state.cameras);
}

async function saveTripwire() {
  const editing = state.editingTripwire;
  if (!editing || editing.points.length !== 2) {
    alert('Click two points on the camera frame first.');
    return;
  }
  const res = await fetch('/api/cameras/' + encodeURIComponent(editing.cameraId) + '/tripwire', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pt1: editing.points[0], pt2: editing.points[1], id: 'primary'})
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Failed to save tripwire');
    return;
  }
  state.editingTripwire = null;
  await loadState();
}

async function clearTripwire(cameraId) {
  const res = await fetch('/api/cameras/' + encodeURIComponent(cameraId) + '/tripwire', {method:'DELETE'});
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Failed to clear tripwire');
    return;
  }
  await loadState();
}

async function updateCamera(camera, body) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(camera)}/settings`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Failed to update camera');
    return;
  }
  await loadState();
}

function renderFace(face) {
  const badge = document.getElementById('faceStatus');
  badge.textContent = face.available ? `ACTIVE · ${face.known_people} KNOWN` : 'UNAVAILABLE';
  badge.className = `badge ${face.available ? 'good' : 'warn'}`;
  fetch('/api/face/people', {cache: 'no-store'})
    .then(r => r.json())
    .then(data => {
      document.getElementById('peopleList').innerHTML = (data.people || []).length
        ? data.people.map(p => `<div class="person-row"><span><strong>${esc(p.display_name)}</strong><small>${esc(p.person_id)}</small></span><span>${p.sample_count} samples</span></div>`).join('')
        : '<div class="muted">No people enrolled yet.</div>';
    })
    .catch(console.error);
}

function renderLedger(ledger) {
  const ok = ledger.valid === true;
  const badge = document.getElementById('ledgerStatus');
  badge.textContent = ok ? 'CHAIN VALID' : 'CHECK LEDGER';
  badge.className = `badge ${ok ? 'good' : 'warn'}`;
  document.getElementById('ledgerBlocks').textContent = ledger.blocks ?? '0';
  document.getElementById('ledgerHead').textContent = ledger.head_hash ? ledger.head_hash.slice(0, 24) + '…' : '—';
}

function openPairModal() {
  document.getElementById('pairModal').classList.remove('hidden');
  const select = document.getElementById('pairCamera');
  select.innerHTML = state.cameras.map(c => `<option value="${esc(c.camera_id)}">${esc(c.camera_id)} · ${esc(c.name)}</option>`).join('');
  const preferred = state.cameras.find(c => !c.enabled || c.status === 'DISABLED' || c.status === 'WAITING FOR PHONE');
  if (preferred) select.value = preferred.camera_id;
  resetPairModal();
}

function closePairModal() {
  document.getElementById('pairModal').classList.add('hidden');
  state.pairing = null;
}

function resetPairModal() {
  document.getElementById('qrBox').classList.add('hidden');
  document.getElementById('copyPair').classList.add('hidden');
  document.getElementById('qrImage').src = '';
  document.getElementById('qrState').textContent = 'CREATE A PAIRING TO CONTINUE';
  document.getElementById('pairHint').textContent = 'The QR link is short-lived. The phone and laptop must be on the same network.';
}

async function createPairing() {
  const camera_id = document.getElementById('pairCamera').value;
  const btn = document.getElementById('createPair');
  btn.disabled = true;
  document.getElementById('qrState').textContent = 'GENERATING QR…';
  try {
    const res = await fetch('/api/pairing/create', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({camera_id})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not create pairing');
    state.pairing = data;
    document.getElementById('qrImage').src = `/api/pairing/${encodeURIComponent(data.fingerprint ? data.pairing_url.split('/').pop() : '')}/qr.png`;
    document.getElementById('qrImage').onerror = () => { document.getElementById('qrState').textContent = 'QR IMAGE FAILED — USE THE PAIRING LINK BELOW'; };
    document.getElementById('qrBox').classList.remove('hidden');
    document.getElementById('copyPair').classList.remove('hidden');
    document.getElementById('pairCameraLabel').textContent = `${data.camera_id} · ${data.camera_name}`;
    document.getElementById('pairFingerprint').textContent = `PAIR ${data.fingerprint}`;
    document.getElementById('pairExpiry').textContent = `EXPIRES ${new Date(data.expires_at * 1000).toLocaleTimeString()}`;
    document.getElementById('qrState').textContent = 'SCAN WITH PHONE';
    document.getElementById('pairHint').innerHTML = `Open the QR link on the phone. Because the camera page uses HTTPS, the phone may show a certificate warning for this local development server. Continue to the page and allow camera access.`;
    waitForPair(data.token || data.pairing_url.split('/').pop());
  } catch (err) {
    document.getElementById('qrState').textContent = 'PAIRING FAILED';
    document.getElementById('pairHint').textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function waitForPair(token) {
  for (let i = 0; i < 150; i++) {
    await new Promise(r => setTimeout(r, 1000));
    if (!state.pairing) return;
    try {
      const res = await fetch(`/api/pairing/${encodeURIComponent(token)}/status`, {cache:'no-store'});
      const data = await res.json();
      if (!res.ok) return;
      if (data.status === 'ONLINE') {
        document.getElementById('qrState').textContent = `${data.camera_id} CONNECTED`;
        document.getElementById('pairHint').textContent = 'Phone camera is live. Close this dialog or leave it open while operating.';
        await loadState();
        return;
      }
    } catch (e) { return; }
  }
}

async function copyPairingLink() {
  if (!state.pairing?.pairing_url) return;
  try {
    await navigator.clipboard.writeText(state.pairing.pairing_url);
    document.getElementById('pairHint').textContent = 'Pairing link copied.';
  } catch {
    document.getElementById('pairHint').textContent = state.pairing.pairing_url;
  }
}

document.getElementById('enrollForm').addEventListener('submit', async e => {
  e.preventDefault();
  const form = new FormData(e.target);
  const res = await fetch('/api/face/enroll', {method: 'POST', body: form});
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Face registration failed');
    return;
  }
  e.target.reset();
  await loadState();
});

document.getElementById('addCameraBtn').addEventListener('click', openPairModal);
document.getElementById('closePair').addEventListener('click', closePairModal);
document.getElementById('pairModal').addEventListener('click', e => { if (e.target.id === 'pairModal') closePairModal(); });
document.getElementById('createPair').addEventListener('click', createPairing);
document.getElementById('copyPair').addEventListener('click', copyPairingLink);

document.addEventListener('keydown', e => { if (e.key === 'Escape') closePairModal(); });

function tick() {
  document.getElementById('clock').textContent = new Date().toLocaleString();
}

setInterval(tick, 1000);
setInterval(loadState, 3000);
tick();
loadState();
