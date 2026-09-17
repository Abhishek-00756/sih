const state = { cameras: [], pairing: null };

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
    renderCameras(state.cameras);
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
    return `<article class="camera-card">
      <div class="camera-head">
        <div><div class="cam-id">${esc(cam.camera_id)}</div><h3>${esc(cam.name)}</h3></div>
        <span class="badge ${cam.status === 'ONLINE' ? 'good' : cam.status === 'DISABLED' ? '' : 'warn'}">${esc(cam.status)}</span>
      </div>
      <div class="feed-wrap"><img src="/api/cameras/${encodeURIComponent(cam.camera_id)}/stream" alt="${esc(cam.name)} live feed"></div>
      <div class="camera-foot">
        <label class="ai-switch"><input type="checkbox" data-ai="${esc(cam.camera_id)}" ${cam.ai_enabled ? 'checked' : ''}><span></span> AI PROCESSING</label>
        <div class="feature-row">
          ${toggleButton(cam.camera_id, 'geofence', f.geofence, 'Geofence')}
          ${toggleButton(cam.camera_id, 'tripwire', f.tripwire, 'Tripwire')}
          ${toggleButton(cam.camera_id, 'anpr', f.anpr, 'ANPR')}
          ${toggleButton(cam.camera_id, 'face_recognition', f.face_recognition, 'Face ID')}
          ${toggleButton(cam.camera_id, 'enhancement', f.enhancement, 'Enhance')}
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
