const state = { cameras: [] };

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

function tick() {
  document.getElementById('clock').textContent = new Date().toLocaleString();
}

setInterval(tick, 1000);
setInterval(loadState, 3000);
tick();
loadState();
