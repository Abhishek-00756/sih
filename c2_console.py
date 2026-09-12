#!/usr/bin/env python3
"""Command-and-control incident review console.

Serves the SQLite audit trail and timestamped evidence snapshots so operators
can search by Global ID, camera, and time without opening the database by hand.
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
from typing import Optional

from flask import Flask, Response, abort, jsonify, request, send_file

from alert_logger import AlertLogger

ROOT = Path(__file__).resolve().parent

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Border C2 Incident Review</title>
  <style>
    :root { --bg:#0b1220; --panel:#121a2b; --line:#243049; --text:#e8eefc; --muted:#93a0b8; --accent:#e11d48; --ok:#22c55e; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
    header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
    h1 { margin:0; font-size:20px; letter-spacing:.04em; }
    .stats { color:var(--muted); font-size:13px; }
    .wrap { display:grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 62px); }
    aside { border-right:1px solid var(--line); padding:16px; background:var(--panel); }
    label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
    input, select, button { width:100%; padding:8px 10px; border-radius:8px; border:1px solid var(--line); background:#0d1524; color:var(--text); }
    button { background:var(--accent); border:0; font-weight:600; cursor:pointer; margin-top:14px; }
    .list { max-height: calc(100vh - 320px); overflow:auto; margin-top:12px; }
    .card { border:1px solid var(--line); border-radius:10px; padding:10px; margin-bottom:8px; cursor:pointer; }
    .card:hover, .card.active { border-color:var(--accent); }
    .gid { color:var(--accent); font-weight:700; }
    main { padding:18px 24px; }
    .empty { color:var(--muted); padding:40px 0; }
    .meta { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom:16px; }
    .meta div { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
    .meta b { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }
    img { max-width:100%; border:1px solid var(--line); border-radius:12px; background:#000; }
    .shots { display:grid; grid-template-columns: 1fr 260px; gap:16px; }
  </style>
</head>
<body>
  <header>
    <h1>BORDER C2 / INCIDENT REVIEW</h1>
    <div class="stats" id="stats">Loading audit trail...</div>
  </header>
  <div class="wrap">
    <aside>
      <label>Camera ID</label>
      <input id="camera" placeholder="Cam_1_Outpost"/>
      <label>Global ID</label>
      <input id="gid" type="number" placeholder="17"/>
      <label>Alert type</label>
      <select id="atype">
        <option value="">All</option>
        <option>Geofence Intrusion</option>
        <option>GEOFENCE_ENTER</option>
        <option>Loitering</option>
      </select>
      <button id="search">Search evidence</button>
      <div class="list" id="list"></div>
    </aside>
    <main id="detail"><div class="empty">Select an incident to inspect snapshots and metadata.</div></main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    async function loadStats() {
      const s = await fetch('/api/stats').then(r => r.json());
      $('stats').textContent = s.total + ' incidents  |  latest ' + (s.latest || 'n/a');
    }
    async function search() {
      const q = new URLSearchParams();
      if ($('camera').value) q.set('camera_id', $('camera').value);
      if ($('gid').value) q.set('global_id', $('gid').value);
      if ($('atype').value) q.set('alert_type', $('atype').value);
      const rows = await fetch('/api/alerts?' + q.toString()).then(r => r.json());
      $('list').innerHTML = rows.map(r => `
        <div class="card" data-id="${r.id}">
          <div class="gid">GID ${r.global_id}</div>
          <div>${r.camera_id}</div>
          <div style="color:#93a0b8;font-size:12px">${r.timestamp} · ${r.alert_type}</div>
        </div>`).join('') || '<div class="empty">No matching incidents.</div>';
      document.querySelectorAll('.card').forEach(el => el.onclick = () => openAlert(el.dataset.id, el));
      if (rows[0]) openAlert(rows[0].id, document.querySelector('.card'));
    }
    async function openAlert(id, el) {
      document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
      if (el) el.classList.add('active');
      const r = await fetch('/api/alerts/' + id).then(res => res.json());
      const snap = r.snapshot_path ? '/evidence?path=' + encodeURIComponent(r.snapshot_path) : '';
      const crop = r.crop_path ? '/evidence?path=' + encodeURIComponent(r.crop_path) : '';
      $('detail').innerHTML = `
        <div class="meta">
          <div><b>Global ID</b>${r.global_id}</div>
          <div><b>Camera / Outpost</b>${r.camera_id}</div>
          <div><b>Timestamp</b>${r.timestamp}</div>
          <div><b>Alert</b>${r.alert_type}</div>
          <div><b>Zone</b>${r.zone_name || r.zone_id || '-'}</div>
          <div><b>Coordinates</b>${r.footprint || r.bbox}</div>
        </div>
        <div class="shots">
          <div><img src="${snap}" alt="full frame evidence"/></div>
          <div>${crop ? '<img src="'+crop+'" alt="crop evidence"/>' : ''}</div>
        </div>`;
    }
    $('search').onclick = search;
    loadStats().then(search);
  </script>
</body>
</html>
"""


def create_app(
    db_path: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> Flask:
    logger = AlertLogger(
        db_path=db_path or (ROOT / "border_alerts.db"),
        snapshot_dir=snapshot_dir or (ROOT / "alert_snapshots"),
    )
    app = Flask(__name__)
    allowed_roots = [
        logger.snapshot_dir.resolve(),
        ROOT.resolve() / "alert_snapshots",
    ]

    def _safe_file(raw: str) -> Path:
        path = Path(raw).resolve()
        if not any(_is_relative_to(path, root) for root in allowed_roots):
            abort(403)
        if not path.is_file():
            abort(404)
        return path

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/stats")
    def api_stats():
        return jsonify(logger.stats())

    @app.get("/api/alerts")
    def api_alerts():
        gid = request.args.get("global_id")
        rows = logger.search(
            camera_id=request.args.get("camera_id") or None,
            global_id=int(gid) if gid else None,
            alert_type=request.args.get("alert_type") or None,
            limit=int(request.args.get("limit", 100)),
            offset=int(request.args.get("offset", 0)),
        )
        return jsonify(rows)

    @app.get("/api/alerts/<int:alert_id>")
    def api_alert(alert_id: int):
        row = logger.get(alert_id)
        if row is None:
            abort(404)
        return jsonify(row)

    @app.get("/evidence")
    def evidence():
        raw = request.args.get("path") or ""
        path = _safe_file(raw)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return send_file(path, mimetype=mime)

    return app


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2 incident review console")
    p.add_argument("--db", default=str(ROOT / "border_alerts.db"))
    p.add_argument("--snapshots", default=str(ROOT / "alert_snapshots"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(Path(args.db), Path(args.snapshots))
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
