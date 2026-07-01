from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import cos, radians
from statistics import mean
from threading import Lock
from time import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
import json

from flask import Flask, jsonify, request


app = Flask(__name__)
state_lock = Lock()
latest_raw_info: dict[str, Any] = {}
latest_received_at = 0.0


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def raw_lidar_items(raw_info: dict[str, Any]) -> list[dict[str, Any]]:
    points = (
        raw_info.get("lidarPoints")
        or raw_info.get("points")
        or raw_info.get("lidarData")
        or raw_info.get("data")
        or []
    )
    return points if isinstance(points, list) else []


def point_position(item: dict[str, Any]) -> dict[str, float | None]:
    pos = item.get("position", {}) if isinstance(item.get("position"), dict) else {}
    return {
        "x": safe_float(item.get("x", pos.get("x"))),
        "y": safe_float(item.get("y", pos.get("y"))),
        "z": safe_float(item.get("z", pos.get("z"))),
    }


def detected_value(item: dict[str, Any]) -> bool:
    if "isDetected" in item:
        return bool(item.get("isDetected"))
    if "detected" in item:
        return str(item.get("detected")).strip().lower() not in {"false", "0", "no", ""}
    return True


def summarize_raw_lidar(raw_info: dict[str, Any]) -> dict[str, Any]:
    items = raw_lidar_items(raw_info)
    detected = [item for item in items if isinstance(item, dict) and detected_value(item)]

    distances = [
        safe_float(item.get("distance", item.get("range")))
        for item in detected
        if safe_float(item.get("distance", item.get("range"))) is not None
    ]
    angles = [
        safe_float(item.get("angle"))
        for item in detected
        if safe_float(item.get("angle")) is not None
    ]
    verticals = [
        safe_float(item.get("verticalAngle"))
        for item in detected
        if safe_float(item.get("verticalAngle")) is not None
    ]
    channels = {
        item.get("channelIndex")
        for item in detected
        if item.get("channelIndex") is not None
    }

    return {
        "receivedAt": latest_received_at,
        "ageSeconds": round(max(0.0, time() - latest_received_at), 2) if latest_received_at else None,
        "rawRayCount": len(items),
        "detectedRayCount": len(detected),
        "detectionRate": round(len(detected) / len(items), 4) if items else 0.0,
        "distanceMin": round(min(distances), 3) if distances else None,
        "distanceMean": round(mean(distances), 3) if distances else None,
        "distanceMax": round(max(distances), 3) if distances else None,
        "angleMin": round(min(angles), 3) if angles else None,
        "angleMax": round(max(angles), 3) if angles else None,
        "verticalMin": round(min(verticals), 3) if verticals else None,
        "verticalMax": round(max(verticals), 3) if verticals else None,
        "channelCount": len(channels),
    }


def make_raw_point_rows(raw_info: dict[str, Any], limit: int = 600) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_lidar_items(raw_info)):
        if not isinstance(item, dict):
            continue

        distance = safe_float(item.get("distance", item.get("range")))
        angle = safe_float(item.get("angle"))
        vertical = safe_float(item.get("verticalAngle"), 0.0)
        pos = point_position(item)
        horizontal_range = (
            float(distance) * cos(radians(float(vertical or 0.0)))
            if distance is not None
            else None
        )

        rows.append(
            {
                "index": index,
                "detected": detected_value(item),
                "channelIndex": item.get("channelIndex"),
                "angle": round(float(angle), 3) if angle is not None else None,
                "signedAngle": round(((float(angle) + 180.0) % 360.0) - 180.0, 3) if angle is not None else None,
                "verticalAngle": round(float(vertical or 0.0), 3) if vertical is not None else None,
                "distance": round(float(distance), 3) if distance is not None else None,
                "horizontalRange": round(float(horizontal_range), 3) if horizontal_range is not None else None,
                "x": round(float(pos["x"]), 3) if pos["x"] is not None else None,
                "y": round(float(pos["y"]), 3) if pos["y"] is not None else None,
                "z": round(float(pos["z"]), 3) if pos["z"] is not None else None,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def make_distribution(values: list[float], bin_size: float) -> list[dict[str, Any]]:
    bins: dict[float, int] = defaultdict(int)
    for value in values:
        bins[round(round(value / bin_size) * bin_size, 3)] += 1
    return [{"bin": key, "count": count} for key, count in sorted(bins.items())]


def make_channel_distribution(raw_info: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_lidar_items(raw_info):
        if isinstance(item, dict):
            grouped[item.get("channelIndex", "none")].append(item)

    rows = []
    for channel, items in grouped.items():
        detected = [item for item in items if detected_value(item)]
        distances = [
            safe_float(item.get("distance", item.get("range")))
            for item in detected
            if safe_float(item.get("distance", item.get("range"))) is not None
        ]
        verticals = [
            safe_float(item.get("verticalAngle"))
            for item in items
            if safe_float(item.get("verticalAngle")) is not None
        ]
        rows.append(
            {
                "channelIndex": channel,
                "rayCount": len(items),
                "detectedCount": len(detected),
                "verticalAngle": round(mean(verticals), 3) if verticals else None,
                "distanceMin": round(min(distances), 3) if distances else None,
                "distanceMean": round(mean(distances), 3) if distances else None,
                "distanceMax": round(max(distances), 3) if distances else None,
            }
        )
    return sorted(rows, key=lambda row: (row["channelIndex"] is None, str(row["channelIndex"])))


def make_raw_distributions(raw_info: dict[str, Any]) -> dict[str, Any]:
    detected = [
        item
        for item in raw_lidar_items(raw_info)
        if isinstance(item, dict) and detected_value(item)
    ]
    angles = [safe_float(item.get("angle")) for item in detected if safe_float(item.get("angle")) is not None]
    verticals = [safe_float(item.get("verticalAngle")) for item in detected if safe_float(item.get("verticalAngle")) is not None]
    distances = [safe_float(item.get("distance", item.get("range"))) for item in detected if safe_float(item.get("distance", item.get("range"))) is not None]
    return {
        "angleBins10Deg": make_distribution([float(v) for v in angles], 10.0),
        "verticalBins1Deg": make_distribution([float(v) for v in verticals], 1.0),
        "distanceBins5m": make_distribution([float(v) for v in distances], 5.0),
        "channels": make_channel_distribution(raw_info),
    }


def make_top_points(raw_info: dict[str, Any], limit: int = 12000) -> list[dict[str, Any]]:
    points = []
    for row in make_raw_point_rows(raw_info, limit=limit):
        if row["detected"] and row["angle"] is not None and row["distance"] is not None:
            points.append(row)
    return points


def make_front_slice(raw_info: dict[str, Any], half_angle_deg: float = 2.0) -> list[dict[str, Any]]:
    rows = []
    for row in make_raw_point_rows(raw_info, limit=12000):
        signed = row.get("signedAngle")
        if signed is not None and abs(float(signed)) <= half_angle_deg:
            rows.append(row)
    return sorted(rows, key=lambda row: (row.get("verticalAngle") is None, row.get("verticalAngle") or 0.0))


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=1.5) as response:
        return json.loads(response.read().decode("utf-8"))


def pull_from_controller(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    try:
        raw = fetch_json(f"{base}/raw_lidar_status")
    except (OSError, URLError, json.JSONDecodeError):
        raw = {}
    try:
        analyzed = fetch_json(f"{base}/lidar_status")
    except (OSError, URLError, json.JSONDecodeError):
        analyzed = {}
    return {"raw": raw, "analyzed": analyzed}


def build_report(raw_info: dict[str, Any], analyzed: dict[str, Any] | None = None, include_full: bool = False) -> dict[str, Any]:
    analyzed = analyzed or {}
    report = {
        "rawSummary": summarize_raw_lidar(raw_info),
        "rawTopPoints": make_top_points(raw_info),
        "rawPointRows": make_raw_point_rows(raw_info),
        "rawDistributions": make_raw_distributions(raw_info),
        "frontSlice": make_front_slice(raw_info),
        "analyzedSummary": {
            "rawRayCount": analyzed.get("rawRayCount"),
            "rawDetectedPointCount": analyzed.get("rawDetectedPointCount"),
            "rawObjectCount": analyzed.get("rawObjectCount"),
            "provisionalObjectCount": analyzed.get("provisionalObjectCount"),
            "trackedObjectCount": analyzed.get("trackedObjectCount"),
            "confirmedObjectCount": analyzed.get("confirmedObjectCount"),
            "terrainDecision": analyzed.get("terrainDecision", {}),
            "bodyAlignment": analyzed.get("bodyAlignment", {}),
            "primaryFusionTarget": analyzed.get("primaryFusionTarget"),
        },
    }
    if include_full:
        report["analyzed"] = analyzed
        report["rawInfo"] = raw_info
    return report


@app.route("/info", methods=["POST"])
@app.route("/lidar", methods=["POST"])
@app.route("/lidar_data", methods=["POST"])
def ingest_lidar():
    global latest_raw_info, latest_received_at
    data = request.get_json(force=True, silent=True) or {}
    with state_lock:
        latest_raw_info = deepcopy(data)
        latest_received_at = time()
    return jsonify({"status": "OK", "message": "LiDAR payload stored by standalone inspector", "rawRayCount": len(raw_lidar_items(data))})


@app.route("/lidar_only_status", methods=["GET"])
def lidar_only_status():
    source = request.args.get("source", "").strip()
    include_full = str(request.args.get("full", "")).strip().lower() in {"1", "true", "yes"}
    analyzed = {}
    with state_lock:
        raw = deepcopy(latest_raw_info)

    if source:
        pulled = pull_from_controller(source)
        raw = pulled["raw"] or raw
        analyzed = pulled["analyzed"]

    return jsonify(build_report(raw, analyzed, include_full=include_full))


@app.route("/lidar_only_view", methods=["GET"])
def lidar_only_view():
    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Standalone LiDAR Inspector</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#181c1f; --line:#3a4248; --text:#f2f5f7; --muted:#a9b3ba; }
    * { box-sizing: border-box; }
    body { margin: 16px; background: var(--bg); color: var(--text); font-family: Arial, sans-serif; }
    h1, h2, p { margin: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 15px; margin-bottom: 8px; }
    p, .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: minmax(420px, 1fr) minmax(360px, 460px); gap: 12px; align-items: start; margin-top: 14px; }
    .panel { border: 1px solid var(--line); background: var(--panel); padding: 12px; }
    canvas { width: 100%; height: auto; border: 1px solid var(--line); background: #121416; display: block; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin-top: 12px; }
    .card { border: 1px solid var(--line); background: #121619; padding: 8px; min-height: 54px; }
    .card span { display:block; color: var(--muted); font-size: 12px; }
    .card strong { display:block; margin-top: 5px; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #30373d; padding: 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    pre { margin: 0; max-height: 520px; overflow: auto; font-size: 12px; line-height: 1.35; white-space: pre-wrap; }
    .toolbar { display:flex; gap:8px; align-items:center; justify-content:space-between; margin-top: 8px; flex-wrap: wrap; }
    input { background:#111619; border:1px solid var(--line); color:var(--text); padding:7px 9px; min-width: 280px; }
    button { border: 1px solid var(--line); background: #20262a; color: var(--text); padding: 7px 10px; cursor:pointer; }
    .wide { grid-column: 1 / -1; }
    .subgrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .viewgrid { display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 10px; }
    .table-wrap { max-height: 420px; overflow: auto; border: 1px solid #30373d; }
  </style>
</head>
<body>
  <h1>Standalone LiDAR Inspector</h1>
  <p>This server is separate from driving/fire control. POST LiDAR payloads here, or set a source URL to read from an existing controller.</p>
  <div class="toolbar">
    <div>
      <input id="sourceUrl" value="http://127.0.0.1:5000" placeholder="optional source, e.g. http://127.0.0.1:5000">
      <button onclick="refresh()">Refresh</button>
    </div>
    <span class="muted">Local ingest endpoints: POST /info, /lidar, /lidar_data</span>
  </div>
  <div class="cards" id="cards"></div>
  <div class="grid">
    <div class="panel wide">
      <h2>Point Cloud Views</h2>
      <div class="viewgrid">
        <div>
          <p class="muted">Top polar: angle x horizontal range</p>
          <canvas id="top" width="420" height="420"></canvas>
        </div>
        <div>
          <p class="muted">Front angular: left/right angle x vertical angle</p>
          <canvas id="frontAngular" width="420" height="300"></canvas>
        </div>
        <div>
          <p class="muted">Side profile: forward range x height/world Y</p>
          <canvas id="sideProfile" width="420" height="300"></canvas>
        </div>
        <div>
          <p class="muted">World X-Z projection</p>
          <canvas id="worldXZ" width="420" height="300"></canvas>
        </div>
        <div>
          <p class="muted">World X-Y elevation projection</p>
          <canvas id="worldXY" width="420" height="300"></canvas>
        </div>
        <div>
          <p class="muted">Channel x angle heatmap</p>
          <canvas id="channelHeatmap" width="420" height="300"></canvas>
        </div>
      </div>
      <p class="muted">All views use the same raw detected LiDAR rows. Color shows distance: red=near, yellow=mid, green=far.</p>
    </div>
    <div class="panel">
      <h2>Raw Summary</h2>
      <pre id="diagnosis">Waiting...</pre>
    </div>
    <div class="panel wide">
      <h2>Point Cloud Distributions</h2>
      <div class="subgrid">
        <div><p class="muted">Channels</p><div class="table-wrap"><table><thead><tr><th>ch</th><th>rays</th><th>hit</th><th>v</th><th>mean m</th></tr></thead><tbody id="channelRows"></tbody></table></div></div>
        <div><p class="muted">Angle bins 10 deg</p><div class="table-wrap"><table><thead><tr><th>angle</th><th>count</th></tr></thead><tbody id="angleBins"></tbody></table></div></div>
        <div><p class="muted">Distance bins 5 m</p><div class="table-wrap"><table><thead><tr><th>dist</th><th>count</th></tr></thead><tbody id="distanceBins"></tbody></table></div></div>
      </div>
    </div>
    <div class="panel wide">
      <h2>Raw Point Rows</h2>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>hit</th><th>ch</th><th>angle</th><th>signed</th><th>vertical</th><th>dist</th><th>hRange</th><th>x</th><th>y</th><th>z</th></tr></thead><tbody id="pointRows"></tbody></table></div>
    </div>
    <div class="panel wide">
      <h2>Body-Forward Slice (+/-2 deg)</h2>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>ch</th><th>signed</th><th>vertical</th><th>dist</th><th>hRange</th><th>x</th><th>y</th><th>z</th></tr></thead><tbody id="frontSliceRows"></tbody></table></div>
    </div>
    <div class="panel wide">
      <h2>Inspector JSON</h2>
      <pre id="json"></pre>
    </div>
  </div>
<script>
const topCanvas = document.getElementById('top');
const topCtx = topCanvas.getContext('2d');
const frontCanvas = document.getElementById('frontAngular');
const frontCtx = frontCanvas.getContext('2d');
const sideCanvas = document.getElementById('sideProfile');
const sideCtx = sideCanvas.getContext('2d');
const xzCanvas = document.getElementById('worldXZ');
const xzCtx = xzCanvas.getContext('2d');
const xyCanvas = document.getElementById('worldXY');
const xyCtx = xyCanvas.getContext('2d');
const heatCanvas = document.getElementById('channelHeatmap');
const heatCtx = heatCanvas.getContext('2d');
const MAX_DISTANCE = 120;
function fmt(v, d=2) { const n = Number(v); return Number.isFinite(n) ? n.toFixed(d) : 'n/a'; }
function colorByDistance(distance) { const d = Number(distance); if (d <= 20) return '#ff3030'; if (d <= 50) return '#ffca3a'; return '#35e835'; }
function clearCanvas(ctx, canvas) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#121416';
  ctx.fillRect(0,0,canvas.width,canvas.height);
}
function drawAxes(ctx, canvas, xlabel, ylabel) {
  const m = 46, w = canvas.width, h = canvas.height;
  ctx.strokeStyle = '#3f474d';
  ctx.fillStyle = '#bfc7cc';
  ctx.font = '12px Arial';
  ctx.beginPath(); ctx.moveTo(m, m); ctx.lineTo(m, h-m); ctx.lineTo(w-m, h-m); ctx.stroke();
  ctx.fillText(xlabel, w/2-40, h-12);
  ctx.save(); ctx.translate(14, h/2+40); ctx.rotate(-Math.PI/2); ctx.fillText(ylabel, 0, 0); ctx.restore();
}
function rangeOf(values, fallbackMin=0, fallbackMax=1) {
  const nums = values.map(Number).filter(Number.isFinite);
  if (!nums.length) return [fallbackMin, fallbackMax];
  let min = Math.min(...nums), max = Math.max(...nums);
  if (min === max) { min -= 1; max += 1; }
  return [min, max];
}
function scale(value, min, max, outMin, outMax) {
  return outMin + (Number(value) - min) / (max - min) * (outMax - outMin);
}
function polar(angleDeg, distance, cx, cy, radius) {
  const signed = ((Number(angleDeg) + 180) % 360) - 180;
  const a = signed * Math.PI / 180;
  const r = Math.min(Number(distance) || 0, MAX_DISTANCE) / MAX_DISTANCE * radius;
  return {x: cx + Math.sin(a) * r, y: cy - Math.cos(a) * r};
}
function drawTop(report) {
  const ctx = topCtx, w = topCanvas.width, h = topCanvas.height, cx = w/2, cy = h/2, radius = Math.min(w,h)*0.46;
  ctx.clearRect(0,0,w,h);
  ctx.font = '13px Arial'; ctx.strokeStyle = '#4b5359'; ctx.fillStyle = '#bfc7cc';
  for (const d of [30,60,90,120]) { const r = radius*d/MAX_DISTANCE; ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke(); ctx.fillText(d+'m',cx+5,cy-r+16); }
  ctx.strokeStyle='#89939a'; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx,cy-radius); ctx.stroke();
  ctx.fillText('0 deg body-forward',cx+8,cy-radius+16); ctx.fillText('-90',cx-radius+4,cy-8); ctx.fillText('+90',cx+radius-38,cy-8);
  ctx.globalAlpha = 0.55;
  for (const p of report.rawTopPoints || []) {
    const q = polar(p.angle, p.horizontalRange || p.distance, cx, cy, radius);
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.beginPath(); ctx.arc(q.x,q.y,1.5,0,Math.PI*2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#4cc9f0'; ctx.beginPath(); ctx.arc(cx,cy,8,0,Math.PI*2); ctx.fill();
}
function drawFrontAngular(report) {
  const canvas = frontCanvas, ctx = frontCtx, rows = report.rawTopPoints || [];
  clearCanvas(ctx, canvas); drawAxes(ctx, canvas, 'signed angle deg', 'vertical angle deg');
  const m = 46, w = canvas.width, h = canvas.height;
  ctx.strokeStyle = '#2f383d'; ctx.fillStyle = '#9faab1'; ctx.font = '11px Arial';
  for (const a of [-180,-120,-60,0,60,120,180]) {
    const x = scale(a, -180, 180, m, w-m); ctx.beginPath(); ctx.moveTo(x,m); ctx.lineTo(x,h-m); ctx.stroke(); ctx.fillText(String(a), x-12, h-m+16);
  }
  const [vMin, vMax] = rangeOf(rows.map(r => r.verticalAngle), -25, 25);
  for (const p of rows) {
    if (p.verticalAngle === null || p.signedAngle === null) continue;
    const x = scale(p.signedAngle, -180, 180, m, w-m);
    const y = scale(p.verticalAngle, vMax, vMin, m, h-m);
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.beginPath(); ctx.arc(x,y,1.7,0,Math.PI*2); ctx.fill();
  }
}
function drawSideProfile(report) {
  const canvas = sideCanvas, ctx = sideCtx, rows = report.frontSlice || [];
  clearCanvas(ctx, canvas); drawAxes(ctx, canvas, 'horizontal range m', 'world Y');
  const m = 46, w = canvas.width, h = canvas.height;
  const [yMin, yMax] = rangeOf(rows.map(r => r.y), 0, 20);
  for (const p of rows) {
    if (p.horizontalRange === null || p.y === null) continue;
    const x = scale(p.horizontalRange, 0, MAX_DISTANCE, m, w-m);
    const y = scale(p.y, yMax, yMin, m, h-m);
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
  }
}
function drawWorldProjection(report, canvas, ctx, mode) {
  const rows = (report.rawTopPoints || []).filter(p => Number.isFinite(Number(p.x)) && Number.isFinite(Number(mode === 'xz' ? p.z : p.y)));
  clearCanvas(ctx, canvas); drawAxes(ctx, canvas, mode === 'xz' ? 'world X' : 'world X', mode === 'xz' ? 'world Z' : 'world Y');
  const m = 46, w = canvas.width, h = canvas.height;
  const [xMin, xMax] = rangeOf(rows.map(r => r.x), 0, 300);
  const [yMin, yMax] = rangeOf(rows.map(r => mode === 'xz' ? r.z : r.y), 0, 300);
  for (const p of rows) {
    const x = scale(p.x, xMin, xMax, m, w-m);
    const yValue = mode === 'xz' ? p.z : p.y;
    const y = scale(yValue, yMax, yMin, m, h-m);
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.beginPath(); ctx.arc(x,y,1.6,0,Math.PI*2); ctx.fill();
  }
}
function drawChannelHeatmap(report) {
  const canvas = heatCanvas, ctx = heatCtx, rows = report.rawTopPoints || [];
  clearCanvas(ctx, canvas); drawAxes(ctx, canvas, 'angle deg', 'channel');
  const m = 46, w = canvas.width, h = canvas.height;
  const channels = [...new Set(rows.map(r => r.channelIndex).filter(v => v !== null && v !== undefined))].sort((a,b) => Number(a)-Number(b));
  const channelIndex = new Map(channels.map((ch,i) => [String(ch), i]));
  const cellW = Math.max(1, (w-2*m) / 360);
  const cellH = Math.max(2, (h-2*m) / Math.max(1, channels.length));
  for (const p of rows) {
    if (p.angle === null || p.channelIndex === null || p.channelIndex === undefined) continue;
    const chIndex = channelIndex.get(String(p.channelIndex));
    if (chIndex === undefined) continue;
    const x = m + (Number(p.angle) % 360) / 360 * (w-2*m);
    const y = m + chIndex * cellH;
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.globalAlpha = 0.70;
    ctx.fillRect(x,y,Math.max(1.5,cellW),Math.max(1.5,cellH));
  }
  ctx.globalAlpha = 1;
}
function renderCards(report) {
  const r = report.rawSummary || {};
  const items = [['raw rays',r.rawRayCount],['detected',r.detectedRayCount],['channels',r.channelCount],['age sec',r.ageSeconds ?? 'n/a'],['dist min',fmt(r.distanceMin,1)],['dist mean',fmt(r.distanceMean,1)],['angle',fmt(r.angleMin,0)+'..'+fmt(r.angleMax,0)],['vertical',fmt(r.verticalMin,1)+'..'+fmt(r.verticalMax,1)]];
  document.getElementById('cards').innerHTML = items.map(([k,v]) => `<div class="card"><span>${k}</span><strong>${v ?? 'n/a'}</strong></div>`).join('');
}
function renderTables(report) {
  const d = report.rawDistributions || {};
  document.getElementById('channelRows').innerHTML = (d.channels || []).map(r => `<tr><td>${r.channelIndex ?? 'n/a'}</td><td>${r.rayCount}</td><td>${r.detectedCount}</td><td>${fmt(r.verticalAngle,2)}</td><td>${fmt(r.distanceMean,2)}</td></tr>`).join('') || '<tr><td colspan="5">No data</td></tr>';
  document.getElementById('angleBins').innerHTML = (d.angleBins10Deg || []).map(r => `<tr><td>${fmt(r.bin,0)}</td><td>${r.count}</td></tr>`).join('') || '<tr><td colspan="2">No data</td></tr>';
  document.getElementById('distanceBins').innerHTML = (d.distanceBins5m || []).map(r => `<tr><td>${fmt(r.bin,0)}m</td><td>${r.count}</td></tr>`).join('') || '<tr><td colspan="2">No data</td></tr>';
  document.getElementById('pointRows').innerHTML = (report.rawPointRows || []).map(r => `<tr><td>${r.index}</td><td>${r.detected?'Y':'N'}</td><td>${r.channelIndex ?? 'n/a'}</td><td>${fmt(r.angle,1)}</td><td>${fmt(r.signedAngle,1)}</td><td>${fmt(r.verticalAngle,2)}</td><td>${fmt(r.distance,2)}</td><td>${fmt(r.horizontalRange,2)}</td><td>${fmt(r.x,2)}</td><td>${fmt(r.y,2)}</td><td>${fmt(r.z,2)}</td></tr>`).join('') || '<tr><td colspan="11">No data</td></tr>';
  document.getElementById('frontSliceRows').innerHTML = (report.frontSlice || []).map(r => `<tr><td>${r.index}</td><td>${r.channelIndex ?? 'n/a'}</td><td>${fmt(r.signedAngle,1)}</td><td>${fmt(r.verticalAngle,2)}</td><td>${fmt(r.distance,2)}</td><td>${fmt(r.horizontalRange,2)}</td><td>${fmt(r.x,2)}</td><td>${fmt(r.y,2)}</td><td>${fmt(r.z,2)}</td></tr>`).join('') || '<tr><td colspan="9">No data</td></tr>';
}
function renderDiagnosis(report) {
  const r = report.rawSummary || {};
  document.getElementById('diagnosis').textContent = [
    `Raw rays: ${r.rawRayCount} | detected: ${r.detectedRayCount} | detectionRate: ${fmt((r.detectionRate || 0)*100,1)}%`,
    `Distance: min ${fmt(r.distanceMin,2)}m | mean ${fmt(r.distanceMean,2)}m | max ${fmt(r.distanceMax,2)}m`,
    `Angles: ${fmt(r.angleMin,1)}..${fmt(r.angleMax,1)} deg`,
    `Vertical: ${fmt(r.verticalMin,2)}..${fmt(r.verticalMax,2)} deg | channels: ${r.channelCount}`,
    `Age: ${r.ageSeconds ?? 'n/a'} seconds`,
    '',
    'If source URL is set, this standalone page reads raw/analyzed data from that controller without owning its routes.'
  ].join('\n');
}
async function refresh() {
  const source = document.getElementById('sourceUrl').value.trim();
  const url = '/lidar_only_status' + (source ? '?source=' + encodeURIComponent(source) : '');
  const report = await (await fetch(url, {cache:'no-store'})).json();
  drawTop(report);
  drawFrontAngular(report);
  drawSideProfile(report);
  drawWorldProjection(report, xzCanvas, xzCtx, 'xz');
  drawWorldProjection(report, xyCanvas, xyCtx, 'xy');
  drawChannelHeatmap(report);
  renderCards(report); renderTables(report); renderDiagnosis(report);
  document.getElementById('json').textContent = JSON.stringify(report, null, 2);
}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Standalone LiDAR inspector: http://127.0.0.1:5051/lidar_only_view")
    print("Standalone LiDAR JSON: http://127.0.0.1:5051/lidar_only_status")
    app.run(host="0.0.0.0", port=5051, threaded=True)
