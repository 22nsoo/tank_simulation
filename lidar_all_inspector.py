from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from math import cos, radians
from statistics import mean
from typing import Any

from flask import jsonify

import aim_drive_fire_modes as controller


aim = controller.aim
app = controller.app


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def raw_lidar_items(raw_info: dict[str, Any]) -> list[dict[str, Any]]:
    points = raw_info.get("lidarPoints", [])
    return points if isinstance(points, list) else []


def summarize_raw_lidar(raw_info: dict[str, Any]) -> dict[str, Any]:
    items = raw_lidar_items(raw_info)
    detected = [item for item in items if isinstance(item, dict) and bool(item.get("isDetected", False))]

    distances = [
        safe_float(item.get("distance"))
        for item in detected
        if safe_float(item.get("distance")) is not None
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
        if isinstance(item, dict) and item.get("channelIndex") is not None
    }

    return {
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


def make_raw_top_points(raw_info: dict[str, Any], limit: int = 12000) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in raw_lidar_items(raw_info):
        if not isinstance(item, dict) or not bool(item.get("isDetected", False)):
            continue

        distance = safe_float(item.get("distance"))
        angle = safe_float(item.get("angle"))
        vertical = safe_float(item.get("verticalAngle"), 0.0)
        if distance is None or angle is None:
            continue

        horizontal_range = float(distance) * cos(radians(float(vertical or 0.0)))
        points.append(
            {
                "angle": round(float(angle), 3),
                "verticalAngle": round(float(vertical or 0.0), 3),
                "distance": round(float(distance), 3),
                "horizontalRange": round(horizontal_range, 3),
                "channelIndex": item.get("channelIndex"),
            }
        )

        if len(points) >= limit:
            break

    return points


def point_position(item: dict[str, Any]) -> dict[str, float | None]:
    pos = item.get("position", {}) if isinstance(item.get("position"), dict) else {}
    return {
        "x": safe_float(pos.get("x")),
        "y": safe_float(pos.get("y")),
        "z": safe_float(pos.get("z")),
    }


def make_raw_point_rows(raw_info: dict[str, Any], limit: int = 600) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_lidar_items(raw_info)):
        if not isinstance(item, dict):
            continue

        distance = safe_float(item.get("distance"))
        angle = safe_float(item.get("angle"))
        vertical = safe_float(item.get("verticalAngle"))
        detected = bool(item.get("isDetected", False))
        pos = point_position(item)
        horizontal_range = (
            float(distance) * cos(radians(float(vertical or 0.0)))
            if distance is not None
            else None
        )

        rows.append(
            {
                "index": index,
                "detected": detected,
                "channelIndex": item.get("channelIndex"),
                "angle": round(float(angle), 3) if angle is not None else None,
                "verticalAngle": round(float(vertical), 3) if vertical is not None else None,
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


def make_distribution(values: list[float], bin_size: float, limit: int | None = None) -> list[dict[str, Any]]:
    bins: dict[float, list[float]] = defaultdict(list)
    for value in values:
        center = round((round(value / bin_size) * bin_size), 3)
        bins[center].append(value)
    rows = [
        {"bin": key, "count": len(items)}
        for key, items in sorted(bins.items(), key=lambda pair: pair[0])
    ]
    return rows[:limit] if limit else rows


def make_channel_distribution(raw_info: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_lidar_items(raw_info):
        if isinstance(item, dict):
            grouped[item.get("channelIndex", "none")].append(item)

    rows = []
    for channel, items in grouped.items():
        detected = [item for item in items if bool(item.get("isDetected", False))]
        distances = [
            safe_float(item.get("distance"))
            for item in detected
            if safe_float(item.get("distance")) is not None
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
        if isinstance(item, dict) and bool(item.get("isDetected", False))
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
    distances = [
        safe_float(item.get("distance"))
        for item in detected
        if safe_float(item.get("distance")) is not None
    ]
    return {
        "angleBins10Deg": make_distribution([float(v) for v in angles], 10.0),
        "verticalBins1Deg": make_distribution([float(v) for v in verticals], 1.0),
        "distanceBins5m": make_distribution([float(v) for v in distances], 5.0),
        "channels": make_channel_distribution(raw_info),
    }


def make_front_slice(raw_info: dict[str, Any], half_angle_deg: float = 2.0) -> list[dict[str, Any]]:
    rows = []
    for row in make_raw_point_rows(raw_info, limit=20000):
        angle = row.get("angle")
        if angle is None:
            continue
        signed = ((float(angle) + 180.0) % 360.0) - 180.0
        if abs(signed) <= half_angle_deg:
            row["signedAngle"] = round(signed, 3)
            rows.append(row)
    rows.sort(key=lambda item: (item.get("verticalAngle") is None, item.get("verticalAngle") or 0.0))
    return rows


def filter_constants() -> dict[str, Any]:
    names = [
        "MAX_DISTANCE_M",
        "OBJECT_VERTICAL_MIN_DEG",
        "OBJECT_VERTICAL_MAX_DEG",
        "OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M",
        "OBJECT_AZIMUTH_BIN_WIDTH_DEG",
        "OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG",
        "OBJECT_CLUSTER_MAX_DISTANCE_GAP_M",
        "BULKY_MIN_WIDTH_M",
        "THIN_MAX_WIDTH_M",
        "TERRAIN_VERTICAL_MIN_DEG",
        "TERRAIN_VERTICAL_MAX_DEG",
        "TERRAIN_FRONT_LIMIT_DEG",
        "TERRAIN_ANALYSIS_MAX_DISTANCE_M",
        "PRIORITY_NEAR_MAX_DISTANCE_M",
        "BODY_ALIGN_TARGET_MAX_DISTANCE_M",
    ]
    return {name: getattr(aim, name, None) for name in names}


def build_lidar_report() -> dict[str, Any]:
    with aim.state_lock:
        raw_info = deepcopy(aim.latest_raw_info)
        scan = deepcopy(aim.latest_state)
        vision = deepcopy(aim.vision_state)
        action_debug = deepcopy(aim.action_debug_state)
        body_alignment = deepcopy(aim.latest_state.get("bodyAlignment", {}))
        priority = deepcopy(aim.latest_state.get("fusionPriorityQueue", []))
        fire_status = aim.fire_readiness_status()

    raw_summary = summarize_raw_lidar(raw_info)
    raw_top_points = make_raw_top_points(raw_info)
    raw_point_rows = make_raw_point_rows(raw_info)
    raw_distributions = make_raw_distributions(raw_info)
    front_slice = make_front_slice(raw_info)
    raw_objects = scan.get("rawObjects", []) if isinstance(scan.get("rawObjects"), list) else []
    terrain_connected = [obj for obj in raw_objects if obj.get("terrainConnected")]
    object_like = [obj for obj in raw_objects if not obj.get("terrainConnected")]

    return {
        "rawSummary": raw_summary,
        "rawTopPoints": raw_top_points,
        "rawPointRows": raw_point_rows,
        "rawDistributions": raw_distributions,
        "frontSlice": front_slice,
        "filters": filter_constants(),
        "scan": scan,
        "counts": {
            "rawObjectCount": scan.get("rawObjectCount", len(raw_objects)),
            "objectCandidatePointCount": scan.get("objectCandidatePointCount"),
            "terrainConnectedRawObjects": len(terrain_connected),
            "objectLikeRawObjects": len(object_like),
            "provisionalObjectCount": scan.get("provisionalObjectCount", 0),
            "trackedObjectCount": scan.get("trackedObjectCount", 0),
            "confirmedObjectCount": scan.get("confirmedObjectCount", 0),
            "fusionPriorityQueueCount": scan.get("fusionPriorityQueueCount", len(priority)),
        },
        "classification": {
            "rawObjects": raw_objects,
            "terrainConnectedRawObjects": terrain_connected,
            "objectLikeRawObjects": object_like,
            "provisionalObjects": scan.get("provisionalObjects", []),
            "trackedObjects": scan.get("trackedObjects", []),
            "confirmedObjects": scan.get("confirmedObjects", []),
            "fusionPriorityQueue": priority,
            "primaryFusionTarget": scan.get("primaryFusionTarget"),
        },
        "terrain": {
            "decision": scan.get("terrainDecision", {}),
            "sectors": scan.get("terrainSectors", []),
            "contourPoints": scan.get("contourPoints", []),
            "frontVerticalProfile": scan.get("frontVerticalProfile", {}),
            "localGroundY": scan.get("localGroundY"),
            "estimatedGroundY": scan.get("estimatedGroundY"),
            "lidarOriginY": scan.get("lidarOriginY"),
            "localGroundGridCellCount": scan.get("localGroundGridCellCount"),
        },
        "alignment": body_alignment,
        "vision": vision,
        "fireControl": fire_status,
        "actionDebug": action_debug,
        "modeState": controller.mode_state(),
        "rawInfo": raw_info,
    }


@app.route("/lidar_all_status", methods=["GET"])
def lidar_all_status():
    return jsonify(build_lidar_report())


@app.route("/lidar_all_view", methods=["GET"])
def lidar_all_view():
    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LiDAR All Inspector</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#181c1f; --line:#3a4248; --text:#f2f5f7; --muted:#a9b3ba; --green:#35e835; --yellow:#ffca3a; --red:#ff3030; --cyan:#4cc9f0; }
    * { box-sizing: border-box; }
    body { margin: 16px; background: var(--bg); color: var(--text); font-family: Arial, sans-serif; }
    h1, h2, p { margin: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 15px; margin-bottom: 8px; }
    p, .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: minmax(560px, 1fr) minmax(420px, 520px); gap: 14px; align-items: start; margin-top: 14px; }
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
    .toolbar { display:flex; gap:8px; align-items:center; justify-content:space-between; margin-top: 8px; }
    button { border: 1px solid var(--line); background: #20262a; color: var(--text); padding: 7px 10px; cursor:pointer; }
    .wide { grid-column: 1 / -1; }
    .subgrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .table-wrap { max-height: 420px; overflow: auto; border: 1px solid #30373d; }
  </style>
</head>
<body>
  <h1>LiDAR All Inspector</h1>
  <p>Raw input, terrain, object candidates, tracks, fusion priority, body alignment, fire state, and mode state in one place.</p>
  <div class="cards" id="cards"></div>
  <div class="grid">
    <div class="panel">
      <h2>Raw 360 Point Cloud + Analyzed Markers</h2>
      <canvas id="top" width="900" height="900"></canvas>
      <p class="muted">Small dots are raw detected LiDAR hits. Bright green dots are contour points. Hollow/solid rings are provisional/confirmed object candidates.</p>
    </div>
    <div class="panel">
      <div class="toolbar">
        <h2>Current Diagnosis</h2>
        <button onclick="refresh()">Refresh</button>
      </div>
      <pre id="diagnosis">Waiting...</pre>
    </div>
    <div class="panel">
      <h2>Objects and Priority Queue</h2>
      <table>
        <thead><tr><th>stage</th><th>label</th><th>dist</th><th>angle</th><th>width</th><th>reason</th></tr></thead>
        <tbody id="objects"></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Point Cloud Distributions</h2>
      <div class="subgrid">
        <div>
          <p class="muted">Channels</p>
          <div class="table-wrap"><table><thead><tr><th>ch</th><th>rays</th><th>hit</th><th>v</th><th>mean m</th></tr></thead><tbody id="channelRows"></tbody></table></div>
        </div>
        <div>
          <p class="muted">Angle bins 10 deg</p>
          <div class="table-wrap"><table><thead><tr><th>angle</th><th>count</th></tr></thead><tbody id="angleBins"></tbody></table></div>
        </div>
        <div>
          <p class="muted">Distance bins 5 m</p>
          <div class="table-wrap"><table><thead><tr><th>dist</th><th>count</th></tr></thead><tbody id="distanceBins"></tbody></table></div>
        </div>
      </div>
    </div>
    <div class="panel wide">
      <h2>Raw Point Rows</h2>
      <p class="muted">First 600 raw rows, including detected flag, channel, azimuth, vertical angle, distance, horizontal range, and world position.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>hit</th><th>ch</th><th>angle</th><th>vertical</th><th>dist</th><th>hRange</th><th>x</th><th>y</th><th>z</th></tr></thead>
        <tbody id="pointRows"></tbody>
      </table></div>
    </div>
    <div class="panel wide">
      <h2>Body-Forward Slice</h2>
      <p class="muted">Rows within +/-2 degrees of body-forward, sorted by vertical angle. This shows exactly what the side profile is built from.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>ch</th><th>signed</th><th>vertical</th><th>dist</th><th>hRange</th><th>x</th><th>y</th><th>z</th></tr></thead>
        <tbody id="frontSliceRows"></tbody>
      </table></div>
    </div>
    <div class="panel">
      <h2>Full JSON</h2>
      <pre id="json"></pre>
    </div>
  </div>
<script>
const topCanvas = document.getElementById('top');
const topCtx = topCanvas.getContext('2d');
const MAX_DISTANCE = 120;

function fmt(v, d=2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : 'n/a';
}
function colorByDistance(distance) {
  const d = Number(distance);
  if (d <= 20) return '#ff3030';
  if (d <= 50) return '#ffca3a';
  return '#35e835';
}
function objectColor(kind) {
  if (kind === 'thin') return '#4cc9f0';
  if (kind === 'bulky') return '#f72585';
  return '#f1fa8c';
}
function polar(angleDeg, distance, cx, cy, radius) {
  const a = Number(angleDeg) * Math.PI / 180;
  const r = Math.min(Number(distance) || 0, MAX_DISTANCE) / MAX_DISTANCE * radius;
  return {x: cx + Math.sin(a) * r, y: cy - Math.cos(a) * r};
}
function drawTop(report) {
  const ctx = topCtx, w = topCanvas.width, h = topCanvas.height, cx = w/2, cy = h/2, radius = Math.min(w,h)*0.46;
  ctx.clearRect(0,0,w,h);
  ctx.font = '13px Arial';
  ctx.strokeStyle = '#4b5359';
  ctx.fillStyle = '#bfc7cc';
  for (const d of [30,60,90,120]) {
    const r = radius * d / MAX_DISTANCE;
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke();
    ctx.fillText(d+'m', cx+5, cy-r+16);
  }
  ctx.strokeStyle = '#89939a';
  ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx,cy-radius); ctx.stroke();
  ctx.fillText('0 deg body-forward', cx+8, cy-radius+16);
  ctx.fillText('-90', cx-radius+4, cy-8);
  ctx.fillText('+90', cx+radius-38, cy-8);

  ctx.save();
  ctx.globalAlpha = 0.45;
  for (const p of report.rawTopPoints || []) {
    const q = polar(p.angle, p.horizontalRange || p.distance, cx, cy, radius);
    ctx.fillStyle = colorByDistance(p.distance);
    ctx.beginPath(); ctx.arc(q.x,q.y,1.5,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();

  for (const p of report.terrain?.contourPoints || []) {
    const q = polar(p.angle, p.distance, cx, cy, radius);
    ctx.fillStyle = '#35e835';
    ctx.beginPath(); ctx.arc(q.x,q.y,2.7,0,Math.PI*2); ctx.fill();
  }

  for (const sector of report.terrain?.sectors || []) {
    const boundary = Math.max(3, Math.min(Number(sector.hazardBoundaryRange || 20), MAX_DISTANCE));
    const qRadius = radius * boundary / MAX_DISTANCE;
    const center = Number(sector.centerAngle || 0) * Math.PI / 180;
    const half = 4.5 * Math.PI / 180;
    ctx.strokeStyle = sector.state === 'blocked' ? '#ff3030' : (sector.state === 'caution' ? '#ffca3a' : '#35e835');
    ctx.lineWidth = sector.state === 'blocked' ? 10 : 6;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(18, qRadius), -Math.PI/2 + center - half, -Math.PI/2 + center + half);
    ctx.stroke();
  }
  ctx.lineWidth = 1;

  const drawObj = (obj, solid) => {
    const q = polar(obj.centerAngle, obj.medianDistance || obj.nearestDistance, cx, cy, radius);
    ctx.strokeStyle = objectColor(obj.geometryClass);
    ctx.fillStyle = objectColor(obj.geometryClass);
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(q.x,q.y, solid ? 9 : 6, 0, Math.PI*2);
    if (solid) ctx.fill(); else ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = obj.terrainConnected ? '#ffca3a' : '#fff';
    const label = (obj.candidateLabel || 'UK') + ' ' + fmt(obj.nearestDistance,1) + 'm';
    ctx.fillText(label, q.x+10, q.y-8);
  };
  for (const obj of report.classification?.provisionalObjects || []) drawObj(obj, false);
  for (const obj of report.classification?.confirmedObjects || []) drawObj(obj, true);

  ctx.fillStyle = '#4cc9f0';
  ctx.beginPath(); ctx.arc(cx,cy,8,0,Math.PI*2); ctx.fill();
}
function renderCards(report) {
  const c = report.counts || {};
  const r = report.rawSummary || {};
  const terrain = report.terrain?.decision || {};
  const alignment = report.alignment || {};
  const items = [
    ['raw rays', r.rawRayCount],
    ['detected', r.detectedRayCount],
    ['objects', c.rawObjectCount],
    ['provisional', c.provisionalObjectCount],
    ['confirmed', c.confirmedObjectCount],
    ['priority', c.fusionPriorityQueueCount],
    ['terrain', terrain.state || 'n/a'],
    ['align', alignment.reason || 'n/a'],
  ];
  document.getElementById('cards').innerHTML = items.map(([k,v]) => `<div class="card"><span>${k}</span><strong>${v ?? 'n/a'}</strong></div>`).join('');
}
function renderObjects(report) {
  const rows = [];
  const add = (stage, list) => {
    for (const o of list || []) rows.push({stage, ...o});
  };
  add('raw', report.classification?.rawObjects);
  add('provisional', report.classification?.provisionalObjects);
  add('confirmed', report.classification?.confirmedObjects);
  add('priority', report.classification?.fusionPriorityQueue);
  document.getElementById('objects').innerHTML = rows.slice(0, 80).map(o => `
    <tr>
      <td>${o.stage}${o.terrainConnected ? '<br>terrain' : ''}</td>
      <td>${o.candidateLabel || o.geometryClass || 'n/a'}</td>
      <td>${fmt(o.nearestDistance,1)}</td>
      <td>${fmt(o.centerAngle,1)}</td>
      <td>${fmt(o.estimatedWidth,2)}</td>
      <td>${o.shapeReason || o.fusionPriorityReason || o.terrainSuppressionReason || ''}</td>
    </tr>`).join('') || '<tr><td colspan="6">No objects</td></tr>';
}
function renderPointDetails(report) {
  const distributions = report.rawDistributions || {};
  const channels = distributions.channels || [];
  document.getElementById('channelRows').innerHTML = channels.map(row => `
    <tr><td>${row.channelIndex ?? 'n/a'}</td><td>${row.rayCount}</td><td>${row.detectedCount}</td><td>${fmt(row.verticalAngle,2)}</td><td>${fmt(row.distanceMean,2)}</td></tr>
  `).join('') || '<tr><td colspan="5">No channel data</td></tr>';

  document.getElementById('angleBins').innerHTML = (distributions.angleBins10Deg || []).map(row => `
    <tr><td>${fmt(row.bin,0)}</td><td>${row.count}</td></tr>
  `).join('') || '<tr><td colspan="2">No angle bins</td></tr>';

  document.getElementById('distanceBins').innerHTML = (distributions.distanceBins5m || []).map(row => `
    <tr><td>${fmt(row.bin,0)}m</td><td>${row.count}</td></tr>
  `).join('') || '<tr><td colspan="2">No distance bins</td></tr>';

  const pointCell = value => value === null || value === undefined ? 'n/a' : value;
  document.getElementById('pointRows').innerHTML = (report.rawPointRows || []).map(row => `
    <tr>
      <td>${row.index}</td><td>${row.detected ? 'Y' : 'N'}</td><td>${pointCell(row.channelIndex)}</td>
      <td>${fmt(row.angle,1)}</td><td>${fmt(row.verticalAngle,2)}</td><td>${fmt(row.distance,2)}</td><td>${fmt(row.horizontalRange,2)}</td>
      <td>${fmt(row.x,2)}</td><td>${fmt(row.y,2)}</td><td>${fmt(row.z,2)}</td>
    </tr>
  `).join('') || '<tr><td colspan="10">No point rows</td></tr>';

  document.getElementById('frontSliceRows').innerHTML = (report.frontSlice || []).map(row => `
    <tr>
      <td>${row.index}</td><td>${pointCell(row.channelIndex)}</td><td>${fmt(row.signedAngle,2)}</td>
      <td>${fmt(row.verticalAngle,2)}</td><td>${fmt(row.distance,2)}</td><td>${fmt(row.horizontalRange,2)}</td>
      <td>${fmt(row.x,2)}</td><td>${fmt(row.y,2)}</td><td>${fmt(row.z,2)}</td>
    </tr>
  `).join('') || '<tr><td colspan="9">No body-forward slice rows</td></tr>';
}
function renderDiagnosis(report) {
  const r = report.rawSummary || {};
  const c = report.counts || {};
  const t = report.terrain?.decision || {};
  const a = report.alignment || {};
  const fire = report.fireControl || {};
  const lines = [
    `Raw rays: ${r.rawRayCount} | detected: ${r.detectedRayCount} | detectionRate: ${fmt((r.detectionRate || 0)*100,1)}%`,
    `Distance: min ${fmt(r.distanceMin,2)}m | mean ${fmt(r.distanceMean,2)}m | max ${fmt(r.distanceMax,2)}m`,
    `Angles: ${fmt(r.angleMin,1)}..${fmt(r.angleMax,1)} deg | vertical: ${fmt(r.verticalMin,1)}..${fmt(r.verticalMax,1)} deg | channels: ${r.channelCount}`,
    `Terrain: ${t.state || 'n/a'} | action ${t.recommendedAction || 'n/a'} | reason ${t.reason || 'n/a'}`,
    `Objects: raw ${c.rawObjectCount} | terrain-connected ${c.terrainConnectedRawObjects} | object-like ${c.objectLikeRawObjects}`,
    `Tracks: provisional ${c.provisionalObjectCount} | tracked ${c.trackedObjectCount} | confirmed ${c.confirmedObjectCount} | priority ${c.fusionPriorityQueueCount}`,
    `Body alignment: enabled=${a.enabled} | aligned=${a.aligned} | reason=${a.reason || 'n/a'}`,
    `Fire: ready=${fire.ready} | reason=${fire.reason || 'n/a'} | lidarYaw=${fire.lidarYawErrorDeg ?? 'n/a'} | dist=${fire.lidarDistance ?? 'n/a'}`,
    '',
    'Important filters:',
    JSON.stringify(report.filters || {}, null, 2),
  ];
  document.getElementById('diagnosis').textContent = lines.join('\n');
}
async function refresh() {
  const report = await (await fetch('/lidar_all_status', {cache:'no-store'})).json();
  drawTop(report);
  renderCards(report);
  renderObjects(report);
  renderPointDetails(report);
  renderDiagnosis(report);
  document.getElementById('json').textContent = JSON.stringify(report, null, 2);
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("LiDAR all inspector: http://127.0.0.1:5000/lidar_all_view")
    print("LiDAR all JSON: http://127.0.0.1:5000/lidar_all_status")
    app.run(host="0.0.0.0", port=5000, threaded=True)
