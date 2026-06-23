const stateUrl = "/api/state";
const cameraImage = document.querySelector("#cameraImage");
const overlayCanvas = document.querySelector("#overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");
const mapCanvas = document.querySelector("#mapCanvas");
const mapCtx = mapCanvas.getContext("2d");

const el = (id) => document.querySelector(`#${id}`);
const MAP_SIZE = 300;
const MAP_PAD = 24;
let latestDashboardData = null;

const plotLayoutBase = {
  paper_bgcolor: "#0b0f0e",
  plot_bgcolor: "#0b0f0e",
  font: { color: "#edf3ee", family: "Segoe UI, Malgun Gothic, sans-serif" },
  margin: { l: 48, r: 20, t: 42, b: 42 },
  xaxis: { gridcolor: "#22302b", zerolinecolor: "#33413c" },
  yaxis: { gridcolor: "#22302b", zerolinecolor: "#33413c" },
};

function fmt(value, digits = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "-";
}

function num(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function actionLabel(action) {
  if (!action) return "-";

  const parts = [];
  for (const key of ["moveWS", "moveAD", "turretQE", "turretRF"]) {
    const value = action[key];
    if (value && value.command) parts.push(`${key}:${value.command}(${value.weight})`);
  }
  if (action.fire) parts.push("FIRE");
  return parts.length ? parts.join(" / ") : "STOP";
}

function mapMetrics() {
  const rect = mapCanvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const scaleX = (width - MAP_PAD * 2) / MAP_SIZE;
  const scaleZ = (height - MAP_PAD * 2) / MAP_SIZE;

  return {
    rect,
    width,
    height,
    scaleX,
    scaleZ,
    toX: (x) => MAP_PAD + x * scaleX,
    toY: (z) => height - MAP_PAD - z * scaleZ,
    toWorld: (clientX, clientY) => {
      const x = (clientX - rect.left - MAP_PAD) / scaleX;
      const z = (height - MAP_PAD - (clientY - rect.top)) / scaleZ;
      return {
        x: clamp(x, 0, MAP_SIZE),
        z: clamp(z, 0, MAP_SIZE),
      };
    },
  };
}

function drawOverlay(data) {
  const detections = data.detections || [];
  const roi = data.roi;
  const rect = overlayCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  overlayCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  overlayCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

  if (roi) {
    const xMin = clamp(Number(roi.x_min), 0, 1);
    const xMax = clamp(Number(roi.x_max), 0, 1);
    const yMin = clamp(Number(roi.y_min), 0, 1);
    const yMax = clamp(Number(roi.y_max), 0, 1);

    if (xMax > xMin && yMax > yMin) {
      const x = xMin * rect.width;
      const y = yMin * rect.height;
      const w = (xMax - xMin) * rect.width;
      const h = (yMax - yMin) * rect.height;

      overlayCtx.save();
      overlayCtx.strokeStyle = "#ff3b30";
      overlayCtx.lineWidth = 3;
      overlayCtx.setLineDash([8, 6]);
      overlayCtx.strokeRect(x, y, w, h);
      overlayCtx.setLineDash([]);
      overlayCtx.fillStyle = "rgba(255, 59, 48, 0.85)";
      overlayCtx.fillRect(x, Math.max(0, y - 24), 58, 24);
      overlayCtx.fillStyle = "#ffffff";
      overlayCtx.font = "13px Segoe UI";
      overlayCtx.fillText("ROI", x + 8, Math.max(17, y - 7));
      overlayCtx.restore();
    }
  }

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
    const iw = det.imageWidth || rect.width;
    const ih = det.imageHeight || rect.height;
    const x = (x1 / iw) * rect.width;
    const y = (y1 / ih) * rect.height;
    const w = ((x2 - x1) / iw) * rect.width;
    const h = ((y2 - y1) / ih) * rect.height;

    overlayCtx.strokeStyle = det.color || "#35d07f";
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([]);
    overlayCtx.strokeRect(x, y, w, h);
    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.72)";
    overlayCtx.fillRect(x, Math.max(0, y - 24), Math.max(110, w), 24);
    overlayCtx.fillStyle = "#edf3ee";
    overlayCtx.font = "13px Segoe UI";
    const label = det.displayName || (
      det.distanceText
        ? `${det.className} ${det.distanceText}`
        : `${det.className} ${(Number(det.confidence || 0) * 100).toFixed(0)}%`
    );
    overlayCtx.fillText(label, x + 6, Math.max(17, y - 7));
  }

  overlayCtx.strokeStyle = "rgba(255,255,255,0.42)";
  overlayCtx.setLineDash([5, 5]);
  overlayCtx.beginPath();
  overlayCtx.moveTo(rect.width / 2, 0);
  overlayCtx.lineTo(rect.width / 2, rect.height);
  overlayCtx.moveTo(0, rect.height / 2);
  overlayCtx.lineTo(rect.width, rect.height / 2);
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
}

function drawObstacleBox(obs, metrics, margin = 0, strokeOnly = false) {
  const xMin = clamp(Number(obs.x_min) - margin, 0, MAP_SIZE);
  const xMax = clamp(Number(obs.x_max) + margin, 0, MAP_SIZE);
  const zMin = clamp(Number(obs.z_min) - margin, 0, MAP_SIZE);
  const zMax = clamp(Number(obs.z_max) + margin, 0, MAP_SIZE);

  const x = metrics.toX(xMin);
  const y = metrics.toY(zMax);
  const w = (xMax - xMin) * metrics.scaleX;
  const h = (zMax - zMin) * metrics.scaleZ;

  if (strokeOnly) mapCtx.strokeRect(x, y, w, h);
  else mapCtx.fillRect(x, y, w, h);
}

function drawMap(data) {
  const metrics = mapMetrics();
  const dpr = window.devicePixelRatio || 1;

  mapCanvas.width = Math.max(1, Math.floor(metrics.width * dpr));
  mapCanvas.height = Math.max(1, Math.floor(metrics.height * dpr));
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mapCtx.clearRect(0, 0, metrics.width, metrics.height);

  mapCtx.strokeStyle = "#22302b";
  mapCtx.lineWidth = 1;
  for (let i = 0; i <= MAP_SIZE; i += 50) {
    mapCtx.beginPath();
    mapCtx.moveTo(metrics.toX(i), metrics.toY(0));
    mapCtx.lineTo(metrics.toX(i), metrics.toY(MAP_SIZE));
    mapCtx.moveTo(metrics.toX(0), metrics.toY(i));
    mapCtx.lineTo(metrics.toX(MAP_SIZE), metrics.toY(i));
    mapCtx.stroke();
  }

  const mapObstacles = (data.map_obstacles && data.map_obstacles.length)
    ? data.map_obstacles
    : (data.active_obstacles && data.active_obstacles.length)
      ? data.active_obstacles
      : (data.obstacles || []);

  for (const obs of mapObstacles) {
    mapCtx.fillStyle = obs.source === "lidar"
      ? "rgba(53, 208, 127, 0.24)"
      : obs.source === "slam"
        ? "rgba(38, 166, 154, 0.24)"
        : "rgba(229, 186, 79, 0.35)";
    drawObstacleBox(obs, metrics);
  }

  mapCtx.save();
  mapCtx.strokeStyle = "rgba(255, 107, 94, 0.72)";
  mapCtx.lineWidth = 1.5;
  mapCtx.setLineDash([6, 5]);
  const marginObstacles = (data.obstacle_margins && data.obstacle_margins.length)
    ? data.obstacle_margins
    : (data.active_obstacles || []).map((obs) => ({
      ...obs,
      margin: Number(data.navigation_debug?.obstacle_clear_margin || 0),
    }));
  for (const obs of marginObstacles) {
    if (obs.source === "lidar" || obs.source === "slam") continue;
    drawObstacleBox(obs, metrics, Number(obs.margin || 0), true);
  }
  mapCtx.restore();

  const path = data.path || [];
  if (path.length > 1) {
    mapCtx.strokeStyle = "#35d07f";
    mapCtx.lineWidth = 2;
    mapCtx.beginPath();
    mapCtx.moveTo(metrics.toX(Number(path[0].x || 0)), metrics.toY(Number(path[0].z || 0)));
    for (const point of path.slice(1)) {
      mapCtx.lineTo(metrics.toX(Number(point.x || 0)), metrics.toY(Number(point.z || 0)));
    }
    mapCtx.stroke();
  }

  const waypoint = path[data.path_index || 0];
  if (waypoint) {
    mapCtx.fillStyle = "#edf3ee";
    mapCtx.beginPath();
    mapCtx.arc(metrics.toX(Number(waypoint.x || 0)), metrics.toY(Number(waypoint.z || 0)), 4, 0, Math.PI * 2);
    mapCtx.fill();
  }

  const pos = data.position || {};
  mapCtx.fillStyle = "#35d07f";
  mapCtx.beginPath();
  mapCtx.arc(metrics.toX(Number(pos.x || 0)), metrics.toY(Number(pos.z || 0)), 7, 0, Math.PI * 2);
  mapCtx.fill();

  const dest = data.destination || {};
  mapCtx.strokeStyle = "#ff6b5e";
  mapCtx.lineWidth = 3;
  mapCtx.beginPath();
  mapCtx.arc(metrics.toX(Number(dest.x || 0)), metrics.toY(Number(dest.z || 0)), 9, 0, Math.PI * 2);
  mapCtx.stroke();
}

async function postDestination(x, z) {
  const destX = Math.round(clamp(Number(x), 0, MAP_SIZE) * 10) / 10;
  const destZ = Math.round(clamp(Number(z), 0, MAP_SIZE) * 10) / 10;

  el("destX").value = destX;
  el("destZ").value = destZ;

  await fetch("/set_destination", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x: destX, y: 0, z: destZ }),
  });

  await refresh();
}

async function handleMapClick(event) {
  const point = mapMetrics().toWorld(event.clientX, event.clientY);
  await postDestination(point.x, point.z);
}

function updateTargetStatus(data) {
  const status = data.target_status || { state: "no_target", label: "표적 없음" };
  const target = data.primary_target;
  const strip = el("targetStrip");
  const firePanel = el("fireConfirmPanel");
  const fireButton = el("confirmFireBtn");
  const state = status.state || "no_target";

  strip.dataset.state = state;
  el("targetStateBadge").textContent = status.label || "표적 없음";
  el("targetStateText").textContent = target
    ? `${target.className} ${(Number(target.confidence || 0) * 100).toFixed(1)}%`
    : "-";

  el("targetErrorText").textContent = Number.isFinite(Number(status.error_x))
    ? `x ${fmt(status.error_x, 3)} / y ${fmt(status.error_y, 3)}`
    : "-";

  el("targetConfidenceText").textContent = Number.isFinite(Number(status.confidence))
    ? `${(Number(status.confidence) * 100).toFixed(1)}%`
    : "-";

  const candidate = data.target_candidate;
  el("targetCandidateText").textContent = candidate
    ? `${candidate.className} ${(Number(candidate.confidence || 0) * 100).toFixed(1)}% / ${data.target_candidate_hits || 0}/${data.target_thresholds?.lock_frames || 2}`
    : "-";

  const readyToAsk = state === "awaiting_fire_confirm" || state === "fire_ready";
  firePanel.dataset.ready = readyToAsk ? "true" : "false";
  fireButton.disabled = !readyToAsk;

  if (state === "aiming") {
    el("fireConfirmText").textContent = data.target_retained
      ? "표적 유지 중: 전차 정지, 포탑 조준 중"
      : "표적 발견: 전차 정지, 포탑 조준 중";
  } else if (readyToAsk) {
    el("fireConfirmText").textContent = data.target_retained
      ? "유지 표적 조준 완료. 발사하겠습니까?"
      : "조준 완료. 발사하겠습니까?";
  } else if (state === "firing") {
    el("fireConfirmText").textContent = "발사 명령 전송됨";
  } else {
    el("fireConfirmText").textContent = "표적 발견 시 조준 후 발사 승인을 기다립니다";
  }
}

function updateDriveLogs(data) {
  el("logFileText").textContent = data.drive_log_file || "drive_logs/drive_log.csv";

  const logs = (data.drive_logs || []).slice(-8).reverse();
  const body = el("driveLogBody");
  body.innerHTML = "";

  if (!logs.length) {
    body.innerHTML = '<tr><td colspan="6">로그 대기 중</td></tr>';
    return;
  }

  for (const row of logs) {
    const tr = document.createElement("tr");
    const command = [row.move_ws, row.move_ad, row.turret_qe, row.turret_rf]
      .filter(Boolean)
      .join(" / ") || "STOP";

    tr.innerHTML = `
      <td>${String(row.timestamp || row.timestamp_text || "").slice(11, 19)}</td>
      <td>x ${fmt(row.x)} / z ${fmt(row.z)}</td>
      <td>${command}${row.fire ? " / FIRE" : ""}</td>
      <td>${row.target_label || "-"}</td>
      <td>${row.lidar_source || "-"} / near ${row.lidar_near_count || 0}</td>
      <td>${row.event || "-"}</td>
    `;

    body.appendChild(tr);
  }
}

function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

async function loadDbTable() {
  const table = el("dbTableSelect").value;
  const limit = Number(el("dbLimitInput").value || 100);
  const head = el("dbTableHead");
  const body = el("dbTableBody");

  try {
    const res = await fetch(`/api/db/table/${encodeURIComponent(table)}?limit=${limit}`, {
      cache: "no-store",
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "조회 실패");

    el("dbStatusText").textContent = data.available
      ? `${data.table} / ${data.rows.length} rows`
      : `DB 연결 실패: ${data.error || "unknown"}`;

    head.innerHTML = `<tr>${data.columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;

    if (!data.rows.length) {
      body.innerHTML = `<tr><td colspan="${data.columns.length}">데이터 없음</td></tr>`;
      return;
    }

    body.innerHTML = data.rows.map((row) => `
      <tr>
        ${data.columns.map((column) => `<td>${formatCell(row[column])}</td>`).join("")}
      </tr>
    `).join("");
  } catch (err) {
    el("dbStatusText").textContent = `DB 조회 실패: ${err.message}`;
    head.innerHTML = "";
    body.innerHTML = "<tr><td>조회 실패</td></tr>";
  }
}

async function clearLogs() {
  const ok = window.confirm("CSV, PostgreSQL 로그, lidar_data CSV를 초기화하고 capture_images는 최신 10개만 남길까요?");
  if (!ok) return;

  const button = el("clearLogsBtn");
  button.disabled = true;
  button.textContent = "초기화 중";

  try {
    const res = await fetch("/api/logs/clear", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "초기화 실패");

    el("driveLogBody").innerHTML = '<tr><td colspan="6">로그 대기 중</td></tr>';
    el("dbStatusText").textContent =
      `초기화 완료 / LiDAR CSV ${data.cleared?.lidar_csv ?? 0}개 삭제 / 캡처 ${data.cleared?.capture_images ?? 0}개 정리`;
    await refreshAnalytics();
    await loadDbTable();
    await refresh();
  } catch (err) {
    el("dbStatusText").textContent = `로그 초기화 실패: ${err.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "로그+라이다 초기화";
  }
}

async function confirmFire() {
  const button = el("confirmFireBtn");
  button.disabled = true;
  button.textContent = "승인됨";

  try {
    const res = await fetch("/api/fire/confirm", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "발사 승인 실패");
    el("fireConfirmText").textContent = "발사 승인 전송됨";
    await refresh();
  } catch (err) {
    el("fireConfirmText").textContent = `발사 승인 실패: ${err.message}`;
    button.disabled = false;
  } finally {
    setTimeout(() => {
      button.textContent = "발사";
    }, 700);
  }
}

function addChatMessage(role, text) {
  const log = el("chatLog");
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  message.textContent = text;
  log.appendChild(message);
  log.scrollTop = log.scrollHeight;
  return message;
}

async function sendChat(event) {
  event.preventDefault();
  const input = el("chatInput");
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  addChatMessage("user", question);
  const pending = addChatMessage("assistant", "분석 중...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: question,
        state: latestDashboardData,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== "OK") throw new Error(data.message || "chat failed");
    pending.textContent = data.answer;
  } catch (err) {
    pending.classList.add("error");
    pending.textContent = `챗봇 오류: ${err.message}`;
  }
}

function updateText(data) {
  const detections = data.detections || [];

  el("modelName").textContent = data.model || "-";
  el("detectCount").textContent = detections.length;
  el("pathState").textContent = `${data.path_index || 0} / ${(data.path || []).length}`;
  el("eventText").textContent = data.latest_event || "-";
  el("captureName").textContent = data.capture_name || "no frame";

  const pos = data.position || {};
  const turret = data.turret || {};

  el("positionText").textContent = `x ${fmt(pos.x)} / y ${fmt(pos.y)} / z ${fmt(pos.z)}`;
  el("turretText").textContent = `x ${fmt(turret.x)} / y ${fmt(turret.y)}`;
  el("actionText").textContent = actionLabel(data.latest_action);

  const target = data.primary_target;
  el("targetText").textContent = target
    ? `${target.className} ${(Number(target.confidence || 0) * 100).toFixed(1)}%`
    : "표적 없음";

  const lidar = data.lidar || {};
  const lidarQuality = lidar.quality || {};
  const detectionRate = Number(lidarQuality.detection_rate || 0) * 100;
  el("lidarText").textContent =
    `NAV ${data.use_lidar_navigation ? "ON" : "OFF"} / ` +
    `${lidar.source || "none"} ${lidarQuality.grade || "no_data"} / ` +
    `det ${detectionRate.toFixed(1)}% / near ${lidar.near_count || 0} / ` +
    `front ${fmt(lidarQuality.front_min_distance, 1)}m / ch ${lidarQuality.channel_count || 0}`;

  const database = data.database || {};
  el("databaseText").textContent = database.available
    ? `ON / ${database.name}`
    : `OFF / ${database.error || "not connected"}`;

  const toggle = el("lidarNavToggle");
  if (toggle) toggle.checked = !!data.use_lidar_navigation;

  el("bulletText").textContent = data.latest_bullet ? JSON.stringify(data.latest_bullet) : "-";
  el("collisionText").textContent = data.latest_collision ? JSON.stringify(data.latest_collision) : "-";

  const debug = data.navigation_debug || {};
  el("obstacleDebugText").textContent =
    `active ${debug.active_obstacle_count || 0} / map ${(data.map_obstacles || []).length}`;

  el("marginDebugText").textContent =
    `cell ${fmt(debug.obstacle_cell_margin)} / ` +
    `clear ${fmt(debug.obstacle_clear_margin)} / ` +
    `size ${fmt(debug.obstacle_size_margin_factor, 2)} max ${fmt(debug.obstacle_max_extra_margin)} / ` +
    `lidar ${fmt(debug.lidar_point_margin)}`;

  updateTargetStatus(data);
  updateDriveLogs(data);
}

async function refreshAnalytics() {
  if (!window.Plotly) return;

  const [driveRes, detectionRes] = await Promise.all([
    fetch("/api/history/drive?limit=500", { cache: "no-store" }),
    fetch("/api/history/detections?limit=500", { cache: "no-store" }),
  ]);

  const driveRows = (await driveRes.json()).rows || [];
  const detections = (await detectionRes.json()).rows || [];
  const drive = driveRows.filter((row) => {
    const x = num(row.x);
    const z = num(row.z);
    return x !== null && z !== null && x >= 0 && x <= MAP_SIZE && z >= 0 && z <= MAP_SIZE;
  });
  const times = drive.map((row) => row.created_at || row.timestamp_text || row.timestamp);

  Plotly.react("trajectoryPlot", [
    {
      x: drive.map((row) => num(row.x)),
      y: drive.map((row) => num(row.z)),
      mode: "lines+markers",
      type: "scatter",
      name: "전차 궤적",
      line: { color: "#35d07f", width: 3 },
      marker: { size: 4 },
    },
    {
      x: drive.map((row) => num(row.dest_x)),
      y: drive.map((row) => num(row.dest_z)),
      mode: "markers",
      type: "scatter",
      name: "목적지",
      marker: { color: "#ff6b5e", size: 8, symbol: "circle-open" },
    },
  ], {
    ...plotLayoutBase,
    title: "전차 이동 궤적",
    xaxis: { ...plotLayoutBase.xaxis, title: "X", range: [0, MAP_SIZE] },
    yaxis: { ...plotLayoutBase.yaxis, title: "Z", range: [0, MAP_SIZE], scaleanchor: "x" },
  }, { responsive: true, displaylogo: false });

  Plotly.react("lidarPlot", [
    {
      x: times,
      y: drive.map((row) => num(row.lidar_min_distance)),
      type: "scatter",
      mode: "lines",
      name: "최소거리",
      line: { color: "#e5ba4f", width: 2 },
    },
    {
      x: times,
      y: drive.map((row) => num(row.lidar_near_count)),
      type: "scatter",
      mode: "lines",
      name: "근접점",
      yaxis: "y2",
      line: { color: "#35d07f", width: 2 },
    },
  ], {
    ...plotLayoutBase,
    title: "LiDAR 근접 위험",
    yaxis: { ...plotLayoutBase.yaxis, title: "최소거리" },
    yaxis2: { title: "근접점", overlaying: "y", side: "right", gridcolor: "#22302b" },
  }, { responsive: true, displaylogo: false });

  Plotly.react("commandPlot", [
    {
      x: times,
      y: drive.map((row) => num(row.move_ws_weight)),
      type: "scatter",
      mode: "lines",
      name: "전후진",
      line: { color: "#35d07f", width: 2 },
    },
    {
      x: times,
      y: drive.map((row) => num(row.move_ad_weight)),
      type: "scatter",
      mode: "lines",
      name: "좌우회전",
      line: { color: "#ff6b5e", width: 2 },
    },
  ], {
    ...plotLayoutBase,
    title: "주행 명령 가중치",
    yaxis: { ...plotLayoutBase.yaxis, title: "weight", range: [0, 1] },
  }, { responsive: true, displaylogo: false });

  Plotly.react("targetPlot", [
    {
      x: detections.map((row) => row.created_at),
      y: detections.map((row) => num(row.confidence)),
      type: "scatter",
      mode: "markers",
      name: "탐지 신뢰도",
      marker: { color: "#e5ba4f", size: 6 },
    },
    {
      x: times,
      y: drive.map((row) => num(row.target_confidence)),
      type: "scatter",
      mode: "lines",
      name: "주 표적",
      line: { color: "#35d07f", width: 2 },
    },
  ], {
    ...plotLayoutBase,
    title: "표적 탐지 신뢰도",
    yaxis: { ...plotLayoutBase.yaxis, title: "confidence", range: [0, 1] },
  }, { responsive: true, displaylogo: false });
}

async function refresh() {
  try {
    const res = await fetch(stateUrl, { cache: "no-store" });
    const data = await res.json();
    latestDashboardData = data;

    updateText(data);

    if (data.capture_name) {
      if (!cameraImage.src.endsWith("/video_feed")) cameraImage.src = "/video_feed";
      cameraImage.style.display = "block";
      el("emptyFrame").style.display = "none";
    } else {
      cameraImage.removeAttribute("src");
      cameraImage.style.display = "none";
      el("emptyFrame").style.display = "grid";
    }

    drawOverlay(data);
    drawMap(data);
  } catch (err) {
    el("eventText").textContent = `연결 실패: ${err.message}`;
  }
}

el("setDestBtn").addEventListener("click", async () => {
  await postDestination(el("destX").value, el("destZ").value);
});

mapCanvas.addEventListener("click", handleMapClick);

mapCanvas.addEventListener("mousemove", (event) => {
  const point = mapMetrics().toWorld(event.clientX, event.clientY);
  mapCanvas.title = `목적지 설정: X ${point.x.toFixed(1)}, Z ${point.z.toFixed(1)}`;
});

el("lidarNavToggle").addEventListener("change", async (event) => {
  await fetch("/set_lidar_navigation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: event.target.checked }),
  });

  refresh();
});

el("refreshAnalyticsBtn").addEventListener("click", refreshAnalytics);
el("loadDbTableBtn").addEventListener("click", loadDbTable);
el("dbTableSelect").addEventListener("change", loadDbTable);
el("clearLogsBtn").addEventListener("click", clearLogs);
el("confirmFireBtn").addEventListener("click", confirmFire);
el("chatForm").addEventListener("submit", sendChat);

window.addEventListener("resize", () => {
  refresh();
  refreshAnalytics();
});

refresh();
refreshAnalytics();
loadDbTable();

setInterval(refresh, 1000);
setInterval(refreshAnalytics, 5000);
