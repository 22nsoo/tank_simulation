from __future__ import annotations

import heapq
import json
import math
import re
from copy import deepcopy
from time import monotonic
from typing import Any

from flask import jsonify, request

import fire_logic_pasted as aim
from actions import make_action
from config import (
    ANGLE_THRESHOLD,
    GRID_SIZE,
    GOAL_THRESHOLD,
    LOOKAHEAD_WAYPOINTS,
    MAP_X_MAX,
    MAP_X_MIN,
    MAP_Z_MAX,
    MAP_Z_MIN,
    OBSTACLE_CELL_MARGIN,
    OBSTACLE_CACHE_PATH,
    OBSTACLE_CLEAR_MARGIN,
    OBSTACLE_ESCAPE_MARGIN,
    OBSTACLE_MAX_EXTRA_MARGIN,
    OBSTACLE_SIZE_MARGIN_FACTOR,
    OBSTACLE_SLOW_DISTANCE,
    START_CLEAR_RADIUS,
    REPLAN_DISTANCE,
    WAYPOINT_THRESHOLD,
)
from geometry import distance_2d, get_target_angle, grid_to_world, is_inside_map, normalize_angle, world_to_grid


app = aim.app
original_get_action = app.view_functions["get_action"]
original_info = app.view_functions["info"]
original_init = app.view_functions["init"]
original_lidar_view = app.view_functions["lidar_view"]

OBJECT_RECOGNITION_DISABLED = True


def disable_object_recognition() -> None:
    """Keep this backup build LiDAR/fire-control only; never load or run YOLO."""
    aim.USE_YOLO_FOR_AIM = False
    aim.USE_YOLO_FIRE_GUARD = False
    aim.REQUIRE_YOLO_ENEMY_TANK_FOR_FIRE = False
    aim.yolo_model = None
    with aim.state_lock:
        aim.vision_state.update({
            "detections": [],
            "target": None,
            "lidarFusion": {
                "status": "disabled",
                "reason": "object_recognition_disabled",
                "vision": None,
            },
            "modelLoaded": False,
            "disabled": True,
            "reason": "object_recognition_disabled",
        })
        aim.latest_state["visionDetections"] = []
        aim.latest_state["visionTarget"] = None
        aim.latest_state["lidarVisionFusion"] = deepcopy(aim.vision_state["lidarFusion"])


disable_object_recognition()

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_cached_obstacles() -> list[dict[str, Any]]:
    if not OBSTACLE_CACHE_PATH.exists():
        return []
    try:
        with OBSTACLE_CACHE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        items = data.get("obstacles", []) if isinstance(data, dict) else []
        return normalize_obstacles(items)
    except (OSError, json.JSONDecodeError):
        try:
            text = OBSTACLE_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        recovered = []
        for match in re.findall(r"\{[^{}]*\"x_min\"[^{}]*\}", text):
            try:
                recovered.append(json.loads(match))
            except json.JSONDecodeError:
                pass
        return normalize_obstacles(recovered)


def save_cached_obstacles(items: list[dict[str, Any]]) -> None:
    try:
        with OBSTACLE_CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump({"obstacles": items}, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


def normalize_obstacles(items: Any) -> list[dict[str, Any]]:
    cleaned = []
    for obs in items if isinstance(items, list) else []:
        if not isinstance(obs, dict):
            continue
        try:
            cleaned.append({
                "x_min": max(MAP_X_MIN, safe_float(obs["x_min"])),
                "x_max": min(MAP_X_MAX, safe_float(obs["x_max"])),
                "z_min": max(MAP_Z_MIN, safe_float(obs["z_min"])),
                "z_max": min(MAP_Z_MAX, safe_float(obs["z_max"])),
                "source": obs.get("source", "json"),
            })
        except KeyError:
            pass
    return cleaned


mode = "drive"
destination: dict[str, float] | None = {"x": 100.0, "y": 0.0, "z": 250.0}
latest_position = {"x": 60.0, "y": 10.0, "z": 27.23}
latest_body_yaw = 0.0
obstacles: list[dict[str, Any]] = load_cached_obstacles()
active_navigation_obstacles: list[dict[str, Any]] = []
current_path: list[tuple[float, float]] = []
current_path_index = 0
last_nav_action = make_action("STOP", 1.0)
last_mode_update = monotonic()


def parse_position(data: dict[str, Any]) -> dict[str, float]:
    pos = data.get("position") or data.get("playerPosition") or data.get("player_position") or {}
    if not isinstance(pos, dict):
        pos = {}
    return {
        "x": safe_float(data.get("x", data.get("playerPosX", data.get("Player_Pos_X", pos.get("x")))), latest_position["x"]),
        "y": safe_float(data.get("y", data.get("playerPosY", data.get("Player_Pos_Y", pos.get("y")))), latest_position["y"]),
        "z": safe_float(data.get("z", data.get("playerPosZ", data.get("Player_Pos_Z", pos.get("z")))), latest_position["z"]),
    }


def parse_body_yaw(data: dict[str, Any]) -> float:
    for key in ("playerBodyX", "Player_Body_X", "player_body_x", "bodyX", "body_x", "bodyYaw"):
        if key in data:
            return safe_float(data[key], latest_body_yaw)
    body = data.get("body") or data.get("playerBody") or {}
    return safe_float(body.get("x") if isinstance(body, dict) else None, latest_body_yaw)


def obstacle_margin(obs: dict[str, Any], base: float) -> float:
    width = max(0.0, safe_float(obs.get("x_max")) - safe_float(obs.get("x_min")))
    depth = max(0.0, safe_float(obs.get("z_max")) - safe_float(obs.get("z_min")))
    return base + min(OBSTACLE_MAX_EXTRA_MARGIN, max(width, depth) * OBSTACLE_SIZE_MARGIN_FACTOR)


def is_obstacle_cell(gx: int, gz: int) -> bool:
    x, z = grid_to_world(gx, gz)
    if distance_2d(latest_position["x"], latest_position["z"], x, z) <= START_CLEAR_RADIUS:
        return False
    active_obstacles = active_navigation_obstacles or obstacles
    for obs in active_obstacles:
        margin = obstacle_margin(obs, OBSTACLE_CELL_MARGIN)
        if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
            return True
    return False


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dz = abs(a[1] - b[1])
    return (dx + dz) + (math.sqrt(2) - 2) * min(dx, dz)


def astar(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    open_set = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            rev = []
            walk = current
            while walk in came_from:
                rev.append(walk)
                walk = came_from[walk]
            return [start] + list(reversed(rev))
        for dx, dz in directions:
            neighbor = (current[0] + dx, current[1] + dz)
            if not is_inside_map(*neighbor) or is_obstacle_cell(*neighbor):
                continue
            if dx and dz and (is_obstacle_cell(current[0] + dx, current[1]) or is_obstacle_cell(current[0], current[1] + dz)):
                continue
            tentative = g_score[current] + (math.sqrt(2) if dx and dz else 1.0)
            if neighbor not in g_score or tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(open_set, (tentative + heuristic(neighbor, goal), neighbor))
    return []


def point_blocked(x: float, z: float, active_obstacles: list[dict[str, Any]], margin: float = OBSTACLE_CLEAR_MARGIN) -> bool:
    for obs in active_obstacles:
        if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
            return True
    return False


def distance_to_obstacle_box(x: float, z: float, obs: dict[str, Any]) -> float:
    dx = max(obs["x_min"] - x, 0.0, x - obs["x_max"])
    dz = max(obs["z_min"] - z, 0.0, z - obs["z_max"])
    return math.sqrt(dx * dx + dz * dz)


def nearest_obstacle_distance(x: float, z: float, active_obstacles: list[dict[str, Any]]) -> float | None:
    if not active_obstacles:
        return None
    return min(distance_to_obstacle_box(x, z, obs) for obs in active_obstacles)


def nearest_obstacle_center(pos_x: float, pos_z: float, active_obstacles: list[dict[str, Any]]) -> tuple[float, float] | None:
    if not active_obstacles:
        return None
    return min(
        (((obs["x_min"] + obs["x_max"]) / 2.0, (obs["z_min"] + obs["z_max"]) / 2.0) for obs in active_obstacles),
        key=lambda center: distance_2d(pos_x, pos_z, center[0], center[1]),
    )


def has_clear_path(ax: float, az: float, bx: float, bz: float, active_obstacles: list[dict[str, Any]]) -> bool:
    if not active_obstacles:
        return True
    samples = max(2, int(distance_2d(ax, az, bx, bz) / (GRID_SIZE / 2.0)))
    for i in range(samples + 1):
        r = i / samples
        x = ax + (bx - ax) * r
        z = az + (bz - az) * r
        for obs in active_obstacles:
            margin = obstacle_margin(obs, OBSTACLE_CLEAR_MARGIN)
            if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
                return False
    return True


def drive_to(tx: float, tz: float) -> dict[str, Any]:
    error = normalize_angle(get_target_angle(latest_position["x"], latest_position["z"], tx, tz) - latest_body_yaw)
    dist = distance_2d(latest_position["x"], latest_position["z"], tx, tz)
    obstacle_distance = nearest_obstacle_distance(latest_position["x"], latest_position["z"], active_navigation_obstacles)
    near_obstacle = obstacle_distance is not None and obstacle_distance <= OBSTACLE_SLOW_DISTANCE
    if dist < WAYPOINT_THRESHOLD:
        return make_action("STOP", 1.0)
    if near_obstacle and abs(error) > 45:
        return make_action("", 0.0, "D" if error > 0 else "A", 0.75)
    if near_obstacle and abs(error) > ANGLE_THRESHOLD:
        return make_action("W", 0.16, "D" if error > 0 else "A", 0.55)
    if near_obstacle:
        return make_action("W", 0.28)
    if abs(error) > 70:
        return make_action("W", 0.22, "D" if error > 0 else "A", 0.8)
    if abs(error) > 35:
        return make_action("W", 0.18, "D" if error > 0 else "A", 0.65)
    if abs(error) > ANGLE_THRESHOLD:
        return make_action("W", 0.35, "D" if error > 0 else "A", 0.45)
    return make_action("W", 0.50)


def advance_path_index_to_nearest(pos_x: float, pos_z: float) -> None:
    global current_path_index
    if not current_path:
        return
    search_end = min(len(current_path), current_path_index + 8)
    nearest_index = min(
        range(current_path_index, search_end),
        key=lambda index: distance_2d(pos_x, pos_z, current_path[index][0], current_path[index][1]),
    )
    current_path_index = max(current_path_index, nearest_index)


def decide_escape_action(pos_x: float, pos_z: float) -> dict[str, Any]:
    goal_x = destination["x"] if destination else pos_x
    goal_z = destination["z"] if destination else pos_z
    error = normalize_angle(get_target_angle(pos_x, pos_z, goal_x, goal_z) - latest_body_yaw)
    return make_action("W", 0.16, "D" if error > 0 else "A", 0.85)


def choose_visible_waypoint(pos_x: float, pos_z: float, active_obstacles: list[dict[str, Any]]) -> tuple[float, float]:
    max_index = min(len(current_path) - 1, current_path_index + LOOKAHEAD_WAYPOINTS)
    min_index = min(len(current_path) - 1, current_path_index + 1)
    for index in range(max_index, min_index - 1, -1):
        waypoint_x, waypoint_z = current_path[index]
        if has_clear_path(pos_x, pos_z, waypoint_x, waypoint_z, active_obstacles):
            return waypoint_x, waypoint_z
    return current_path[current_path_index]


def decide_drive_action() -> dict[str, Any]:
    global active_navigation_obstacles, current_path, current_path_index
    if destination is None:
        return make_action("STOP", 1.0)
    active_navigation_obstacles = obstacles
    gx, gz = destination["x"], destination["z"]
    if point_blocked(latest_position["x"], latest_position["z"], active_navigation_obstacles, OBSTACLE_ESCAPE_MARGIN):
        nearest = nearest_obstacle_center(latest_position["x"], latest_position["z"], active_navigation_obstacles)
        if nearest:
            away_angle = get_target_angle(nearest[0], nearest[1], latest_position["x"], latest_position["z"])
            error = normalize_angle(away_angle - latest_body_yaw)
            return make_action("S", 0.45, "D" if error > 0 else "A", 0.7)
        return make_action("S", 0.4)
    if distance_2d(latest_position["x"], latest_position["z"], gx, gz) < GOAL_THRESHOLD:
        return make_action("STOP", 1.0)
    if has_clear_path(latest_position["x"], latest_position["z"], gx, gz, active_navigation_obstacles):
        current_path = []
        current_path_index = 0
        return drive_to(gx, gz)
    if current_path:
        sx, sz = current_path[0]
        if distance_2d(latest_position["x"], latest_position["z"], sx, sz) > REPLAN_DISTANCE:
            current_path = []
            current_path_index = 0
    if not current_path:
        grid_path = astar(world_to_grid(latest_position["x"], latest_position["z"]), world_to_grid(gx, gz))
        if not grid_path:
            return decide_escape_action(latest_position["x"], latest_position["z"])
        current_path = [grid_to_world(x, z) for x, z in grid_path]
        current_path_index = 0
    if current_path_index >= len(current_path):
        return make_action("STOP", 1.0)
    advance_path_index_to_nearest(latest_position["x"], latest_position["z"])
    while current_path_index < len(current_path) - 1 and distance_2d(latest_position["x"], latest_position["z"], *current_path[current_path_index]) < WAYPOINT_THRESHOLD:
        current_path_index += 1
    waypoint_x, waypoint_z = choose_visible_waypoint(latest_position["x"], latest_position["z"], active_navigation_obstacles)
    return drive_to(waypoint_x, waypoint_z)


def set_mode_value(value: str) -> str:
    global mode, current_path, current_path_index, last_mode_update
    mode = "fire" if str(value).lower() in {"fire", "aim", "attack", "shoot"} else "drive"
    current_path = []
    current_path_index = 0
    last_mode_update = monotonic()
    return mode


def combined_info():
    global latest_position, latest_body_yaw
    data = request.get_json(force=True, silent=True) or {}
    latest_position = parse_position(data)
    latest_body_yaw = parse_body_yaw(data)
    return original_info()


def combined_get_action():
    global latest_position, latest_body_yaw, last_nav_action
    data = request.get_json(force=True, silent=True) or {}
    latest_position = parse_position(data)
    latest_body_yaw = parse_body_yaw(data)
    if mode == "fire":
        return original_get_action()
    last_nav_action = decide_drive_action()
    return jsonify(last_nav_action)


def combined_init():
    global obstacles, active_navigation_obstacles, current_path, current_path_index
    set_mode_value("drive")
    obstacles = load_cached_obstacles()
    active_navigation_obstacles = []
    current_path = []
    current_path_index = 0
    if obstacles:
        save_cached_obstacles(obstacles)
    return original_init()


def combined_lidar_view():
    html = original_lidar_view()
    if not isinstance(html, str):
        return html
    drive_bar = """
  <div style="margin:0 0 12px 0">
    <button onclick="fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'drive'})}).then(()=>location.href='/mode_view')" style="padding:8px 12px;border:1px solid #555;background:#222;color:#eee;border-radius:6px;cursor:pointer">Drive mode</button>
  </div>
"""
    return html.replace("<body>", "<body>" + drive_bar, 1)


def update_obstacle():
    global obstacles, current_path, current_path_index
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("obstacles", data if isinstance(data, list) else [])
    obstacles = normalize_obstacles(items)
    save_cached_obstacles(obstacles)
    current_path = []
    current_path_index = 0
    return jsonify({"status": "OK", "message": "Obstacle data received", "mode": mode, "obstacleCount": len(obstacles), "obstacles": obstacles})


def detect_disabled():
    disable_object_recognition()
    payload = {
        "status": "disabled",
        "message": "Object recognition is disabled in this backup build.",
        "detections": [],
        "target": None,
        "lidarVisionFusion": deepcopy(aim.vision_state.get("lidarFusion")),
    }
    if request.args.get("format") == "debug":
        return jsonify(payload)
    return jsonify([])


@app.route("/mode", methods=["GET", "POST"])
def mode_route():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        set_mode_value(data.get("mode", request.args.get("mode", mode)))
    return jsonify(mode_state())


@app.route("/set_destination", methods=["POST"])
def set_destination():
    global destination, current_path, current_path_index
    data = request.get_json(force=True, silent=True) or {}
    destination = {"x": safe_float(data.get("x")), "y": safe_float(data.get("y"), 0.0), "z": safe_float(data.get("z"))}
    current_path = []
    current_path_index = 0
    return jsonify({"status": "OK", "destination": destination})


@app.route("/mode_state")
def mode_state_route():
    return jsonify(mode_state())


def mode_state() -> dict[str, Any]:
    obstacle_margins = [
        {
            "x_min": obs["x_min"],
            "x_max": obs["x_max"],
            "z_min": obs["z_min"],
            "z_max": obs["z_max"],
            "margin": obstacle_margin(obs, OBSTACLE_CLEAR_MARGIN),
        }
        for obs in obstacles
    ]
    return {
        "mode": mode,
        "position": latest_position,
        "bodyYaw": latest_body_yaw,
        "destination": destination,
        "obstacles": obstacles,
        "obstacleMargins": obstacle_margins,
        "path": [{"x": x, "z": z} for x, z in current_path],
        "pathIndex": current_path_index,
        "lastNavAction": last_nav_action,
        "lastModeUpdate": last_mode_update,
    }


@app.route("/mode_view")
def mode_view():
    return r"""<!doctype html><html><head><meta charset="utf-8"><title>Drive / Fire Mode</title>
<style>body{margin:18px;background:#101214;color:#eee;font-family:Arial,sans-serif}button{margin-right:8px;padding:8px 12px}canvas{background:#171a1d;border:1px solid #555}pre{white-space:pre-wrap}</style></head>
<body><h2>Drive / Fire Mode</h2><button onclick="setMode('drive')">Drive mode</button><button onclick="setMode('fire')">Fire mode</button><canvas id="map" width="720" height="720"></canvas><pre id="s" style="display:none"></pre>
<script>
const c=document.getElementById('map'),ctx=c.getContext('2d'),s=document.getElementById('s'),M=300,P=24;
function x(v){return P+v/M*(c.width-2*P)} function y(v){return c.height-P-v/M*(c.height-2*P)}
function wx(px){return Math.max(0,Math.min(M,(px-P)/(c.width-2*P)*M))}
function wz(py){return Math.max(0,Math.min(M,(c.height-P-py)/(c.height-2*P)*M))}
async function setMode(m){await fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});if(m==='fire')location.href='/lidar_view';}
async function setDestFromEvent(e){const r=c.getBoundingClientRect();const dx=wx(e.clientX-r.left),dz=wz(e.clientY-r.top);await fetch('/set_destination',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:dx,y:0,z:dz})});await setMode('drive');}
async function tick(){const d=await (await fetch('/mode_state',{cache:'no-store'})).json();draw(d);}
function draw(d){ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#333';for(let i=0;i<=300;i+=50){ctx.beginPath();ctx.moveTo(x(i),y(0));ctx.lineTo(x(i),y(300));ctx.moveTo(x(0),y(i));ctx.lineTo(x(300),y(i));ctx.stroke();}
ctx.save();ctx.strokeStyle='rgba(255,180,80,.85)';ctx.setLineDash([6,5]);for(const o of d.obstacleMargins||[]){const m=Number(o.margin||0),xmin=Math.max(0,o.x_min-m),xmax=Math.min(M,o.x_max+m),zmin=Math.max(0,o.z_min-m),zmax=Math.min(M,o.z_max+m);ctx.strokeRect(x(xmin),y(zmax),(xmax-xmin)*(c.width-2*P)/M,(zmax-zmin)*(c.height-2*P)/M);}ctx.restore();
for(const o of d.obstacles||[]){ctx.fillStyle='rgba(255,80,80,.55)';ctx.fillRect(x(o.x_min),y(o.z_max),(o.x_max-o.x_min)*(c.width-2*P)/M,(o.z_max-o.z_min)*(c.height-2*P)/M);}
ctx.strokeStyle='#35d07f';ctx.lineWidth=2;ctx.beginPath();(d.path||[]).forEach((p,i)=>{if(i===0)ctx.moveTo(x(p.x),y(p.z));else ctx.lineTo(x(p.x),y(p.z));});ctx.stroke();ctx.lineWidth=1;
if(d.destination){ctx.strokeStyle='#ffdd55';ctx.beginPath();ctx.arc(x(d.destination.x),y(d.destination.z),8,0,Math.PI*2);ctx.stroke();}
ctx.fillStyle=d.mode==='fire'?'#ff4444':'#44d66b';ctx.beginPath();ctx.arc(x(d.position.x||0),y(d.position.z||0),7,0,Math.PI*2);ctx.fill();ctx.fillText('mode: '+d.mode,20,20);}
c.addEventListener('click',setDestFromEvent);
setInterval(tick,300);tick();</script></body></html>"""


app.view_functions["info"] = combined_info
app.view_functions["get_action"] = combined_get_action
app.view_functions["init"] = combined_init
app.view_functions["lidar_view"] = combined_lidar_view
app.view_functions["update_obstacle"] = update_obstacle
app.view_functions["detect"] = detect_disabled


if __name__ == "__main__":
    print("Mode view: http://127.0.0.1:5000/mode_view")
    print("LiDAR/fire view: http://127.0.0.1:5000/lidar_view")
    app.run(host="0.0.0.0", port=5000, threaded=True)
