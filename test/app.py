from flask import Flask, Response, jsonify, request, send_from_directory

import base64
import csv
import heapq
import json
import math
import os
import urllib.error
import urllib.request
from datetime import datetime
from time import sleep, time

from ultralytics import YOLO
from slam_mapper import SlamMapper

from config import *
from actions import make_action, action_command, action_weight
from geometry import (
    distance_2d,
    normalize_angle,
    get_target_angle,
    world_to_grid,
    grid_to_world,
    is_inside_map,
)

from database import (
    DB_TABLES,
    init_database,
    db_execute,
    db_fetch_all,
    record_event,
    json_ready,
    get_db_status,
)

from lidar import (
    latest_lidar_file,
    parse_bool,
    empty_lidar_summary,
    lidar_raw_items,
    normalize_lidar_points,
    analyze_lidar_quality,
    points_to_obstacles,
    read_lidar_points_from_file,
)


def load_cached_obstacles():
    if not OBSTACLE_CACHE_PATH.exists():
        return []
    try:
        with OBSTACLE_CACHE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("obstacles", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_cached_obstacles(items):
    try:
        with OBSTACLE_CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump({"obstacles": items}, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


def prune_capture_images(keep=10):
    if not CAPTURE_DIR.exists():
        return 0

    images = sorted(CAPTURE_DIR.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    deleted = 0
    for path in images[keep:]:
        try:
            path.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def parse_limit(default=300, maximum=2000):
    try:
        return max(1, min(maximum, int(request.args.get("limit", default))))
    except (TypeError, ValueError):
        return default


def read_drive_logs_from_csv(limit):
    if not DRIVE_LOG_PATH.exists():
        return []
    try:
        with DRIVE_LOG_PATH.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return rows[-limit:]
    except OSError:
        return []


app = Flask(__name__, static_folder=None)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
print(f"[MODEL] Loaded model path: {MODEL_PATH}")

# config.py의 ATTACK_TARGET_CLASSES는 class_id -> class_name 매핑으로 사용합니다.
MODEL_CLASS_NAMES = ATTACK_TARGET_CLASSES

# 실제 공격 대상으로 lock/aim/fire 처리할 클래스입니다.
ATTACK_TARGET_NAMES = {
    "Enemy",
    "Tank_enemy",
}

# 화면 표시 및 탐지 로그에 남길 클래스입니다.
VISIBLE_DETECTION_NAMES = ATTACK_TARGET_NAMES | {
    "Ally",
    "House",
    "Rock",
    "Rock_L",
    "Tent",
    "car",
}


# 시뮬레이터와 웹 대시보드가 함께 보는 런타임 상태입니다.
destination = {"x": 100.0, "y": 0.0, "z": 250.0}
obstacles = load_cached_obstacles()
active_navigation_obstacles = []
current_path = []
current_path_index = 0
latest_player_body_yaw = 0.0
latest_position = {"x": 0.0, "y": 0.0, "z": 0.0}
latest_turret = {"x": 0.0, "y": 0.0}
latest_action = None
latest_detections = []
latest_detection_time = 0.0
target_candidate = None
target_candidate_hits = 0
target_candidate_time = 0.0
retained_target = None
retained_target_time = 0.0
latest_frame_name = None
latest_frame_bytes = b""
latest_event = "server ready"
latest_bullet = None
latest_collision = None
latest_lidar = {
    "source": "none",
    "file": None,
    "mtime": 0.0,
    "point_count": 0,
    "detected_count": 0,
    "near_count": 0,
    "min_distance": None,
    "obstacle_count": 0,
}
latest_lidar_points = []
latest_lidar_api_time = 0.0
latest_target_status = {"state": "no_target", "label": "탐지 없음"}
latest_drive_logs = []
use_lidar_navigation = False
fire_approval_until = 0.0
last_update_time = time()


def reset_runtime_map_state():
    global destination, obstacles, active_navigation_obstacles, current_path, current_path_index
    global latest_player_body_yaw, latest_position, latest_turret, latest_action
    global latest_detections, latest_detection_time, target_candidate, target_candidate_hits, target_candidate_time
    global retained_target, retained_target_time, latest_frame_name, latest_frame_bytes
    global latest_event, latest_bullet, latest_collision, latest_lidar, latest_lidar_points, latest_lidar_api_time
    global latest_target_status, latest_drive_logs, use_lidar_navigation, fire_approval_until, last_update_time

    destination = {"x": 100.0, "y": 0.0, "z": 250.0}
    obstacles = load_cached_obstacles()
    active_navigation_obstacles = []
    current_path = []
    current_path_index = 0
    latest_player_body_yaw = 0.0
    latest_position = {"x": 60.0, "y": 10.0, "z": 27.23}
    latest_turret = {"x": 0.0, "y": 0.0}
    latest_action = None
    latest_detections = []
    latest_detection_time = 0.0
    target_candidate = None
    target_candidate_hits = 0
    target_candidate_time = 0.0
    retained_target = None
    retained_target_time = 0.0
    latest_frame_name = None
    latest_frame_bytes = b""
    latest_event = "map initialized"
    latest_bullet = None
    latest_collision = None
    latest_lidar = empty_lidar_summary()
    latest_lidar_points = []
    latest_lidar_api_time = 0.0
    slam_mapper.clear()
    latest_target_status = {"state": "no_target", "label": "탐지 없음"}
    latest_drive_logs = []
    use_lidar_navigation = False
    fire_approval_until = 0.0
    last_update_time = time()

# 주행/탐지 튜닝값은 config.py에서 import합니다.

slam_mapper = SlamMapper(
    x_min=MAP_X_MIN,
    x_max=MAP_X_MAX,
    z_min=MAP_Z_MIN,
    z_max=MAP_Z_MAX,
    grid_size=GRID_SIZE,
    point_margin=LIDAR_POINT_MARGIN,
    cache_path=SLAM_CACHE_PATH,
)


def is_in_roi_bbox(bbox, image_w, image_h):
    x1, y1, x2, y2 = bbox

    center_x = ((x1 + x2) / 2.0) / image_w
    center_y = ((y1 + y2) / 2.0) / image_h

    return (
        ROI_X_MIN <= center_x <= ROI_X_MAX and
        ROI_Y_MIN <= center_y <= ROI_Y_MAX
    )


def is_obstacle_cell(gx, gz):
    x, z = grid_to_world(gx, gz)
    if distance_2d(x, z, latest_position["x"], latest_position["z"]) <= START_CLEAR_RADIUS:
        return False
    active_obstacles = active_navigation_obstacles or obstacles
    for obs in active_obstacles:
        # 장애물 크기에 따라 margin을 다르게 주되 상한을 둬서 과도한 팽창을 막습니다.
        margin = obstacle_margin(obs, OBSTACLE_CELL_MARGIN)
        if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
            return True
    return False


def obstacle_size(obs):
    """장애물 박스의 가로/세로 크기와 대표 크기를 반환합니다."""
    width = max(0.0, float(obs["x_max"]) - float(obs["x_min"]))
    depth = max(0.0, float(obs["z_max"]) - float(obs["z_min"]))
    return width, depth, max(width, depth)


def obstacle_margin(obs, base_margin):
    """작은 장애물은 덜, 큰 장애물은 조금 더 부풀리는 동적 회피 margin입니다."""
    _, _, size = obstacle_size(obs)
    extra = min(OBSTACLE_MAX_EXTRA_MARGIN, size * OBSTACLE_SIZE_MARGIN_FACTOR)
    return base_margin + extra


def obstacle_margins_for_web(active_obstacles):
    """웹 지도에서 각 장애물 점선 margin을 실제 직선 경로 검사 기준과 맞춥니다."""
    return [
        {
            "x_min": obs["x_min"],
            "x_max": obs["x_max"],
            "z_min": obs["z_min"],
            "z_max": obs["z_max"],
            "source": obs.get("source"),
            "margin": obstacle_margin(obs, OBSTACLE_CLEAR_MARGIN),
        }
        for obs in active_obstacles
    ]


def is_point_in_obstacle_margin(x, z, active_obstacles, margin):
    """월드 좌표가 장애물 확장 박스 안에 있는지 확인합니다."""
    for obs in active_obstacles:
        if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
            return True
    return False


def distance_to_obstacle_box(x, z, obs):
    """점과 축 정렬 장애물 박스 사이 최단 거리를 계산합니다."""
    dx = max(obs["x_min"] - x, 0.0, x - obs["x_max"])
    dz = max(obs["z_min"] - z, 0.0, z - obs["z_max"])
    return math.sqrt(dx * dx + dz * dz)


def nearest_obstacle_distance(x, z, active_obstacles):
    """가장 가까운 장애물 박스까지의 거리입니다."""
    if not active_obstacles:
        return None
    return min(distance_to_obstacle_box(x, z, obs) for obs in active_obstacles)


def nearest_obstacle_center(pos_x, pos_z, active_obstacles):
    """가장 가까운 장애물 중심을 찾아 비상 회피 방향 계산에 사용합니다."""
    if not active_obstacles:
        return None
    return min(
        (
            ((obs["x_min"] + obs["x_max"]) / 2.0, (obs["z_min"] + obs["z_max"]) / 2.0)
            for obs in active_obstacles
        ),
        key=lambda center: distance_2d(pos_x, pos_z, center[0], center[1]),
    )


def heuristic(a, b):
    # 대각 이동을 허용하는 격자에서 실제 이동 비용과 비슷한 휴리스틱입니다.
    dx = abs(a[0] - b[0])
    dz = abs(a[1] - b[1])
    return (dx + dz) + (math.sqrt(2) - 2) * min(dx, dz)


def astar(start, goal):
    """장애물 셀을 피해 현재 위치에서 목적지까지 격자 경로를 찾습니다."""
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [start]
            walk = current
            rev = []
            while walk in came_from:
                rev.append(walk)
                walk = came_from[walk]
            return path + list(reversed(rev))

        for dx, dz in directions:
            neighbor = (current[0] + dx, current[1] + dz)
            if not is_inside_map(*neighbor) or is_obstacle_cell(*neighbor):
                continue
            # 대각선 이동은 두 장애물 사이 모서리를 비집고 지나가는 corner cutting을 막습니다.
            if dx and dz and (is_obstacle_cell(current[0] + dx, current[1]) or is_obstacle_cell(current[0], current[1] + dz)):
                continue
            move_cost = math.sqrt(2) if dx and dz else 1.0
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), neighbor))
    return []


def decide_move_to_waypoint(pos_x, pos_z, target_x, target_z):
    """목표 지점 방향으로 차체를 돌리면서 전진하도록 W/A/D 명령을 결정합니다."""
    target_angle = get_target_angle(pos_x, pos_z, target_x, target_z)
    error = normalize_angle(target_angle - latest_player_body_yaw)
    dist = distance_2d(pos_x, pos_z, target_x, target_z)
    obstacle_distance = nearest_obstacle_distance(pos_x, pos_z, active_navigation_obstacles)
    near_obstacle = obstacle_distance is not None and obstacle_distance <= OBSTACLE_SLOW_DISTANCE

    if dist < WAYPOINT_THRESHOLD:
        return make_action("STOP", 1.0)
    if not use_lidar_navigation:
        if near_obstacle and abs(error) > 45:
            return make_action("", 0.0, "D" if error > 0 else "A", 0.75)
        if near_obstacle and abs(error) > ANGLE_THRESHOLD:
            return make_action("W", 0.25, "D" if error > 0 else "A", 0.55)
        if near_obstacle:
            return make_action("W", 0.45)
        if abs(error) > 70:
            return make_action("W", 0.35, "D" if error > 0 else "A", 0.8)
        if abs(error) > 35:
            return make_action("W", 0.25, "D" if error > 0 else "A", 0.65)
        if abs(error) > ANGLE_THRESHOLD:
            return make_action("W", 0.55, "D" if error > 0 else "A", 0.45)
        return make_action("W", 0.8)

    if abs(error) > 100:
        return make_action("W", 0.3, "D" if error > 0 else "A", 0.45)
    if abs(error) > 55:
        return make_action("W", 0.55, "D" if error > 0 else "A", 0.35)
    if abs(error) > 25:
        return make_action("W", 0.7, "D" if error > 0 else "A", 0.25)
    if abs(error) > ANGLE_THRESHOLD:
        return make_action("W", 0.8, "D" if error > 0 else "A", 0.15)
    return make_action("W", 0.8)


def advance_path_index_to_nearest(pos_x, pos_z):
    """전차가 경로 옆으로 흘렀을 때 이미 지난 waypoint를 다시 쫓지 않게 합니다."""
    global current_path_index

    if not current_path:
        return

    search_end = min(len(current_path), current_path_index + 8)
    candidates = range(current_path_index, search_end)
    nearest_index = min(
        candidates,
        key=lambda index: distance_2d(pos_x, pos_z, current_path[index][0], current_path[index][1]),
    )
    current_path_index = max(current_path_index, nearest_index)


def decide_escape_action(pos_x, pos_z):
    """경로를 못 찾는 경우 제자리 정지 대신 천천히 빠져나오는 비상 명령입니다."""
    goal_x = destination["x"] if destination else pos_x
    goal_z = destination["z"] if destination else pos_z
    target_angle = get_target_angle(pos_x, pos_z, goal_x, goal_z)
    error = normalize_angle(target_angle - latest_player_body_yaw)
    turn = "D" if error > 0 else "A"

    min_distance = latest_lidar.get("min_distance") if use_lidar_navigation else None
    if min_distance is not None and min_distance < START_CLEAR_RADIUS:
        return make_action("S", 0.35, turn, 0.75)
    return make_action("W", 0.25, turn, 0.85)


def get_body_yaw(data, fallback):
    # 서버 샘플/시뮬레이터 버전에 따라 차체 yaw 키 이름이 달라서 여러 형태를 허용합니다.
    for key in ("playerBodyX", "Player_Body_X", "player_body_x", "bodyX", "body_x"):
        if key in data:
            return float(data[key])

    for key in ("body", "playerBody", "player_body"):
        body = data.get(key)
        if isinstance(body, dict) and "x" in body:
            return float(body["x"])

    return fallback


def load_lidar_obstacles(pos_x=None, pos_z=None):
    global latest_lidar

    # 최근 API 데이터가 있으면 CSV/JSON 파일보다 우선 사용합니다.
    if latest_lidar_points and time() - latest_lidar_api_time <= LIDAR_API_STALE_SECONDS:
        lidar_obstacles, near_count, min_distance = points_to_obstacles(
            latest_lidar_points,
            pos_x,
            pos_z,
        )

        quality = analyze_lidar_quality(
            latest_lidar_points,
            raw_count=latest_lidar.get("quality", {}).get("raw_count", len(latest_lidar_points)),
            source="api",
            mtime=latest_lidar_api_time,
        )

        if use_lidar_navigation:
            slam_mapper.update(
                latest_lidar_points,
                pos_x,
                pos_z,
                latest_player_body_yaw,
                scan_id=("api", latest_lidar_api_time),
            )
        else:
            lidar_obstacles = []

        latest_lidar = {
            "source": "api",
            "file": None,
            "mtime": latest_lidar_api_time,
            "point_count": len(latest_lidar_points),
            "detected_count": len(latest_lidar_points),
            "near_count": near_count,
            "min_distance": min_distance,
            "obstacle_count": len(lidar_obstacles),
            "slam_obstacle_count": slam_mapper.summary()["obstacle_count"],
            "quality": quality,
        }

        return lidar_obstacles

    # API 데이터가 있긴 하지만 오래된 경우에는 상태만 stale로 표시하고 경로계획에는 쓰지 않습니다.
    if latest_lidar_points:
        _, near_count, min_distance = points_to_obstacles(
            latest_lidar_points,
            pos_x,
            pos_z,
        )

        quality = analyze_lidar_quality(
            latest_lidar_points,
            raw_count=latest_lidar.get("quality", {}).get("raw_count", len(latest_lidar_points)),
            source="api",
            mtime=latest_lidar_api_time,
        )
        quality["grade"] = "stale"

        latest_lidar = {
            "source": "api",
            "file": None,
            "mtime": latest_lidar_api_time,
            "point_count": len(latest_lidar_points),
            "detected_count": len(latest_lidar_points),
            "near_count": near_count,
            "min_distance": min_distance,
            "obstacle_count": 0,
            "slam_obstacle_count": slam_mapper.summary()["obstacle_count"],
            "quality": quality,
        }

        return []

    # API 데이터가 없으면 lidar_data 폴더의 최신 CSV/JSON 파일을 읽습니다.
    path = latest_lidar_file()

    if not path:
        latest_lidar = empty_lidar_summary()
        return []

    stat = path.stat()

    try:
        points, point_count, detected_count = read_lidar_points_from_file(path)
    except (OSError, json.JSONDecodeError):
        latest_lidar = empty_lidar_summary(source="file_error")
        return []

    obstacles_from_lidar, near_count, min_distance = points_to_obstacles(
        points,
        pos_x,
        pos_z,
    )

    quality = analyze_lidar_quality(
        points,
        raw_count=point_count,
        source="file",
        mtime=stat.st_mtime,
    )

    if use_lidar_navigation:
        slam_mapper.update(
            points,
            pos_x,
            pos_z,
            latest_player_body_yaw,
            scan_id=("file", path.name, stat.st_mtime),
        )
    else:
        obstacles_from_lidar = []

    latest_lidar = {
        "source": "file",
        "file": path.name,
        "mtime": stat.st_mtime,
        "point_count": point_count,
        "detected_count": detected_count,
        "near_count": near_count,
        "min_distance": min_distance,
        "obstacle_count": len(obstacles_from_lidar),
        "slam_obstacle_count": slam_mapper.summary()["obstacle_count"],
        "quality": quality,
    }

    return obstacles_from_lidar


def get_active_obstacles(pos_x=None, pos_z=None):
    """주행 판단에 실제로 사용할 장애물 목록을 만듭니다."""
    lidar_obstacles = load_lidar_obstacles(pos_x, pos_z)
    slam_obstacles = slam_mapper.get_obstacles(pos_x, pos_z, SLAM_NEAR_RADIUS)
    if not use_lidar_navigation:
        return obstacles
    return obstacles + slam_obstacles + lidar_obstacles


def has_clear_path(pos_x, pos_z, goal_x, goal_z, active_obstacles):
    """현재 위치와 목표 사이 직선 경로가 장애물 박스를 통과하는지 검사합니다."""
    if not active_obstacles:
        return True

    samples = max(2, int(distance_2d(pos_x, pos_z, goal_x, goal_z) / (GRID_SIZE / 2.0)))
    for index in range(samples + 1):
        ratio = index / samples
        x = pos_x + (goal_x - pos_x) * ratio
        z = pos_z + (goal_z - pos_z) * ratio
        for obs in active_obstacles:
            margin = obstacle_margin(obs, OBSTACLE_CLEAR_MARGIN)
            if obs["x_min"] - margin <= x <= obs["x_max"] + margin and obs["z_min"] - margin <= z <= obs["z_max"] + margin:
                return False
    return True


def choose_visible_waypoint(pos_x, pos_z, active_obstacles):
    """경로 위에서 직선으로 갈 수 있는 가장 먼 waypoint를 골라 불필요한 꺾임을 줄입니다."""
    max_index = min(len(current_path) - 1, current_path_index + LOOKAHEAD_WAYPOINTS)
    min_index = min(len(current_path) - 1, current_path_index + 1)

    for index in range(max_index, min_index - 1, -1):
        waypoint_x, waypoint_z = current_path[index]
        if has_clear_path(pos_x, pos_z, waypoint_x, waypoint_z, active_obstacles):
            return waypoint_x, waypoint_z

    return current_path[current_path_index]


def select_primary_target(active_only=True):
    """주행을 멈추고 조준할 만큼 안정적으로 확정된 공격 표적만 반환합니다."""
    now = time()
    if retained_target and (not active_only or now - retained_target_time <= TARGET_HOLD_SECONDS):
        return retained_target
    return None


def target_center(target):
    """탐지 박스 중심을 정규화 좌표로 반환합니다."""
    x1, y1, x2, y2 = target["bbox"]
    image_w = max(float(target.get("imageWidth", 1)), 1.0)
    image_h = max(float(target.get("imageHeight", 1)), 1.0)
    return ((x1 + x2) / 2.0 / image_w, (y1 + y2) / 2.0 / image_h)


def is_same_target(left, right):
    """연속 프레임에서 같은 표적인지 클래스와 화면 중심 거리로 판정합니다."""
    if not left or not right or left.get("className") != right.get("className"):
        return False
    lx, ly = target_center(left)
    rx, ry = target_center(right)
    return abs(lx - rx) <= TARGET_MATCH_TOLERANCE and abs(ly - ry) <= TARGET_MATCH_TOLERANCE


def analyze_target(target):
    """탐지 박스가 조준 중심에 얼마나 가까운지 표적 상태로 정리합니다."""
    if not target:
        return {"state": "no_target", "label": "탐지 없음"}

    x1, y1, x2, y2 = target["bbox"]
    image_w = max(float(target.get("imageWidth", 1)), 1.0)
    image_h = max(float(target.get("imageHeight", 1)), 1.0)
    center_x = ((x1 + x2) / 2.0) / image_w
    center_y = ((y1 + y2) / 2.0) / image_h
    error_x = center_x - 0.5
    error_y = center_y - 0.5
    aligned = abs(error_x) <= TARGET_CENTER_TOLERANCE and abs(error_y) <= TARGET_CENTER_TOLERANCE
    fire_ready = aligned and target["confidence"] >= FIRE_CONFIDENCE

    if fire_ready:
        state = "fire_ready"
        label = "사격 가능"
    elif aligned:
        state = "aligned"
        label = "조준 정렬"
    else:
        state = "aiming"
        label = "조준 중"

    return {
        "state": state,
        "label": label,
        "className": target["className"],
        "confidence": target["confidence"],
        "error_x": error_x,
        "error_y": error_y,
        "center_x": center_x,
        "center_y": center_y,
    }


def decide_attack_action(target, fire_approved=False, base_action=None, force_stop=False):
    """표적을 조준하되, 발사 단계가 아니면 기존 주행 명령을 유지할 수 있습니다."""
    target_status = analyze_target(target)
    error_x = target_status["error_x"]
    error_y = target_status["error_y"]
    base_action = base_action or make_action()

    qe = ""
    rf = ""
    if error_x < -TARGET_CENTER_TOLERANCE:
        qe = "Q"
    elif error_x > TARGET_CENTER_TOLERANCE:
        qe = "E"
    if error_y < -TARGET_CENTER_TOLERANCE:
        rf = "R"
    elif error_y > TARGET_CENTER_TOLERANCE:
        rf = "F"

    should_fire = target_status["state"] == "fire_ready" and fire_approved
    if should_fire:
        return make_action("STOP", 1.0, "", 0.0, qe, 0.7 if qe else 0.0, rf, 0.6 if rf else 0.0, True)
    if force_stop:
        return make_action("STOP", 1.0, "", 0.0, qe, 0.7 if qe else 0.0, rf, 0.6 if rf else 0.0, False)
    return make_action(
        base_action["moveWS"]["command"],
        base_action["moveWS"]["weight"],
        base_action["moveAD"]["command"],
        base_action["moveAD"]["weight"],
        qe,
        0.7 if qe else 0.0,
        rf,
        0.6 if rf else 0.0,
        False,
    )


def decide_navigation_action(pos_x, pos_z):
    """목적지까지의 주행 명령을 결정합니다. 직선 주행이 가능하면 A*를 쓰지 않습니다."""
    global active_navigation_obstacles, current_path, current_path_index

    if destination is None:
        return make_action("STOP", 1.0)

    active_navigation_obstacles = get_active_obstacles(pos_x, pos_z)
    if is_point_in_obstacle_margin(pos_x, pos_z, active_navigation_obstacles, OBSTACLE_ESCAPE_MARGIN):
        nearest = nearest_obstacle_center(pos_x, pos_z, active_navigation_obstacles)
        if nearest:
            away_angle = get_target_angle(nearest[0], nearest[1], pos_x, pos_z)
            error = normalize_angle(away_angle - latest_player_body_yaw)
            return make_action("S", 0.45, "D" if error > 0 else "A", 0.7)
        return make_action("S", 0.4)

    goal_x = destination["x"]
    goal_z = destination["z"]
    if distance_2d(pos_x, pos_z, goal_x, goal_z) < GOAL_THRESHOLD:
        return make_action("STOP", 1.0)

    if has_clear_path(pos_x, pos_z, goal_x, goal_z, active_navigation_obstacles):
        # 장애물이 없거나 직선으로 비켜 갈 수 있으면 최단 직선 경로를 우선합니다.
        current_path = []
        current_path_index = 0
        return decide_move_to_waypoint(pos_x, pos_z, goal_x, goal_z)

    if current_path:
        path_start_x, path_start_z = current_path[0]
        if distance_2d(pos_x, pos_z, path_start_x, path_start_z) > REPLAN_DISTANCE:
            current_path = []
            current_path_index = 0

    if not current_path:
        # 직선 경로가 막힌 경우에만 격자 기반 우회 경로를 새로 계산합니다.
        grid_path = astar(world_to_grid(pos_x, pos_z), world_to_grid(goal_x, goal_z))
        if not grid_path:
            return decide_escape_action(pos_x, pos_z)
        current_path = [grid_to_world(gx, gz) for gx, gz in grid_path]
        current_path_index = 0

    if current_path_index >= len(current_path):
        return make_action("STOP", 1.0)

    advance_path_index_to_nearest(pos_x, pos_z)

    waypoint_x, waypoint_z = current_path[current_path_index]
    while current_path_index < len(current_path) - 1 and distance_2d(pos_x, pos_z, waypoint_x, waypoint_z) < WAYPOINT_THRESHOLD:
        current_path_index += 1
        waypoint_x, waypoint_z = current_path[current_path_index]

    waypoint_x, waypoint_z = choose_visible_waypoint(pos_x, pos_z, active_navigation_obstacles)

    return decide_move_to_waypoint(pos_x, pos_z, waypoint_x, waypoint_z)


def latest_capture():
    """웹 화면에 보여줄 최신 카메라 캡처 이미지를 찾습니다."""
    prune_capture_images(keep=10)
    if not CAPTURE_DIR.exists():
        return None
    images = sorted(CAPTURE_DIR.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    return images[0] if images else None


def encode_image(path):
    if not path or not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def mjpeg_frames():
    last_frame = None
    while True:
        if latest_frame_bytes and latest_frame_bytes != last_frame:
            last_frame = latest_frame_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n\r\n" +
                latest_frame_bytes +
                b"\r\n"
            )
        sleep(0.03)


def append_drive_log(action, target_status):
    """주행 판단 결과를 CSV와 최근 로그 버퍼에 동시에 남깁니다."""
    global latest_drive_logs

    DRIVE_LOG_DIR.mkdir(exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "x": latest_position["x"],
        "y": latest_position["y"],
        "z": latest_position["z"],
        "dest_x": destination["x"] if destination else None,
        "dest_z": destination["z"] if destination else None,
        "body_yaw": latest_player_body_yaw,
        "move_ws": action_command(action, "moveWS"),
        "move_ws_weight": action_weight(action, "moveWS"),
        "move_ad": action_command(action, "moveAD"),
        "move_ad_weight": action_weight(action, "moveAD"),
        "turret_qe": action_command(action, "turretQE"),
        "turret_rf": action_command(action, "turretRF"),
        "fire": bool((action or {}).get("fire", False)),
        "path_index": current_path_index,
        "path_length": len(current_path),
        "obstacle_count": len(active_navigation_obstacles),
        "lidar_source": latest_lidar.get("source"),
        "lidar_points": latest_lidar.get("point_count"),
        "lidar_near_count": latest_lidar.get("near_count"),
        "lidar_min_distance": latest_lidar.get("min_distance"),
        "target_state": target_status.get("state"),
        "target_label": target_status.get("label"),
        "target_class": target_status.get("className"),
        "target_confidence": target_status.get("confidence"),
        "collision": bool(latest_collision),
        "event": latest_event,
    }

    fieldnames = list(row.keys())
    write_header = not DRIVE_LOG_PATH.exists()
    with DRIVE_LOG_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    latest_drive_logs = (latest_drive_logs + [row])[-30:]
    db_execute(
        """
        INSERT INTO drive_logs (
            timestamp_text, x, y, z, dest_x, dest_z, body_yaw,
            move_ws, move_ws_weight, move_ad, move_ad_weight, turret_qe, turret_rf, fire,
            path_index, path_length, obstacle_count,
            lidar_source, lidar_points, lidar_near_count, lidar_min_distance,
            target_state, target_label, target_class, target_confidence,
            collision, event, action_json, raw_json
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s::jsonb, %s::jsonb
        )
        """,
        (
            row["timestamp"], row["x"], row["y"], row["z"], row["dest_x"], row["dest_z"], row["body_yaw"],
            row["move_ws"], row["move_ws_weight"], row["move_ad"], row["move_ad_weight"],
            row["turret_qe"], row["turret_rf"], row["fire"],
            row["path_index"], row["path_length"], row["obstacle_count"],
            row["lidar_source"], row["lidar_points"], row["lidar_near_count"], row["lidar_min_distance"],
            row["target_state"], row["target_label"], row["target_class"], row["target_confidence"],
            row["collision"], row["event"],
            json.dumps(action or {}, ensure_ascii=False),
            json.dumps(row, ensure_ascii=False),
        ),
    )
    db_execute(
        """
        INSERT INTO lidar_summaries (
            source, file_name, point_count, detected_count, near_count, min_distance, obstacle_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            latest_lidar.get("source"),
            latest_lidar.get("file"),
            latest_lidar.get("point_count"),
            latest_lidar.get("detected_count"),
            latest_lidar.get("near_count"),
            latest_lidar.get("min_distance"),
            latest_lidar.get("obstacle_count"),
        ),
    )


@app.route("/")
def dashboard():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/web/<path:path>")
def web_asset(path):
    return send_from_directory(WEB_DIR, path)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/state")
def api_state():
    """웹 대시보드가 주기적으로 읽는 통합 상태 API입니다."""
    capture = latest_capture()
    lidar_obstacles = load_lidar_obstacles(latest_position["x"], latest_position["z"])
    slam_obstacles = slam_mapper.get_obstacles(latest_position["x"], latest_position["z"], SLAM_NEAR_RADIUS)
    active_obstacles = obstacles + slam_obstacles + lidar_obstacles if use_lidar_navigation else obstacles
    # LiDAR 회피 OFF일 때는 시뮬레이터가 보낸 정적 장애물만 A*와 지도에 사용합니다.
    map_obstacles = active_obstacles
    slam_summary = slam_mapper.summary()
    return jsonify({
        "model": MODEL_PATH.name,
        "position": latest_position,
        "turret": latest_turret,
        "body_yaw": latest_player_body_yaw,
        "destination": destination,
        "obstacles": obstacles,
        "active_obstacles": active_obstacles,
        "map_obstacles": map_obstacles,
        "obstacle_margins": obstacle_margins_for_web(active_obstacles),
        "slam_obstacles": slam_obstacles,
        "slam": slam_summary,
        "lidar": latest_lidar,
        "use_lidar_navigation": use_lidar_navigation,
        "path": [{"x": x, "z": z} for x, z in current_path],
        "path_index": current_path_index,
        "navigation_debug": {
            "grid_size": GRID_SIZE,
            "obstacle_cell_margin": OBSTACLE_CELL_MARGIN,
            "obstacle_clear_margin": OBSTACLE_CLEAR_MARGIN,
            "obstacle_size_margin_factor": OBSTACLE_SIZE_MARGIN_FACTOR,
            "obstacle_max_extra_margin": OBSTACLE_MAX_EXTRA_MARGIN,
            "start_clear_radius": START_CLEAR_RADIUS,
            "obstacle_slow_distance": OBSTACLE_SLOW_DISTANCE,
            "lidar_point_margin": LIDAR_POINT_MARGIN,
            "slam_near_radius": SLAM_NEAR_RADIUS,
            "lookahead_waypoints": LOOKAHEAD_WAYPOINTS,
            "active_obstacle_count": len(active_obstacles),
        },
        "roi": {
            "x_min": ROI_X_MIN,
            "x_max": ROI_X_MAX,
            "y_min": ROI_Y_MIN,
            "y_max": ROI_Y_MAX,
        },
        "detections": latest_detections,
        "primary_target": select_primary_target(active_only=False),
        "target_active": select_primary_target(active_only=True) is not None,
        "target_status": latest_target_status,
        "target_hold_seconds": TARGET_HOLD_SECONDS,
        "target_retained": bool(retained_target and time() - retained_target_time <= TARGET_HOLD_SECONDS),
        "target_candidate": target_candidate,
        "target_candidate_hits": target_candidate_hits,
        "target_thresholds": {
            "candidate_confidence": TARGET_CANDIDATE_CONFIDENCE,
            "lock_confidence": TARGET_LOCK_CONFIDENCE,
            "lock_frames": TARGET_LOCK_FRAMES,
            "fire_confidence": FIRE_CONFIDENCE,
        },
        "fire_approval_active": time() <= fire_approval_until,
        "drive_log_file": str(DRIVE_LOG_PATH.relative_to(BASE_DIR)),
        "drive_logs": latest_drive_logs,
        "database": get_db_status(),
        "latest_action": latest_action,
        "latest_event": latest_event,
        "latest_bullet": latest_bullet,
        "latest_collision": latest_collision,
        "last_update_time": last_update_time,
        "capture_name": latest_frame_name,
        "capture_image": encode_image(capture),
    })


def chat_context_snapshot():
    target = select_primary_target(active_only=False)
    lidar_quality = latest_lidar.get("quality", {}) if isinstance(latest_lidar, dict) else {}
    return {
        "position": latest_position,
        "turret": latest_turret,
        "body_yaw": latest_player_body_yaw,
        "destination": destination,
        "path_index": current_path_index,
        "path_length": len(current_path),
        "use_lidar_navigation": use_lidar_navigation,
        "lidar": {
            "source": latest_lidar.get("source"),
            "point_count": latest_lidar.get("point_count"),
            "near_count": latest_lidar.get("near_count"),
            "min_distance": latest_lidar.get("min_distance"),
            "quality_grade": lidar_quality.get("grade"),
            "front_min_distance": lidar_quality.get("front_min_distance"),
        },
        "target_status": latest_target_status,
        "primary_target": target,
        "target_retained": bool(retained_target and time() - retained_target_time <= TARGET_HOLD_SECONDS),
        "target_candidate": target_candidate,
        "target_candidate_hits": target_candidate_hits,
        "fire_approval_active": time() <= fire_approval_until,
        "latest_action": latest_action,
        "latest_event": latest_event,
        "latest_bullet": latest_bullet,
        "latest_collision": latest_collision,
        "recent_drive_logs": latest_drive_logs[-5:],
    }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"status": "ERROR", "message": "질문을 입력하세요."}), 400

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return jsonify({
            "status": "ERROR",
            "message": "OPENAI_API_KEY 또는 LLM_API_KEY 환경변수를 설정해야 합니다.",
        }), 503

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    context = chat_context_snapshot()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 Tank Challenge 전장 콘솔의 전술 보조 챗봇이다. "
                    "한국어로 짧고 실용적으로 답하고, 모르는 값은 추측하지 않는다. "
                    "발사/이동 명령을 직접 실행한다고 말하지 말고 현재 상태 해석과 점검 포인트를 제공한다."
                ),
            },
            {
                "role": "user",
                "content": (
                    "현재 전장 상태 JSON:\n"
                    f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                    f"사용자 질문: {message}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return jsonify({"status": "ERROR", "message": f"LLM API 오류: {detail}"}), 502
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return jsonify({"status": "ERROR", "message": f"LLM API 연결 실패: {exc}"}), 502

    answer = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not answer:
        return jsonify({"status": "ERROR", "message": "LLM 응답이 비어 있습니다."}), 502

    return jsonify({"status": "OK", "answer": answer, "model": model_name})


@app.route("/api/lidar/quality")
def api_lidar_quality():
    load_lidar_obstacles(latest_position["x"], latest_position["z"])
    return jsonify({
        "use_lidar_navigation": use_lidar_navigation,
        "lidar": latest_lidar,
    })


@app.route("/detect", methods=["POST"])
def detect():
    global latest_detections, latest_detection_time
    global target_candidate, target_candidate_hits, target_candidate_time
    global retained_target, retained_target_time
    global latest_frame_name, latest_frame_bytes
    global latest_event, last_update_time

    # 시뮬레이터가 보내는 카메라 프레임을 YOLO로 탐지해 표적/장애물 후보로 저장합니다.
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = BASE_DIR / "temp_image.jpg"
    latest_frame_bytes = image.read()
    image_path.write_bytes(latest_frame_bytes)
    latest_frame_name = image.filename or image_path.name

    results = model(str(image_path))
    result = results[0]
    boxes = result.boxes.data.cpu().numpy()
    height, width = result.orig_shape

    # YOLO 모델 내부 names를 fallback으로 사용합니다.
    model_names = getattr(model, "names", {})

    detections = []
    for box in boxes:
        class_id = int(box[5])

        # 우선 config.py의 class_id -> class_name 매핑을 사용하고,
        # 없으면 YOLO model.names를 fallback으로 사용합니다.
        class_name = MODEL_CLASS_NAMES.get(
            class_id,
            model_names.get(class_id, str(class_id)),
        )

        bbox = [float(coord) for coord in box[:4]]
        confidence = float(box[4])

        # 필요한 클래스만 사용합니다.
        if class_name not in VISIBLE_DETECTION_NAMES:
            continue

        # ROI 밖이면 무시합니다.
        if not is_in_roi_bbox(bbox, width, height):
            continue

        is_attack_target = class_name in ATTACK_TARGET_NAMES

        detections.append({
            "className": class_name,
            "classId": class_id,
            "bbox": bbox,
            "confidence": confidence,
            "color": "#00FF66" if is_attack_target else "#FFD166",
            "filled": False,
            "updateBoxWhileMoving": False,
            "imageWidth": width,
            "imageHeight": height,
        })

    latest_detections = detections
    latest_detection_time = time()

    # 공격 대상 후보만 분리합니다.
    attack_targets = [
        detection
        for detection in detections
        if detection["className"] in ATTACK_TARGET_NAMES
        and detection["confidence"] >= TARGET_CANDIDATE_CONFIDENCE
    ]

    lock_targets = [
        detection
        for detection in attack_targets
        if detection["confidence"] >= TARGET_LOCK_CONFIDENCE
    ]

    if lock_targets:
        best_target = max(lock_targets, key=lambda item: item["confidence"]).copy()

        if is_same_target(target_candidate, best_target):
            target_candidate_hits += 1
        else:
            target_candidate_hits = 1

        target_candidate = best_target
        target_candidate_time = latest_detection_time

        if target_candidate_hits >= TARGET_LOCK_FRAMES:
            retained_target = best_target.copy()
            retained_target["retained"] = False
            retained_target["lockHits"] = target_candidate_hits
            retained_target_time = latest_detection_time

    elif attack_targets:
        target_candidate = max(attack_targets, key=lambda item: item["confidence"]).copy()
        target_candidate_hits = 0
        target_candidate_time = latest_detection_time

    elif retained_target and latest_detection_time - retained_target_time <= TARGET_HOLD_SECONDS:
        retained_target = retained_target.copy()
        retained_target["retained"] = True

    elif latest_detection_time - retained_target_time > TARGET_HOLD_SECONDS:
        retained_target = None
        target_candidate = None
        target_candidate_hits = 0

    latest_event = f"{len(detections)} detection(s)"
    last_update_time = time()

    for det in detections:
        db_execute(
            """
            INSERT INTO detections (
                frame_name, class_name, confidence, bbox, image_width, image_height
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                latest_frame_name,
                det["className"],
                det["confidence"],
                json.dumps(det["bbox"]),
                det["imageWidth"],
                det["imageHeight"],
            ),
        )

    return jsonify(detections)

@app.route("/api/history/drive")
def history_drive():
    limit = parse_limit()
    rows = db_fetch_all(
        """
        SELECT created_at, timestamp_text, x, y, z, dest_x, dest_z, body_yaw,
               move_ws, move_ws_weight, move_ad, move_ad_weight, turret_qe, turret_rf,
               fire, path_index, path_length, obstacle_count, lidar_source, lidar_points,
               lidar_near_count, lidar_min_distance, target_state, target_label,
               target_class, target_confidence, collision, event
        FROM drive_logs
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    if rows:
        rows = list(reversed(rows))
    else:
        rows = read_drive_logs_from_csv(limit)
    return jsonify({"source": "postgres" if get_db_status()["available"] and rows else "csv", "rows": json_ready(rows)})


@app.route("/api/history/detections")
def history_detections():
    limit = parse_limit()
    rows = db_fetch_all(
        """
        SELECT created_at, frame_name, class_name, confidence, bbox, image_width, image_height
        FROM detections
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    return jsonify({"source": "postgres" if get_db_status()["available"] else "none", "rows": json_ready(list(reversed(rows)))})


@app.route("/api/history/lidar")
def history_lidar():
    limit = parse_limit()
    rows = db_fetch_all(
        """
        SELECT created_at, source, file_name, point_count, detected_count,
               near_count, min_distance, obstacle_count
        FROM lidar_summaries
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    return jsonify({"source": "postgres" if get_db_status()["available"] else "none", "rows": json_ready(list(reversed(rows)))})


@app.route("/api/history/events")
def history_events():
    limit = parse_limit()
    rows = db_fetch_all(
        """
        SELECT created_at, event_type, message, payload
        FROM events
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    return jsonify({"source": "postgres" if get_db_status()["available"] else "none", "rows": json_ready(list(reversed(rows)))})


@app.route("/api/db/tables")
def db_tables():
    return jsonify({
        "available": init_database(),
        "error": get_db_status()["error"],
        "tables": list(DB_TABLES.keys()),
    })


@app.route("/api/db/table/<table_name>")
def db_table(table_name):
    if table_name not in DB_TABLES:
        return jsonify({"error": "table not allowed"}), 400

    limit = parse_limit(default=100, maximum=500)
    columns = DB_TABLES[table_name]
    query = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY id DESC LIMIT %s"
    rows = db_fetch_all(query, (limit,))
    return jsonify({
        "available": get_db_status()["available"],
        "error": get_db_status()["error"],
        "table": table_name,
        "columns": columns,
        "rows": json_ready(rows),
    })


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    global latest_drive_logs, latest_event, last_update_time

    cleared = {
        "csv": False,
        "lidar_csv": 0,
        "capture_images": 0,
        "slam": False,
        "postgres": False,
    }

    try:
        if DRIVE_LOG_PATH.exists():
            DRIVE_LOG_PATH.unlink()
        cleared["csv"] = True
    except OSError as exc:
        return jsonify({"status": "ERROR", "message": f"CSV log clear failed: {exc}"}), 500

    latest_drive_logs = []
    slam_mapper.clear()
    cleared["slam"] = True
    try:
        deleted = 0
        if LIDAR_DIR.exists():
            for path in LIDAR_DIR.glob("*.csv"):
                path.unlink()
                deleted += 1
        cleared["lidar_csv"] = deleted
    except OSError as exc:
        return jsonify({"status": "ERROR", "message": f"LiDAR CSV clear failed: {exc}"}), 500

    cleared["capture_images"] = prune_capture_images(keep=10)

    if init_database():
        cleared["postgres"] = db_execute(
            "TRUNCATE drive_logs, detections, lidar_summaries, events RESTART IDENTITY"
        )

    latest_event = "logs cleared"
    last_update_time = time()
    return jsonify({
        "status": "OK",
        "cleared": cleared,
        "database": {
            "available": get_db_status()["available"],
            "error": get_db_status()["error"],
        },
    })


@app.route("/api/fire/confirm", methods=["POST"])
def confirm_fire():
    """웹 대시보드에서 승인한 사격 명령을 짧은 시간 동안만 유효하게 둡니다."""
    global fire_approval_until, latest_event

    fire_approval_until = time() + FIRE_APPROVAL_SECONDS
    latest_event = "fire approved"
    record_event("fire_approved", "fire approved", {"valid_seconds": FIRE_APPROVAL_SECONDS})
    return jsonify({
        "status": "OK",
        "message": "fire approved",
        "valid_seconds": FIRE_APPROVAL_SECONDS,
    })


@app.route("/stereo_image", methods=["POST"])
def stereo_image():
    left_image = request.files.get("left_image")
    right_image = request.files.get("right_image")
    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400
    left_image.save(BASE_DIR / "temp_left.jpg")
    right_image.save(BASE_DIR / "temp_right.jpg")
    return jsonify({"result": "success"})


@app.route("/info", methods=["POST"])
def info():
    global latest_player_body_yaw, last_update_time, latest_event

    # /get_action 사이에도 yaw만 따로 들어오는 경우가 있어 최신 차체 방향을 갱신합니다.
    data = request.get_json(force=True, silent=True) or {}
    latest_player_body_yaw = get_body_yaw(data, latest_player_body_yaw)
    latest_event = "telemetry updated"
    last_update_time = time()
    return jsonify({"status": "success", "message": "Data received", "control": ""})


@app.route("/get_action", methods=["POST"])
def get_action():
    global latest_player_body_yaw, latest_position, latest_turret, latest_action, latest_target_status, last_update_time, latest_event, fire_approval_until

    # 시뮬레이터의 메인 제어 루프: 위치/포탑 상태를 받고 바로 다음 행동을 반환합니다.
    data = request.get_json(force=True, silent=True) or {}
    latest_player_body_yaw = get_body_yaw(data, latest_player_body_yaw)
    position = data.get("position", {})
    turret = data.get("turret", {})
    latest_position = {
        "x": float(position.get("x", 0)),
        "y": float(position.get("y", 0)),
        "z": float(position.get("z", 0)),
    }
    latest_turret = {
        "x": float(turret.get("x", 0)),
        "y": float(turret.get("y", 0)),
    }

    nav_action = decide_navigation_action(latest_position["x"], latest_position["z"])
    target = select_primary_target()
    target_status = analyze_target(target)
    if target:
        fire_approved = time() <= fire_approval_until
        should_stop_for_fire = False
        action = decide_attack_action(target, fire_approved, nav_action, force_stop=should_stop_for_fire)
        if action.get("fire"):
            target_status["state"] = "firing"
            target_status["label"] = "사격"
            fire_approval_until = 0.0
        elif target_status["state"] == "fire_ready":
            target_status["state"] = "awaiting_fire_confirm"
            target_status["label"] = "발사 승인 대기"
        elif should_stop_for_fire:
            latest_event = "target engagement"
        else:
            latest_event = "navigation with target tracking"
    else:
        fire_approval_until = 0.0
        action = nav_action
        latest_event = "navigation"

    latest_action = action
    latest_target_status = target_status
    last_update_time = time()
    append_drive_log(action, target_status)
    return jsonify(action)


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global latest_bullet, latest_event, last_update_time

    latest_bullet = request.get_json(force=True, silent=True) or {}
    latest_event = "bullet impact"
    last_update_time = time()
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    global destination, current_path, current_path_index, latest_event, last_update_time

    # 웹에서 목적지를 바꾸면 기존 경로는 버리고 다음 tick에서 새로 판단합니다.
    data = request.get_json(force=True, silent=True) or {}
    try:
        if "destination" in data:
            x, y, z = map(float, str(data["destination"]).split(","))
        else:
            x = float(data["x"])
            y = float(data.get("y", 0))
            z = float(data["z"])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"status": "ERROR", "message": f"Invalid destination: {exc}"}), 400

    destination = {"x": x, "y": y, "z": z}
    current_path = []
    current_path_index = 0
    latest_event = "destination updated"
    last_update_time = time()
    record_event("destination", latest_event, destination)
    return jsonify({"status": "OK", "destination": destination})


@app.route("/set_lidar_navigation", methods=["POST"])
def set_lidar_navigation():
    global current_path, current_path_index, latest_event, last_update_time, use_lidar_navigation

    # 라이다를 지도 표시에만 쓸지, 실제 경로계획에도 반영할지 전환합니다.
    data = request.get_json(force=True, silent=True) or {}
    use_lidar_navigation = bool(data.get("enabled", False))
    current_path = []
    current_path_index = 0
    latest_event = "lidar navigation enabled" if use_lidar_navigation else "lidar navigation disabled"
    last_update_time = time()
    record_event("lidar_navigation", latest_event, {"enabled": use_lidar_navigation})
    return jsonify({"status": "OK", "use_lidar_navigation": use_lidar_navigation})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    global obstacles, current_path, current_path_index, latest_event, last_update_time

    # 시뮬레이터가 제공하는 정적/명시적 장애물 목록입니다.
    data = request.get_json(force=True, silent=True) or {}
    obstacles = data.get("obstacles", [])
    save_cached_obstacles(obstacles)
    current_path = []
    current_path_index = 0
    latest_event = "obstacles updated"
    last_update_time = time()
    record_event("obstacles", latest_event, {"count": len(obstacles)})
    return jsonify({"status": "OK", "message": "Obstacle data received"})


@app.route("/update_lidar", methods=["POST"])
@app.route("/lidar", methods=["POST"])
@app.route("/lidar_data", methods=["POST"])
def update_lidar():
    global current_path, current_path_index, latest_event, latest_lidar, latest_lidar_api_time, latest_lidar_points, last_update_time

    # CSV 저장 없이 실시간으로 라이다 점군을 받는 API입니다.
    data = request.get_json(force=True, silent=True) or {}
    raw_count = len(lidar_raw_items(data))
    latest_lidar_points = normalize_lidar_points(data)
    latest_lidar_api_time = time()
    if use_lidar_navigation:
        current_path = []
        current_path_index = 0
    quality = analyze_lidar_quality(latest_lidar_points, raw_count=raw_count, source="api", mtime=latest_lidar_api_time)
    latest_lidar = {
        "source": "api",
        "file": None,
        "mtime": latest_lidar_api_time,
        "point_count": len(latest_lidar_points),
        "detected_count": len(latest_lidar_points),
        "near_count": sum(1 for point in latest_lidar_points if point["distance"] <= LIDAR_NEAR_DISTANCE),
        "min_distance": min((point["distance"] for point in latest_lidar_points), default=None),
        "obstacle_count": 0,
        "slam_obstacle_count": slam_mapper.summary()["obstacle_count"],
        "quality": quality,
    }
    latest_event = f"lidar api updated ({len(latest_lidar_points)} points)"
    last_update_time = latest_lidar_api_time
    record_event("lidar_api", latest_event, {"point_count": len(latest_lidar_points), "quality": quality})
    return jsonify({
        "status": "OK",
        "source": "api",
        "point_count": len(latest_lidar_points),
        "quality": quality,
        "use_lidar_navigation": use_lidar_navigation,
    })


@app.route("/collision", methods=["POST"])
def collision():
    global latest_collision, latest_event, last_update_time

    latest_collision = request.get_json(force=True, silent=True) or {}
    latest_event = "collision detected"
    last_update_time = time()
    record_event("collision", latest_event, latest_collision)
    return jsonify({"status": "OK", "message": "Collision data received"})


@app.route("/init", methods=["GET"])
def init():
    reset_runtime_map_state()
    return jsonify({
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": True,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": True,
        "saveSnapshot": True,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True,
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": ""})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "server running",
        "model": MODEL_PATH.name,
        "destination": destination,
        "obstacle_count": len(obstacles),
        "path_length": len(current_path),
        "path_index": current_path_index,
        "detections": len(latest_detections),
        "lidar_source": latest_lidar.get("source"),
        "lidar_points": latest_lidar.get("point_count"),
        "use_lidar_navigation": use_lidar_navigation,
        "database": get_db_status(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
