from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TERRAIN_CSV = BASE_DIR / "hill_map_height.csv"

latest_info: dict[str, Any] = {}
latest_obstacles: list[dict[str, Any]] = []
last_fire_time = 0.0
aim_ready_since: Optional[float] = None

GRAVITY = 9.81
MUZZLE_SPEED = 59.0
DISTANCE_SPEED_CALIBRATION = (
    (60.0, 64.715),
    (80.0, 66.883),
    (100.0, 63.284),
    (120.0, 61.558),
)

PLAYER_START = {"x": 150.0, "y": 10.0, "z": 150.0}
OBSTACLE_TARGET_MODE = True
OBSTACLE_PREFAB_NAME = "Rock002_HillFire_80m_Front"
OBSTACLE_TARGET_X = 150.0
OBSTACLE_TARGET_Z = 230.0
OBSTACLE_TARGET_Y_OFFSET = 0.0
DEFAULT_TARGET_HEIGHT_ABOVE_TERRAIN = 1.45
MUZZLE_HEIGHT_ABOVE_PLAYER_POS = 1.0

MIN_PITCH_DEG = -5.0
MAX_PITCH_DEG = 10.0
BALLISTIC_MIN_THETA_DEG = -10.0
BALLISTIC_MAX_THETA_DEG = 35.0

TURRET_YAW_DEADBAND_DEG = 0.45
BODY_YAW_DEADBAND_DEG = 8.0
BODY_FIRE_TOLERANCE_DEG = 12.0
PITCH_FIRE_TOLERANCE_DEG = 0.22
TARGET_VERTICAL_HIT_TOLERANCE_M = 1.25
FIRE_COOLDOWN_SECONDS = 1.0
AIM_STABLE_SECONDS = 0.35

TERRAIN_CLEARANCE_M = 0.75
TRACE_STEP_M = 1.0
TRACE_SKIP_START_M = 5.0
TRACE_SKIP_END_M = 2.5


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pos_from_dict(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return dict(fallback)
    return {
        "x": safe_float(value.get("x"), fallback["x"]),
        "y": safe_float(value.get("y"), fallback["y"]),
        "z": safe_float(value.get("z"), fallback["z"]),
    }


def effective_muzzle_speed(distance_m: float) -> float:
    points = DISTANCE_SPEED_CALIBRATION
    distance_m = float(distance_m)

    if distance_m <= points[0][0]:
        return points[0][1]
    if distance_m >= points[-1][0]:
        return points[-1][1]

    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if d0 <= distance_m <= d1:
            ratio = (distance_m - d0) / (d1 - d0)
            return v0 + ratio * (v1 - v0)

    return MUZZLE_SPEED


@dataclass
class TerrainMap:
    path: Path
    grid: dict[tuple[int, int], float]
    min_x: int
    max_x: int
    min_z: int
    max_z: int
    loaded_at: float

    @classmethod
    def load(cls, path: Path = TERRAIN_CSV) -> "TerrainMap":
        grid: dict[tuple[int, int], float] = {}
        min_x = min_z = 10**9
        max_x = max_z = -10**9

        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(round(safe_float(row.get("x"))))
                z = int(round(safe_float(row.get("z"))))
                y = safe_float(row.get("y"), math.nan)
                if math.isnan(y):
                    continue
                grid[(x, z)] = y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_z = min(min_z, z)
                max_z = max(max_z, z)

        if not grid:
            raise RuntimeError(f"empty terrain grid: {path}")

        return cls(
            path=path,
            grid=grid,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            loaded_at=time.time(),
        )

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "gridCount": len(self.grid),
            "xRange": [self.min_x, self.max_x],
            "zRange": [self.min_z, self.max_z],
            "loadedAt": self.loaded_at,
        }

    def nearest_height(self, x: float, z: float) -> Optional[float]:
        key = (int(round(x)), int(round(z)))
        return self.grid.get(key)

    def height_at(self, x: float, z: float) -> Optional[float]:
        if x < self.min_x or x > self.max_x or z < self.min_z or z > self.max_z:
            return None

        x0 = math.floor(x)
        z0 = math.floor(z)
        x1 = x0 + 1
        z1 = z0 + 1

        h00 = self.grid.get((x0, z0))
        h10 = self.grid.get((x1, z0))
        h01 = self.grid.get((x0, z1))
        h11 = self.grid.get((x1, z1))

        if None in (h00, h10, h01, h11):
            return self.nearest_height(x, z)

        tx = x - x0
        tz = z - z0
        h0 = h00 * (1.0 - tx) + h10 * tx
        h1 = h01 * (1.0 - tx) + h11 * tx
        return h0 * (1.0 - tz) + h1 * tz


terrain_map: Optional[TerrainMap] = None


def get_terrain(force: bool = False) -> TerrainMap:
    global terrain_map
    if terrain_map is None or force:
        terrain_map = TerrainMap.load(TERRAIN_CSV)
    return terrain_map


def calc_target_angle_and_distance(
    player_pos: dict[str, float],
    target_pos: dict[str, float],
) -> tuple[float, float]:
    dx = target_pos["x"] - player_pos["x"]
    dz = target_pos["z"] - player_pos["z"]
    distance = math.hypot(dx, dz)
    target_world_angle = math.degrees(math.atan2(dx, dz))
    return target_world_angle, distance


def current_obstacle_target(terrain: Optional[TerrainMap] = None) -> dict[str, Any]:
    terrain = terrain or get_terrain()
    terrain_y = terrain.height_at(OBSTACLE_TARGET_X, OBSTACLE_TARGET_Z)
    y = (
        terrain_y + OBSTACLE_TARGET_Y_OFFSET
        if terrain_y is not None
        else PLAYER_START["y"]
    )
    position = {
        "x": OBSTACLE_TARGET_X,
        "y": y,
        "z": OBSTACLE_TARGET_Z,
    }
    return {
        "prefabName": OBSTACLE_PREFAB_NAME,
        "name": OBSTACLE_PREFAB_NAME,
        "type": "Rock002",
        "position": position,
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "terrainY": terrain_y,
    }


def active_target_pos(info: dict[str, Any]) -> dict[str, float]:
    if OBSTACLE_TARGET_MODE:
        obstacle = current_obstacle_target()
        return pos_from_dict(obstacle.get("position"), PLAYER_START)

    enemy_fallback = {
        "x": PLAYER_START["x"],
        "y": PLAYER_START["y"],
        "z": PLAYER_START["z"] + 80.0,
    }
    return pos_from_dict(info.get("enemyPos"), enemy_fallback)


def ballistic_theta_solutions(
    range_m: float,
    dy: float,
    muzzle_speed: float,
    gravity: float = GRAVITY,
) -> list[dict[str, Any]]:
    if range_m < 1e-6:
        return [{"thetaDeg": 0.0, "mode": "zero_range", "discriminant": 0.0}]

    v = muzzle_speed
    g = gravity
    discriminant = v**4 - g * (g * range_m**2 + 2.0 * dy * v**2)

    if discriminant < 0.0:
        return [{
            "thetaDeg": BALLISTIC_MAX_THETA_DEG,
            "mode": "unreachable_use_max_pitch",
            "discriminant": discriminant,
        }]

    sqrt_d = math.sqrt(discriminant)
    low = math.degrees(math.atan((v**2 - sqrt_d) / (g * range_m)))
    high = math.degrees(math.atan((v**2 + sqrt_d) / (g * range_m)))
    return [
        {"thetaDeg": low, "mode": "low_angle_solution", "discriminant": discriminant},
        {"thetaDeg": high, "mode": "high_angle_solution", "discriminant": discriminant},
    ]


def projectile_y_at_range(
    muzzle_y: float,
    forward_m: float,
    theta_deg: float,
    muzzle_speed: float,
    gravity: float = GRAVITY,
) -> float:
    theta_rad = math.radians(theta_deg)
    cos_theta = max(1e-6, math.cos(theta_rad))
    return (
        muzzle_y
        + forward_m * math.tan(theta_rad)
        - gravity * forward_m**2 / (2.0 * muzzle_speed**2 * cos_theta**2)
    )


def trace_trajectory_against_terrain(
    terrain: TerrainMap,
    player_pos: dict[str, float],
    target_pos: dict[str, float],
    theta_deg: float,
    muzzle_speed: float,
    step_m: float = TRACE_STEP_M,
    terrain_clearance_m: float = TERRAIN_CLEARANCE_M,
) -> dict[str, Any]:
    dx = target_pos["x"] - player_pos["x"]
    dz = target_pos["z"] - player_pos["z"]
    range_m = math.hypot(dx, dz)
    if range_m < 1e-6:
        return {"clear": False, "reason": "zero_range", "samples": 0}

    ux = dx / range_m
    uz = dz / range_m
    muzzle_y = player_pos["y"] + MUZZLE_HEIGHT_ABOVE_PLAYER_POS
    samples = max(2, int(math.ceil(range_m / max(0.25, step_m))))
    worst_clearance = float("inf")
    terrain_hit: Optional[dict[str, Any]] = None
    min_clearance_sample: Optional[dict[str, Any]] = None

    for i in range(1, samples + 1):
        s = min(range_m, i * range_m / samples)
        x = player_pos["x"] + ux * s
        z = player_pos["z"] + uz * s
        projectile_y = projectile_y_at_range(
            muzzle_y,
            s,
            theta_deg,
            muzzle_speed,
        )
        terrain_y = terrain.height_at(x, z)
        if terrain_y is None:
            continue

        clearance = projectile_y - terrain_y
        if clearance < worst_clearance:
            worst_clearance = clearance
            min_clearance_sample = {
                "x": round(x, 3),
                "y": round(projectile_y, 3),
                "z": round(z, 3),
                "terrainY": round(terrain_y, 3),
                "rangeM": round(s, 3),
                "clearanceM": round(clearance, 3),
            }

        is_middle = TRACE_SKIP_START_M <= s <= max(TRACE_SKIP_START_M, range_m - TRACE_SKIP_END_M)
        if is_middle and clearance < terrain_clearance_m:
            terrain_hit = {
                "x": round(x, 3),
                "y": round(projectile_y, 3),
                "z": round(z, 3),
                "terrainY": round(terrain_y, 3),
                "rangeM": round(s, 3),
                "clearanceM": round(clearance, 3),
            }
            break

    target_projectile_y = projectile_y_at_range(
        muzzle_y,
        range_m,
        theta_deg,
        muzzle_speed,
    )
    target_error_y = target_projectile_y - target_pos["y"]

    return {
        "clear": terrain_hit is None,
        "reason": "clear" if terrain_hit is None else "terrain_blocked",
        "blockedAt": terrain_hit,
        "minClearance": min_clearance_sample,
        "worstClearanceM": None if worst_clearance == float("inf") else round(worst_clearance, 3),
        "targetProjectileY": round(target_projectile_y, 3),
        "targetErrorY": round(target_error_y, 3),
        "samples": samples,
    }


def solve_hill_fire_solution(
    player_pos: dict[str, float],
    target_pos: dict[str, float],
    terrain: Optional[TerrainMap] = None,
) -> dict[str, Any]:
    terrain = terrain or get_terrain()
    target = dict(target_pos)

    terrain_target_y = terrain.height_at(target["x"], target["z"])
    if terrain_target_y is not None and target.get("y", 0.0) <= terrain_target_y + 0.2:
        target["y"] = terrain_target_y + DEFAULT_TARGET_HEIGHT_ABOVE_TERRAIN

    target_world_angle, range_m = calc_target_angle_and_distance(player_pos, target)
    dy = target["y"] - (player_pos["y"] + MUZZLE_HEIGHT_ABOVE_PLAYER_POS)
    muzzle_speed = effective_muzzle_speed(range_m)

    raw_solutions = ballistic_theta_solutions(range_m, dy, muzzle_speed)
    candidates: list[dict[str, Any]] = []

    for raw in raw_solutions:
        theta = clamp(
            safe_float(raw.get("thetaDeg")),
            BALLISTIC_MIN_THETA_DEG,
            BALLISTIC_MAX_THETA_DEG,
        )
        sim_pitch = clamp(theta, MIN_PITCH_DEG, MAX_PITCH_DEG)
        pitch_limited = sim_pitch != theta
        trace = trace_trajectory_against_terrain(
            terrain,
            player_pos,
            target,
            sim_pitch,
            muzzle_speed,
        )
        target_error_m = abs(safe_float(trace.get("targetErrorY")))
        target_reachable = target_error_m <= TARGET_VERTICAL_HIT_TOLERANCE_M
        candidates.append({
            "thetaDeg": round(theta, 4),
            "desiredPitchDeg": round(sim_pitch, 4),
            "mode": raw.get("mode"),
            "discriminant": round(safe_float(raw.get("discriminant")), 4),
            "pitchLimited": pitch_limited,
            "terrainClear": trace["clear"],
            "targetReachable": target_reachable,
            "trace": trace,
        })

    clear_candidates = [
        item for item in candidates
        if item["terrainClear"]
        and item["targetReachable"]
        and not item["pitchLimited"]
    ]

    if clear_candidates:
        selected = min(clear_candidates, key=lambda item: abs(item["thetaDeg"]))
        status = "ok"
    else:
        selected = min(
            candidates,
            key=lambda item: (
                item["trace"].get("blockedAt") is not None,
                abs(safe_float(item["trace"].get("targetErrorY"))),
            ),
        ) if candidates else None
        status = "blocked"

    return {
        "status": status,
        "targetWorldAngleDeg": round(target_world_angle, 4),
        "rangeM": round(range_m, 4),
        "dyM": round(dy, 4),
        "muzzleSpeed": round(muzzle_speed, 4),
        "playerPos": {k: round(v, 4) for k, v in player_pos.items()},
        "targetPos": {k: round(v, 4) for k, v in target.items()},
        "targetTerrainY": None if terrain_target_y is None else round(terrain_target_y, 4),
        "selected": selected,
        "candidates": candidates,
    }


def make_default_action(reason: str = "idle") -> dict[str, Any]:
    return {
        "moveWS": {"command": "STOP", "weight": 0.0},
        "moveAD": {"command": "STOP", "weight": 0.0},
        "turretQE": {"command": "STOP", "weight": 0.0},
        "turretRF": {"command": "STOP", "weight": 0.0},
        "fire": False,
        "debug": {"reason": reason},
    }


def make_hill_map_action(info: dict[str, Any]) -> dict[str, Any]:
    global aim_ready_since, last_fire_time

    player_pos = pos_from_dict(info.get("playerPos"), PLAYER_START)
    target_pos = active_target_pos(info)

    solution = solve_hill_fire_solution(player_pos, target_pos)
    selected = solution.get("selected") or {}
    desired_pitch = safe_float(selected.get("desiredPitchDeg"))
    target_world_angle = safe_float(solution.get("targetWorldAngleDeg"))

    player_body_yaw = safe_float(info.get("playerBodyX"))
    player_turret_yaw = safe_float(info.get("playerTurretX"))
    player_turret_pitch = safe_float(info.get("playerTurretY"))

    body_error = normalize_angle(target_world_angle - player_body_yaw)
    turret_error = normalize_angle(target_world_angle - player_turret_yaw)
    pitch_error = desired_pitch - player_turret_pitch

    action = make_default_action("hill_map_formula")

    if abs(body_error) > BODY_YAW_DEADBAND_DEG:
        action["moveAD"] = {
            "command": "D" if body_error > 0.0 else "A",
            "weight": round(clamp(abs(body_error) / 90.0, 0.08, 0.35), 3),
        }

    if abs(turret_error) > TURRET_YAW_DEADBAND_DEG:
        action["turretQE"] = {
            "command": "E" if turret_error > 0.0 else "Q",
            "weight": round(clamp(abs(turret_error) / 60.0, 0.04, 0.28), 3),
        }

    if abs(pitch_error) > PITCH_FIRE_TOLERANCE_DEG:
        action["turretRF"] = {
            "command": "R" if pitch_error > 0.0 else "F",
            "weight": round(clamp(abs(pitch_error) / 5.0, 0.045, 0.24), 3),
        }

    now = time.time()
    aim_aligned = (
        solution["status"] == "ok"
        and abs(body_error) < BODY_FIRE_TOLERANCE_DEG
        and abs(turret_error) < 1.2
        and abs(pitch_error) < PITCH_FIRE_TOLERANCE_DEG
    )

    if aim_aligned:
        if aim_ready_since is None:
            aim_ready_since = now
    else:
        aim_ready_since = None

    aim_stable = aim_ready_since is not None and now - aim_ready_since >= AIM_STABLE_SECONDS
    if aim_aligned and aim_stable and now - last_fire_time >= FIRE_COOLDOWN_SECONDS:
        action["fire"] = True
        last_fire_time = now

    action["debug"] = {
        "reason": "hill_map_formula",
        "solutionStatus": solution["status"],
        "rangeM": solution["rangeM"],
        "targetWorldAngleDeg": round(target_world_angle, 3),
        "desiredPitchDeg": round(desired_pitch, 3),
        "playerBodyYawDeg": round(player_body_yaw, 3),
        "playerTurretYawDeg": round(player_turret_yaw, 3),
        "playerTurretPitchDeg": round(player_turret_pitch, 3),
        "bodyErrorDeg": round(body_error, 3),
        "turretErrorDeg": round(turret_error, 3),
        "pitchErrorDeg": round(pitch_error, 3),
        "aimAligned": aim_aligned,
        "aimStable": aim_stable,
        "fire": action["fire"],
        "hillFireSolution": solution,
        "targetMode": "obstacle" if OBSTACLE_TARGET_MODE else "enemy",
        "activeObstacle": current_obstacle_target() if OBSTACLE_TARGET_MODE else None,
    }
    return action


@app.route("/init", methods=["GET"])
def init() -> Any:
    global aim_ready_since, last_fire_time
    terrain = get_terrain()
    obstacle = current_obstacle_target(terrain)
    aim_ready_since = None
    last_fire_time = 0.0
    return jsonify({
        "terrainIndex": 3,
        "startMode": "start",
        "blStartX": PLAYER_START["x"],
        "blStartY": PLAYER_START["y"],
        "blStartZ": PLAYER_START["z"],
        "rdStartX": 290.0 if OBSTACLE_TARGET_MODE else 230.0,
        "rdStartY": PLAYER_START["y"],
        "rdStartZ": 290.0 if OBSTACLE_TARGET_MODE else 230.0,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveLog": True,
        "saveLidarData": False,
        "destoryObstaclesOnHit": True,
        "obstacles": [obstacle],
        "activeTarget": obstacle,
    })


@app.route("/start", methods=["GET"])
def start() -> Any:
    return jsonify({"control": "start"})


@app.route("/info", methods=["POST"])
def info() -> Any:
    global latest_info
    latest_info = request.get_json(silent=True) or {}
    return jsonify({"status": "success", "control": "start"})


@app.route("/get_action", methods=["POST"])
def get_action() -> Any:
    if not latest_info:
        return jsonify(make_default_action("latest_info_empty"))
    return jsonify(make_hill_map_action(latest_info))


@app.route("/hill_fire_solution", methods=["GET", "POST"])
def hill_fire_solution() -> Any:
    payload = request.get_json(silent=True) or {}
    info_payload = payload or latest_info
    player_pos = pos_from_dict(info_payload.get("playerPos"), PLAYER_START)
    if "targetPos" in info_payload:
        target_pos = pos_from_dict(info_payload.get("targetPos"), active_target_pos(info_payload))
    else:
        target_pos = active_target_pos(info_payload)
    return jsonify(solve_hill_fire_solution(player_pos, target_pos))


@app.route("/current_target", methods=["GET"])
def current_target() -> Any:
    return jsonify({
        "targetMode": "obstacle" if OBSTACLE_TARGET_MODE else "enemy",
        "target": current_obstacle_target() if OBSTACLE_TARGET_MODE else None,
    })


@app.route("/terrain_status", methods=["GET"])
def terrain_status() -> Any:
    force = request.args.get("reload") in ("1", "true", "yes")
    terrain = get_terrain(force=force)
    return jsonify({"status": "ok", "terrain": terrain.status()})


@app.route("/terrain_height", methods=["GET"])
def terrain_height() -> Any:
    terrain = get_terrain()
    x = safe_float(request.args.get("x"))
    z = safe_float(request.args.get("z"))
    return jsonify({"x": x, "z": z, "y": terrain.height_at(x, z)})


@app.route("/detect", methods=["POST"])
def detect() -> Any:
    return jsonify([])


@app.route("/update_bullet", methods=["POST"])
def update_bullet() -> Any:
    return jsonify({"status": "OK", "message": "prototype does not log impacts yet"})


@app.route("/collision", methods=["POST"])
def collision() -> Any:
    return jsonify({"status": "OK"})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle() -> Any:
    global latest_obstacles
    data = request.get_json(silent=True) or {}
    obstacles = data.get("obstacles", [])
    latest_obstacles = obstacles if isinstance(obstacles, list) else []
    return jsonify({"status": "OK", "obstacleCount": len(latest_obstacles)})


if __name__ == "__main__":
    terrain = get_terrain()
    print(f"[HILL MAP FIRE] loaded terrain: {terrain.status()}")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
