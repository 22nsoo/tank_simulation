from flask import Flask, request, jsonify
import math
import time
import csv
import os
from collections import deque
from datetime import datetime

app = Flask(__name__)

# ============================================================
# 전역 상태
# ============================================================

latest_info = {}
latest_bullet = {}
latest_collision = {}
latest_obstacles = []

last_fire_time = 0.0
FIRE_COOLDOWN = 1.0
SHOT_PENDING_TIMEOUT = 5.0

# Reset/respawn position for the player tank.
PLAYER_START_X = 150.0
PLAYER_START_Y = 10.0
PLAYER_START_Z = 150.0


# ============================================================
# 120m ~ 44m, 4m 단위 동일 장애물 20개 타깃 시나리오 생성
# ============================================================

def make_obstacle_scenarios():
    """
    NewMap_120m_to_40m_4m_varied.map과 같은 규칙으로 타깃 생성.

    거리:
    120, 116, 112, ..., 44m (총 20개)

    방향:
    golden-angle 방식으로 360도에 분산.
    0deg = +Z, 90deg = +X
    """
    scenarios = []
    angle_step = 137.507764

    for i, distance in enumerate(range(120, 43, -4)):
        bearing = (i * angle_step) % 360.0

        scenarios.append({
            "distance": float(distance),
            "bearing": float(bearing),
            "name": f"{distance}m_{int(round(bearing))}deg"
        })

    return tuple(scenarios)


ENEMY_SPAWN_SCENARIOS = make_obstacle_scenarios()

ENEMY_SPAWN_DISTANCES = tuple(
    scenario["distance"] for scenario in ENEMY_SPAWN_SCENARIOS
)

enemy_spawn_index = 0
next_enemy_spawn_scenario = None

# Static obstacle target mode.
OBSTACLE_TARGET_MODE = True
OBSTACLE_TARGET_Y = 9.502197265625
obstacle_target_index = 0
obstacle_targets_completed = 0
obstacle_test_completed = False

# Keep tank stationary while aiming/firing.
STATIONARY_FIRE_MODE = True
TEST_SINGLE_SHOT_PER_SPAWN = False
shot_fired_for_current_spawn = False
SPAWN_FIRE_ARM_DELAY_SECONDS = 1.5
AIM_STABLE_SECONDS = 0.5
spawn_initialized_at = 0.0
aim_ready_since = None

# Yaw PD controller.
TURRET_YAW_DEADBAND_DEG = 0.5
TURRET_YAW_KP = 0.0053
TURRET_YAW_KD = 0.0020
TURRET_YAW_SPEED_PER_WEIGHT = 43.498
BODY_YAW_SPEED_PER_WEIGHT = 37.254
BODY_YAW_KP = 0.0039
BODY_YAW_DEADBAND_DEG = 8.0
BODY_FIRE_TOLERANCE_DEG = 10.0

previous_turret_error = None
previous_yaw_control_time = None
previous_yaw_target_index = None

pending_shots = deque()
shot_id_counter = 0

# 실행할 때마다 shot_log_1.csv, shot_log_2.csv ...
SHOT_LOG_BASE = "shot_log"
SHOT_LOG_EXT = ".csv"
SHOT_LOG_PATH = None

CONTROL_LOG_BASE = "control_log"
CONTROL_LOG_PATH = None


# ============================================================
# 탄도 기반 포각 계산 + 자동 튜닝 설정
# ============================================================

GRAVITY = 9.81

MUZZLE_SPEED = 59.0

DISTANCE_SPEED_CALIBRATION = (
    (60.0, 64.715),
    (80.0, 66.883),
    (100.0, 63.284),
    (120.0, 61.558),
)

PITCH_CORRECTION_SIGN = 1.0
PITCH_OFFSET_DEG = 0.0
PITCH_BIAS_DEG = 0.0
IMPACT_BIAS_TUNING_ENABLED = False

PITCH_TUNE_GAIN = 0.35
MAX_BIAS_UPDATE_DEG = 2.0
RANGE_ERROR_TOLERANCE = 3.0

# Static-obstacle pitch control tuning.
# Larger errors should move the gun faster, while small errors still slow down
# before entering the fire tolerance band.
PITCH_CONTROL_ERROR_SCALE_DEG = 4.0
PITCH_MIN_WEIGHT = 0.055
PITCH_MAX_WEIGHT_FLAT = 0.22
PITCH_MAX_WEIGHT_SENSITIVE = 0.28
PITCH_SENSITIVE_RANGE_DERIVATIVE = 80.0

MIN_PITCH_DEG = -10.0
MAX_PITCH_DEG = 35.0

SIM_MIN_TURRET_PITCH_DEG = -5.0
SIM_MAX_TURRET_PITCH_DEG = 10.0


# ============================================================
# CSV 파일명 자동 생성
# ============================================================

def get_next_log_path(base, ext=".csv"):
    idx = 1

    while True:
        path = f"{base}_{idx}{ext}"

        if not os.path.exists(path):
            return path

        idx += 1


def get_next_shot_log_path(base=SHOT_LOG_BASE, ext=SHOT_LOG_EXT):
    return get_next_log_path(base, ext)


# ============================================================
# Control log
# ============================================================

def init_control_log():
    global CONTROL_LOG_PATH

    if CONTROL_LOG_PATH is not None:
        return

    CONTROL_LOG_PATH = get_next_log_path(CONTROL_LOG_BASE, SHOT_LOG_EXT)

    with open(CONTROL_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "target_type",
            "target_name",
            "target_index",
            "target_distance",
            "target_bearing",
            "distance",
            "player_turret_pitch",
            "desired_pitch",
            "pitch_error",
            "pitch_tolerance",
            "turret_rf_command",
            "turret_rf_weight",
            "turret_error",
            "turret_error_rate",
            "turret_pd_effort_raw",
            "turret_pd_effort",
            "body_equivalent_turret_weight",
            "turret_qe_command",
            "turret_qe_weight",
            "body_error",
            "body_signed_effort",
            "predicted_body_yaw_rate",
            "body_coarse_turn_enabled",
            "move_ad_command",
            "move_ad_weight",
            "aim_aligned",
            "spawn_arm_ready",
            "aim_stable_ready",
            "fire",
        ])

    print(f"[CONTROL LOG CREATED] {CONTROL_LOG_PATH}")


def log_control_action(info, action):
    global CONTROL_LOG_PATH

    if CONTROL_LOG_PATH is None:
        init_control_log()

    debug = action.get("debug", {})
    turret_rf = action.get("turretRF", {})
    turret_qe = action.get("turretQE", {})
    move_ad = action.get("moveAD", {})

    with open(CONTROL_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            info.get("targetType", "enemy"),
            info.get("targetName"),
            info.get("targetIndex"),
            info.get("targetDistance"),
            info.get("targetBearing"),
            debug.get("distance"),
            debug.get("player_turret_pitch"),
            debug.get("desired_pitch"),
            debug.get("pitch_error"),
            debug.get("pitch_fire_tolerance"),
            turret_rf.get("command"),
            turret_rf.get("weight"),
            debug.get("turret_error"),
            debug.get("turret_error_rate"),
            debug.get("turret_pd_effort_raw"),
            debug.get("turret_pd_effort"),
            debug.get("body_equivalent_turret_weight"),
            turret_qe.get("command"),
            turret_qe.get("weight"),
            debug.get("body_error"),
            debug.get("body_signed_effort"),
            debug.get("predicted_body_yaw_rate"),
            debug.get("body_coarse_turn_enabled"),
            move_ad.get("command"),
            move_ad.get("weight"),
            debug.get("aim_aligned"),
            debug.get("spawn_arm_ready"),
            debug.get("aim_stable_ready"),
            action.get("fire", False),
        ])


# ============================================================
# 기본 유틸
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def effective_muzzle_speed(distance_m):
    distance_m = float(distance_m)
    points = DISTANCE_SPEED_CALIBRATION

    if distance_m <= points[0][0]:
        return points[0][1]

    if distance_m >= points[-1][0]:
        return points[-1][1]

    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if d0 <= distance_m <= d1:
            ratio = (distance_m - d0) / (d1 - d0)
            return v0 + ratio * (v1 - v0)

    return MUZZLE_SPEED


def safe_get_pos(pos, key, default=None):
    if not isinstance(pos, dict):
        return default
    return pos.get(key, default)


def make_default_action():
    return {
        "moveWS": {"command": "STOP", "weight": 0.0},
        "moveAD": {"command": "STOP", "weight": 0.0},
        "turretQE": {"command": "STOP", "weight": 0.0},
        "turretRF": {"command": "STOP", "weight": 0.0},
        "fire": False
    }


def current_obstacle_target():
    scenario = ENEMY_SPAWN_SCENARIOS[
        obstacle_target_index % len(ENEMY_SPAWN_SCENARIOS)
    ]

    distance = float(scenario["distance"])
    bearing = float(scenario["bearing"])
    bearing_rad = math.radians(bearing)

    return {
        "name": f"Rock002_{int(distance)}m_{int(round(bearing))}deg",
        "distance": distance,
        "bearing": bearing,
        "position": {
            "x": PLAYER_START_X + distance * math.sin(bearing_rad),
            "y": OBSTACLE_TARGET_Y,
            "z": PLAYER_START_Z + distance * math.cos(bearing_rad),
        },
    }


def info_with_active_target(info):
    if not OBSTACLE_TARGET_MODE:
        return info

    target = current_obstacle_target()
    aimed_info = dict(info)

    aimed_info["enemyPos"] = dict(target["position"])
    aimed_info["enemyHealth"] = 100.0
    aimed_info["targetType"] = "obstacle"
    aimed_info["targetName"] = target["name"]
    aimed_info["targetIndex"] = obstacle_target_index
    aimed_info["targetDistance"] = target["distance"]
    aimed_info["targetBearing"] = target["bearing"]

    return aimed_info


def calc_target_angle_and_distance(player_pos, enemy_pos):
    dx = float(enemy_pos["x"]) - float(player_pos["x"])
    dz = float(enemy_pos["z"]) - float(player_pos["z"])

    distance = math.sqrt(dx * dx + dz * dz)
    target_world_angle = math.degrees(math.atan2(dx, dz))

    return target_world_angle, distance


# ============================================================
# 수식 기반 탄도 함수
# ============================================================

def ballistic_theta_low_angle(R, dy, muzzle_speed=MUZZLE_SPEED, gravity=GRAVITY):
    if R < 1e-6:
        return 0.0, "zero_range", 0.0

    v = muzzle_speed
    g = gravity

    discriminant = v**4 - g * (g * R**2 + 2 * dy * v**2)

    if discriminant < 0:
        return MAX_PITCH_DEG, "unreachable_use_max_pitch", discriminant

    sqrt_d = math.sqrt(discriminant)

    tan_theta_low = (v**2 - sqrt_d) / (g * R)
    theta_rad = math.atan(tan_theta_low)
    theta_deg = math.degrees(theta_rad)

    return theta_deg, "low_angle_solution", discriminant


def calc_desired_pitch_ballistic(player_pos, enemy_pos):
    global PITCH_BIAS_DEG

    px = float(player_pos["x"])
    py = float(player_pos["y"])
    pz = float(player_pos["z"])

    ex = float(enemy_pos["x"])
    ey = float(enemy_pos["y"])
    ez = float(enemy_pos["z"])

    dx = ex - px
    dz = ez - pz

    R = math.sqrt(dx * dx + dz * dz)
    dy = ey - py

    calibrated_speed = effective_muzzle_speed(R)

    theta_raw, mode, discriminant = ballistic_theta_low_angle(
        R,
        dy,
        muzzle_speed=calibrated_speed,
    )

    theta_with_bias = theta_raw + PITCH_BIAS_DEG
    theta_with_bias = clamp(theta_with_bias, MIN_PITCH_DEG, MAX_PITCH_DEG)

    desired_pitch_unclamped = (
        PITCH_OFFSET_DEG + PITCH_CORRECTION_SIGN * theta_with_bias
    )

    desired_pitch = clamp(
        desired_pitch_unclamped,
        SIM_MIN_TURRET_PITCH_DEG,
        SIM_MAX_TURRET_PITCH_DEG
    )

    debug = {
        "ballistic_R": round(R, 4),
        "ballistic_dy": round(dy, 4),
        "ballistic_mode": mode,
        "ballistic_discriminant": round(discriminant, 4),
        "theta_raw_deg": round(theta_raw, 4),
        "theta_with_bias_deg": round(theta_with_bias, 4),
        "pitch_bias_deg": round(PITCH_BIAS_DEG, 4),
        "desired_pitch_sim": round(desired_pitch, 4),
        "desired_pitch_unclamped": round(desired_pitch_unclamped, 4),
        "pitch_limited": desired_pitch != desired_pitch_unclamped,
        "muzzle_speed": round(calibrated_speed, 4),
        "speed_calibration": "distance_linear_interpolation",
        "pitch_correction_sign": PITCH_CORRECTION_SIGN
    }

    return desired_pitch, debug


def predict_range_flat(theta_deg, muzzle_speed=MUZZLE_SPEED, gravity=GRAVITY):
    theta_rad = math.radians(theta_deg)
    return (muzzle_speed ** 2 / gravity) * math.sin(2.0 * theta_rad)


def range_derivative_numeric(theta_deg, muzzle_speed=MUZZLE_SPEED):
    eps_rad = math.radians(0.2)
    theta_rad = math.radians(theta_deg)

    theta_plus_deg = math.degrees(theta_rad + eps_rad)
    theta_minus_deg = math.degrees(theta_rad - eps_rad)

    r_plus = predict_range_flat(theta_plus_deg, muzzle_speed=muzzle_speed)
    r_minus = predict_range_flat(theta_minus_deg, muzzle_speed=muzzle_speed)

    derivative = (r_plus - r_minus) / (2.0 * eps_rad)

    return derivative


def dynamic_pitch_tolerance(desired_theta_deg, muzzle_speed=MUZZLE_SPEED):
    derivative = abs(
        range_derivative_numeric(
            desired_theta_deg,
            muzzle_speed=muzzle_speed,
        )
    )

    if derivative < 1e-6:
        return 0.15

    tol_rad = RANGE_ERROR_TOLERANCE / derivative
    tol_deg = math.degrees(tol_rad)

    return clamp(tol_deg, 0.08, 0.5)


def update_pitch_bias_from_impact(shot, impact_data):
    global PITCH_BIAS_DEG

    player_pos = shot.get("player_pos", {})
    enemy_pos = shot.get("enemy_pos", {})

    impact_x = impact_data.get("x")
    impact_z = impact_data.get("z")

    if None in [
        player_pos.get("x"),
        player_pos.get("z"),
        enemy_pos.get("x"),
        enemy_pos.get("z"),
        impact_x,
        impact_z
    ]:
        return None

    px = float(player_pos["x"])
    pz = float(player_pos["z"])

    ex = float(enemy_pos["x"])
    ez = float(enemy_pos["z"])

    ix = float(impact_x)
    iz = float(impact_z)

    target_range = math.sqrt((ex - px) ** 2 + (ez - pz) ** 2)
    impact_range = math.sqrt((ix - px) ** 2 + (iz - pz) ** 2)

    range_error = target_range - impact_range

    theta_used = shot.get("theta_with_bias_deg")
    if theta_used is None:
        theta_used = abs(float(shot.get("desired_pitch", 0.0)))

    shot_speed = float(
        shot.get("muzzle_speed", MUZZLE_SPEED) or MUZZLE_SPEED
    )

    derivative = range_derivative_numeric(
        theta_used,
        muzzle_speed=shot_speed,
    )

    if abs(derivative) < 1e-6:
        return None

    delta_theta_rad = PITCH_TUNE_GAIN * (range_error / derivative)
    delta_theta_deg = math.degrees(delta_theta_rad)
    delta_theta_deg = PITCH_CORRECTION_SIGN * delta_theta_deg

    delta_theta_deg = clamp(
        delta_theta_deg,
        -MAX_BIAS_UPDATE_DEG,
        MAX_BIAS_UPDATE_DEG
    )

    old_bias = PITCH_BIAS_DEG

    if IMPACT_BIAS_TUNING_ENABLED:
        PITCH_BIAS_DEG += delta_theta_deg
    else:
        delta_theta_deg = 0.0

    PITCH_BIAS_DEG = clamp(PITCH_BIAS_DEG, -10.0, 15.0)

    tune_debug = {
        "target_range": round(target_range, 4),
        "impact_range": round(impact_range, 4),
        "range_error": round(range_error, 4),
        "theta_used": round(theta_used, 4),
        "range_derivative": round(derivative, 4),
        "delta_theta_deg": round(delta_theta_deg, 4),
        "old_pitch_bias_deg": round(old_bias, 4),
        "new_pitch_bias_deg": round(PITCH_BIAS_DEG, 4),
        "bias_tuning_enabled": IMPACT_BIAS_TUNING_ENABLED,
    }

    print("[PITCH AUTO TUNE]", tune_debug)

    return tune_debug


# ============================================================
# CSV 로그
# ============================================================

def init_shot_log():
    global SHOT_LOG_PATH

    if SHOT_LOG_PATH is not None:
        return

    SHOT_LOG_PATH = get_next_shot_log_path()

    with open(SHOT_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "shot_id",
            "fire_time",
            "impact_time",
            "flight_time_sec",

            "hit",
            "impact_x",
            "impact_y",
            "impact_z",

            "player_x_fire",
            "player_y_fire",
            "player_z_fire",

            "enemy_x_fire",
            "enemy_y_fire",
            "enemy_z_fire",
            "enemy_health_fire",

            "target_type",
            "target_name",
            "target_index",
            "target_distance",
            "target_bearing",

            "distance_fire",
            "target_world_angle_fire",

            "player_body_yaw_fire",
            "player_turret_yaw_fire",
            "player_turret_pitch_fire",

            "body_error_fire",
            "turret_error_fire",
            "desired_pitch_fire",
            "pitch_error_fire",
            "pitch_fire_tolerance",

            "ballistic_R_fire",
            "ballistic_dy_fire",
            "ballistic_discriminant_fire",
            "ballistic_mode_fire",
            "theta_raw_deg_fire",
            "theta_with_bias_deg_fire",
            "pitch_bias_deg_fire",
            "muzzle_speed_fire",

            "impact_error_to_enemy_fire",
            "range_error_fire",
            "z_shortfall_fire",
            "forward_error_fire",
            "lateral_error_fire",
            "vertical_error_fire",

            "tune_target_range",
            "tune_impact_range",
            "tune_range_error",
            "tune_theta_used",
            "tune_range_derivative",
            "tune_delta_theta_deg",
            "tune_old_pitch_bias_deg",
            "tune_new_pitch_bias_deg",

            "action_moveWS_command",
            "action_moveWS_weight",
            "action_moveAD_command",
            "action_moveAD_weight",
            "action_turretQE_command",
            "action_turretQE_weight",
            "action_turretRF_command",
            "action_turretRF_weight",
        ])

    print(f"[SHOT LOG CREATED] {SHOT_LOG_PATH}")


def save_pending_shot(info, debug, action):
    global shot_id_counter

    shot_id_counter += 1

    player_pos = info.get("playerPos", {})
    enemy_pos = info.get("enemyPos", {})
    ballistic = debug.get("ballistic", {})

    shot = {
        "shot_id": shot_id_counter,
        "fire_time": time.time(),
        "fire_time_iso": datetime.now().isoformat(timespec="milliseconds"),

        "player_pos": {
            "x": safe_get_pos(player_pos, "x"),
            "y": safe_get_pos(player_pos, "y"),
            "z": safe_get_pos(player_pos, "z"),
        },

        "enemy_pos": {
            "x": safe_get_pos(enemy_pos, "x"),
            "y": safe_get_pos(enemy_pos, "y"),
            "z": safe_get_pos(enemy_pos, "z"),
        },

        "enemy_health": info.get("enemyHealth"),

        "target_type": info.get("targetType", "enemy"),
        "target_name": info.get("targetName"),
        "target_index": info.get("targetIndex"),
        "target_distance": info.get("targetDistance"),
        "target_bearing": info.get("targetBearing"),

        "distance": debug.get("distance"),
        "target_world_angle": debug.get("target_world_angle"),

        "player_body_yaw": debug.get("player_body_yaw"),
        "player_turret_yaw": debug.get("player_turret_yaw"),
        "player_turret_pitch": debug.get("player_turret_pitch"),

        "body_error": debug.get("body_error"),
        "turret_error": debug.get("turret_error"),
        "desired_pitch": debug.get("desired_pitch"),
        "pitch_error": debug.get("pitch_error"),
        "pitch_fire_tolerance": debug.get("pitch_fire_tolerance"),

        "ballistic_R": ballistic.get("ballistic_R"),
        "ballistic_dy": ballistic.get("ballistic_dy"),
        "ballistic_discriminant": ballistic.get("ballistic_discriminant"),
        "ballistic_mode": ballistic.get("ballistic_mode"),
        "theta_raw_deg": ballistic.get("theta_raw_deg"),
        "theta_with_bias_deg": ballistic.get("theta_with_bias_deg"),
        "pitch_bias_deg": ballistic.get("pitch_bias_deg"),
        "muzzle_speed": ballistic.get("muzzle_speed"),

        "action_moveWS": action.get("moveWS", {}),
        "action_moveAD": action.get("moveAD", {}),
        "action_turretQE": action.get("turretQE", {}),
        "action_turretRF": action.get("turretRF", {}),
    }

    pending_shots.append(shot)

    while len(pending_shots) > 20:
        pending_shots.popleft()

    print(
        f"[SHOT SAVED] "
        f"file={SHOT_LOG_PATH} "
        f"shot_id={shot['shot_id']} "
        f"target={shot['target_name']} "
        f"distance={shot['distance']} "
        f"turret_error={shot['turret_error']} "
        f"pitch_error={shot['pitch_error']} "
        f"desired_pitch={shot['desired_pitch']}"
    )


def reset_obstacle_target_aim_state():
    global shot_fired_for_current_spawn, aim_ready_since, last_fire_time
    global previous_turret_error, previous_yaw_control_time
    global previous_yaw_target_index

    shot_fired_for_current_spawn = False
    aim_ready_since = None
    last_fire_time = time.time()
    previous_turret_error = None
    previous_yaw_control_time = None
    previous_yaw_target_index = None


def advance_obstacle_target():
    global obstacle_target_index, obstacle_targets_completed
    global obstacle_test_completed

    obstacle_targets_completed += 1

    if obstacle_targets_completed >= len(ENEMY_SPAWN_SCENARIOS):
        obstacle_test_completed = True
        pending_shots.clear()
        reset_obstacle_target_aim_state()
        print(
            f"[OBSTACLE TEST COMPLETE] "
            f"shots={obstacle_targets_completed}"
        )
        return

    obstacle_target_index += 1

    pending_shots.clear()
    reset_obstacle_target_aim_state()

    print(
        f"[NEXT TARGET] index={obstacle_target_index} "
        f"target={current_obstacle_target()['name']}"
    )


def log_bullet_impact(impact_data):
    global SHOT_LOG_PATH

    if SHOT_LOG_PATH is None:
        init_shot_log()

    impact_time = time.time()
    impact_time_iso = datetime.now().isoformat(timespec="milliseconds")

    if pending_shots:
        shot = pending_shots.popleft()
    else:
        shot = None

    impact_x = impact_data.get("x")
    impact_y = impact_data.get("y")
    impact_z = impact_data.get("z")
    hit = impact_data.get("hit", "unknown")

    if shot is not None:
        player_pos = shot.get("player_pos", {})
        enemy_pos = shot.get("enemy_pos", {})

        enemy_x = enemy_pos.get("x")
        enemy_y = enemy_pos.get("y")
        enemy_z = enemy_pos.get("z")

        px = player_pos.get("x")
        pz = player_pos.get("z")

        impact_error = None
        range_error = None
        z_shortfall = None
        forward_error = None
        lateral_error = None
        vertical_error = None

        if None not in [impact_x, impact_y, impact_z, enemy_x, enemy_y, enemy_z]:
            dx = float(impact_x) - float(enemy_x)
            dy = float(impact_y) - float(enemy_y)
            dz = float(impact_z) - float(enemy_z)
            impact_error = math.sqrt(dx * dx + dy * dy + dz * dz)
            z_shortfall = float(enemy_z) - float(impact_z)

        if None not in [px, pz, enemy_x, enemy_z, impact_x, impact_z]:
            target_dx = float(enemy_x) - float(px)
            target_dz = float(enemy_z) - float(pz)
            target_range = math.sqrt(target_dx ** 2 + target_dz ** 2)
            impact_range = math.sqrt(
                (float(impact_x) - float(px)) ** 2
                + (float(impact_z) - float(pz)) ** 2
            )
            range_error = target_range - impact_range

            if target_range > 1e-6:
                forward_x = target_dx / target_range
                forward_z = target_dz / target_range
                right_x = forward_z
                right_z = -forward_x
                error_x = float(impact_x) - float(enemy_x)
                error_z = float(impact_z) - float(enemy_z)

                forward_error = (
                    error_x * forward_x + error_z * forward_z
                )
                lateral_error = (
                    error_x * right_x + error_z * right_z
                )

        if None not in [impact_y, enemy_y]:
            vertical_error = float(impact_y) - float(enemy_y)

        tune_debug = update_pitch_bias_from_impact(shot, impact_data)

        flight_time = impact_time - shot["fire_time"]

        action_moveWS = shot.get("action_moveWS", {})
        action_moveAD = shot.get("action_moveAD", {})
        action_turretQE = shot.get("action_turretQE", {})
        action_turretRF = shot.get("action_turretRF", {})

        row = [
            shot["shot_id"],
            shot["fire_time_iso"],
            impact_time_iso,
            round(flight_time, 4),

            hit,
            impact_x,
            impact_y,
            impact_z,

            shot["player_pos"].get("x"),
            shot["player_pos"].get("y"),
            shot["player_pos"].get("z"),

            shot["enemy_pos"].get("x"),
            shot["enemy_pos"].get("y"),
            shot["enemy_pos"].get("z"),
            shot.get("enemy_health"),

            shot.get("target_type"),
            shot.get("target_name"),
            shot.get("target_index"),
            shot.get("target_distance"),
            shot.get("target_bearing"),

            shot.get("distance"),
            shot.get("target_world_angle"),

            shot.get("player_body_yaw"),
            shot.get("player_turret_yaw"),
            shot.get("player_turret_pitch"),

            shot.get("body_error"),
            shot.get("turret_error"),
            shot.get("desired_pitch"),
            shot.get("pitch_error"),
            shot.get("pitch_fire_tolerance"),

            shot.get("ballistic_R"),
            shot.get("ballistic_dy"),
            shot.get("ballistic_discriminant"),
            shot.get("ballistic_mode"),
            shot.get("theta_raw_deg"),
            shot.get("theta_with_bias_deg"),
            shot.get("pitch_bias_deg"),
            shot.get("muzzle_speed"),

            impact_error,
            range_error,
            z_shortfall,
            forward_error,
            lateral_error,
            vertical_error,

            tune_debug.get("target_range") if tune_debug else None,
            tune_debug.get("impact_range") if tune_debug else None,
            tune_debug.get("range_error") if tune_debug else None,
            tune_debug.get("theta_used") if tune_debug else None,
            tune_debug.get("range_derivative") if tune_debug else None,
            tune_debug.get("delta_theta_deg") if tune_debug else None,
            tune_debug.get("old_pitch_bias_deg") if tune_debug else None,
            tune_debug.get("new_pitch_bias_deg") if tune_debug else None,

            action_moveWS.get("command"),
            action_moveWS.get("weight"),
            action_moveAD.get("command"),
            action_moveAD.get("weight"),
            action_turretQE.get("command"),
            action_turretQE.get("weight"),
            action_turretRF.get("command"),
            action_turretRF.get("weight"),
        ]

        print(
            f"[IMPACT LOGGED] "
            f"file={SHOT_LOG_PATH} "
            f"shot_id={shot['shot_id']} "
            f"target={shot.get('target_name')} "
            f"hit={hit} "
            f"impact_error={impact_error} "
            f"range_error={range_error} "
            f"forward_error={forward_error} "
            f"lateral_error={lateral_error} "
            f"vertical_error={vertical_error}"
        )

    else:
        row = [
            "",
            "",
            impact_time_iso,
            "",

            hit,
            impact_x,
            impact_y,
            impact_z,

            "", "", "",
            "", "", "", "",

            "", "", "", "", "",

            "", "",

            "", "", "",

            "", "", "", "", "",

            "", "", "", "", "", "", "", "",

            "", "", "",

            "", "", "", "", "", "", "", "",

            "", "",
            "", "",
            "", "",
            "", "",
        ]

        print(f"[IMPACT ONLY] file={SHOT_LOG_PATH} hit={hit}")

    with open(SHOT_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if shot is not None and OBSTACLE_TARGET_MODE:
        advance_obstacle_target()


# ============================================================
# 자동 조준 핵심
# ============================================================

def clear_stale_pending_shot_if_needed(now):
    if not pending_shots:
        return

    oldest_shot = pending_shots[0]

    if now - oldest_shot["fire_time"] > SHOT_PENDING_TIMEOUT:
        print(
            f"[PENDING SHOT TIMEOUT] "
            f"clearing stale shot_id={oldest_shot.get('shot_id')}"
        )
        pending_shots.clear()
        reset_obstacle_target_aim_state()


def make_aim_action(info):
    global last_fire_time, shot_fired_for_current_spawn, aim_ready_since
    global previous_turret_error, previous_yaw_control_time
    global previous_yaw_target_index

    action = make_default_action()

    if OBSTACLE_TARGET_MODE and obstacle_test_completed:
        action["debug"] = {
            "reason": "20 obstacle test complete",
            "targetsCompleted": obstacle_targets_completed,
            "targetCount": len(ENEMY_SPAWN_SCENARIOS),
        }
        return action

    player_pos = info.get("playerPos")
    enemy_pos = info.get("enemyPos")

    if not player_pos or not enemy_pos:
        action["debug"] = {"reason": "missing playerPos or enemyPos"}
        return action

    enemy_health = float(info.get("enemyHealth", 100.0))

    if enemy_health <= 0:
        action["debug"] = {"reason": "enemy dead", "enemyHealth": enemy_health}
        return action

    player_body_yaw = float(info.get("playerBodyX", 0.0))
    player_turret_yaw = float(info.get("playerTurretX", 0.0))
    player_turret_pitch = float(info.get("playerTurretY", 0.0))

    target_world_angle, distance = calc_target_angle_and_distance(
        player_pos=player_pos,
        enemy_pos=enemy_pos
    )

    body_error = normalize_angle(target_world_angle - player_body_yaw)
    turret_error = normalize_angle(target_world_angle - player_turret_yaw)

    desired_pitch, ballistic_debug = calc_desired_pitch_ballistic(
        player_pos=player_pos,
        enemy_pos=enemy_pos
    )

    pitch_error = desired_pitch - player_turret_pitch

    abs_body_error = abs(body_error)
    abs_turret_error = abs(turret_error)
    abs_pitch_error = abs(pitch_error)

    # --------------------------------------------------------
    # stale pending shot timeout
    # --------------------------------------------------------
    now = time.time()
    clear_stale_pending_shot_if_needed(now)

    # --------------------------------------------------------
    # 차체 yaw 제어
    # --------------------------------------------------------
    body_signed_effort = clamp(
        BODY_YAW_KP * body_error,
        -0.35,
        0.35,
    )

    if (
        abs_body_error > BODY_YAW_DEADBAND_DEG
        and abs(body_signed_effort) >= 0.001
    ):
        action["moveAD"] = {
            "command": "D" if body_signed_effort > 0 else "A",
            "weight": round(abs(body_signed_effort), 3),
        }
    else:
        body_signed_effort = 0.0

    predicted_body_yaw_rate = (
        BODY_YAW_SPEED_PER_WEIGHT * body_signed_effort
    )

    # --------------------------------------------------------
    # 포탑 yaw PD 제어
    # --------------------------------------------------------
    yaw_control_time = time.monotonic()
    yaw_target_index = info.get("targetIndex")

    if yaw_target_index != previous_yaw_target_index:
        previous_turret_error = None
        previous_yaw_control_time = None

    yaw_dt = (
        yaw_control_time - previous_yaw_control_time
        if previous_yaw_control_time is not None
        else 0.0
    )

    turret_error_rate = (
        normalize_angle(turret_error - previous_turret_error) / yaw_dt
        if previous_turret_error is not None and yaw_dt > 1e-3
        else 0.0
    )

    if distance < 40:
        max_turret_weight = 0.12
    elif distance < 90:
        max_turret_weight = 0.22
    else:
        max_turret_weight = 0.32

    turret_pd_effort_raw = (
        TURRET_YAW_KP * turret_error
        + TURRET_YAW_KD * turret_error_rate
    )

    body_equivalent_turret_weight = (
        predicted_body_yaw_rate / TURRET_YAW_SPEED_PER_WEIGHT
    )

    turret_pd_effort = clamp(
        turret_pd_effort_raw - body_equivalent_turret_weight,
        -max_turret_weight,
        max_turret_weight,
    )

    if (
        abs_turret_error > TURRET_YAW_DEADBAND_DEG
        and abs(turret_pd_effort) >= 0.001
    ):
        action["turretQE"] = {
            "command": "E" if turret_pd_effort > 0 else "Q",
            "weight": round(abs(turret_pd_effort), 3)
        }

    previous_turret_error = turret_error
    previous_yaw_control_time = yaw_control_time
    previous_yaw_target_index = yaw_target_index

    # --------------------------------------------------------
    # 포신 pitch 제어
    # --------------------------------------------------------
    if abs_pitch_error > 0.2:
        theta_for_gain = ballistic_debug.get("theta_with_bias_deg", abs(desired_pitch))
        calibrated_speed = float(
            ballistic_debug.get("muzzle_speed", MUZZLE_SPEED)
        )

        dR_dtheta = abs(
            range_derivative_numeric(
                theta_for_gain,
                muzzle_speed=calibrated_speed,
            )
        )

        max_pitch_weight = (
            PITCH_MAX_WEIGHT_SENSITIVE
            if dR_dtheta > PITCH_SENSITIVE_RANGE_DERIVATIVE
            else PITCH_MAX_WEIGHT_FLAT
        )

        pitch_weight = clamp(
            abs_pitch_error / PITCH_CONTROL_ERROR_SCALE_DEG * max_pitch_weight,
            PITCH_MIN_WEIGHT,
            max_pitch_weight
        )

        action["turretRF"] = {
            "command": "R" if pitch_error > 0 else "F",
            "weight": round(pitch_weight, 3)
        }

    # 가까운 거리에서는 차체 흔들림 방지
    if distance < 50 and abs_turret_error < 25:
        action["moveAD"] = {"command": "STOP", "weight": 0.0}

    # --------------------------------------------------------
    # 이동 제어
    # --------------------------------------------------------
    if STATIONARY_FIRE_MODE:
        action["moveWS"] = {"command": "STOP", "weight": 0.0}
    elif distance > 120 and abs_body_error < 20:
        action["moveWS"] = {"command": "W", "weight": 0.25}
    elif distance > 70 and abs_body_error < 15:
        action["moveWS"] = {"command": "W", "weight": 0.12}
    else:
        action["moveWS"] = {"command": "STOP", "weight": 0.0}

    # --------------------------------------------------------
    # 발사 조건
    # --------------------------------------------------------
    theta_for_tol = ballistic_debug.get("theta_with_bias_deg", abs(desired_pitch))

    pitch_tol = dynamic_pitch_tolerance(
        theta_for_tol,
        muzzle_speed=float(
            ballistic_debug.get("muzzle_speed", MUZZLE_SPEED)
        ),
    )

    aim_aligned = (
        20 < distance < 200
        and abs_body_error < BODY_FIRE_TOLERANCE_DEG
        and abs_turret_error < 1.5
        and abs_pitch_error < pitch_tol
    )

    if aim_aligned:
        if aim_ready_since is None:
            aim_ready_since = now
    else:
        aim_ready_since = None

    spawn_arm_ready = (
        now - spawn_initialized_at >= SPAWN_FIRE_ARM_DELAY_SECONDS
    )

    aim_stable_ready = (
        aim_ready_since is not None
        and now - aim_ready_since >= AIM_STABLE_SECONDS
    )

    can_fire = (
        aim_aligned
        and spawn_arm_ready
        and aim_stable_ready
        and now - last_fire_time > FIRE_COOLDOWN
        and not pending_shots
        and (
            not TEST_SINGLE_SHOT_PER_SPAWN
            or not shot_fired_for_current_spawn
        )
    )

    if can_fire:
        action["fire"] = True
        last_fire_time = now
        shot_fired_for_current_spawn = True

    # --------------------------------------------------------
    # debug
    # --------------------------------------------------------
    body_coarse_turn_enabled = (
        abs_body_error > BODY_YAW_DEADBAND_DEG
    )

    action["debug"] = {
        "distance": round(distance, 2),
        "target_world_angle": round(target_world_angle, 2),

        "player_body_yaw": round(player_body_yaw, 2),
        "player_turret_yaw": round(player_turret_yaw, 2),
        "player_turret_pitch": round(player_turret_pitch, 2),

        "body_error": round(body_error, 2),
        "turret_error": round(turret_error, 2),
        "turret_error_rate": round(turret_error_rate, 3),
        "turret_pd_effort_raw": round(turret_pd_effort_raw, 4),
        "turret_pd_effort": round(turret_pd_effort, 4),
        "body_signed_effort": round(body_signed_effort, 4),
        "predicted_body_yaw_rate": round(predicted_body_yaw_rate, 3),
        "body_equivalent_turret_weight": round(
            body_equivalent_turret_weight,
            4,
        ),
        "body_coarse_turn_enabled": body_coarse_turn_enabled,

        "desired_pitch": round(desired_pitch, 2),
        "pitch_error": round(pitch_error, 2),
        "pitch_fire_tolerance": round(pitch_tol, 4),

        "aim_aligned": aim_aligned,
        "spawn_arm_ready": spawn_arm_ready,
        "aim_stable_ready": aim_stable_ready,
        "spawn_elapsed_seconds": round(now - spawn_initialized_at, 3),
        "aim_stable_elapsed_seconds": (
            round(now - aim_ready_since, 3)
            if aim_ready_since is not None
            else 0.0
        ),

        "enemy_health": enemy_health,
        "fire": action["fire"],

        "ballistic": ballistic_debug,

        "moveWS": action["moveWS"],
        "moveAD": action["moveAD"],
        "turretQE": action["turretQE"],
        "turretRF": action["turretRF"],
    }

    if action["fire"]:
        save_pending_shot(info, action["debug"], action)

    return action


# ============================================================
# Endpoint
# ============================================================

@app.route("/init", methods=["GET"])
def init():
    global enemy_spawn_index, next_enemy_spawn_scenario
    global last_fire_time, shot_fired_for_current_spawn
    global spawn_initialized_at, aim_ready_since
    global previous_turret_error, previous_yaw_control_time
    global previous_yaw_target_index
    global obstacle_target_index, obstacle_targets_completed
    global obstacle_test_completed

    # reset 시 stale shot만 정리.
    # 리셋하면 동일 장애물 20개 검증을 첫 표적부터 다시 시작한다.
    pending_shots.clear()
    obstacle_target_index = 0
    obstacle_targets_completed = 0
    obstacle_test_completed = False

    last_fire_time = 0.0
    shot_fired_for_current_spawn = False
    spawn_initialized_at = time.time()
    aim_ready_since = None

    previous_turret_error = None
    previous_yaw_control_time = None
    previous_yaw_target_index = None

    if next_enemy_spawn_scenario is not None:
        enemy_scenario = next_enemy_spawn_scenario
        next_enemy_spawn_scenario = None
    else:
        enemy_scenario = ENEMY_SPAWN_SCENARIOS[
            enemy_spawn_index % len(ENEMY_SPAWN_SCENARIOS)
        ]
        enemy_spawn_index += 1

    enemy_distance = float(enemy_scenario["distance"])
    enemy_bearing = float(enemy_scenario["bearing"])
    enemy_bearing_rad = math.radians(enemy_bearing)

    enemy_start_x = PLAYER_START_X + enemy_distance * math.sin(enemy_bearing_rad)
    enemy_start_z = PLAYER_START_Z + enemy_distance * math.cos(enemy_bearing_rad)

    print(
        f"[INIT] obstacle_target_index={obstacle_target_index} "
        f"active_target={current_obstacle_target()['name']}"
    )

    return jsonify({
        "startMode": "start",

        "blStartX": PLAYER_START_X,
        "blStartY": PLAYER_START_Y,
        "blStartZ": PLAYER_START_Z,

        # obstacle target mode에서는 적 전차를 구석에 둔다.
        "rdStartX": 290.0 if OBSTACLE_TARGET_MODE else enemy_start_x,
        "rdStartY": PLAYER_START_Y,
        "rdStartZ": 290.0 if OBSTACLE_TARGET_MODE else enemy_start_z,

        "trackingMode": True,
        "detectMode": False,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": True,

        "stereoCameraMode": False,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,

        "destoryObstaclesOnHit": True
    })


@app.route("/set_test_distance/<float:distance>", methods=["POST", "GET"])
def set_test_distance(distance):
    global next_enemy_spawn_scenario

    if distance not in ENEMY_SPAWN_DISTANCES:
        return jsonify({
            "status": "error",
            "message": f"distance must be one of {ENEMY_SPAWN_DISTANCES}"
        }), 400

    next_enemy_spawn_scenario = next(
        scenario
        for scenario in ENEMY_SPAWN_SCENARIOS
        if scenario["distance"] == float(distance)
    )

    return jsonify({
        "status": "ready",
        "nextEnemyDistance": next_enemy_spawn_scenario["distance"],
        "nextEnemyBearing": next_enemy_spawn_scenario["bearing"],
        "nextEnemyDirection": next_enemy_spawn_scenario["name"],
        "message": "Reset the game to apply this distance."
    })


@app.route("/set_target_index/<int:index>", methods=["POST", "GET"])
def set_target_index(index):
    global obstacle_target_index, obstacle_targets_completed
    global obstacle_test_completed

    if not (0 <= index < len(ENEMY_SPAWN_SCENARIOS)):
        return jsonify({
            "status": "error",
            "message": f"index must be 0 ~ {len(ENEMY_SPAWN_SCENARIOS) - 1}"
        }), 400

    obstacle_target_index = index
    obstacle_targets_completed = index
    obstacle_test_completed = False
    reset_obstacle_target_aim_state()

    return jsonify({
        "status": "ready",
        "targetIndex": obstacle_target_index,
        "targetsCompleted": obstacle_targets_completed,
        "testCompleted": obstacle_test_completed,
        "target": current_obstacle_target(),
    })


@app.route("/current_target", methods=["GET"])
def current_target():
    return jsonify({
        "obstacleTargetMode": OBSTACLE_TARGET_MODE,
        "targetIndex": obstacle_target_index,
        "targetCount": len(ENEMY_SPAWN_SCENARIOS),
        "targetsCompleted": obstacle_targets_completed,
        "testCompleted": obstacle_test_completed,
        "target": current_obstacle_target(),
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({
        "control": "start"
    })


@app.route("/info", methods=["POST"])
def info():
    global latest_info

    data = request.get_json()
    latest_info = data

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": "start"
    })


@app.route("/get_action", methods=["POST"])
def get_action():
    data = request.get_json()

    if not latest_info:
        action = make_default_action()
        action["debug"] = {
            "reason": "latest_info is empty",
            "get_action_data": data
        }
        return jsonify(action)

    aimed_info = info_with_active_target(latest_info)
    action = make_aim_action(aimed_info)
    log_control_action(aimed_info, action)

    return jsonify(action)


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    global latest_bullet

    data = request.get_json()
    latest_bullet = data

    print("[BULLET]", data)

    log_bullet_impact(data)

    return jsonify({
        "status": "OK",
        "message": "Bullet impact data received",
        "log_file": SHOT_LOG_PATH,
        "next_target": current_obstacle_target() if OBSTACLE_TARGET_MODE else None,
    })


@app.route("/collision", methods=["POST"])
def collision():
    global latest_collision

    data = request.get_json()
    latest_collision = data

    print("[COLLISION]", data)

    return jsonify({
        "status": "OK"
    })


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    global latest_obstacles

    data = request.get_json()
    latest_obstacles = data.get("obstacles", [])

    print("[OBSTACLE]", data)

    return jsonify({
        "status": "OK"
    })


@app.route("/detect", methods=["POST"])
def detect():
    return jsonify([])


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    init_shot_log()
    init_control_log()

    print(f"[TARGET COUNT] {len(ENEMY_SPAWN_SCENARIOS)}")
    for i, scenario in enumerate(ENEMY_SPAWN_SCENARIOS):
        print(
            f"  #{i:02d} "
            f"distance={scenario['distance']} "
            f"bearing={round(scenario['bearing'], 2)} "
            f"name={scenario['name']}"
        )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
