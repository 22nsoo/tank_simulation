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

enemy_motion_samples = deque(maxlen=12)
enemy_velocity_x = 0.0
enemy_velocity_z = 0.0
enemy_accel_x = 0.0
enemy_accel_z = 0.0
enemy_motion_sample_count = 0
enemy_motion_last_position = None
enemy_motion_last_time = None

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
OBSTACLE_TARGET_MODE = False
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
MOVING_AIM_STABLE_SECONDS = 0.12
MOVING_TURRET_FIRE_TOLERANCE_DEG = 2.5
MOVING_BODY_FIRE_TOLERANCE_DEG = 14.0
MOVING_PITCH_TOLERANCE_MULTIPLIER = 1.25
MOVING_LEAD_BOOST_FULL_DISTANCE_M = 4.0
MOVING_YAW_MAX_WEIGHT_BOOST = 0.65
MOVING_PITCH_MAX_WEIGHT_BOOST = 0.30
MOVING_PREDICTIVE_FIRE_LOOKAHEAD_SECONDS = 0.18
MOVING_PREDICTIVE_FIRE_STABLE_SECONDS = 0.04
MOVING_PREFIRE_TURRET_WINDOW_DEG = 6.0
MOVING_PREFIRE_PITCH_TOLERANCE_MULTIPLIER = 1.8
MOVING_MIN_YAW_CLOSING_RATE_DEG_S = 2.5
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

# Moving-target lead prediction.
ENEMY_VELOCITY_EMA_ALPHA = 0.35
ENEMY_ACCEL_EMA_ALPHA = 0.25
ENEMY_MAX_VALID_SPEED_MPS = 8.0
ENEMY_MAX_VALID_ACCEL_MPS2 = 8.0
ENEMY_MOTION_MIN_SAMPLES = 3
FIRE_SYSTEM_DELAY_SECONDS = 0.10
FLIGHT_TIME_CORRECTION_FACTOR = 1.06
MAX_FLIGHT_TIME_SECONDS = 4.0
MAX_LEAD_DISTANCE_M = 35.0
MAX_ACCEL_LEAD_DISTANCE_M = 8.0
INTERCEPT_SOLVER_ITERATIONS = 8
MAX_CONTROL_PREDICTION_SECONDS = 15.0
CONTROL_TIME_SAFETY_FACTOR = 1.20
PITCH_SPEED_PER_WEIGHT = 4.562
PITCH_CONTROL_SIM_STEP_SECONDS = 0.05
MOTION_REVERSAL_HOLD_SECONDS = 0.5

previous_turret_error = None
previous_yaw_control_time = None
previous_yaw_target_index = None

pending_shots = deque()
shot_id_counter = 0
MAX_MOVING_TARGET_SHOTS = 15
IGNORE_MOVING_TARGET_HEALTH_FOR_SHOT_LIMIT_TEST = True
moving_target_shots_fired = 0
enemy_motion_reversal_hold_until = 0.0

# 실행할 때마다 shot_log_1.csv, shot_log_2.csv ...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOVING_LOG_DIR = os.path.join(SCRIPT_DIR, "moving_target_logs")

SHOT_LOG_BASE = os.path.join(MOVING_LOG_DIR, "moving_shot_log")
SHOT_LOG_EXT = ".csv"
SHOT_LOG_PATH = None

CONTROL_LOG_BASE = os.path.join(MOVING_LOG_DIR, "moving_control_log")
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

# Moving-target pitch control tuning.
# Keep the same formula as the static-obstacle tuning so the measured pitch
# response and the prediction model use one consistent R/F command curve.
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
    log_directory = os.path.dirname(base)
    if log_directory:
        os.makedirs(log_directory, exist_ok=True)

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
            "motion_ready",
            "motion_sample_count",
            "reversal_hold_remaining",
            "enemy_velocity_x",
            "enemy_velocity_z",
            "enemy_speed",
            "enemy_accel_x",
            "enemy_accel_z",
            "enemy_accel",
            "intercept_solver",
            "predicted_control_time",
            "predicted_yaw_control_time",
            "predicted_pitch_control_time",
            "target_angle_rate_deg_s",
            "yaw_closing_rate_deg_s",
            "predicted_flight_time",
            "flight_time_correction_factor",
            "predicted_total_intercept_time",
            "lead_distance",
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
            "predicted_turret_yaw_rate",
            "body_equivalent_turret_weight",
            "turret_qe_command",
            "turret_qe_weight",
            "body_error",
            "body_signed_effort",
            "predicted_body_yaw_rate",
            "body_coarse_turn_enabled",
            "pitch_signed_effort",
            "predicted_pitch_rate",
            "move_ad_command",
            "move_ad_weight",
            "aim_aligned",
            "predictive_aim_aligned",
            "fire_alignment_ready",
            "predictive_fire_lookahead_seconds",
            "predicted_turret_error_at_fire",
            "predicted_pitch_error_at_fire",
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
    moving_target = debug.get("moving_target", {})

    with open(CONTROL_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            info.get("targetType", "enemy"),
            info.get("targetName"),
            info.get("targetIndex"),
            info.get("targetDistance"),
            info.get("targetBearing"),
            debug.get("distance"),
            moving_target.get("motion_ready"),
            moving_target.get("motion_sample_count"),
            moving_target.get("reversal_hold_remaining"),
            moving_target.get("enemy_velocity_x"),
            moving_target.get("enemy_velocity_z"),
            moving_target.get("enemy_speed"),
            moving_target.get("enemy_accel_x"),
            moving_target.get("enemy_accel_z"),
            moving_target.get("enemy_accel"),
            moving_target.get("intercept_solver"),
            moving_target.get("predicted_control_time"),
            moving_target.get("predicted_yaw_control_time"),
            moving_target.get("predicted_pitch_control_time"),
            moving_target.get("target_angle_rate_deg_s"),
            moving_target.get("yaw_closing_rate_deg_s"),
            moving_target.get("predicted_flight_time"),
            moving_target.get("flight_time_correction_factor"),
            moving_target.get("predicted_total_intercept_time"),
            moving_target.get("lead_distance"),
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
            debug.get("predicted_turret_yaw_rate"),
            debug.get("body_equivalent_turret_weight"),
            turret_qe.get("command"),
            turret_qe.get("weight"),
            debug.get("body_error"),
            debug.get("body_signed_effort"),
            debug.get("predicted_body_yaw_rate"),
            debug.get("body_coarse_turn_enabled"),
            debug.get("pitch_signed_effort"),
            debug.get("predicted_pitch_rate"),
            move_ad.get("command"),
            move_ad.get("weight"),
            debug.get("aim_aligned"),
            debug.get("predictive_aim_aligned"),
            debug.get("fire_alignment_ready"),
            debug.get("predictive_fire_lookahead_seconds"),
            debug.get("predicted_turret_error_at_fire"),
            debug.get("predicted_pitch_error_at_fire"),
            debug.get("spawn_arm_ready"),
            debug.get("aim_stable_ready"),
            action.get("fire", False),
        ])


# ============================================================
# 기본 유틸
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def moving_lead_boost_factor(moving_target_debug):
    if not isinstance(moving_target_debug, dict):
        return 0.0

    try:
        lead_distance = float(moving_target_debug.get("lead_distance", 0.0))
    except (TypeError, ValueError):
        return 0.0

    return clamp(
        lead_distance / MOVING_LEAD_BOOST_FULL_DISTANCE_M,
        0.0,
        1.0,
    )


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


def reset_enemy_motion_tracking():
    global enemy_velocity_x, enemy_velocity_z
    global enemy_accel_x, enemy_accel_z
    global enemy_motion_sample_count
    global enemy_motion_last_position, enemy_motion_last_time
    global enemy_motion_reversal_hold_until

    enemy_motion_samples.clear()
    enemy_velocity_x = 0.0
    enemy_velocity_z = 0.0
    enemy_accel_x = 0.0
    enemy_accel_z = 0.0
    enemy_motion_sample_count = 0
    enemy_motion_last_position = None
    enemy_motion_last_time = None
    enemy_motion_reversal_hold_until = 0.0


def update_enemy_motion_tracking(info, sample_time=None):
    global enemy_velocity_x, enemy_velocity_z
    global enemy_accel_x, enemy_accel_z
    global enemy_motion_sample_count
    global enemy_motion_last_position, enemy_motion_last_time
    global enemy_motion_reversal_hold_until

    enemy_pos = info.get("enemyPos") if isinstance(info, dict) else None
    if not isinstance(enemy_pos, dict):
        return

    try:
        current = {
            "x": float(enemy_pos["x"]),
            "y": float(enemy_pos.get("y", PLAYER_START_Y)),
            "z": float(enemy_pos["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return

    now = time.monotonic() if sample_time is None else float(sample_time)

    if enemy_motion_last_position is not None and enemy_motion_last_time is not None:
        dt = now - enemy_motion_last_time

        if 0.02 <= dt <= 1.0:
            raw_vx = (
                current["x"] - enemy_motion_last_position["x"]
            ) / dt
            raw_vz = (
                current["z"] - enemy_motion_last_position["z"]
            ) / dt
            raw_speed = math.hypot(raw_vx, raw_vz)

            if raw_speed <= ENEMY_MAX_VALID_SPEED_MPS:
                previous_velocity_x = enemy_velocity_x
                previous_velocity_z = enemy_velocity_z
                velocity_dot = (
                    raw_vx * enemy_velocity_x
                    + raw_vz * enemy_velocity_z
                )
                filtered_speed = math.hypot(
                    enemy_velocity_x,
                    enemy_velocity_z,
                )
                direction_reversed = (
                    enemy_motion_sample_count >= ENEMY_MOTION_MIN_SAMPLES
                    and raw_speed >= 0.2
                    and filtered_speed >= 0.2
                    and velocity_dot < 0.0
                )

                if direction_reversed:
                    enemy_velocity_x = 0.0
                    enemy_velocity_z = 0.0
                    enemy_accel_x = 0.0
                    enemy_accel_z = 0.0
                    enemy_motion_sample_count = 0
                    enemy_motion_samples.clear()
                    enemy_motion_reversal_hold_until = (
                        now + MOTION_REVERSAL_HOLD_SECONDS
                    )

                alpha = ENEMY_VELOCITY_EMA_ALPHA
                enemy_velocity_x = (
                    alpha * raw_vx
                    + (1.0 - alpha) * enemy_velocity_x
                )
                enemy_velocity_z = (
                    alpha * raw_vz
                    + (1.0 - alpha) * enemy_velocity_z
                )

                raw_ax = (raw_vx - previous_velocity_x) / dt
                raw_az = (raw_vz - previous_velocity_z) / dt
                raw_accel = math.hypot(raw_ax, raw_az)

                if raw_accel <= ENEMY_MAX_VALID_ACCEL_MPS2:
                    accel_alpha = ENEMY_ACCEL_EMA_ALPHA
                    enemy_accel_x = (
                        accel_alpha * raw_ax
                        + (1.0 - accel_alpha) * enemy_accel_x
                    )
                    enemy_accel_z = (
                        accel_alpha * raw_az
                        + (1.0 - accel_alpha) * enemy_accel_z
                    )
                else:
                    enemy_accel_x = 0.0
                    enemy_accel_z = 0.0

                enemy_motion_sample_count += 1
                enemy_motion_samples.append({
                    "time": now,
                    "x": current["x"],
                    "z": current["z"],
                    "vx": raw_vx,
                    "vz": raw_vz,
                })
            else:
                # A large jump is a respawn/teleport, not target velocity.
                enemy_velocity_x = 0.0
                enemy_velocity_z = 0.0
                enemy_accel_x = 0.0
                enemy_accel_z = 0.0
                enemy_motion_sample_count = 0
                enemy_motion_samples.clear()

    enemy_motion_last_position = current
    enemy_motion_last_time = now


def max_turret_yaw_weight_for_distance(distance):
    if distance < 40:
        return 0.12
    if distance < 90:
        return 0.22
    return 0.32


def estimate_pitch_control_time(
    current_pitch,
    desired_pitch,
    theta_for_gain,
    muzzle_speed,
    lead_boost=0.0,
):
    pitch = float(current_pitch)
    elapsed = 0.0
    step = PITCH_CONTROL_SIM_STEP_SECONDS

    dR_dtheta = abs(
        range_derivative_numeric(
            theta_for_gain,
            muzzle_speed=muzzle_speed,
        )
    )
    max_pitch_weight = (
        PITCH_MAX_WEIGHT_SENSITIVE
        if dR_dtheta > PITCH_SENSITIVE_RANGE_DERIVATIVE
        else PITCH_MAX_WEIGHT_FLAT
    )
    max_pitch_weight *= (
        1.0 + MOVING_PITCH_MAX_WEIGHT_BOOST * clamp(lead_boost, 0.0, 1.0)
    )

    while elapsed < MAX_CONTROL_PREDICTION_SECONDS:
        error = desired_pitch - pitch
        if abs(error) <= 0.2:
            break

        weight = clamp(
            abs(error) / PITCH_CONTROL_ERROR_SCALE_DEG * max_pitch_weight,
            PITCH_MIN_WEIGHT,
            max_pitch_weight,
        )
        pitch_rate = PITCH_SPEED_PER_WEIGHT * weight
        pitch += math.copysign(pitch_rate * step, error)
        elapsed += step

    return min(elapsed, MAX_CONTROL_PREDICTION_SECONDS)


def estimate_aim_control_time(
    info,
    predicted_enemy_pos,
    lead_distance=0.0,
    target_velocity_x=0.0,
    target_velocity_z=0.0,
):
    player_pos = info.get("playerPos", {})

    try:
        current_body_yaw = float(info.get("playerBodyX", 0.0))
        current_turret_yaw = float(info.get("playerTurretX", 0.0))
        current_pitch = float(info.get("playerTurretY", 0.0))
        target_yaw, distance = calc_target_angle_and_distance(
            player_pos,
            predicted_enemy_pos,
        )
        dx = float(predicted_enemy_pos["x"]) - float(player_pos["x"])
        dz = float(predicted_enemy_pos["z"]) - float(player_pos["z"])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    turret_error = normalize_angle(target_yaw - current_turret_yaw)
    body_error = normalize_angle(target_yaw - current_body_yaw)
    range_sq = max(dx * dx + dz * dz, 1e-6)
    target_angle_rate_deg_s = math.degrees(
        (dz * float(target_velocity_x) - dx * float(target_velocity_z))
        / range_sq
    )
    turret_weight = max_turret_yaw_weight_for_distance(distance)
    lead_boost = clamp(
        lead_distance / MOVING_LEAD_BOOST_FULL_DISTANCE_M,
        0.0,
        1.0,
    )
    turret_weight *= (
        1.0 + MOVING_YAW_MAX_WEIGHT_BOOST * lead_boost
    )
    body_weight = min(0.35, abs(BODY_YAW_KP * body_error))

    yaw_speed = (
        TURRET_YAW_SPEED_PER_WEIGHT * turret_weight
        + BODY_YAW_SPEED_PER_WEIGHT * body_weight
    )
    turret_error_sign = 1.0 if turret_error >= 0.0 else -1.0
    yaw_closing_rate = max(
        MOVING_MIN_YAW_CLOSING_RATE_DEG_S,
        yaw_speed - turret_error_sign * target_angle_rate_deg_s,
    )
    yaw_time = (
        abs(turret_error) / yaw_closing_rate
        if yaw_closing_rate > 1e-6
        else MAX_CONTROL_PREDICTION_SECONDS
    )

    desired_pitch, ballistic = calc_desired_pitch_ballistic(
        player_pos,
        predicted_enemy_pos,
    )
    pitch_time = estimate_pitch_control_time(
        current_pitch=current_pitch,
        desired_pitch=desired_pitch,
        theta_for_gain=ballistic.get(
            "theta_with_bias_deg",
            abs(desired_pitch),
        ),
        muzzle_speed=float(
            ballistic.get("muzzle_speed", MUZZLE_SPEED)
        ),
        lead_boost=lead_boost,
    )

    control_time = clamp(
        max(yaw_time, pitch_time) * CONTROL_TIME_SAFETY_FACTOR,
        0.0,
        MAX_CONTROL_PREDICTION_SECONDS,
    )
    return (
        control_time,
        yaw_time,
        pitch_time,
        target_angle_rate_deg_s,
        yaw_closing_rate,
    )


def predict_enemy_intercept(info, observed_enemy_pos):
    player_pos = info.get("playerPos", {})

    try:
        px = float(player_pos["x"])
        py = float(player_pos["y"])
        pz = float(player_pos["z"])
        ex = float(observed_enemy_pos["x"])
        ey = float(observed_enemy_pos["y"])
        ez = float(observed_enemy_pos["z"])
    except (KeyError, TypeError, ValueError):
        return observed_enemy_pos, {
            "motion_ready": False,
            "reason": "invalid moving-target position",
        }

    reversal_hold_remaining = max(
        0.0,
        enemy_motion_reversal_hold_until - time.monotonic(),
    )
    motion_ready = (
        enemy_motion_sample_count >= ENEMY_MOTION_MIN_SAMPLES
        and reversal_hold_remaining <= 0.0
    )
    vx = enemy_velocity_x if motion_ready else 0.0
    vz = enemy_velocity_z if motion_ready else 0.0
    ax = enemy_accel_x if motion_ready else 0.0
    az = enemy_accel_z if motion_ready else 0.0
    predicted_x = ex
    predicted_z = ez
    flight_time = 0.0
    control_time = 0.0
    yaw_control_time = 0.0
    pitch_control_time = 0.0
    target_angle_rate_deg_s = 0.0
    yaw_closing_rate_deg_s = 0.0
    total_intercept_time = 0.0
    estimated_lead_distance = 0.0

    # Paper-style iterative intercept solver:
    # future target position = p + v*t + 0.5*a*t^2.
    # The future position changes range, flight time, and control delay, so
    # iterate until the implied intercept time settles.
    for _ in range(INTERCEPT_SOLVER_ITERATIONS):
        predicted_position = {
            "x": predicted_x,
            "y": ey,
            "z": predicted_z,
        }
        (
            control_time,
            yaw_control_time,
            pitch_control_time,
            target_angle_rate_deg_s,
            yaw_closing_rate_deg_s,
        ) = estimate_aim_control_time(
            info,
            predicted_position,
            lead_distance=estimated_lead_distance,
            target_velocity_x=vx,
            target_velocity_z=vz,
        )

        horizontal_range = math.hypot(predicted_x - px, predicted_z - pz)
        muzzle_speed = effective_muzzle_speed(horizontal_range)
        theta_deg, mode, _ = ballistic_theta_low_angle(
            horizontal_range,
            ey - py,
            muzzle_speed=muzzle_speed,
        )

        cos_theta = math.cos(math.radians(theta_deg))
        if mode != "low_angle_solution" or abs(cos_theta) < 1e-6:
            break

        raw_flight_time = horizontal_range / (muzzle_speed * cos_theta)
        flight_time = clamp(
            raw_flight_time * FLIGHT_TIME_CORRECTION_FACTOR,
            0.0,
            MAX_FLIGHT_TIME_SECONDS,
        )
        total_intercept_time = (
            control_time
            + MOVING_AIM_STABLE_SECONDS
            + FIRE_SYSTEM_DELAY_SECONDS
            + flight_time
        )
        lead_time = total_intercept_time
        velocity_lead_dx = vx * lead_time
        velocity_lead_dz = vz * lead_time
        accel_lead_dx = 0.5 * ax * lead_time * lead_time
        accel_lead_dz = 0.5 * az * lead_time * lead_time
        accel_lead_distance = math.hypot(accel_lead_dx, accel_lead_dz)

        if accel_lead_distance > MAX_ACCEL_LEAD_DISTANCE_M:
            scale = MAX_ACCEL_LEAD_DISTANCE_M / accel_lead_distance
            accel_lead_dx *= scale
            accel_lead_dz *= scale

        lead_dx = velocity_lead_dx + accel_lead_dx
        lead_dz = velocity_lead_dz + accel_lead_dz
        lead_distance = math.hypot(lead_dx, lead_dz)
        estimated_lead_distance = lead_distance

        if lead_distance > MAX_LEAD_DISTANCE_M:
            scale = MAX_LEAD_DISTANCE_M / lead_distance
            lead_dx *= scale
            lead_dz *= scale

        predicted_x = ex + lead_dx
        predicted_z = ez + lead_dz

    speed = math.hypot(vx, vz)
    accel = math.hypot(ax, az)
    lead_distance = math.hypot(predicted_x - ex, predicted_z - ez)

    return {
        "x": predicted_x,
        "y": ey,
        "z": predicted_z,
    }, {
        "motion_ready": motion_ready,
        "motion_sample_count": enemy_motion_sample_count,
        "reversal_hold_remaining": round(
            reversal_hold_remaining,
            4,
        ),
        "enemy_velocity_x": round(vx, 4),
        "enemy_velocity_z": round(vz, 4),
        "enemy_speed": round(speed, 4),
        "enemy_accel_x": round(ax, 4),
        "enemy_accel_z": round(az, 4),
        "enemy_accel": round(accel, 4),
        "predicted_control_time": round(control_time, 4),
        "predicted_yaw_control_time": round(yaw_control_time, 4),
        "predicted_pitch_control_time": round(pitch_control_time, 4),
        "target_angle_rate_deg_s": round(target_angle_rate_deg_s, 4),
        "yaw_closing_rate_deg_s": round(yaw_closing_rate_deg_s, 4),
        "predicted_flight_time": round(flight_time, 4),
        "flight_time_correction_factor": (
            FLIGHT_TIME_CORRECTION_FACTOR
        ),
        "predicted_total_intercept_time": round(
            total_intercept_time,
            4,
        ),
        "lead_distance": round(lead_distance, 4),
        "intercept_solver": "iterative_p_vt_half_at2",
        "intercept_solver_iterations": INTERCEPT_SOLVER_ITERATIONS,
        "max_accel_lead_distance": MAX_ACCEL_LEAD_DISTANCE_M,
        "observed_enemy_x": ex,
        "observed_enemy_y": ey,
        "observed_enemy_z": ez,
        "predicted_enemy_x": round(predicted_x, 4),
        "predicted_enemy_y": ey,
        "predicted_enemy_z": round(predicted_z, 4),
    }


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
        aimed_info = dict(info)
        observed_enemy_pos = info.get("enemyPos")
        player_pos = info.get("playerPos")

        if isinstance(observed_enemy_pos, dict) and isinstance(player_pos, dict):
            predicted_pos, motion_debug = predict_enemy_intercept(
                info,
                observed_enemy_pos,
            )
            aimed_info["enemyObservedPos"] = dict(observed_enemy_pos)
            aimed_info["enemyPos"] = predicted_pos
            aimed_info["movingTarget"] = motion_debug
            aimed_info["targetType"] = "moving_enemy"
            aimed_info["targetName"] = "EnemyTank"
            aimed_info["targetIndex"] = 0
            aimed_info["targetDistance"] = math.hypot(
                float(observed_enemy_pos["x"]) - float(player_pos["x"]),
                float(observed_enemy_pos["z"]) - float(player_pos["z"]),
            )

        return aimed_info

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

            "observed_enemy_x_fire",
            "observed_enemy_y_fire",
            "observed_enemy_z_fire",
            "enemy_velocity_x_fire",
            "enemy_velocity_z_fire",
            "enemy_speed_fire",
            "enemy_accel_x_fire",
            "enemy_accel_z_fire",
            "enemy_accel_fire",
            "intercept_solver_fire",
            "reversal_hold_remaining_fire",
            "predicted_control_time_fire",
            "predicted_yaw_control_time_fire",
            "predicted_pitch_control_time_fire",
            "target_angle_rate_deg_s_fire",
            "yaw_closing_rate_deg_s_fire",
            "predicted_flight_time_fire",
            "flight_time_correction_factor_fire",
            "predicted_total_intercept_time_fire",
            "lead_distance_fire",

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
            "predictive_aim_aligned_fire",
            "fire_alignment_ready_fire",
            "predictive_fire_lookahead_seconds_fire",
            "predicted_turret_yaw_rate_fire",
            "predicted_turret_error_at_fire",
            "predicted_pitch_rate_fire",
            "predicted_pitch_error_at_fire",

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
    moving_target = debug.get("moving_target", {})
    observed_enemy_pos = info.get("enemyObservedPos", enemy_pos)

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

        "observed_enemy_pos": {
            "x": safe_get_pos(observed_enemy_pos, "x"),
            "y": safe_get_pos(observed_enemy_pos, "y"),
            "z": safe_get_pos(observed_enemy_pos, "z"),
        },
        "enemy_velocity_x": moving_target.get("enemy_velocity_x"),
        "enemy_velocity_z": moving_target.get("enemy_velocity_z"),
        "enemy_speed": moving_target.get("enemy_speed"),
        "enemy_accel_x": moving_target.get("enemy_accel_x"),
        "enemy_accel_z": moving_target.get("enemy_accel_z"),
        "enemy_accel": moving_target.get("enemy_accel"),
        "intercept_solver": moving_target.get("intercept_solver"),
        "reversal_hold_remaining": moving_target.get(
            "reversal_hold_remaining"
        ),
        "predicted_control_time": moving_target.get(
            "predicted_control_time"
        ),
        "predicted_yaw_control_time": moving_target.get(
            "predicted_yaw_control_time"
        ),
        "predicted_pitch_control_time": moving_target.get(
            "predicted_pitch_control_time"
        ),
        "target_angle_rate_deg_s": moving_target.get(
            "target_angle_rate_deg_s"
        ),
        "yaw_closing_rate_deg_s": moving_target.get(
            "yaw_closing_rate_deg_s"
        ),
        "predicted_flight_time": moving_target.get("predicted_flight_time"),
        "flight_time_correction_factor": moving_target.get(
            "flight_time_correction_factor"
        ),
        "predicted_total_intercept_time": moving_target.get(
            "predicted_total_intercept_time"
        ),
        "lead_distance": moving_target.get("lead_distance"),

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
        "predictive_aim_aligned": debug.get("predictive_aim_aligned"),
        "fire_alignment_ready": debug.get("fire_alignment_ready"),
        "predictive_fire_lookahead_seconds": debug.get(
            "predictive_fire_lookahead_seconds"
        ),
        "predicted_turret_yaw_rate": debug.get("predicted_turret_yaw_rate"),
        "predicted_turret_error_at_fire": debug.get(
            "predicted_turret_error_at_fire"
        ),
        "predicted_pitch_rate": debug.get("predicted_pitch_rate"),
        "predicted_pitch_error_at_fire": debug.get(
            "predicted_pitch_error_at_fire"
        ),

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

            shot["observed_enemy_pos"].get("x"),
            shot["observed_enemy_pos"].get("y"),
            shot["observed_enemy_pos"].get("z"),
            shot.get("enemy_velocity_x"),
            shot.get("enemy_velocity_z"),
            shot.get("enemy_speed"),
            shot.get("enemy_accel_x"),
            shot.get("enemy_accel_z"),
            shot.get("enemy_accel"),
            shot.get("intercept_solver"),
            shot.get("reversal_hold_remaining"),
            shot.get("predicted_control_time"),
            shot.get("predicted_yaw_control_time"),
            shot.get("predicted_pitch_control_time"),
            shot.get("target_angle_rate_deg_s"),
            shot.get("yaw_closing_rate_deg_s"),
            shot.get("predicted_flight_time"),
            shot.get("flight_time_correction_factor"),
            shot.get("predicted_total_intercept_time"),
            shot.get("lead_distance"),

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
            shot.get("predictive_aim_aligned"),
            shot.get("fire_alignment_ready"),
            shot.get("predictive_fire_lookahead_seconds"),
            shot.get("predicted_turret_yaw_rate"),
            shot.get("predicted_turret_error_at_fire"),
            shot.get("predicted_pitch_rate"),
            shot.get("predicted_pitch_error_at_fire"),

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

            "", "", "", "", "", "", "", "", "", "", "", "", "",

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
    global moving_target_shots_fired

    action = make_default_action()

    if (
        not OBSTACLE_TARGET_MODE
        and moving_target_shots_fired >= MAX_MOVING_TARGET_SHOTS
    ):
        action["debug"] = {
            "reason": "15 moving-target shots complete",
            "shotsFired": moving_target_shots_fired,
            "maxShots": MAX_MOVING_TARGET_SHOTS,
        }
        return action

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

    ignore_enemy_health_for_test = (
        IGNORE_MOVING_TARGET_HEALTH_FOR_SHOT_LIMIT_TEST
        and info.get("targetType") == "moving_enemy"
    )

    if enemy_health <= 0 and not ignore_enemy_health_for_test:
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
    moving_target_debug = info.get("movingTarget", {})

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
    predicted_turret_yaw_rate = 0.0

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

    lead_boost = moving_lead_boost_factor(moving_target_debug)
    yaw_max_weight_multiplier = 1.0

    if info.get("targetType") == "moving_enemy":
        yaw_max_weight_multiplier += (
            MOVING_YAW_MAX_WEIGHT_BOOST * lead_boost
        )
        max_turret_weight *= yaw_max_weight_multiplier

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
    predicted_turret_yaw_rate = (
        TURRET_YAW_SPEED_PER_WEIGHT * turret_pd_effort
        + predicted_body_yaw_rate
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
    pitch_signed_effort = 0.0
    predicted_pitch_rate = 0.0

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

        pitch_max_weight_multiplier = 1.0

        if info.get("targetType") == "moving_enemy":
            pitch_max_weight_multiplier += (
                MOVING_PITCH_MAX_WEIGHT_BOOST * lead_boost
            )
            max_pitch_weight *= pitch_max_weight_multiplier

        pitch_weight = clamp(
            abs_pitch_error / PITCH_CONTROL_ERROR_SCALE_DEG * max_pitch_weight,
            PITCH_MIN_WEIGHT,
            max_pitch_weight
        )
        pitch_signed_effort = math.copysign(pitch_weight, pitch_error)
        predicted_pitch_rate = PITCH_SPEED_PER_WEIGHT * pitch_signed_effort

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

    is_moving_target = info.get("targetType") == "moving_enemy"
    turret_fire_tolerance = (
        MOVING_TURRET_FIRE_TOLERANCE_DEG
        if is_moving_target
        else 1.5
    )
    body_fire_tolerance = (
        MOVING_BODY_FIRE_TOLERANCE_DEG
        if is_moving_target
        else BODY_FIRE_TOLERANCE_DEG
    )
    effective_pitch_tol = (
        pitch_tol * MOVING_PITCH_TOLERANCE_MULTIPLIER
        if is_moving_target
        else pitch_tol
    )
    required_aim_stable_seconds = (
        MOVING_AIM_STABLE_SECONDS
        if is_moving_target
        else AIM_STABLE_SECONDS
    )

    aim_aligned = (
        20 < distance < 200
        and abs_body_error < body_fire_tolerance
        and abs_turret_error < turret_fire_tolerance
        and abs_pitch_error < effective_pitch_tol
    )
    predictive_lookahead = (
        MOVING_PREDICTIVE_FIRE_LOOKAHEAD_SECONDS
        if is_moving_target
        else 0.0
    )
    predicted_turret_yaw_at_fire = (
        player_turret_yaw
        + predicted_turret_yaw_rate * predictive_lookahead
    )
    predicted_pitch_at_fire = (
        player_turret_pitch
        + predicted_pitch_rate * predictive_lookahead
    )
    predicted_turret_error_at_fire = normalize_angle(
        target_world_angle - predicted_turret_yaw_at_fire
    )
    predicted_pitch_error_at_fire = (
        desired_pitch - predicted_pitch_at_fire
    )
    predictive_aim_aligned = (
        is_moving_target
        and 20 < distance < 200
        and abs_body_error < body_fire_tolerance
        and abs_turret_error < MOVING_PREFIRE_TURRET_WINDOW_DEG
        and abs(predicted_turret_error_at_fire) < turret_fire_tolerance
        and abs(predicted_pitch_error_at_fire)
        < pitch_tol * MOVING_PREFIRE_PITCH_TOLERANCE_MULTIPLIER
    )
    fire_alignment_ready = aim_aligned or predictive_aim_aligned

    if fire_alignment_ready:
        if aim_ready_since is None:
            aim_ready_since = now
    else:
        aim_ready_since = None

    spawn_arm_ready = (
        now - spawn_initialized_at >= SPAWN_FIRE_ARM_DELAY_SECONDS
    )

    aim_stable_ready = (
        aim_ready_since is not None
        and now - aim_ready_since >= (
            MOVING_PREDICTIVE_FIRE_STABLE_SECONDS
            if predictive_aim_aligned
            else required_aim_stable_seconds
        )
    )

    can_fire = (
        fire_alignment_ready
        and moving_target_shots_fired < MAX_MOVING_TARGET_SHOTS
        and (
            OBSTACLE_TARGET_MODE
            or moving_target_debug.get("motion_ready", False)
        )
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
        moving_target_shots_fired += 1

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
        "predicted_turret_yaw_rate": round(
            predicted_turret_yaw_rate,
            3,
        ),
        "lead_boost": round(lead_boost, 4),
        "yaw_max_weight_multiplier": round(
            yaw_max_weight_multiplier,
            4,
        ),
        "max_turret_weight": round(max_turret_weight, 4),
        "body_signed_effort": round(body_signed_effort, 4),
        "predicted_body_yaw_rate": round(predicted_body_yaw_rate, 3),
        "body_equivalent_turret_weight": round(
            body_equivalent_turret_weight,
            4,
        ),
        "body_coarse_turn_enabled": body_coarse_turn_enabled,

        "desired_pitch": round(desired_pitch, 2),
        "pitch_error": round(pitch_error, 2),
        "pitch_signed_effort": round(pitch_signed_effort, 4),
        "predicted_pitch_rate": round(predicted_pitch_rate, 3),
        "pitch_fire_tolerance": round(pitch_tol, 4),
        "effective_pitch_fire_tolerance": round(effective_pitch_tol, 4),
        "turret_fire_tolerance": round(turret_fire_tolerance, 4),
        "body_fire_tolerance": round(body_fire_tolerance, 4),
        "required_aim_stable_seconds": round(
            required_aim_stable_seconds,
            4,
        ),

        "aim_aligned": aim_aligned,
        "predictive_aim_aligned": predictive_aim_aligned,
        "fire_alignment_ready": fire_alignment_ready,
        "predictive_fire_lookahead_seconds": round(
            predictive_lookahead,
            4,
        ),
        "predicted_turret_error_at_fire": round(
            predicted_turret_error_at_fire,
            3,
        ),
        "predicted_pitch_error_at_fire": round(
            predicted_pitch_error_at_fire,
            3,
        ),
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
        "moving_target_shots_fired": moving_target_shots_fired,
        "moving_target_max_shots": MAX_MOVING_TARGET_SHOTS,
        "moving_target": moving_target_debug,

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
    global moving_target_shots_fired

    # reset 시 stale shot만 정리.
    # 리셋하면 동일 장애물 20개 검증을 첫 표적부터 다시 시작한다.
    pending_shots.clear()
    reset_enemy_motion_tracking()
    moving_target_shots_fired = 0
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
        f"[INIT] moving enemy target "
        f"spawn_distance={enemy_distance} "
        f"spawn_bearing={enemy_bearing}"
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
    update_enemy_motion_tracking(data)

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": "start"
    })


@app.route("/moving_target_status", methods=["GET"])
def moving_target_status():
    reversal_hold_remaining = max(
        0.0,
        enemy_motion_reversal_hold_until - time.monotonic(),
    )
    return jsonify({
        "mode": "moving_enemy_lead",
        "motionReady": (
            enemy_motion_sample_count >= ENEMY_MOTION_MIN_SAMPLES
            and reversal_hold_remaining <= 0.0
        ),
        "sampleCount": enemy_motion_sample_count,
        "reversalHoldRemaining": round(
            reversal_hold_remaining,
            4,
        ),
        "shotsFired": moving_target_shots_fired,
        "maxShots": MAX_MOVING_TARGET_SHOTS,
        "shotLimitReached": (
            moving_target_shots_fired >= MAX_MOVING_TARGET_SHOTS
        ),
        "velocity": {
            "x": round(enemy_velocity_x, 4),
            "z": round(enemy_velocity_z, 4),
            "speed": round(
                math.hypot(enemy_velocity_x, enemy_velocity_z),
                4,
            ),
        },
        "acceleration": {
            "x": round(enemy_accel_x, 4),
            "z": round(enemy_accel_z, 4),
            "accel": round(
                math.hypot(enemy_accel_x, enemy_accel_z),
                4,
            ),
        },
        "latestPrediction": (
            info_with_active_target(latest_info).get("movingTarget", {})
            if latest_info
            else {}
        ),
        "shotLog": SHOT_LOG_PATH,
        "controlLog": CONTROL_LOG_PATH,
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
