from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime
from math import atan, atan2, cos, degrees, floor, hypot, isfinite, radians, tan
from pathlib import Path
from statistics import median
from threading import Lock
from time import monotonic
from typing import Any
from io import BytesIO
import base64
import json
import re

from flask import Flask, jsonify, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# =============================================================================
# 0. TUNING PARAMETERS
# =============================================================================
MAX_DISTANCE_M = 120.0

# ----------------------------------------------------------------------------
# Object candidates. These are NOT final semantic detections.
# TH = thin candidate (person/tree/pole-like), BK = bulky candidate
# (tank/rock/wall-like), UK = unknown candidate. YOLO fusion comes later.
# ----------------------------------------------------------------------------
OBJECT_VERTICAL_MIN_DEG = -10.0
OBJECT_VERTICAL_MAX_DEG = -0.1
OBJECT_DETECTION_MAX_DISTANCE_M = 120.0
OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M = 0.35
OBJECT_AZIMUTH_BIN_WIDTH_DEG = 1.0
OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG = 2.1
OBJECT_CLUSTER_MAX_DISTANCE_GAP_M = 3.0
THIN_MAX_WIDTH_M = 2.20
BULKY_MIN_WIDTH_M = 2.50
THIN_MAX_DISTANCE_M = 70.0
BULKY_CONFIDENT_DISTANCE_M = 100.0

# Terrain-connected clusters are hidden from stable object tracks. They remain
# available under rawObjects for debugging.
TERRAIN_OBJECT_OVERLAP_ANGLE_DEG = 7.0
TERRAIN_OBJECT_OVERLAP_RANGE_M = 5.0
# Suppress only low-profile terrain-connected clusters. Tall people or tanks
# near a slope must remain visible as candidates.
TERRAIN_CONNECTED_LOW_HEIGHT_M = 0.35
TERRAIN_CONNECTED_HAZARD_LOW_HEIGHT_M = 0.95
LOCAL_GROUND_LOOKUP_MAX_RANGE_GAP_M = 4.0

# ----------------------------------------------------------------------------
# Local terrain model and hazard detection
# ----------------------------------------------------------------------------
TERRAIN_VERTICAL_MIN_DEG = 0.5
TERRAIN_VERTICAL_MAX_DEG = 22.5
TERRAIN_FRONT_LIMIT_DEG = 60.0
TERRAIN_ANALYSIS_MAX_DISTANCE_M = 35.0
TERRAIN_SECTOR_WIDTH_DEG = 10.0
TERRAIN_RANGE_BIN_M = 1.5
LOCAL_GROUND_GRID_ANGLE_WIDTH_DEG = 5.0
LOCAL_GROUND_GRID_RANGE_BIN_M = 2.0

PASSABLE_UP_SLOPE_MAX_DEG = 12.0
CAUTION_UP_SLOPE_MAX_DEG = 20.0
PASSABLE_DOWN_SLOPE_MAX_DEG = 10.0
CAUTION_DOWN_SLOPE_MAX_DEG = 18.0
CAUTION_UP_STEP_M = 0.30
BLOCKED_UP_STEP_M = 0.60
CAUTION_DROP_M = 0.30
BLOCKED_DROP_M = 0.65

WALL_RANGE_BIN_M = 1.2
WALL_MIN_HEIGHT_SPAN_M = 1.0
WALL_MIN_UNIQUE_CHANNELS = 3

# Expected-ground model: ignore shallow rays that would naturally hit flat
# ground beyond the analysis window. This reduces false cliff alarms.
LOCAL_GROUND_STEEP_MIN_DEG = 8.0
LOCAL_GROUND_NEAR_MAX_DISTANCE_M = 12.0
LOCAL_GROUND_MIN_POINT_COUNT = 6
EXPECTED_GROUND_MIN_DISTANCE_M = 1.5
EXPECTED_GROUND_MAX_DISTANCE_M = 30.0
EXPECTED_GROUND_MIN_RAY_COUNT = 8
EXPECTED_GROUND_RANGE_TOLERANCE_M = 2.5
EXPECTED_GROUND_DELAY_RATIO = 1.35
CLIFF_CAUTION_EXPECTED_MISS_RATIO = 0.35
CLIFF_BLOCKED_EXPECTED_MISS_RATIO = 0.60
CLIFF_CAUTION_DELAYED_RETURN_RATIO = 0.35
CLIFF_BLOCKED_DELAYED_RETURN_RATIO = 0.60
CLIFF_MIN_PROFILE_GAP_M = 4.5
CLIFF_NEAR_GROUND_EVIDENCE_M = 12.0

# ----------------------------------------------------------------------------
# Temporal stabilization and tracking
# ----------------------------------------------------------------------------
TERRAIN_HISTORY_SIZE = 5
TERRAIN_BLOCKED_CONFIRM_FRAMES = 2
TERRAIN_CAUTION_CONFIRM_FRAMES = 2
TERRAIN_PASSABLE_CONFIRM_FRAMES = 3
DEAD_END_FRONT_LIMIT_DEG = 55.0
DEAD_END_BLOCKED_RATIO = 0.70
DEAD_END_MAX_PASSABLE_SECTORS = 1

TRACK_HISTORY_SIZE = 5
# Show provisional candidates immediately. Confirm stable tracks after two
# matching frames for responsive monitoring.
TRACK_CONFIRM_HITS = 2
TRACK_MAX_MISSES = 4
TRACK_ASSOCIATION_DISTANCE_M = 4.0
PROVISIONAL_OBJECT_LIMIT = 30
TRACK_PROCESS_NOISE = 0.8
TRACK_MEASUREMENT_NOISE = 1.5
IMPACT_HISTORY_SIZE = 20

# ----------------------------------------------------------------------------
# LiDAR -> YOLO fusion scheduling priority
# ----------------------------------------------------------------------------
# Keep every 360-degree candidate, but schedule wide BK candidates first.
# A 50 m threshold represents the requested close-object range. Change this
# to 40.0 if the team decides to use the stricter close-range threshold.
PRIORITY_NEAR_MAX_DISTANCE_M = 45.0
PRIORITY_MAX_QUEUE_SIZE = 20
PRIORITY_DUPLICATE_ANGLE_TOLERANCE_DEG = 3.0
PRIORITY_DUPLICATE_DISTANCE_TOLERANCE_M = 4.0

# ----------------------------------------------------------------------------
# Automatic body alignment toward the most dangerous close target
# ----------------------------------------------------------------------------
# Current LiDAR-only fallback:
#   BK? means bulky tank-or-rock-like candidate, not a final tank label.
# After YOLO fusion, semantic tank labels automatically outrank fallback BK?.
AUTO_BODY_ALIGN_ENABLED = True
BODY_ALIGN_USE_LIDAR_BULKY_FALLBACK = True
BODY_ALIGN_CONFIRMED_ONLY = True
BODY_ALIGN_TARGET_MAX_DISTANCE_M = 45.0
BODY_ALIGN_LOCK_RELEASE_DISTANCE_M = 52.0
BODY_ALIGN_DEADBAND_DEG = 0.4
BODY_ALIGN_SLOW_ZONE_DEG = 18.0
BODY_ALIGN_MEDIUM_ZONE_DEG = 45.0
BODY_ALIGN_FAST_ZONE_DEG = 90.0
BODY_ALIGN_WEIGHT_SLOW = 0.08
BODY_ALIGN_WEIGHT_MEDIUM = 0.14
BODY_ALIGN_WEIGHT_FAST = 0.25
BODY_ALIGN_WEIGHT_MAX = 0.35
BODY_ALIGN_USE_PRIMARY_FOR_RECOGNITION = True

# ----------------------------------------------------------------------------
# UI and controller
# ----------------------------------------------------------------------------
CONTOUR_ANGLE_BIN_DEG = 3.0
CONTOUR_MAX_DISTANCE_M = 120.0
FRONT_CLEARANCE_HALF_WIDTH_DEG = 15.0

# Front angular point-cloud view. The front canvas shows the LiDAR as if the
# driver were looking forward. X = body-relative azimuth, Y = vertical angle.
FRONT_VIEW_HORIZONTAL_LIMIT_DEG = 60.0
FRONT_VIEW_VERTICAL_MIN_DEG = -22.5
FRONT_VIEW_VERTICAL_MAX_DEG = 22.5
FRONT_VIEW_MAX_DISTANCE_M = 120.0

# Centerline vertical profile. Select the horizontal beam closest to body-forward
# 0 degrees and visualize all available vertical channels as a side cross-section.
FRONT_PROFILE_TARGET_ANGLE_DEG = 0.0

# ----------------------------------------------------------------------------
# YOLO visual target recognition and aim assist
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "v11_best_model.pt"
VISION_CONFIDENCE_MIN = 0.25
YOLO_CLASS_TO_LIDAR_GEOMETRY = {
    "ally": "thin",
    "enemy": "thin",
    "rock": "bulky",
    "rock_l": "bulky",
    "tank_ally_back": "bulky",
    "tank_ally_front": "bulky",
    "tank_ally_side": "bulky",
    "tank_enemy_back": "bulky",
    "tank_enemy_front": "bulky",
    "tank_enemy_side": "bulky",
    "tent": "bulky",
    "car": "bulky",
}
VISION_TARGET_CLASSES = {
    "enemy",
    "tank_enemy_back",
    "tank_enemy_front",
    "tank_enemy_side",
    "car",
}
VISION_TARGET_HOLD_SECONDS = 1.2
VISION_AIM_DEADBAND_X = 0.015
VISION_AIM_DEADBAND_Y = 0.025
VISION_AIM_ZERO_STEP_Y = 0.01
VISION_TURRET_WEIGHT_MIN = 0.06
VISION_TURRET_WEIGHT_MAX = 0.18
VISION_BODY_WEIGHT_MIN = 0.06
VISION_BODY_WEIGHT_MAX = 0.18
LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG = 10.0
FIRE_APPROVAL_SECONDS = 1.5

# LiDAR-only aiming mode. YOLO detection/mapping is ignored for aiming and firing.
# Use this while YOLO class recognition is unstable.
USE_YOLO_FOR_AIM = False
USE_YOLO_FIRE_GUARD = False
# LiDAR yaw control: turret starts correcting above this deadband.
LIDAR_TURRET_YAW_DEADBAND_DEG = 0.15
LIDAR_TURRET_YAW_WEIGHT_MIN = 0.05
LIDAR_TURRET_YAW_WEIGHT_MAX = 0.14

# LiDAR-only fire yaw guard: farther targets require tighter angular error.
# This keeps the allowed lateral miss distance roughly bounded.
LIDAR_FIRE_LATERAL_TOLERANCE_M = 0.30
LIDAR_FIRE_YAW_DEADBAND_MIN_DEG = 0.15
LIDAR_FIRE_YAW_DEADBAND_MAX_DEG = 0.45
# Kept for compatibility/debug display fallback; dynamic deadband is used for firing.
LIDAR_FIRE_YAW_DEADBAND_DEG = 0.30

AUTO_BALLISTIC_PITCH_ENABLED = True
# Distance-based pitch offset: far targets get more elevation than near targets.
BALLISTIC_PITCH_OFFSET_NEAR_DEG = 0.20
BALLISTIC_PITCH_OFFSET_FAR_DEG = 0.70
BALLISTIC_PITCH_OFFSET_FAR_START_M = 50.0
BALLISTIC_PITCH_OFFSET_FAR_END_M = 85.0
BALLISTIC_PITCH_DEADBAND_DEG = 0.05
BALLISTIC_PITCH_WEIGHT_MIN = 0.10
BALLISTIC_PITCH_WEIGHT_MAX = 0.25
BALLISTIC_DISTANCE_PITCH_TABLE = [
    (20.8, -5.00),
    (22.0, -4.72),
    (24.2, -3.86),
    (31.0, -2.00),
    (33.2, -1.54),
    (36.6, -0.89),
    (41.1, -0.20),
    (45.5, 0.55),
    (50.0, 1.13),
    (55.6, 1.82),
    (60.1, 2.51),
    (64.5, 2.96),
    (70.1, 3.60),
    (75.6, 4.29),
    (83.3, 5.17),
    (89.9, 5.82),
    (97.6, 6.60),
    (100.8, 7.00),
    (104.1, 7.36),
    (107.3, 7.65),
    (108.4, 7.83),
    (111.7, 8.08),
    (115.9, 8.55),
    (120.2, 8.98),
    (129.8, 9.76),
]
FRONT_PROFILE_MAX_SELECT_ANGLE_ERROR_DEG = 2.0
FRONT_PROFILE_MAX_DISTANCE_M = 120.0
FRONT_PROFILE_FIT_MAX_DISTANCE_M = 35.0
FRONT_PROFILE_MIN_FIT_POINTS = 3

PRINT_INTERVAL_SECONDS = 0.5
AUTO_DRIVE_ENABLED = False

state_lock = Lock()
last_print_time = 0.0
yolo_model = None
terrain_history: deque[dict[float, dict[str, Any]]] = deque(maxlen=TERRAIN_HISTORY_SIZE)
latest_raw_info: dict[str, Any] = {}
latest_state: dict[str, Any] = {
    "simulationTime": None,
    "terrainSectors": [],
    "terrainDecision": {},
    "contourPoints": [],
    "frontVerticalProfile": {},
    "rawObjects": [],
    "trackedObjects": [],
    "fusionPriorityQueue": [],
    "primaryFusionTarget": None,
    "bodyAlignment": {},
    "visionTarget": None,
    "visionDetections": [],
    "lidarVisionFusion": None,
    "impactMarkers": [],
}

body_alignment_state: dict[str, Any] = {
    "enabled": AUTO_BODY_ALIGN_ENABLED,
    "lockedTrackId": None,
    "target": None,
    "moveAD": {"command": "", "weight": 0.0},
    "aligned": False,
    "reason": "waiting_for_target",
}

action_debug_state: dict[str, Any] = {
    "getActionRequestCount": 0,
    "lastRequestBody": {},
    "lastResponse": {},
    "lastRequestedAt": None,
}

fire_control_state: dict[str, Any] = {
    "approvedUntil": 0.0,
    "approvedAt": None,
    "lastFiredAt": None,
    "fireCount": 0,
}
aim_zero_state: dict[str, Any] = {
    "offsetY": 0.0,
    "updatedAt": None,
}

vision_state: dict[str, Any] = {
    "target": None,
    "detections": [],
    "lastDetectedAt": 0.0,
    "modelPath": str(MODEL_PATH),
    "modelLoaded": False,
    "lidarFusion": None,
}
recognized_lidar_objects: dict[int, dict[str, Any]] = {}
impact_history: deque[dict[str, Any]] = deque(maxlen=IMPACT_HISTORY_SIZE)
next_impact_id = 1


# =============================================================================
# 1. BASIC HELPERS
# =============================================================================
def normalize_signed_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_distance_deg(a: float, b: float) -> float:
    return abs(normalize_signed_angle(a - b))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(str(MODEL_PATH))
        vision_state["modelLoaded"] = True
    return yolo_model


def aim_weight(error_ratio: float, deadband: float, low: float, high: float) -> float:
    magnitude = abs(error_ratio)
    if magnitude <= deadband:
        return 0.0
    scaled = clamp((magnitude - deadband) / max(0.001, 0.5 - deadband), 0.0, 1.0)
    return round(low + (high - low) * scaled, 3)


def interpolate_ballistic_pitch(distance_m: float | None) -> float | None:
    if distance_m is None:
        return None
    table = BALLISTIC_DISTANCE_PITCH_TABLE
    distance = float(distance_m)
    if distance <= table[0][0]:
        return table[0][1]
    if distance >= table[-1][0]:
        return table[-1][1]
    for (left_distance, left_pitch), (right_distance, right_pitch) in zip(table, table[1:]):
        if left_distance <= distance <= right_distance:
            ratio = (distance - left_distance) / max(0.001, right_distance - left_distance)
            return left_pitch + (right_pitch - left_pitch) * ratio
    return None


def distance_based_pitch_offset(distance_m: float | None) -> float:
    """Return extra elevation for the current target distance.

    Near targets keep a small offset. Far targets receive a larger offset so
    shots do not fall short when distance grows.
    """
    if distance_m is None:
        return BALLISTIC_PITCH_OFFSET_NEAR_DEG

    distance = float(distance_m)
    if distance <= BALLISTIC_PITCH_OFFSET_FAR_START_M:
        return BALLISTIC_PITCH_OFFSET_NEAR_DEG
    if distance >= BALLISTIC_PITCH_OFFSET_FAR_END_M:
        return BALLISTIC_PITCH_OFFSET_FAR_DEG

    ratio = (
        (distance - BALLISTIC_PITCH_OFFSET_FAR_START_M)
        / max(0.001, BALLISTIC_PITCH_OFFSET_FAR_END_M - BALLISTIC_PITCH_OFFSET_FAR_START_M)
    )
    return BALLISTIC_PITCH_OFFSET_NEAR_DEG + (
        BALLISTIC_PITCH_OFFSET_FAR_DEG - BALLISTIC_PITCH_OFFSET_NEAR_DEG
    ) * ratio


def distance_based_fire_yaw_deadband(distance_m: float | None) -> float:
    """Return a distance-aware yaw deadband for firing.

    A fixed angular tolerance is too loose at long range. This converts a
    lateral tolerance in meters into an angular threshold and clamps it.
    """
    if distance_m is None or distance_m <= 0:
        return LIDAR_FIRE_YAW_DEADBAND_MAX_DEG

    yaw_deg = degrees(atan(LIDAR_FIRE_LATERAL_TOLERANCE_M / float(distance_m)))
    return clamp(
        yaw_deg,
        LIDAR_FIRE_YAW_DEADBAND_MIN_DEG,
        LIDAR_FIRE_YAW_DEADBAND_MAX_DEG,
    )


def ballistic_pitch_control(lidar_target: dict[str, Any] | None = None) -> dict[str, Any]:
    lidar_target = lidar_target or current_lidar_fusion_target(latest_state)
    turret = latest_turret_from_info(latest_raw_info)
    current_pitch = safe_float(turret.get("pitch"))
    distance = safe_float((lidar_target or {}).get("nearestDistance"))
    target_pitch = interpolate_ballistic_pitch(distance)
    pitch_offset = distance_based_pitch_offset(distance)
    if target_pitch is not None:
        target_pitch += pitch_offset

    status = {
        "enabled": AUTO_BALLISTIC_PITCH_ENABLED,
        "ready": False,
        "reason": "waiting_for_lidar_target",
        "distance": round_or_none(distance),
        "currentPitch": round_or_none(current_pitch),
        "targetPitch": round_or_none(target_pitch),
        "pitchOffset": round_or_none(pitch_offset),
        "pitchError": None,
        "turretRF": {"command": "", "weight": 0.0},
    }
    if not AUTO_BALLISTIC_PITCH_ENABLED:
        status["reason"] = "ballistic_pitch_disabled"
        return status
    if not lidar_target or distance is None:
        return status
    if current_pitch is None or target_pitch is None:
        status["reason"] = "missing_current_or_target_pitch"
        return status

    pitch_error = float(target_pitch) - float(current_pitch)
    status["pitchError"] = round(pitch_error, 3)
    if abs(pitch_error) <= BALLISTIC_PITCH_DEADBAND_DEG:
        status["ready"] = True
        status["reason"] = "pitch_inside_ballistic_deadband"
        return status

    weight = aim_weight(
        pitch_error / 10.0,
        BALLISTIC_PITCH_DEADBAND_DEG / 10.0,
        BALLISTIC_PITCH_WEIGHT_MIN,
        BALLISTIC_PITCH_WEIGHT_MAX,
    )
    # R raises pitch in the current simulator mapping; F lowers pitch.
    status["turretRF"] = {"command": "R" if pitch_error > 0 else "F", "weight": weight}
    status["reason"] = "raise_pitch_to_ballistic_solution" if pitch_error > 0 else "lower_pitch_to_ballistic_solution"
    return status


def vision_aim_commands(target: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not target:
        empty = {"command": "", "weight": 0.0}
        return deepcopy(empty), deepcopy(empty), deepcopy(empty)

    error_x = float(target.get("errorX", 0.0))
    raw_error_y = float(target.get("errorY", 0.0))
    error_y = raw_error_y - float(aim_zero_state.get("offsetY", 0.0) or 0.0)
    body_weight = aim_weight(error_x, VISION_AIM_DEADBAND_X, VISION_BODY_WEIGHT_MIN, VISION_BODY_WEIGHT_MAX)
    turret_yaw_weight = aim_weight(error_x, VISION_AIM_DEADBAND_X, VISION_TURRET_WEIGHT_MIN, VISION_TURRET_WEIGHT_MAX)
    turret_pitch_weight = aim_weight(error_y, VISION_AIM_DEADBAND_Y, VISION_TURRET_WEIGHT_MIN, VISION_TURRET_WEIGHT_MAX)

    move_ad = {"command": "", "weight": 0.0}
    turret_qe = {"command": "", "weight": 0.0}
    turret_rf = {"command": "", "weight": 0.0}

    if body_weight:
        move_ad = {"command": "D" if error_x > 0 else "A", "weight": body_weight}
    if turret_yaw_weight:
        turret_qe = {"command": "E" if error_x > 0 else "Q", "weight": turret_yaw_weight}
    if turret_pitch_weight:
        turret_rf = {"command": "F" if error_y > 0 else "R", "weight": turret_pitch_weight}

    return move_ad, turret_qe, turret_rf


def lidar_turret_yaw_control(lidar_target: dict[str, Any] | None) -> dict[str, Any]:
    """LiDAR-only turret yaw correction.

    Q = turret left, E = turret right.
    The command is generated from the target's body-relative angle.
    """
    if not lidar_target:
        return {"command": "", "weight": 0.0}

    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
        999.0,
    )
    if angle_error is None:
        return {"command": "", "weight": 0.0}

    angle_error = normalize_signed_angle(float(angle_error))
    if abs(angle_error) <= LIDAR_TURRET_YAW_DEADBAND_DEG:
        return {"command": "", "weight": 0.0}

    scaled_error = min(abs(angle_error) / 10.0, 1.0)
    weight = round(
        LIDAR_TURRET_YAW_WEIGHT_MIN
        + (LIDAR_TURRET_YAW_WEIGHT_MAX - LIDAR_TURRET_YAW_WEIGHT_MIN) * scaled_error,
        3,
    )
    return {"command": "E" if angle_error > 0.0 else "Q", "weight": weight}


def active_vision_target() -> dict[str, Any] | None:
    if not USE_YOLO_FOR_AIM:
        return None
    target = vision_state.get("target")
    detected_at = float(vision_state.get("lastDetectedAt", 0.0) or 0.0)
    if target and monotonic() - detected_at <= VISION_TARGET_HOLD_SECONDS:
        return deepcopy(target)
    return None


def fire_readiness_status(now: float | None = None) -> dict[str, Any]:
    now = monotonic() if now is None else now
    target = active_vision_target()
    lidar_target = current_lidar_fusion_target(latest_state)
    fusion = vision_state.get("lidarFusion") or latest_state.get("lidarVisionFusion") or {}
    alignment = latest_state.get("bodyAlignment", {}) or {}

    error_x = abs(float(target.get("errorX", 999.0))) if target else 999.0
    raw_error_y = float(target.get("errorY", 999.0)) if target else 999.0
    adjusted_error_y = raw_error_y - float(aim_zero_state.get("offsetY", 0.0) or 0.0)
    error_y = abs(adjusted_error_y)
    vision_ready = bool(target) and error_x <= VISION_AIM_DEADBAND_X
    fusion_ready = bool(fusion.get("isAttackTarget")) and bool(fusion.get("alignedForFusion"))
    body_ready = bool(bool(alignment.get("aligned")) or (
        alignment.get("target")
        and abs(float(alignment["target"].get("bodyRelativeAngleErrorDeg", 999.0))) <= BODY_ALIGN_DEADBAND_DEG
    ))
    pitch_status = ballistic_pitch_control(lidar_target)
    pitch_ready = bool(pitch_status.get("ready"))

    # Extra yaw guard for LiDAR-only firing. Body alignment can be considered
    # "aligned" for driving at about 1 deg, but firing needs a tighter yaw gate.
    aim_angle_error = 999.0
    if lidar_target:
        raw_aim_angle_error = safe_float(
            lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
            999.0,
        )
        aim_angle_error = abs(float(raw_aim_angle_error if raw_aim_angle_error is not None else 999.0))
    lidar_distance = safe_float((lidar_target or {}).get("nearestDistance"))
    fire_yaw_deadband = distance_based_fire_yaw_deadband(lidar_distance)
    lidar_yaw_ready = aim_angle_error <= fire_yaw_deadband

    # LiDAR-only mode: YOLO class/mapping does not block aiming or firing.
    fusion_status = str(fusion.get("status", ""))
    fusion_has_vision = bool(fusion.get("vision"))
    known_non_attack = bool(USE_YOLO_FIRE_GUARD and fusion_has_vision and not bool(fusion.get("isAttackTarget")))
    geometry_mismatch = bool(USE_YOLO_FIRE_GUARD and fusion_status == "recognized_geometry_mismatch")

    lidar_ready = (
        bool(lidar_target)
        and body_ready
        and pitch_ready
        and lidar_yaw_ready
        and not known_non_attack
        and not geometry_mismatch
    )
    ready = bool(lidar_ready or (USE_YOLO_FOR_AIM and vision_ready and fusion_ready and body_ready and pitch_ready))
    approved = now <= float(fire_control_state.get("approvedUntil", 0.0) or 0.0)

    if vision_ready and fusion_ready and body_ready:
        reason = "ready_to_fire_yolo_confirmed"
    elif lidar_ready:
        reason = "ready_to_fire_lidar_aligned"
    elif known_non_attack:
        reason = "blocked_by_yolo_non_attack_class"
    elif geometry_mismatch:
        reason = "blocked_by_lidar_yolo_geometry_mismatch"
    elif not pitch_ready:
        reason = pitch_status.get("reason", "pitch_not_ready")
    elif not lidar_yaw_ready:
        reason = "lidar_yaw_not_ready"
    elif not lidar_target:
        reason = "no_lidar_target"
    elif not fusion_ready:
        reason = "lidar_yolo_fusion_not_attack_ready"
    elif not body_ready:
        reason = "body_not_aligned"
    else:
        reason = "aim_not_centered"

    return {
        "ready": ready,
        "approved": approved,
        "fireOnNextAction": bool(ready and approved),
        "reason": reason,
        "approvalSeconds": FIRE_APPROVAL_SECONDS,
        "approvedUntil": fire_control_state.get("approvedUntil"),
        "approvedAt": fire_control_state.get("approvedAt"),
        "lastFiredAt": fire_control_state.get("lastFiredAt"),
        "fireCount": fire_control_state.get("fireCount", 0),
        "target": deepcopy(target),
        "aimError": {
            "x": round(error_x, 4) if target else None,
            "rawY": round(raw_error_y, 4) if target else None,
            "adjustedY": round(adjusted_error_y, 4) if target else None,
            "y": round(error_y, 4) if target else None,
            "deadbandX": VISION_AIM_DEADBAND_X,
            "deadbandY": VISION_AIM_DEADBAND_Y,
            "zeroOffsetY": round(float(aim_zero_state.get("offsetY", 0.0) or 0.0), 4),
        },
        "aimZero": deepcopy(aim_zero_state),
        "ballisticPitch": pitch_status,
        "fusionReady": fusion_ready,
        "bodyReady": body_ready,
        "lidarReady": lidar_ready,
        "lidarYawReady": lidar_yaw_ready,
        "lidarYawErrorDeg": round(aim_angle_error, 3) if aim_angle_error != 999.0 else None,
        "lidarFireYawDeadbandDeg": round(fire_yaw_deadband, 3),
        "lidarDistance": round_or_none(lidar_distance),
        "pitchReady": pitch_ready,
        "visionReady": vision_ready,
        "knownNonAttack": known_non_attack,
        "geometryMismatch": geometry_mismatch,
        "fusionStatus": fusion_status,
    }


def current_lidar_fusion_target(scan: dict[str, Any] | None = None) -> dict[str, Any] | None:
    scan = scan or latest_state
    alignment = scan.get("bodyAlignment", {}) or {}
    if alignment.get("target"):
        return deepcopy(alignment["target"])
    if scan.get("primaryFusionTarget"):
        return deepcopy(scan["primaryFusionTarget"])
    return None


def build_lidar_vision_fusion(
    lidar_target: dict[str, Any] | None,
    vision_target: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not lidar_target:
        return None

    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
        999.0,
    )
    aligned_for_fusion = abs(float(angle_error or 999.0)) <= LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG
    fusion = {
        "status": "waiting_for_alignment" if not aligned_for_fusion else "waiting_for_detection",
        "alignedForFusion": aligned_for_fusion,
        "maxBodyErrorDeg": LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG,
        "lidar": {
            "trackId": lidar_target.get("trackId"),
            "candidateLabel": lidar_target.get("candidateLabel"),
            "geometryClass": lidar_target.get("geometryClass"),
            "nearestDistance": lidar_target.get("nearestDistance"),
            "centerAngle": lidar_target.get("centerAngle"),
            "bodyRelativeAngleErrorDeg": angle_error,
            "fusionPriorityRank": lidar_target.get("fusionPriorityRank"),
            "fusionPriorityTier": lidar_target.get("fusionPriorityTier"),
        },
        "vision": None,
        "semanticClass": None,
        "isAttackTarget": False,
    }

    if not aligned_for_fusion or not vision_target:
        return fusion

    class_name = str(vision_target.get("className", "unknown"))
    semantic = normalize_yolo_class_name(class_name)
    mapped_geometry = lidar_geometry_for_yolo_class(class_name)
    lidar_geometry = str(lidar_target.get("geometryClass", "unknown"))
    geometry_matches = mapped_geometry is None or mapped_geometry == lidar_geometry
    is_attack_target = is_attack_yolo_class(class_name)
    fusion.update({
        "status": (
            "recognized_attack_target"
            if is_attack_target and geometry_matches
            else (
                "recognized_geometry_mismatch"
                if not geometry_matches
                else "recognized_non_attack_target"
            )
        ),
        "vision": deepcopy(vision_target),
        "semanticClass": class_name,
        "mappedLidarGeometry": mapped_geometry,
        "lidarGeometryMatchesYolo": geometry_matches,
        "isAttackTarget": is_attack_target and geometry_matches,
    })
    return fusion


def normalize_yolo_class_name(class_name: Any) -> str:
    return str(class_name or "").strip().lower()


def lidar_geometry_for_yolo_class(class_name: Any) -> str | None:
    return YOLO_CLASS_TO_LIDAR_GEOMETRY.get(normalize_yolo_class_name(class_name))


def is_attack_yolo_class(class_name: Any) -> bool:
    semantic = normalize_yolo_class_name(class_name)
    return semantic in VISION_TARGET_CLASSES or semantic.startswith("tank_enemy_")


def enrich_with_recognition(obj: dict[str, Any]) -> dict[str, Any]:
    track_id = obj.get("trackId")
    if track_id is None:
        return obj
    recognition = recognized_lidar_objects.get(int(track_id))
    if not recognition:
        return obj

    enriched = deepcopy(obj)
    class_name = recognition.get("className", "unknown")
    enriched.update({
        "semanticClass": class_name,
        "recognizedClass": class_name,
        "recognizedConfidence": recognition.get("confidence"),
        "recognizedAt": recognition.get("recognizedAt"),
        "recognizedBy": "yolo_lidar_geometry_mapping",
        "isAttackTarget": is_attack_yolo_class(class_name),
    })
    return enriched


def remember_recognized_lidar_object(
    lidar_target: dict[str, Any] | None,
    vision_target: dict[str, Any] | None,
    fusion: dict[str, Any] | None,
) -> None:
    if not lidar_target or not vision_target or not fusion:
        return
    if not fusion.get("lidarGeometryMatchesYolo", False):
        return
    track_id = lidar_target.get("trackId")
    if track_id is None:
        return

    class_name = str(vision_target.get("className", "unknown"))
    recognized_lidar_objects[int(track_id)] = {
        "trackId": int(track_id),
        "className": class_name,
        "confidence": vision_target.get("confidence"),
        "mappedLidarGeometry": lidar_geometry_for_yolo_class(class_name),
        "isAttackTarget": fusion.get("isAttackTarget", False),
        "recognizedAt": datetime.now().isoformat(timespec="milliseconds"),
    }


def refresh_latest_state_recognitions() -> None:
    for key in ("trackedObjects", "confirmedObjects", "fusionPriorityQueue"):
        latest_state[key] = [
            enrich_with_recognition(obj)
            for obj in latest_state.get(key, [])
        ]
    if latest_state.get("primaryFusionTarget"):
        latest_state["primaryFusionTarget"] = enrich_with_recognition(latest_state["primaryFusionTarget"])
    alignment = latest_state.get("bodyAlignment", {})
    if alignment.get("target"):
        alignment["target"] = enrich_with_recognition(alignment["target"])
        latest_state["bodyAlignment"] = alignment


def record_impact(data: dict[str, Any]) -> dict[str, Any]:
    marker = make_impact_marker(data)
    with state_lock:
        impact_history.append(marker)
        latest_state["impactMarkers"] = list(impact_history)
    return marker


def bin_center(value: float, width: float, offset: float = 0.0) -> float:
    index = floor((value - offset) / width)
    return offset + (index + 0.5) * width


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def first_present_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = safe_float(data.get(key))
        if value is not None:
            return value
    return None


def nested_position(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("position", "impactPosition", "hitPosition", "bulletPosition", "location", "point"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data if any(key in data for key in ("x", "X", "z", "Z")) else None


def body_yaw_from_info(data: dict[str, Any]) -> float:
    yaw = first_present_float(
        data,
        (
            "playerBodyX",
            "Player_Body_X",
            "player_body_x",
            "bodyX",
            "body_x",
            "bodyYaw",
            "body_yaw",
        ),
    )
    if yaw is not None:
        return yaw
    rotation = data.get("lidarRotation", {}) or {}
    if isinstance(rotation, dict):
        return safe_float(rotation.get("y"), 0.0) or 0.0
    return 0.0


def latest_turret_from_info(data: dict[str, Any]) -> dict[str, float | None]:
    turret = data.get("turret", {}) or data.get("playerTurret", {}) or data.get("Player_Turret", {}) or {}
    if not isinstance(turret, dict):
        turret = {}
    yaw = first_present_float(
        data,
        (
            "playerTurretX",
            "Player_Turret_X",
            "player_turret_x",
            "turretX",
            "turret_x",
            "turretYaw",
            "turret_yaw",
        ),
    )
    pitch = first_present_float(
        data,
        (
            "playerTurretY",
            "Player_Turret_Y",
            "player_turret_y",
            "turretY",
            "turret_y",
            "turretPitch",
            "turret_pitch",
        ),
    )
    if yaw is None:
        yaw = first_present_float(turret, ("x", "X", "yaw", "Yaw"))
    if pitch is None:
        pitch = first_present_float(turret, ("y", "Y", "pitch", "Pitch"))
    return {"yaw": round_or_none(yaw), "pitch": round_or_none(pitch)}


def make_impact_marker(data: dict[str, Any]) -> dict[str, Any]:
    global next_impact_id

    position = nested_position(data) or {}
    angle = first_present_float(data, ("angle", "centerAngle", "bodyRelativeAngleDeg"))
    distance = first_present_float(data, ("distance", "nearestDistance", "range"))
    world_x = first_present_float(position, ("x", "X"))
    world_y = first_present_float(position, ("y", "Y"))
    world_z = first_present_float(position, ("z", "Z"))

    with state_lock:
        origin = deepcopy(latest_state.get("lidarOrigin", {}) or {})
        raw_info = deepcopy(latest_raw_info)
        last_action = deepcopy(action_debug_state.get("lastResponse", {}))
        last_fire_status = deepcopy(action_debug_state.get("lastFireStatus", {}))
        aim_zero = deepcopy(aim_zero_state)

    origin_x = first_present_float(origin, ("x", "X"))
    origin_z = first_present_float(origin, ("z", "Z"))
    if angle is None and distance is None and None not in (world_x, world_z, origin_x, origin_z):
        dx = float(world_x) - float(origin_x)
        dz = float(world_z) - float(origin_z)
        distance = hypot(dx, dz)
        world_angle = degrees(atan2(dx, dz))
        angle = normalize_signed_angle(world_angle - body_yaw_from_info(raw_info))

    marker_id = next_impact_id
    next_impact_id += 1

    return {
        "id": marker_id,
        "receivedAt": datetime.now().isoformat(timespec="milliseconds"),
        "objectName": data.get("objectName") or data.get("name") or data.get("target"),
        "angle": round_or_none(angle),
        "distance": round_or_none(distance),
        "position": {
            "x": round_or_none(world_x),
            "y": round_or_none(world_y),
            "z": round_or_none(world_z),
        },
        "turret": latest_turret_from_info(raw_info),
        "bodyYaw": round_or_none(body_yaw_from_info(raw_info)),
        "lastAction": {
            "turretQE": last_action.get("turretQE", {"command": "", "weight": 0.0}),
            "turretRF": last_action.get("turretRF", {"command": "", "weight": 0.0}),
            "moveAD": last_action.get("moveAD", {"command": "", "weight": 0.0}),
            "fire": bool(last_action.get("fire", False)),
        },
        "aimZero": aim_zero,
        "fireStatus": last_fire_status,
        "raw": deepcopy(data),
    }


def quantile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    ratio = max(0.0, min(1.0, ratio))
    index = int(round((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


def state_severity(state: str) -> int:
    return {"unknown": 0, "passable": 1, "caution": 2, "blocked": 3}.get(state, 0)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


# =============================================================================
# 2. PARSE RAYS
# =============================================================================
def parse_lidar_rays(data: dict[str, Any]) -> list[dict[str, Any]]:
    rays: list[dict[str, Any]] = []

    for raw in data.get("lidarPoints", []):
        if not isinstance(raw, dict):
            continue

        distance = safe_float(raw.get("distance"))
        angle = safe_float(raw.get("angle"))
        vertical_angle = safe_float(raw.get("verticalAngle"))
        if None in (distance, angle, vertical_angle):
            continue
        if not (0.0 < float(distance) <= MAX_DISTANCE_M):
            continue

        position = raw.get("position", {}) or {}
        x = safe_float(position.get("x"))
        y = safe_float(position.get("y"))
        z = safe_float(position.get("z"))

        rays.append(
            {
                "isDetected": bool(raw.get("isDetected", False)),
                "angle": normalize_signed_angle(float(angle)),
                "verticalAngle": float(vertical_angle),
                "distance": float(distance),
                "horizontalRange": float(distance) * cos(radians(float(vertical_angle))),
                "channelIndex": raw.get("channelIndex"),
                "position": {"x": x, "y": y, "z": z},
            }
        )

    return rays


def detected_rays(rays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        ray
        for ray in rays
        if (
            ray.get("isDetected", False)
            and ray["position"].get("x") is not None
            and ray["position"].get("y") is not None
            and ray["position"].get("z") is not None
        )
    ]


# =============================================================================
# 3. LOCAL GROUND MODEL
# =============================================================================
def estimate_flat_ground_y(hits: list[dict[str, Any]]) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in hits
        if (
            TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG
            and ray["horizontalRange"] <= TERRAIN_ANALYSIS_MAX_DISTANCE_M
        )
    )
    return quantile(heights, 0.25)


def estimate_local_ground_y(
    rays: list[dict[str, Any]],
    fallback_ground_y: float | None,
) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in rays
        if (
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
            and ray["verticalAngle"] >= LOCAL_GROUND_STEEP_MIN_DEG
            and ray["horizontalRange"] <= LOCAL_GROUND_NEAR_MAX_DISTANCE_M
        )
    )
    if len(heights) >= LOCAL_GROUND_MIN_POINT_COUNT:
        return float(median(heights))
    return fallback_ground_y


def local_grid_angle_center(angle_deg: float) -> float:
    return round(bin_center(angle_deg, LOCAL_GROUND_GRID_ANGLE_WIDTH_DEG, offset=-180.0), 3)


def local_grid_range_center(horizontal_range_m: float) -> float:
    return round(bin_center(horizontal_range_m, LOCAL_GROUND_GRID_RANGE_BIN_M), 3)


def build_local_ground_grid(hits: list[dict[str, Any]]) -> dict[float, list[dict[str, float]]]:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)

    for ray in hits:
        if not (TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        if ray["horizontalRange"] > TERRAIN_ANALYSIS_MAX_DISTANCE_M:
            continue
        y = ray["position"].get("y")
        if y is None:
            continue
        grouped[(local_grid_angle_center(ray["angle"]), local_grid_range_center(ray["horizontalRange"]))].append(float(y))

    grid: dict[float, list[dict[str, float]]] = defaultdict(list)
    for (angle_center, range_center), ys in grouped.items():
        grid[angle_center].append({"range": range_center, "groundY": min(ys)})

    for values in grid.values():
        values.sort(key=lambda item: item["range"])
    return dict(grid)


def lookup_local_ground_y(
    grid: dict[float, list[dict[str, float]]],
    angle_deg: float,
    horizontal_range_m: float,
    fallback_ground_y: float | None,
) -> float | None:
    angle_candidates = sorted(
        grid.keys(),
        key=lambda key: angular_distance_deg(float(key), angle_deg),
    )[:3]

    best: tuple[float, float] | None = None
    for angle_key in angle_candidates:
        for entry in grid.get(angle_key, []):
            range_gap = abs(float(entry["range"]) - horizontal_range_m)
            angle_gap = angular_distance_deg(float(angle_key), angle_deg)
            score = range_gap + 0.25 * angle_gap
            if range_gap <= LOCAL_GROUND_LOOKUP_MAX_RANGE_GAP_M:
                if best is None or score < best[0]:
                    best = (score, float(entry["groundY"]))

    return best[1] if best is not None else fallback_ground_y


# =============================================================================
# 4. TERRAIN HAZARD ANALYZER
# =============================================================================
def terrain_sector_center(angle_deg: float) -> float:
    return round(bin_center(angle_deg, TERRAIN_SECTOR_WIDTH_DEG, offset=-TERRAIN_FRONT_LIMIT_DEG), 3)


def build_ground_profile(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        y = point["position"].get("y")
        if y is None:
            continue
        grouped[round(bin_center(point["horizontalRange"], TERRAIN_RANGE_BIN_M), 3)].append(point)

    profile: list[dict[str, float]] = []
    for range_center, bin_points in grouped.items():
        lowest = min(bin_points, key=lambda item: float(item["position"]["y"]))
        profile.append({"horizontalRange": float(range_center), "height": float(lowest["position"]["y"])})
    profile.sort(key=lambda item: item["horizontalRange"])
    return profile


def analyze_profile_metrics(profile: list[dict[str, float]]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "maxUpSlopeDeg": None,
        "maxDownSlopeDeg": None,
        "maxUpStep": None,
        "maxDrop": None,
        "maxProfileGap": None,
        "maxUpStepRange": None,
        "maxDropRange": None,
    }
    if len(profile) < 2:
        return metrics

    for left, right in zip(profile, profile[1:]):
        dx = right["horizontalRange"] - left["horizontalRange"]
        dy = right["height"] - left["height"]
        if dx <= 0.3:
            continue
        slope = degrees(atan2(dy, dx))
        if slope >= 0:
            if metrics["maxUpSlopeDeg"] is None or slope > float(metrics["maxUpSlopeDeg"]):
                metrics["maxUpSlopeDeg"] = slope
        else:
            down = abs(slope)
            if metrics["maxDownSlopeDeg"] is None or down > float(metrics["maxDownSlopeDeg"]):
                metrics["maxDownSlopeDeg"] = down
        if dy >= 0:
            if metrics["maxUpStep"] is None or dy > float(metrics["maxUpStep"]):
                metrics["maxUpStep"] = dy
                metrics["maxUpStepRange"] = right["horizontalRange"]
        else:
            drop = abs(dy)
            if metrics["maxDrop"] is None or drop > float(metrics["maxDrop"]):
                metrics["maxDrop"] = drop
                metrics["maxDropRange"] = right["horizontalRange"]
        if metrics["maxProfileGap"] is None or dx > float(metrics["maxProfileGap"]):
            metrics["maxProfileGap"] = dx
    return metrics


def detect_wall_stack(points: list[dict[str, Any]]) -> tuple[bool, float, int, float | None]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        y = point["position"].get("y")
        if y is None:
            continue
        grouped[round(bin_center(point["horizontalRange"], WALL_RANGE_BIN_M), 3)].append(point)

    best_span = 0.0
    best_channels = 0
    best_range: float | None = None
    for range_center, bin_points in grouped.items():
        ys = [float(point["position"]["y"]) for point in bin_points]
        channels = {point.get("channelIndex") for point in bin_points}
        span = max(ys) - min(ys)
        if span > best_span:
            best_span = span
            best_channels = len(channels)
            best_range = float(range_center)
    return (
        best_span >= WALL_MIN_HEIGHT_SPAN_M and best_channels >= WALL_MIN_UNIQUE_CHANNELS,
        best_span,
        best_channels,
        best_range,
    )


def expected_flat_ground_range(sensor_height_m: float | None, vertical_angle_deg: float) -> float | None:
    if sensor_height_m is None or sensor_height_m <= 0.05 or vertical_angle_deg <= 0.0:
        return None
    tangent = tan(radians(vertical_angle_deg))
    return sensor_height_m / tangent if tangent > 0.0 else None


def classify_terrain_sector_raw(
    sector_angle: float,
    sector_rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    downward = [
        ray for ray in sector_rays
        if TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG
    ]
    hit_points = [
        ray for ray in downward
        if ray.get("isDetected", False)
        and ray["horizontalRange"] <= TERRAIN_ANALYSIS_MAX_DISTANCE_M
        and ray["position"].get("y") is not None
    ]
    profile = build_ground_profile(hit_points)
    metrics = analyze_profile_metrics(profile)
    is_wall, wall_span, wall_channels, wall_range = detect_wall_stack(hit_points)

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )

    expected_ranges: list[float] = []
    missing_expected_ranges: list[float] = []
    delayed_ranges: list[float] = []

    for ray in downward:
        expected_range = expected_flat_ground_range(sensor_height, ray["verticalAngle"])
        if expected_range is None or not (EXPECTED_GROUND_MIN_DISTANCE_M <= expected_range <= EXPECTED_GROUND_MAX_DISTANCE_M):
            continue
        expected_ranges.append(expected_range)
        if not ray.get("isDetected", False):
            missing_expected_ranges.append(expected_range)
            continue
        hit_y = ray["position"].get("y")
        if hit_y is None:
            missing_expected_ranges.append(expected_range)
            continue
        tolerance = max(EXPECTED_GROUND_RANGE_TOLERANCE_M, expected_range * (EXPECTED_GROUND_DELAY_RATIO - 1.0))
        if (
            ray["horizontalRange"] > expected_range + tolerance
            and local_ground_y is not None
            and float(hit_y) < float(local_ground_y) - CAUTION_DROP_M
        ):
            delayed_ranges.append(ray["horizontalRange"])

    expected_count = len(expected_ranges)
    miss_ratio = len(missing_expected_ranges) / expected_count if expected_count else None
    delayed_ratio = len(delayed_ranges) / expected_count if expected_count else None
    nearest_range = min((point["horizontalRange"] for point in hit_points), default=None)
    farthest_range = max((point["horizontalRange"] for point in hit_points), default=None)
    has_near_ground = any(point["horizontalRange"] <= CLIFF_NEAR_GROUND_EVIDENCE_M for point in hit_points)

    max_up_slope = metrics["maxUpSlopeDeg"]
    max_down_slope = metrics["maxDownSlopeDeg"]
    max_up_step = metrics["maxUpStep"]
    max_drop = metrics["maxDrop"]
    profile_gap = metrics["maxProfileGap"]

    hazard_range = nearest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M
    state = "unknown"
    reason = "insufficient_terrain_points"
    hazard_type = "unknown"

    if is_wall:
        state, reason, hazard_type = "blocked", "wall_like_vertical_stack", "wall_or_obstacle"
        hazard_range = wall_range or hazard_range
    elif max_drop is not None and max_drop >= BLOCKED_DROP_M:
        state, reason, hazard_type = "blocked", "cliff_or_pit_drop", "cliff_or_pit"
        hazard_range = metrics["maxDropRange"] or hazard_range
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio >= CLIFF_BLOCKED_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "blocked", "missing_expected_ground_returns_possible_cliff", "possible_cliff"
        hazard_range = min(missing_expected_ranges, default=hazard_range)
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and delayed_ratio is not None and delayed_ratio >= CLIFF_BLOCKED_DELAYED_RETURN_RATIO:
        state, reason, hazard_type = "blocked", "delayed_lower_ground_returns_possible_pit", "possible_pit"
        hazard_range = min(delayed_ranges, default=hazard_range)
    elif profile_gap is not None and profile_gap >= CLIFF_MIN_PROFILE_GAP_M and has_near_ground and (
        (miss_ratio is not None and miss_ratio >= CLIFF_CAUTION_EXPECTED_MISS_RATIO)
        or (delayed_ratio is not None and delayed_ratio >= CLIFF_CAUTION_DELAYED_RETURN_RATIO)
    ):
        state, reason, hazard_type = "blocked", "terrain_profile_gap_with_drop_evidence", "possible_cliff"
    elif max_up_step is not None and max_up_step >= BLOCKED_UP_STEP_M:
        state, reason, hazard_type = "blocked", "large_upward_step", "step_or_wall"
        hazard_range = metrics["maxUpStepRange"] or hazard_range
    elif max_up_slope is not None and max_up_slope > CAUTION_UP_SLOPE_MAX_DEG:
        state, reason, hazard_type = "blocked", "steep_uphill", "steep_slope"
    elif max_down_slope is not None and max_down_slope > CAUTION_DOWN_SLOPE_MAX_DEG:
        state, reason, hazard_type = "blocked", "steep_downhill", "steep_slope"
    elif max_drop is not None and max_drop >= CAUTION_DROP_M:
        state, reason, hazard_type = "caution", "moderate_drop", "drop"
        hazard_range = metrics["maxDropRange"] or hazard_range
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio >= CLIFF_CAUTION_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "caution", "reduced_expected_ground_returns", "possible_cliff"
        hazard_range = min(missing_expected_ranges, default=hazard_range)
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and delayed_ratio is not None and delayed_ratio >= CLIFF_CAUTION_DELAYED_RETURN_RATIO:
        state, reason, hazard_type = "caution", "delayed_lower_ground_returns", "possible_pit"
        hazard_range = min(delayed_ranges, default=hazard_range)
    elif max_up_step is not None and max_up_step >= CAUTION_UP_STEP_M:
        state, reason, hazard_type = "caution", "moderate_upward_step", "step"
        hazard_range = metrics["maxUpStepRange"] or hazard_range
    elif max_up_slope is not None and max_up_slope > PASSABLE_UP_SLOPE_MAX_DEG:
        state, reason, hazard_type = "caution", "moderate_uphill", "slope"
    elif max_down_slope is not None and max_down_slope > PASSABLE_DOWN_SLOPE_MAX_DEG:
        state, reason, hazard_type = "caution", "moderate_downhill", "slope"
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio < CLIFF_CAUTION_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "passable", "expected_ground_returns_consistent", "ground"
        hazard_range = farthest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M
    elif len(profile) >= 2:
        state, reason, hazard_type = "passable", "continuous_gentle_profile", "ground"
        hazard_range = farthest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M

    return {
        "centerAngle": round(sector_angle, 3),
        "rawState": state,
        "state": state,
        "rawReason": reason,
        "reason": reason,
        "hazardType": hazard_type,
        "hazardBoundaryRange": round(float(hazard_range), 3),
        "nearestHorizontalRange": round_or_none(nearest_range),
        "farthestHorizontalRange": round_or_none(farthest_range),
        "localGroundY": round_or_none(local_ground_y),
        "sensorHeightAboveLocalGround": round_or_none(sensor_height),
        "expectedGroundRayCount": expected_count,
        "missingExpectedGroundCount": len(missing_expected_ranges),
        "expectedGroundMissRatio": round_or_none(miss_ratio),
        "delayedGroundReturnCount": len(delayed_ranges),
        "delayedGroundReturnRatio": round_or_none(delayed_ratio),
        "hitPointCount": len(hit_points),
        "profilePointCount": len(profile),
        "maxUpSlopeDeg": round_or_none(max_up_slope),
        "maxDownSlopeDeg": round_or_none(max_down_slope),
        "maxUpStep": round_or_none(max_up_step),
        "maxDrop": round_or_none(max_drop),
        "maxProfileGap": round_or_none(profile_gap),
        "wallHeightSpan": round(wall_span, 3),
        "wallUniqueChannelCount": wall_channels,
    }


def stabilize_terrain_sectors(raw_sectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terrain_history.append({float(sector["centerAngle"]): deepcopy(sector) for sector in raw_sectors})
    stabilized: list[dict[str, Any]] = []

    for raw in raw_sectors:
        angle = float(raw["centerAngle"])
        history = [frame[angle] for frame in terrain_history if angle in frame]
        states = [item["rawState"] for item in history]
        counts = Counter(states)
        blocked_count = counts["blocked"]
        caution_count = counts["caution"]
        passable_count = counts["passable"]

        stable_state = raw["rawState"]
        reason = raw["rawReason"]
        if blocked_count >= TERRAIN_BLOCKED_CONFIRM_FRAMES:
            stable_state = "blocked"
            blocked_reasons = [item["rawReason"] for item in history if item["rawState"] == "blocked"]
            reason = Counter(blocked_reasons).most_common(1)[0][0]
        elif raw["rawState"] == "blocked":
            stable_state = "caution"
            reason = "pending_confirmation_" + raw["rawReason"]
        elif blocked_count + caution_count >= TERRAIN_CAUTION_CONFIRM_FRAMES:
            stable_state = "caution"
            caution_reasons = [item["rawReason"] for item in history if item["rawState"] in ("blocked", "caution")]
            reason = Counter(caution_reasons).most_common(1)[0][0]
        elif passable_count >= TERRAIN_PASSABLE_CONFIRM_FRAMES:
            stable_state = "passable"
            reason = "temporal_passable_confirmation"
        else:
            stable_state = raw["rawState"] if raw["rawState"] != "passable" else "unknown"
            reason = raw["rawReason"] if stable_state != "unknown" else "collecting_temporal_evidence"

        hazard_ranges = [
            float(item["hazardBoundaryRange"])
            for item in history
            if item.get("hazardBoundaryRange") is not None
            and item["rawState"] in (stable_state, "blocked" if stable_state == "caution" else stable_state)
        ]
        stable = deepcopy(raw)
        stable["state"] = stable_state
        stable["reason"] = reason
        stable["historyCount"] = len(history)
        stable["blockedVoteCount"] = blocked_count
        stable["cautionVoteCount"] = caution_count
        stable["passableVoteCount"] = passable_count
        if hazard_ranges:
            stable["hazardBoundaryRange"] = round(float(median(hazard_ranges)), 3)
        stabilized.append(stable)
    return stabilized


def analyze_terrain(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for ray in rays:
        if abs(ray["angle"]) > TERRAIN_FRONT_LIMIT_DEG:
            continue
        if not (TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        grouped[terrain_sector_center(ray["angle"])].append(ray)
        count += 1

    raw_sectors: list[dict[str, Any]] = []
    angle = -TERRAIN_FRONT_LIMIT_DEG + TERRAIN_SECTOR_WIDTH_DEG / 2.0
    while angle < TERRAIN_FRONT_LIMIT_DEG:
        raw_sectors.append(classify_terrain_sector_raw(angle, grouped.get(round(angle, 3), []), local_ground_y, lidar_origin_y))
        angle += TERRAIN_SECTOR_WIDTH_DEG
    return stabilize_terrain_sectors(raw_sectors), count


def summarize_front_terrain_decision(sectors: list[dict[str, Any]]) -> dict[str, Any]:
    front = [sector for sector in sectors if abs(float(sector["centerAngle"])) <= FRONT_CLEARANCE_HALF_WIDTH_DEG]
    blocked = [sector for sector in front if sector["state"] == "blocked"]
    caution = [sector for sector in front if sector["state"] == "caution"]
    unknown = [sector for sector in front if sector["state"] == "unknown"]

    dead_zone = [sector for sector in sectors if abs(float(sector["centerAngle"])) <= DEAD_END_FRONT_LIMIT_DEG]
    dead_blocked = [sector for sector in dead_zone if sector["state"] == "blocked"]
    dead_passable = [sector for sector in dead_zone if sector["state"] == "passable"]
    blocked_ratio = len(dead_blocked) / len(dead_zone) if dead_zone else 0.0
    dead_end = blocked_ratio >= DEAD_END_BLOCKED_RATIO and len(dead_passable) <= DEAD_END_MAX_PASSABLE_SECTORS

    if dead_end:
        state, action, reason = "blocked", "stop_or_turn", "possible_dead_end"
    elif blocked:
        state, action, reason = "blocked", "stop_or_turn", blocked[0]["reason"]
    elif caution:
        state, action, reason = "caution", "slow_forward", caution[0]["reason"]
    elif unknown:
        state, action, reason = "unknown", "slow_or_recheck", unknown[0]["reason"]
    elif front and all(sector["state"] == "passable" for sector in front):
        state, action, reason = "passable", "forward", "front_path_is_gentle"
    else:
        state, action, reason = "unknown", "slow_or_recheck", "no_front_sector_evidence"

    return {
        "state": state,
        "recommendedAction": action,
        "reason": reason,
        "deadEndDetected": dead_end,
        "deadEndBlockedRatio": round(blocked_ratio, 3),
        "frontSectorCount": len(front),
    }


# =============================================================================
# 5. OBJECT CANDIDATES AND TEMPORAL TRACKS
# =============================================================================
def make_object_azimuth_summaries(
    hits: list[dict[str, Any]],
    ground_grid: dict[float, list[dict[str, float]]],
    fallback_ground_y: float | None,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for point in hits:
        if not (OBJECT_VERTICAL_MIN_DEG <= point["verticalAngle"] <= OBJECT_VERTICAL_MAX_DEG):
            continue
        if point["distance"] > OBJECT_DETECTION_MAX_DISTANCE_M:
            continue
        y = point["position"].get("y")
        if y is None:
            continue
        ground_y = lookup_local_ground_y(ground_grid, point["angle"], point["horizontalRange"], fallback_ground_y)
        above = float(y) - float(ground_y) if ground_y is not None else None
        if above is not None and above < OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M:
            continue
        enriched = deepcopy(point)
        enriched["localGroundY"] = ground_y
        enriched["heightAboveLocalGround"] = above
        grouped[round(bin_center(point["angle"], OBJECT_AZIMUTH_BIN_WIDTH_DEG), 3)].append(enriched)
        count += 1

    summaries: list[dict[str, Any]] = []
    for azimuth, points in grouped.items():
        nearest = min(point["distance"] for point in points)
        surface = [point for point in points if point["distance"] <= nearest + OBJECT_CLUSTER_MAX_DISTANCE_GAP_M]
        summaries.append({
            "azimuth": float(azimuth),
            "nearestDistance": min(point["distance"] for point in surface),
            "medianDistance": float(median(point["distance"] for point in surface)),
            "points": surface,
        })
    summaries.sort(key=lambda item: item["azimuth"])
    return summaries, count


def can_merge_object_bins(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        angular_distance_deg(right["azimuth"], left["azimuth"]) <= OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG
        and abs(float(right["medianDistance"]) - float(left["medianDistance"])) <= OBJECT_CLUSTER_MAX_DISTANCE_GAP_M
    )


def classify_candidate(width: float, distance: float, point_count: int) -> tuple[str, str, str]:
    if point_count < 2:
        return "unknown", "low", "too_few_points"
    if width >= BULKY_MIN_WIDTH_M:
        return "bulky", "high" if distance <= BULKY_CONFIDENT_DISTANCE_M and point_count >= 4 else "medium", "wide_visible_footprint"
    if width <= THIN_MAX_WIDTH_M:
        if distance > THIN_MAX_DISTANCE_M:
            return "unknown", "low", "thin_candidate_beyond_reliable_range"
        return "thin", "high" if point_count >= 3 else "medium", "narrow_visible_footprint"
    return "unknown", "medium", "ambiguous_visible_width"


def overlap_with_hazard(center_angle: float, distance: float, sectors: list[dict[str, Any]]) -> bool:
    for sector in sectors:
        if sector["state"] not in ("blocked", "caution"):
            continue
        if angular_distance_deg(float(sector["centerAngle"]), center_angle) > TERRAIN_OBJECT_OVERLAP_ANGLE_DEG:
            continue
        boundary = safe_float(sector.get("hazardBoundaryRange"))
        if boundary is not None and abs(boundary - distance) <= TERRAIN_OBJECT_OVERLAP_RANGE_M:
            return True
    return False


def summarize_object_cluster(cluster: list[dict[str, Any]], object_id: int, sectors: list[dict[str, Any]]) -> dict[str, Any]:
    points = [point for summary in cluster for point in summary["points"]]
    distances = [float(point["distance"]) for point in points]
    angles = [float(summary["azimuth"]) for summary in cluster]
    ys = [float(point["position"]["y"]) for point in points]
    xs = [float(point["position"]["x"]) for point in points]
    zs = [float(point["position"]["z"]) for point in points]
    above_values = [float(point["heightAboveLocalGround"]) for point in points if point.get("heightAboveLocalGround") is not None]

    median_distance = float(median(distances))
    angular_width = max(OBJECT_AZIMUTH_BIN_WIDTH_DEG, max(angles) - min(angles) + OBJECT_AZIMUTH_BIN_WIDTH_DEG)
    width = 2.0 * median_distance * tan(radians(angular_width / 2.0))
    height = max(ys) - min(ys)
    center_angle = float(median(angles))
    geometry, confidence, reason = classify_candidate(width, median_distance, len(points))
    median_above = float(median(above_values)) if above_values else None
    hazard_overlap = overlap_with_hazard(center_angle, median_distance, sectors)

    # v8.3 suppressed candidates too aggressively:
    #   hazard_overlap OR median_above < 0.80
    # This can hide real objects placed on slopes or near walls.
    close_to_ground = (
        median_above is not None
        and median_above < TERRAIN_CONNECTED_LOW_HEIGHT_M
    )
    low_profile_hazard_overlap = (
        hazard_overlap
        and median_above is not None
        and median_above < TERRAIN_CONNECTED_HAZARD_LOW_HEIGHT_M
    )
    terrain_connected = close_to_ground or low_profile_hazard_overlap

    return {
        "id": object_id,
        "geometryClass": geometry,
        "shapeConfidence": confidence,
        "shapeReason": reason,
        "candidateLabel": {"thin": "TH", "bulky": "BK", "unknown": "UK"}.get(geometry, "UK"),
        "candidateMeaning": {"thin": "person_or_tree_like", "bulky": "tank_or_rock_like", "unknown": "unknown"}.get(geometry, "unknown"),
        "semanticClass": "unassigned_until_yolo_fusion",
        "centerAngle": round(center_angle, 3),
        "angleMin": round(min(angles), 3),
        "angleMax": round(max(angles), 3),
        "nearestDistance": round(min(distances), 3),
        "medianDistance": round(median_distance, 3),
        "estimatedWidth": round(width, 3),
        "estimatedHeight": round(height, 3),
        "medianHeightAboveLocalGround": round_or_none(median_above),
        "pointCount": len(points),
        "azimuthBinCount": len(cluster),
        "worldPosition": {"x": round(float(median(xs)), 3), "y": round(float(median(ys)), 3), "z": round(float(median(zs)), 3)},
        "terrainConnected": terrain_connected,
        "hazardSurfaceOverlap": hazard_overlap,
        "terrainSuppressionReason": (
            "hazard_surface_overlap_low_profile"
            if low_profile_hazard_overlap
            else ("close_to_local_ground" if close_to_ground else None)
        ),
    }


def classify_object_geometry(
    hits: list[dict[str, Any]],
    ground_grid: dict[float, list[dict[str, float]]],
    fallback_ground_y: float | None,
    sectors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    summaries, count = make_object_azimuth_summaries(hits, ground_grid, fallback_ground_y)
    if not summaries:
        return [], count
    clusters: list[list[dict[str, Any]]] = []
    current = [summaries[0]]
    for summary in summaries[1:]:
        if can_merge_object_bins(current[-1], summary):
            current.append(summary)
        else:
            clusters.append(current)
            current = [summary]
    clusters.append(current)
    objects = [summarize_object_cluster(cluster, idx + 1, sectors) for idx, cluster in enumerate(clusters)]
    objects.sort(key=lambda obj: float(obj["nearestDistance"]))
    for idx, obj in enumerate(objects, start=1):
        obj["id"] = idx
    return objects, count


class AxisKalman:
    def __init__(self, position: float) -> None:
        self.position = float(position)
        self.velocity = 0.0
        self.p00, self.p01, self.p10, self.p11 = 4.0, 0.0, 0.0, 4.0

    def predict(self, dt: float) -> None:
        dt = max(0.05, min(dt, 2.0))
        self.position += self.velocity * dt
        q = TRACK_PROCESS_NOISE
        p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + q * dt * dt
        p01 = self.p01 + dt * self.p11
        p10 = self.p10 + dt * self.p11
        p11 = self.p11 + q
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11

    def update(self, measurement: float) -> None:
        residual = float(measurement) - self.position
        s = self.p00 + TRACK_MEASUREMENT_NOISE
        k0, k1 = self.p00 / s, self.p10 / s
        self.position += k0 * residual
        self.velocity += k1 * residual
        p00, p01 = self.p00, self.p01
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p10 = self.p10 - k1 * p00
        self.p11 = self.p11 - k1 * p01
        if abs(self.velocity) < 0.05:
            self.velocity = 0.0


class ObjectTrack:
    def __init__(self, track_id: int, observation: dict[str, Any], sim_time: float) -> None:
        pos = observation["worldPosition"]
        self.id = track_id
        self.kx = AxisKalman(float(pos["x"]))
        self.kz = AxisKalman(float(pos["z"]))
        self.last_time = sim_time
        self.misses = 0
        self.hit_history: deque[int] = deque([1], maxlen=TRACK_HISTORY_SIZE)
        self.labels: deque[str] = deque([observation["geometryClass"]], maxlen=TRACK_HISTORY_SIZE)
        self.last_observation = deepcopy(observation)

    def predict(self, sim_time: float) -> None:
        dt = max(0.05, sim_time - self.last_time)
        self.kx.predict(dt)
        self.kz.predict(dt)
        self.last_time = sim_time

    def distance_to(self, observation: dict[str, Any]) -> float:
        pos = observation["worldPosition"]
        return hypot(float(pos["x"]) - self.kx.position, float(pos["z"]) - self.kz.position)

    def update(self, observation: dict[str, Any]) -> None:
        pos = observation["worldPosition"]
        self.kx.update(float(pos["x"]))
        self.kz.update(float(pos["z"]))
        self.misses = 0
        self.hit_history.append(1)
        self.labels.append(observation["geometryClass"])
        self.last_observation = deepcopy(observation)

    def miss(self) -> None:
        self.misses += 1
        self.hit_history.append(0)

    def to_dict(self) -> dict[str, Any]:
        observation = deepcopy(self.last_observation)
        geometry = Counter(self.labels).most_common(1)[0][0]
        confirmed = sum(self.hit_history) >= TRACK_CONFIRM_HITS and self.misses < TRACK_MAX_MISSES
        observation.update({
            "trackId": self.id,
            "geometryClass": geometry,
            "candidateLabel": {"thin": "TH", "bulky": "BK", "unknown": "UK"}.get(geometry, "UK"),
            "confirmed": confirmed,
            "persistenceHits": sum(self.hit_history),
            "historySize": len(self.hit_history),
            "misses": self.misses,
            "filteredWorldPosition": {"x": round(self.kx.position, 3), "z": round(self.kz.position, 3)},
            "estimatedVelocity": {"vx": round(self.kx.velocity, 3), "vz": round(self.kz.velocity, 3)},
        })
        return observation


class ObjectTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: list[ObjectTrack] = []

    def update(self, observations: list[dict[str, Any]], sim_time: float) -> list[dict[str, Any]]:
        usable = [obj for obj in observations if not obj.get("terrainConnected", False)]
        for track in self.tracks:
            track.predict(sim_time)

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for oi, obs in enumerate(usable):
                distance = track.distance_to(obs)
                if distance <= TRACK_ASSOCIATION_DISTANCE_M:
                    candidates.append((distance, ti, oi))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        for _, ti, oi in candidates:
            if ti in matched_tracks or oi in matched_obs:
                continue
            self.tracks[ti].update(usable[oi])
            matched_tracks.add(ti)
            matched_obs.add(oi)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.miss()

        for oi, obs in enumerate(usable):
            if oi not in matched_obs:
                self.tracks.append(ObjectTrack(self.next_id, obs, sim_time))
                self.next_id += 1

        self.tracks = [track for track in self.tracks if track.misses < TRACK_MAX_MISSES]
        return [track.to_dict() for track in self.tracks]


object_tracker = ObjectTracker()


def reset_runtime_state() -> None:
    """Clear temporal votes, object tracks and body-alignment lock."""
    global object_tracker, latest_state, latest_raw_info, body_alignment_state, action_debug_state, vision_state, recognized_lidar_objects, fire_control_state, aim_zero_state, next_impact_id

    terrain_history.clear()
    impact_history.clear()
    next_impact_id = 1
    object_tracker = ObjectTracker()
    recognized_lidar_objects = {}
    latest_raw_info = {}
    latest_state = {
        "simulationTime": None,
        "terrainSectors": [],
        "terrainDecision": {},
        "contourPoints": [],
        "frontVerticalProfile": {},
        "rawObjects": [],
        "trackedObjects": [],
        "fusionPriorityQueue": [],
        "primaryFusionTarget": None,
        "bodyAlignment": {},
        "visionTarget": None,
        "visionDetections": [],
        "lidarVisionFusion": None,
        "impactMarkers": [],
    }
    fire_control_state = {
        "approvedUntil": 0.0,
        "approvedAt": None,
        "lastFiredAt": None,
        "fireCount": 0,
    }
    aim_zero_state = {
        "offsetY": 0.0,
        "updatedAt": None,
    }
    body_alignment_state = {
        "enabled": AUTO_BODY_ALIGN_ENABLED,
        "lockedTrackId": None,
        "target": None,
        "moveAD": {"command": "", "weight": 0.0},
        "aligned": False,
        "reason": "waiting_for_target",
    }
    action_debug_state = {
        "getActionRequestCount": 0,
        "lastRequestBody": {},
        "lastResponse": {},
        "lastRequestedAt": None,
    }
    vision_state = {
        "target": None,
        "detections": [],
        "lastDetectedAt": 0.0,
        "modelPath": str(MODEL_PATH),
        "modelLoaded": yolo_model is not None,
        "lidarFusion": None,
    }


# =============================================================================
# 6. CONTOUR MAP AND TOP-LEVEL SUMMARY
# =============================================================================
def make_contour_points(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for ray in hits:
        if ray["horizontalRange"] > CONTOUR_MAX_DISTANCE_M:
            continue
        if not (-2.0 <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        grouped[round(bin_center(ray["angle"], CONTOUR_ANGLE_BIN_DEG, offset=-180.0), 3)].append(ray)

    points: list[dict[str, Any]] = []
    for angle, rays in grouped.items():
        nearest = min(rays, key=lambda ray: ray["horizontalRange"])
        points.append({"angle": float(angle), "distance": round(float(nearest["horizontalRange"]), 3)})
    points.sort(key=lambda point: point["angle"])
    return points


def make_front_view_points(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return detected points for a driver-like front angular projection."""
    points: list[dict[str, Any]] = []

    for ray in hits:
        angle = float(ray["angle"])
        vertical = float(ray["verticalAngle"])
        distance = float(ray["distance"])

        if abs(angle) > FRONT_VIEW_HORIZONTAL_LIMIT_DEG:
            continue
        if not (FRONT_VIEW_VERTICAL_MIN_DEG <= vertical <= FRONT_VIEW_VERTICAL_MAX_DEG):
            continue
        if distance > FRONT_VIEW_MAX_DISTANCE_M:
            continue

        points.append(
            {
                "angle": round(angle, 3),
                "verticalAngle": round(vertical, 3),
                "distance": round(distance, 3),
                "horizontalRange": round(float(ray["horizontalRange"]), 3),
                "channelIndex": ray.get("channelIndex"),
            }
        )

    return points



def linear_regression_slope_deg(profile: list[dict[str, float]]) -> float | None:
    """Estimate an approximate local terrain slope from a lower-envelope profile."""
    if len(profile) < FRONT_PROFILE_MIN_FIT_POINTS:
        return None

    xs = [float(point["horizontalRange"]) for point in profile]
    ys = [float(point["height"]) for point in profile]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if denominator <= 1e-9:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return degrees(atan2(slope, 1.0))


def make_front_vertical_profile(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    """
    Select the azimuth closest to body-forward 0 degrees and expose its vertical
    channels for a side-view cross-section.

    Positive verticalAngle points downward in this simulator.
    """
    if not rays:
        return {
            "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
            "selectedAngle": None,
            "channelCount": 0,
            "hitCount": 0,
            "missCount": 0,
            "sensorHeightAboveLocalGround": None,
            "approxGroundSlopeDeg": None,
            "maxUpSlopeDeg": None,
            "maxDownSlopeDeg": None,
            "maxDrop": None,
            "maxUpStep": None,
            "rays": [],
            "groundProfilePoints": [],
        }

    unique_angles = sorted({round(float(ray["angle"]), 3) for ray in rays})
    selected_angle = min(
        unique_angles,
        key=lambda angle: angular_distance_deg(angle, FRONT_PROFILE_TARGET_ANGLE_DEG),
    )

    if angular_distance_deg(selected_angle, FRONT_PROFILE_TARGET_ANGLE_DEG) > FRONT_PROFILE_MAX_SELECT_ANGLE_ERROR_DEG:
        return {
            "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
            "selectedAngle": selected_angle,
            "channelCount": 0,
            "hitCount": 0,
            "missCount": 0,
            "sensorHeightAboveLocalGround": None,
            "approxGroundSlopeDeg": None,
            "maxUpSlopeDeg": None,
            "maxDownSlopeDeg": None,
            "maxDrop": None,
            "maxUpStep": None,
            "rays": [],
            "groundProfilePoints": [],
            "warning": "No near-forward azimuth ray found.",
        }

    selected_rays = [
        ray
        for ray in rays
        if abs(float(ray["angle"]) - selected_angle) <= 0.15
    ]

    selected_rays.sort(
        key=lambda ray: (
            int(ray["channelIndex"]) if isinstance(ray.get("channelIndex"), int) else 9999,
            float(ray["verticalAngle"]),
        )
    )

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )

    output_rays: list[dict[str, Any]] = []
    downward_hits: list[dict[str, Any]] = []

    for ray in selected_rays:
        detected = bool(
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
        )
        position_y = safe_float(ray["position"].get("y"))
        relative_height = (
            float(position_y) - float(local_ground_y)
            if position_y is not None and local_ground_y is not None
            else None
        )
        expected_range = expected_flat_ground_range(
            sensor_height,
            float(ray["verticalAngle"]),
        )

        item = {
            "channelIndex": ray.get("channelIndex"),
            "angle": round(float(ray["angle"]), 3),
            "verticalAngle": round(float(ray["verticalAngle"]), 3),
            "isDetected": detected,
            "distance": round(float(ray["distance"]), 3),
            "horizontalRange": round(float(ray["horizontalRange"]), 3),
            "positionY": round_or_none(position_y),
            "heightAboveLocalGround": round_or_none(relative_height),
            "expectedFlatGroundRange": round_or_none(expected_range),
        }
        output_rays.append(item)

        if (
            detected
            and TERRAIN_VERTICAL_MIN_DEG <= float(ray["verticalAngle"]) <= TERRAIN_VERTICAL_MAX_DEG
            and float(ray["horizontalRange"]) <= FRONT_PROFILE_FIT_MAX_DISTANCE_M
        ):
            downward_hits.append(ray)

    profile = build_ground_profile(downward_hits)
    metrics = analyze_profile_metrics(profile)
    approx_slope = linear_regression_slope_deg(profile)

    profile_output = [
        {
            "horizontalRange": round(float(point["horizontalRange"]), 3),
            "heightAboveLocalGround": (
                round(float(point["height"]) - float(local_ground_y), 3)
                if local_ground_y is not None
                else None
            ),
            "worldY": round(float(point["height"]), 3),
        }
        for point in profile
    ]

    hit_count = sum(1 for ray in output_rays if ray["isDetected"])

    return {
        "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
        "selectedAngle": round(float(selected_angle), 3),
        "channelCount": len(output_rays),
        "hitCount": hit_count,
        "missCount": len(output_rays) - hit_count,
        "sensorHeightAboveLocalGround": round_or_none(sensor_height),
        "approxGroundSlopeDeg": round_or_none(approx_slope),
        "maxUpSlopeDeg": round_or_none(metrics["maxUpSlopeDeg"]),
        "maxDownSlopeDeg": round_or_none(metrics["maxDownSlopeDeg"]),
        "maxDrop": round_or_none(metrics["maxDrop"]),
        "maxUpStep": round_or_none(metrics["maxUpStep"]),
        "rays": output_rays,
        "groundProfilePoints": profile_output,
    }


def make_front_clearance(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    front = [obj for obj in objects if obj.get("confirmed") and abs(float(obj["centerAngle"])) <= FRONT_CLEARANCE_HALF_WIDTH_DEG]
    return min(front, key=lambda obj: float(obj["nearestDistance"]), default=None)



def priority_geometry_rank(geometry_class: str) -> int:
    """
    Smaller number means higher priority.

    BK / bulky candidates come first because they are tank-or-rock-like.
    TH / thin candidates remain available for later person detection.
    """
    return {
        "bulky": 0,
        "thin": 1,
        "unknown": 2,
    }.get(geometry_class, 3)


def is_same_priority_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        angular_distance_deg(
            float(left.get("centerAngle", 0.0)),
            float(right.get("centerAngle", 0.0)),
        )
        <= PRIORITY_DUPLICATE_ANGLE_TOLERANCE_DEG
        and abs(
            float(left.get("medianDistance", left.get("nearestDistance", 0.0)))
            - float(right.get("medianDistance", right.get("nearestDistance", 0.0)))
        )
        <= PRIORITY_DUPLICATE_DISTANCE_TOLERANCE_M
    )


def enrich_priority_candidate(
    obj: dict[str, Any],
    tracking_stage: str,
) -> dict[str, Any]:
    enriched = deepcopy(obj)

    distance = float(
        enriched.get(
            "nearestDistance",
            enriched.get("medianDistance", MAX_DISTANCE_M),
        )
    )
    geometry = str(enriched.get("geometryClass", "unknown"))
    is_close = distance <= PRIORITY_NEAR_MAX_DISTANCE_M
    is_confirmed = tracking_stage == "confirmed"

    # Lexicographic hierarchy:
    # 1. bulky before thin / unknown
    # 2. <= 50 m before farther objects
    # 3. confirmed before provisional
    # 4. shorter distance
    priority_key = (
        priority_geometry_rank(geometry),
        0 if is_close else 1,
        0 if is_confirmed else 1,
        distance,
    )

    enriched.update(
        {
            "trackingStage": tracking_stage,
            "isNearPriorityRange": is_close,
            "fusionPriorityKey": list(priority_key),
            "fusionPriorityTier": (
                "P1_bulky_near"
                if geometry == "bulky" and is_close
                else (
                    "P2_bulky_far"
                    if geometry == "bulky"
                    else (
                        "P3_non_bulky_near"
                        if is_close
                        else "P4_non_bulky_far"
                    )
                )
            ),
            "fusionPriorityReason": (
                "bulky candidate and within close-range threshold"
                if geometry == "bulky" and is_close
                else (
                    "bulky candidate outside close-range threshold"
                    if geometry == "bulky"
                    else (
                        "non-bulky fallback within close-range threshold"
                        if is_close
                        else "non-bulky fallback outside close-range threshold"
                    )
                )
            ),
            "recommendedTurretBodyRelativeAngleDeg": round(
                float(enriched.get("centerAngle", 0.0)),
                3,
            ),
        }
    )
    return enriched


def make_fusion_priority_queue(
    confirmed_objects: list[dict[str, Any]],
    provisional_objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Build a 360-degree LiDAR candidate queue for later YOLO fusion.

    Confirmed and provisional objects are both retained. If a provisional
    candidate overlaps a confirmed track, only the confirmed track is kept.
    """
    candidates: list[dict[str, Any]] = []

    for obj in confirmed_objects:
        candidates.append(enrich_priority_candidate(obj, "confirmed"))

    for obj in provisional_objects:
        provisional = enrich_priority_candidate(obj, "provisional")
        if any(is_same_priority_candidate(provisional, existing) for existing in candidates):
            continue
        candidates.append(provisional)

    candidates.sort(
        key=lambda obj: tuple(obj.get("fusionPriorityKey", [99, 99, 99, 999.0]))
    )

    candidates = candidates[:PRIORITY_MAX_QUEUE_SIZE]

    for index, obj in enumerate(candidates, start=1):
        obj["fusionPriorityRank"] = index
        obj["isPrimaryFusionTarget"] = index == 1

    primary = deepcopy(candidates[0]) if candidates else None
    return candidates, primary



def semantic_is_tank_candidate(obj: dict[str, Any]) -> bool:
    """
    YOLO-fusion-ready semantic check.

    Future semantic examples:
    - Tank_enemy_front
    - Tank_enemy_side
    - Tank_ally_back
    """
    semantic = str(obj.get("semanticClass", "")).strip().lower()
    return "tank" in semantic


def is_body_align_candidate(obj: dict[str, Any]) -> bool:
    distance = safe_float(obj.get("nearestDistance"))
    if distance is None or distance > BODY_ALIGN_LOCK_RELEASE_DISTANCE_M:
        return False

    if BODY_ALIGN_CONFIRMED_ONLY and obj.get("trackingStage") != "confirmed":
        return False

    if semantic_is_tank_candidate(obj):
        return True

    return (
        BODY_ALIGN_USE_LIDAR_BULKY_FALLBACK
        and obj.get("geometryClass") == "bulky"
        and distance <= BODY_ALIGN_TARGET_MAX_DISTANCE_M
    )


def body_align_target_sort_key(obj: dict[str, Any]) -> tuple[int, int, float]:
    """
    Strict tank semantics outrank LiDAR-only BK fallback.
    Within the same class, close and stable objects are selected first.
    """
    semantic_rank = 0 if semantic_is_tank_candidate(obj) else 1
    confirmed_rank = 0 if obj.get("trackingStage") == "confirmed" else 1
    distance = float(obj.get("nearestDistance", MAX_DISTANCE_M))
    return semantic_rank, confirmed_rank, distance


def choose_body_align_target(
    priority_queue: list[dict[str, Any]],
) -> dict[str, Any] | None:
    global body_alignment_state

    candidates = [obj for obj in priority_queue if is_body_align_candidate(obj)]
    if not candidates:
        body_alignment_state["lockedTrackId"] = None
        if BODY_ALIGN_USE_PRIMARY_FOR_RECOGNITION and priority_queue:
            selected = deepcopy(priority_queue[0])
            body_alignment_state["lockedTrackId"] = selected.get("trackId")
            selected["recognitionAlignmentOnly"] = True
            return selected
        return None

    locked_track_id = body_alignment_state.get("lockedTrackId")
    if locked_track_id is not None:
        for obj in candidates:
            if obj.get("trackId") == locked_track_id:
                return deepcopy(obj)

    candidates.sort(key=body_align_target_sort_key)
    selected = deepcopy(candidates[0])
    body_alignment_state["lockedTrackId"] = selected.get("trackId")
    return selected


def body_turn_weight(angle_error_deg: float) -> float:
    error = abs(angle_error_deg)
    if error <= BODY_ALIGN_SLOW_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_SLOW
    if error <= BODY_ALIGN_MEDIUM_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_MEDIUM
    if error <= BODY_ALIGN_FAST_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_FAST
    return BODY_ALIGN_WEIGHT_MAX


def update_body_alignment(
    priority_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate only a body-rotation request.

    Safety behavior:
    - no forward motion while aligning
    - no firing
    - stop rotating inside the deadband
    """
    global body_alignment_state

    enabled = bool(body_alignment_state.get("enabled", AUTO_BODY_ALIGN_ENABLED))
    if not enabled:
        body_alignment_state.update(
            {
                "target": None,
                "moveAD": {"command": "", "weight": 0.0},
                "aligned": False,
                "reason": "automatic_body_alignment_disabled",
            }
        )
        return deepcopy(body_alignment_state)

    target = choose_body_align_target(priority_queue)
    if target is None:
        body_alignment_state.update(
            {
                "target": None,
                "moveAD": {"command": "", "weight": 0.0},
                "aligned": False,
                "reason": "no_confirmed_close_tank_or_bulky_candidate",
            }
        )
        return deepcopy(body_alignment_state)

    angle_error = normalize_signed_angle(float(target.get("centerAngle", 0.0)))
    target["bodyRelativeAngleErrorDeg"] = round(angle_error, 3)

    if abs(angle_error) <= BODY_ALIGN_DEADBAND_DEG:
        move_ad = {"command": "", "weight": 0.0}
        aligned = True
        reason = (
            "recognition_target_inside_alignment_deadband"
            if target.get("recognitionAlignmentOnly")
            else "target_inside_alignment_deadband"
        )
    else:
        # Tank Challenge API:
        # A = body left turn, D = body right turn.
        move_ad = {
            "command": "A" if angle_error < 0.0 else "D",
            "weight": round(body_turn_weight(angle_error), 3),
        }
        aligned = False
        if target.get("recognitionAlignmentOnly"):
            reason = (
                "turn_left_for_lidar_yolo_recognition"
                if angle_error < 0.0
                else "turn_right_for_lidar_yolo_recognition"
            )
        else:
            reason = "turn_left_toward_target" if angle_error < 0.0 else "turn_right_toward_target"

    body_alignment_state.update(
        {
            "target": target,
            "moveAD": move_ad,
            "aligned": aligned,
            "reason": reason,
        }
    )
    return deepcopy(body_alignment_state)


def summarize_lidar(data: dict[str, Any]) -> dict[str, Any]:
    rays = parse_lidar_rays(data)
    hits = detected_rays(rays)
    flat_ground_y = estimate_flat_ground_y(hits)
    lidar_origin = data.get("lidarOrigin", {}) or {}
    lidar_origin_y = safe_float(lidar_origin.get("y"))
    local_ground_y = estimate_local_ground_y(rays, flat_ground_y)
    ground_grid = build_local_ground_grid(hits)

    terrain_sectors, terrain_ray_count = analyze_terrain(rays, local_ground_y, lidar_origin_y)
    terrain_decision = summarize_front_terrain_decision(terrain_sectors)
    raw_objects, object_point_count = classify_object_geometry(hits, ground_grid, local_ground_y, terrain_sectors)

    sim_time = safe_float(data.get("time"), monotonic()) or monotonic()
    provisional_objects = [
        obj for obj in raw_objects
        if not obj.get("terrainConnected", False)
    ][:PROVISIONAL_OBJECT_LIMIT]

    tracked_objects = [
        enrich_with_recognition(obj)
        for obj in object_tracker.update(raw_objects, float(sim_time))
    ]
    confirmed_objects = [obj for obj in tracked_objects if obj.get("confirmed")]

    fusion_priority_queue, primary_fusion_target = make_fusion_priority_queue(
        confirmed_objects=confirmed_objects,
        provisional_objects=provisional_objects,
    )
    body_alignment = update_body_alignment(fusion_priority_queue)

    return {
        "simulationTime": data.get("time"),
        "lidarOrigin": lidar_origin,
        "lidarRotation": data.get("lidarRotation", {}),
        "rawRayCount": len(rays),
        "rawDetectedPointCount": len(hits),
        "estimatedGroundY": round_or_none(flat_ground_y),
        "localGroundY": round_or_none(local_ground_y),
        "lidarOriginY": round_or_none(lidar_origin_y),
        "localGroundGridCellCount": sum(len(values) for values in ground_grid.values()),
        "terrainRayCount": terrain_ray_count,
        "terrainSectors": terrain_sectors,
        "terrainDecision": terrain_decision,
        "contourPoints": make_contour_points(hits),
        "frontVerticalProfile": make_front_vertical_profile(rays, local_ground_y, lidar_origin_y),
        "objectCandidatePointCount": object_point_count,
        "rawObjectCount": len(raw_objects),
        "rawObjects": raw_objects,
        "suppressedTerrainObjectCount": sum(1 for obj in raw_objects if obj.get("terrainConnected")),
        "provisionalObjectCount": len(provisional_objects),
        "provisionalObjects": provisional_objects,
        "trackedObjectCount": len(tracked_objects),
        "confirmedObjectCount": len(confirmed_objects),
        "trackedObjects": tracked_objects,
        "confirmedObjects": confirmed_objects,
        "fusionPriorityQueue": fusion_priority_queue,
        "fusionPriorityQueueCount": len(fusion_priority_queue),
        "primaryFusionTarget": primary_fusion_target,
        "bodyAlignment": body_alignment,
        "visionTarget": deepcopy(vision_state.get("target")),
        "visionDetections": deepcopy(vision_state.get("detections", [])),
        "lidarVisionFusion": deepcopy(vision_state.get("lidarFusion")),
        "impactMarkers": list(impact_history),
        "frontClearance": make_front_clearance(confirmed_objects),
    }


def print_status(scan: dict[str, Any]) -> None:
    global last_print_time
    now = monotonic()
    if now - last_print_time < PRINT_INTERVAL_SECONDS:
        return
    last_print_time = now
    print("\n" + "=" * 112)
    print(
        f"time={scan.get('simulationTime')} | rays={scan.get('rawRayCount')} | hits={scan.get('rawDetectedPointCount')} | "
        f"groundGrid={scan.get('localGroundGridCellCount')} | rawObjects={scan.get('rawObjectCount')} | "
        f"suppressedTerrainObjects={scan.get('suppressedTerrainObjectCount')} | confirmedTracks={scan.get('confirmedObjectCount')}"
    )
    decision = scan.get("terrainDecision", {})
    print(
        f"terrainDecision={decision.get('state')} | action={decision.get('recommendedAction')} | "
        f"reason={decision.get('reason')} | deadEnd={decision.get('deadEndDetected')}"
    )
    primary = scan.get("primaryFusionTarget")
    if primary:
        print(
            "priorityFusionTarget="
            f"#{primary.get('fusionPriorityRank')} | "
            f"{primary.get('candidateLabel')}? | "
            f"geometry={primary.get('geometryClass')} | "
            f"dist={float(primary.get('nearestDistance', 0.0)):.1f}m | "
            f"angle={float(primary.get('centerAngle', 0.0)):+.1f}deg | "
            f"tier={primary.get('fusionPriorityTier')} | "
            f"stage={primary.get('trackingStage')}"
        )
    else:
        print("priorityFusionTarget=none")

    alignment = scan.get("bodyAlignment", {})
    alignment_target = alignment.get("target")
    if alignment_target:
        print(
            "bodyAlignment="
            f"enabled={alignment.get('enabled')} | "
            f"reason={alignment.get('reason')} | "
            f"moveAD={alignment.get('moveAD')} | "
            f"target={alignment_target.get('candidateLabel')}? "
            f"{float(alignment_target.get('nearestDistance', 0.0)):.1f}m "
            f"{float(alignment_target.get('bodyRelativeAngleErrorDeg', 0.0)):+.1f}deg"
        )
    else:
        print(
            "bodyAlignment="
            f"enabled={alignment.get('enabled')} | "
            f"reason={alignment.get('reason')} | target=none"
        )

    for obj in scan.get("confirmedObjects", [])[:20]:
        print(
            f"  {obj['candidateLabel']}{obj['trackId']}? | geometry={obj['geometryClass']} | "
            f"dist={obj['nearestDistance']:.1f}m | angle={obj['centerAngle']:+.1f}deg | "
            f"width={obj['estimatedWidth']:.2f}m | persistence={obj['persistenceHits']}/{obj['historySize']}"
        )


def image_from_request() -> Image.Image | None:
    image_file = request.files.get("image") or request.files.get("file")
    if image_file:
        return Image.open(image_file.stream).convert("RGB")

    data = request.get_json(silent=True) or {}
    encoded = data.get("image") or data.get("frame") or data.get("capture")
    if isinstance(encoded, str):
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        except (OSError, ValueError, TypeError):
            return None
    return None


def detect_visual_targets(image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    model = load_yolo_model()
    width, height = image.size
    results = model.predict(image, conf=VISION_CONFIDENCE_MIN, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        names = result.names or {}
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id))
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            error_x = (center_x - width / 2.0) / max(1.0, width)
            error_y = (center_y - height / 2.0) / max(1.0, height)
            detections.append({
                "classId": class_id,
                "className": class_name,
                "confidence": round(confidence, 4),
                "mappedLidarGeometry": lidar_geometry_for_yolo_class(class_name),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
                "imageSize": {"width": width, "height": height},
                "errorX": round(error_x, 4),
                "errorY": round(error_y, 4),
                "area": round(area, 1),
                "isAimTarget": is_attack_yolo_class(class_name),
            })

    if not detections:
        return [], None

    target_candidates = [item for item in detections if item["isAimTarget"]] or detections
    target_candidates.sort(key=lambda item: (item["confidence"], item["area"]), reverse=True)
    return detections, deepcopy(target_candidates[0])


# =============================================================================
# 7. FLASK ENDPOINTS
# =============================================================================
@app.route("/info", methods=["POST"])
def info():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    scan = summarize_lidar(data)
    with state_lock:
        global latest_state, latest_raw_info
        latest_state = scan
        latest_raw_info = deepcopy(data)
    print_status(scan)
    return jsonify({"status": "success", "control": ""})


@app.route("/detect", methods=["POST"])
def detect():
    image = image_from_request()
    if image is None:
        return jsonify({"status": "error", "message": "No image received"}), 400

    if not USE_YOLO_FOR_AIM:
        with state_lock:
            vision_state["detections"] = []
            vision_state["target"] = None
            vision_state["lidarFusion"] = None
            latest_state["visionDetections"] = []
            latest_state["visionTarget"] = None
            latest_state["lidarVisionFusion"] = None
        if request.args.get("format") == "debug":
            return jsonify({
                "status": "success",
                "mode": "lidar_only_yolo_disabled",
                "detections": [],
                "target": None,
                "lidarVisionFusion": None,
            })
        return jsonify([])

    detections, target = detect_visual_targets(image)
    now = monotonic()
    with state_lock:
        lidar_target = current_lidar_fusion_target(latest_state)
        fusion = build_lidar_vision_fusion(lidar_target, target)
        fused_target = target if fusion and fusion.get("isAttackTarget") else None
        remember_recognized_lidar_object(lidar_target, target, fusion)
        vision_state["detections"] = detections
        vision_state["target"] = fused_target
        vision_state["lidarFusion"] = fusion
        vision_state["lastDetectedAt"] = now if fused_target else vision_state.get("lastDetectedAt", 0.0)
        vision_state["modelPath"] = str(MODEL_PATH)
        vision_state["modelLoaded"] = True
        refresh_latest_state_recognitions()
        latest_state["visionDetections"] = deepcopy(detections)
        latest_state["visionTarget"] = deepcopy(fused_target)
        latest_state["lidarVisionFusion"] = deepcopy(fusion)

    return jsonify({
        "status": "success",
        "model": MODEL_PATH.name,
        "detections": detections,
        "target": fused_target,
        "lidarVisionFusion": fusion,
    }) if request.args.get("format") == "debug" else jsonify([
        {
            "className": item["className"],
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "color": "#00FF00" if item.get("isAimTarget") else "#FFD166",
            "filled": False,
            "updateBoxWhileMoving": True,
        }
        for item in detections
    ])


@app.route("/lidar_status", methods=["GET"])
def lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_state))


@app.route("/lidar_monitor_status", methods=["GET"])
def lidar_monitor_status():
    """
    Lightweight browser payload.

    Excludes raw object arrays and the full front point cloud so that the
    top-view outline and the centerline 64-channel profile can refresh quickly.
    """
    with state_lock:
        scan = deepcopy(latest_state)

    return jsonify(
        {
            "simulationTime": scan.get("simulationTime"),
            "rawRayCount": scan.get("rawRayCount", 0),
            "rawDetectedPointCount": scan.get("rawDetectedPointCount", 0),
            "localGroundGridCellCount": scan.get("localGroundGridCellCount", 0),
            "terrainDecision": scan.get("terrainDecision", {}),
            "terrainSectors": scan.get("terrainSectors", []),
            "contourPoints": scan.get("contourPoints", []),
            "frontVerticalProfile": scan.get("frontVerticalProfile", {}),
            "provisionalObjects": scan.get("provisionalObjects", []),
            "confirmedObjects": scan.get("confirmedObjects", []),
            "rawObjectCount": scan.get("rawObjectCount", 0),
            "suppressedTerrainObjectCount": scan.get("suppressedTerrainObjectCount", 0),
            "provisionalObjectCount": scan.get("provisionalObjectCount", 0),
            "trackedObjectCount": scan.get("trackedObjectCount", 0),
            "confirmedObjectCount": scan.get("confirmedObjectCount", 0),
            "fusionPriorityQueue": scan.get("fusionPriorityQueue", []),
            "fusionPriorityQueueCount": scan.get("fusionPriorityQueueCount", 0),
            "primaryFusionTarget": scan.get("primaryFusionTarget"),
            "bodyAlignment": scan.get("bodyAlignment", {}),
            "visionTarget": scan.get("visionTarget"),
            "visionDetectionCount": len(scan.get("visionDetections", [])),
            "lidarVisionFusion": scan.get("lidarVisionFusion"),
            "impactMarkers": scan.get("impactMarkers", []),
            "fireControl": fire_readiness_status(),
            "aimZero": deepcopy(aim_zero_state),
        }
    )


@app.route("/body_align_status", methods=["GET"])
def body_align_status():
    with state_lock:
        return jsonify(deepcopy(latest_state.get("bodyAlignment", {})))


@app.route("/vision_status", methods=["GET"])
def vision_status():
    with state_lock:
        payload = deepcopy(vision_state)
        payload["activeTarget"] = active_vision_target()
        payload["currentLidarTarget"] = current_lidar_fusion_target(latest_state)
    return jsonify(payload)


@app.route("/fire_status", methods=["GET"])
def fire_status():
    with state_lock:
        return jsonify(fire_readiness_status())


@app.route("/fire_confirm", methods=["POST", "GET"])
def fire_confirm():
    with state_lock:
        status = fire_readiness_status()
        if not status["ready"]:
            return jsonify({"status": "not_ready", "fireControl": status}), 409
        now = monotonic()
        fire_control_state["approvedUntil"] = now + FIRE_APPROVAL_SECONDS
        fire_control_state["approvedAt"] = datetime.now().isoformat(timespec="milliseconds")
        status = fire_readiness_status(now)
    return jsonify({"status": "approved", "fireControl": status})


@app.route("/aim_zero", methods=["POST", "GET"])
def aim_zero():
    raw_action = str(request.args.get("action", "")).strip().lower()
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", raw_action or "set")).strip().lower()
    current = float(aim_zero_state.get("offsetY", 0.0) or 0.0)

    if action == "reset":
        offset = 0.0
    elif action == "capture":
        target = active_vision_target()
        if not target:
            return jsonify({"status": "not_ready", "message": "No active vision target to capture."}), 409
        offset = float(target.get("errorY", 0.0))
    elif action in {"up", "increase"}:
        offset = current + VISION_AIM_ZERO_STEP_Y
    elif action in {"down", "decrease"}:
        offset = current - VISION_AIM_ZERO_STEP_Y
    else:
        offset = safe_float(data.get("offsetY", request.args.get("offsetY")), current) or current

    offset = clamp(float(offset), -0.35, 0.35)
    with state_lock:
        aim_zero_state["offsetY"] = round(offset, 4)
        aim_zero_state["updatedAt"] = datetime.now().isoformat(timespec="milliseconds")
        status = fire_readiness_status()
    return jsonify({"status": "success", "aimZero": deepcopy(aim_zero_state), "fireControl": status})


@app.route("/body_align_enable", methods=["POST", "GET"])
def body_align_enable():
    global body_alignment_state

    raw = str(request.args.get("enabled", "true")).strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}

    with state_lock:
        body_alignment_state["enabled"] = enabled
        if not enabled:
            body_alignment_state["lockedTrackId"] = None
            body_alignment_state["target"] = None
            body_alignment_state["moveAD"] = {"command": "", "weight": 0.0}
            body_alignment_state["aligned"] = False
            body_alignment_state["reason"] = "automatic_body_alignment_disabled"

    return jsonify(
        {
            "status": "success",
            "enabled": enabled,
            "message": "Automatic body alignment setting updated.",
        }
    )


@app.route("/priority_status", methods=["GET"])
def priority_status():
    """Return the 360-degree LiDAR candidate queue for later YOLO fusion."""
    with state_lock:
        return jsonify(
            {
                "nearPriorityMaxDistanceM": PRIORITY_NEAR_MAX_DISTANCE_M,
                "primaryFusionTarget": deepcopy(latest_state.get("primaryFusionTarget")),
                "fusionPriorityQueue": deepcopy(latest_state.get("fusionPriorityQueue", [])),
            }
        )


@app.route("/raw_lidar_status", methods=["GET"])
def raw_lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_raw_info))


@app.route("/front_vertical_profile", methods=["GET"])
def front_vertical_profile():
    with state_lock:
        return jsonify(deepcopy(latest_state.get("frontVerticalProfile", {})))


@app.route("/export_snapshot", methods=["POST", "GET"])
def export_snapshot():
    label = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", request.args.get("label", "snapshot")).strip("_") or "snapshot"
    with state_lock:
        raw_copy = deepcopy(latest_raw_info)
        analyzed_copy = deepcopy(latest_state)
    if not raw_copy:
        return jsonify({"status": "error", "message": "No /info frame has been received yet."}), 400
    output_dir = Path.cwd() / "lidar_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.json"
    path.write_text(json.dumps({"label": label, "rawInfo": raw_copy, "analysis": analyzed_copy}, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"status": "success", "savedPath": str(path), "label": label})


@app.route("/lidar_view", methods=["GET"])
def lidar_view():
    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tank LiDAR Auto Body Alignment v8.7 Tracking Fix</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }
    canvas { background: #181818; border: 1px solid #555; }
    .dashboard { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
    .views { display: flex; flex-direction: column; gap: 18px; }
    .panel-title { font-size: 16px; font-weight: bold; margin: 0 0 6px 0; }
    pre { min-width: 720px; max-width: 980px; white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
    .legend { margin: 4px 0 0 0; font-size: 13px; color: #ddd; }
    .side-panel { display: flex; flex-direction: column; gap: 10px; }
    .firebar { display: flex; align-items: center; gap: 10px; }
    .firebar button { border: 1px solid #7a2424; background: #3a1515; color: #eee; padding: 9px 16px; font-weight: bold; border-radius: 6px; cursor: pointer; }
    .firebar button[data-ready="true"] { background: #cf2f2f; border-color: #ff6b6b; color: #fff; }
    .firebar button:disabled { opacity: 0.45; cursor: not-allowed; }
    .firebar span { color: #ddd; font-size: 13px; }
    .zerobar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .zerobar button { border: 1px solid #555; background: #222; color: #eee; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
    .zerobar span { color: #ddd; font-size: 13px; }
  </style>
</head>
<body>
  <h2>LiDAR Auto Body Alignment v8.7: Tracking Mode Fix + Action Debug</h2>
  <p>Top view: green contour = observed terrain outline. Terrain arcs: green=passable, yellow=caution, red=blocked, gray=unknown.</p>
  <p>Heavy front point-cloud view is disabled to reduce browser rendering and JSON transfer load.</p>
  <p>64-channel profile: select the azimuth closest to body-forward 0°. X = forward horizontal range, Y = height relative to local ground. Cyan polyline = estimated terrain profile.</p>
  <p>Object labels remain LiDAR-only candidates: TH?=thin(person/tree-like), BK?=bulky(tank/rock-like), UK?=unknown. Final person/tank labels require later YOLO fusion.</p>
  <p>Fusion queue priority: BK? bulky candidates first → candidates within 50 m → confirmed tracks → nearer distance. The white ring marks the current fusion priority #1.</p>
  <p>Automatic body alignment: confirmed close tank semantic targets first; before YOLO fusion, confirmed BK? candidates within 50 m are used as a fallback. The orange ring marks the current body-turn target.</p>
  <div class="dashboard">
    <div class="views">
      <div>
        <div class="panel-title">Top view: local terrain outline</div>
        <canvas id="topRadar" width="860" height="860"></canvas>
      </div>
      <div>
        <div class="panel-title">Centerline side profile: nearest body-forward azimuth × vertical channels</div>
        <canvas id="verticalProfile" width="860" height="520"></canvas>
        <div class="legend">Side profile: green ray = detected | gray dashed ray = missed | cyan line = local ground profile | blue point = LiDAR origin</div>
      </div>
    </div>
    <div class="side-panel">
      <div class="firebar">
        <button id="fireButton" disabled data-ready="false">FIRE</button>
        <span id="fireText">Waiting for aim lock</span>
      </div>
      <div class="zerobar">
        <button id="zeroDownButton">Zero -</button>
        <button id="zeroUpButton">Zero +</button>
        <button id="zeroCaptureButton">Use current Y</button>
        <button id="zeroResetButton">Reset zero</button>
        <span id="zeroText">Zero Y 0.000</span>
      </div>
      <pre id="status">Waiting for /info data...</pre>
    </div>
  </div>
<script>
const topCanvas = document.getElementById('topRadar');
const topCtx = topCanvas.getContext('2d');
const profileCanvas = document.getElementById('verticalProfile');
const profileCtx = profileCanvas.getContext('2d');
const statusText = document.getElementById('status');
const fireButton = document.getElementById('fireButton');
const fireText = document.getElementById('fireText');
const zeroText = document.getElementById('zeroText');
const MAX_DISTANCE = 120.0;
const FRONT_HORIZONTAL_LIMIT = 60.0;
const FRONT_VERTICAL_MIN = -22.5;
const FRONT_VERTICAL_MAX = 22.5;
const PROFILE_MAX_DISTANCE = 120.0;
const PROFILE_HEIGHT_MIN = -12.0;
const PROFILE_HEIGHT_MAX = 12.0;

function polar(angleDeg, distance, cx, cy, radius) {
  const a = angleDeg * Math.PI / 180.0;
  const r = Math.min(distance, MAX_DISTANCE) / MAX_DISTANCE * radius;
  return { x: cx + Math.sin(a) * r, y: cy - Math.cos(a) * r };
}
function terrainColor(state) {
  if (state === 'passable') return '#44d62c';
  if (state === 'caution') return '#ffb703';
  if (state === 'blocked') return '#ff3030';
  return '#777777';
}
function objectColor(kind) {
  if (kind === 'thin') return '#4cc9f0';
  if (kind === 'bulky') return '#f72585';
  return '#f1fa8c';
}
function objectLabel(obj) {
  const base = (obj.candidateLabel || 'UK') + (obj.trackId == null ? '?' : obj.trackId + '?');
  return obj.recognizedClass ? obj.recognizedClass + ' <- ' + base : base;
}
function drawImpactMarker(ctx, marker, cx, cy, radius) {
  if (marker.angle == null || marker.distance == null) return;
  const q=polar(Number(marker.angle),Number(marker.distance),cx,cy,radius);
  ctx.save();
  ctx.strokeStyle='#ff3030';
  ctx.fillStyle='#ff3030';
  ctx.lineWidth=4;
  ctx.beginPath();
  ctx.moveTo(q.x-10,q.y-10);
  ctx.lineTo(q.x+10,q.y+10);
  ctx.moveTo(q.x+10,q.y-10);
  ctx.lineTo(q.x-10,q.y+10);
  ctx.stroke();
  ctx.font='bold 13px Arial';
  ctx.fillText('IMPACT '+Number(marker.distance).toFixed(1)+'m',q.x+13,q.y-12);
  ctx.restore();
}
function updateFireControl(control) {
  control = control || {};
  const ready = !!control.ready;
  const approved = !!control.approved;
  fireButton.disabled = !ready;
  fireButton.dataset.ready = ready ? 'true' : 'false';
  if (control.fireOnNextAction) {
    fireText.textContent = 'Approved: next action will fire';
  } else if (approved) {
    fireText.textContent = 'Fire approved';
  } else if (ready) {
    fireText.textContent = 'Aim locked: press FIRE';
  } else {
    fireText.textContent = 'Not ready: '+(control.reason || 'waiting');
  }
}
function updateAimZero(scan) {
  const zero = (scan && scan.aimZero) || {};
  const control = (scan && scan.fireControl) || {};
  const err = control.aimError || {};
  zeroText.textContent =
    'Zero Y '+Number(zero.offsetY || 0).toFixed(3)
    +' | rawY='+(err.rawY == null ? 'n/a' : Number(err.rawY).toFixed(3))
    +' | adjY='+(err.adjustedY == null ? 'n/a' : Number(err.adjustedY).toFixed(3));
}
function distanceColor(distance) {
  if (distance <= 20) return '#ff3030';
  if (distance <= 50) return '#ffca3a';
  return '#44d62c';
}
function fusionColor(status) {
  if (status === 'recognized_attack_target') return '#00ff88';
  if (status === 'recognized_non_attack_target') return '#4cc9f0';
  if (status === 'recognized_geometry_mismatch') return '#ff3030';
  if (status === 'waiting_for_detection') return '#ffd166';
  if (status === 'waiting_for_alignment') return '#ff9f1c';
  return '#aaaaaa';
}
function drawArc(ctx, sector, cx, cy, radius) {
  const boundary = Math.max(3, Math.min(sector.hazardBoundaryRange || 20, 120));
  const r = boundary / MAX_DISTANCE * radius;
  const center = sector.centerAngle * Math.PI / 180.0;
  const half = 4.5 * Math.PI / 180.0;
  ctx.strokeStyle = terrainColor(sector.state);
  ctx.lineWidth = sector.state === 'blocked' ? 11 : 7;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(18, r), -Math.PI / 2 + center - half, -Math.PI / 2 + center + half);
  ctx.stroke();
  ctx.lineWidth = 1;
}
function drawTop(scan) {
  const ctx = topCtx, canvas = topCanvas;
  const w = canvas.width, h = canvas.height, cx = w/2, cy = h/2, radius = Math.min(w,h)*0.46;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#555'; ctx.fillStyle = '#bbb'; ctx.font = '13px Arial';
  for (const d of [30,60,90,120]) { const r=radius*d/MAX_DISTANCE; ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke(); ctx.fillText(d+' m',cx+5,cy-r+16); }
  ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx,cy-radius); ctx.stroke();
  ctx.fillStyle='#eee'; ctx.fillText('0 deg body-forward',cx+8,cy-radius+16); ctx.fillText('-90 deg',cx-radius+4,cy-8); ctx.fillText('+90 deg',cx+radius-60,cy-8);

  for (const p of (scan.contourPoints || [])) { const q=polar(p.angle,p.distance,cx,cy,radius); ctx.fillStyle='#35e835'; ctx.beginPath(); ctx.arc(q.x,q.y,2.5,0,Math.PI*2); ctx.fill(); }
  for (const sector of (scan.terrainSectors || [])) drawArc(ctx,sector,cx,cy,radius);

  if (scan.terrainDecision && scan.terrainDecision.deadEndDetected) {
    ctx.fillStyle='#ff3030'; ctx.font='bold 22px Arial'; ctx.fillText('POSSIBLE DEAD END', 20, 34);
  }

  ctx.font='bold 14px Arial';
  // Provisional candidates are visible immediately as hollow markers.
  ctx.save();
  ctx.globalAlpha=0.60;
  for (const obj of (scan.provisionalObjects || [])) {
    const q=polar(obj.centerAngle,obj.medianDistance,cx,cy,radius);
    ctx.strokeStyle=objectColor(obj.geometryClass); ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(q.x,q.y,5,0,Math.PI*2); ctx.stroke();
    ctx.fillStyle='#ddd'; ctx.fillText('~'+obj.candidateLabel+'? '+obj.nearestDistance.toFixed(1)+'m',q.x+7,q.y+13);
  }
  ctx.restore();

  // Confirmed tracks use a solid marker.
  for (const obj of (scan.confirmedObjects || [])) {
    const q=polar(obj.centerAngle,obj.medianDistance,cx,cy,radius);
    ctx.fillStyle=objectColor(obj.geometryClass); ctx.beginPath(); ctx.arc(q.x,q.y,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=obj.recognizedClass ? '#00ff88' : '#fff';
    ctx.fillText(objectLabel(obj)+' '+obj.nearestDistance.toFixed(1)+'m '+obj.centerAngle.toFixed(1)+'deg',q.x+8,q.y-8);
  }

  // The current LiDAR -> YOLO fusion target is emphasized with a white ring.
  if (scan.primaryFusionTarget) {
    const target=scan.primaryFusionTarget;
    const q=polar(target.centerAngle,target.medianDistance,cx,cy,radius);
    ctx.strokeStyle='#ffffff'; ctx.lineWidth=4;
    ctx.beginPath(); ctx.arc(q.x,q.y,14,0,Math.PI*2); ctx.stroke();
    ctx.lineWidth=1;
    ctx.fillStyle='#ffffff';
    ctx.fillText(
      'PRIORITY #1 '+objectLabel(target)+' '+target.nearestDistance.toFixed(1)+'m',
      q.x+18,q.y+24
    );
  }

  if (scan.bodyAlignment && scan.bodyAlignment.target) {
    const target=scan.bodyAlignment.target;
    const q=polar(target.centerAngle,target.medianDistance,cx,cy,radius);
    ctx.strokeStyle='#ff9f1c'; ctx.lineWidth=4;
    ctx.beginPath(); ctx.arc(q.x,q.y,21,0,Math.PI*2); ctx.stroke();
    ctx.lineWidth=1;
    ctx.fillStyle='#ff9f1c';
    ctx.fillText(
      'BODY TURN '+(scan.bodyAlignment.moveAD.command || 'ALIGNED')
        +' '+objectLabel(target)+' '+target.nearestDistance.toFixed(1)+'m',
      q.x+18,q.y+42
    );
  }

  if (scan.lidarVisionFusion && scan.lidarVisionFusion.lidar) {
    const fusion=scan.lidarVisionFusion;
    const target=fusion.lidar;
    const distance=target.nearestDistance || 0;
    const q=polar(target.centerAngle || 0,distance,cx,cy,radius);
    ctx.strokeStyle=fusionColor(fusion.status); ctx.lineWidth=5;
    ctx.setLineDash(fusion.alignedForFusion ? [] : [9,6]);
    ctx.beginPath(); ctx.arc(q.x,q.y,30,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth=1;
    ctx.fillStyle=fusionColor(fusion.status);
    ctx.font='bold 14px Arial';
    const semantic=fusion.semanticClass || 'YOLO pending';
    ctx.fillText(
      'LiDAR↔YOLO '+fusion.status+' | '+semantic,
      q.x+18,q.y+62
    );
  }

  for (const marker of (scan.impactMarkers || [])) {
    drawImpactMarker(ctx, marker, cx, cy, radius);
  }

  ctx.fillStyle='#4cc9f0'; ctx.beginPath(); ctx.arc(cx,cy,7,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fff'; ctx.font='13px Arial';
  ctx.fillText('Contour: green | Hollow ~=provisional | Solid=confirmed | green label=YOLO mapped | red X=impact',18,h-20);
}
function frontXY(angle, vertical, width, height, margin) {
  const usableW = width - 2 * margin;
  const usableH = height - 2 * margin;
  return {
    x: margin + ((angle + FRONT_HORIZONTAL_LIMIT) / (2 * FRONT_HORIZONTAL_LIMIT)) * usableW,
    y: margin + ((vertical - FRONT_VERTICAL_MIN) / (FRONT_VERTICAL_MAX - FRONT_VERTICAL_MIN)) * usableH
  };
}
function drawFront(scan) {
  const ctx = frontCtx, canvas = frontCanvas;
  const w = canvas.width, h = canvas.height, margin = 42;
  ctx.clearRect(0,0,w,h);
  ctx.font='12px Arial';

  // Grid
  ctx.strokeStyle='#444'; ctx.fillStyle='#bbb';
  for (const a of [-60,-30,0,30,60]) {
    const p=frontXY(a,0,w,h,margin); ctx.beginPath(); ctx.moveTo(p.x,margin); ctx.lineTo(p.x,h-margin); ctx.stroke(); ctx.fillText((a>0?'+':'')+a+'°',p.x-12,h-margin+18);
  }
  for (const v of [-22.5,-10,0,10,22.5]) {
    const p=frontXY(0,v,w,h,margin); ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke(); ctx.fillText((v>0?'+':'')+v+'°',5,p.y+4);
  }
  ctx.fillStyle='#eee'; ctx.fillText('left', margin, 18); ctx.fillText('body-forward 0°', w/2-42, 18); ctx.fillText('right', w-margin-28, 18);
  ctx.fillText('up (- vertical angle)', 8, margin-10); ctx.fillText('down (+ vertical angle)', 8, h-10);

  // Points
  for (const p of (scan.frontViewPoints || [])) {
    const q=frontXY(p.angle,p.verticalAngle,w,h,margin);
    ctx.fillStyle=distanceColor(p.distance);
    const size=p.distance <= 20 ? 3.0 : (p.distance <= 50 ? 2.4 : 1.8);
    ctx.beginPath(); ctx.arc(q.x,q.y,size,0,Math.PI*2); ctx.fill();
  }

  // Draw stabilized hazard sectors as a strip at the bottom of the front view.
  for (const s of (scan.terrainSectors || [])) {
    const left=frontXY(s.centerAngle-4.5,22.5,w,h,margin).x;
    const right=frontXY(s.centerAngle+4.5,22.5,w,h,margin).x;
    ctx.fillStyle=terrainColor(s.state);
    ctx.fillRect(left,h-margin-10,Math.max(2,right-left),8);
  }

  // Draw confirmed object candidates at vertical center. YOLO fusion can later
  // replace these with person/tank bounding boxes.
  ctx.font='bold 13px Arial';
  for (const obj of (scan.confirmedObjects || [])) {
    if (Math.abs(obj.centerAngle) > FRONT_HORIZONTAL_LIMIT) continue;
    const q=frontXY(obj.centerAngle,0,w,h,margin);
    ctx.strokeStyle=objectColor(obj.geometryClass); ctx.lineWidth=2;
    ctx.strokeRect(q.x-18,q.y-22,36,44); ctx.lineWidth=1;
    ctx.fillStyle=obj.recognizedClass ? '#00ff88' : '#fff';
    ctx.fillText(objectLabel(obj),q.x+22,q.y-8);
    ctx.fillText(obj.nearestDistance.toFixed(1)+'m',q.x+22,q.y+10);
  }
}

function profileXY(range, heightAboveGround, width, height, margin) {
  const usableW = width - 2 * margin;
  const usableH = height - 2 * margin;
  const clippedRange = Math.max(0, Math.min(PROFILE_MAX_DISTANCE, range));
  const clippedHeight = Math.max(PROFILE_HEIGHT_MIN, Math.min(PROFILE_HEIGHT_MAX, heightAboveGround));
  return {
    x: margin + clippedRange / PROFILE_MAX_DISTANCE * usableW,
    y: margin + (PROFILE_HEIGHT_MAX - clippedHeight) / (PROFILE_HEIGHT_MAX - PROFILE_HEIGHT_MIN) * usableH
  };
}
function drawVerticalProfile(scan) {
  const ctx = profileCtx, canvas = profileCanvas;
  const w = canvas.width, h = canvas.height, margin = 54;
  const profile = scan.frontVerticalProfile || {};
  const sensorHeight = profile.sensorHeightAboveLocalGround == null ? 1.0 : profile.sensorHeightAboveLocalGround;
  const origin = profileXY(0, sensorHeight, w, h, margin);

  ctx.clearRect(0,0,w,h);
  ctx.font='12px Arial';
  ctx.strokeStyle='#444'; ctx.fillStyle='#bbb';

  for (const d of [0,20,40,60,80,100,120]) {
    const p=profileXY(d,0,w,h,margin);
    ctx.beginPath(); ctx.moveTo(p.x,margin); ctx.lineTo(p.x,h-margin); ctx.stroke();
    ctx.fillText(d+'m',p.x-10,h-margin+18);
  }
  for (const y of [-10,-5,0,5,10]) {
    const p=profileXY(0,y,w,h,margin);
    ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke();
    ctx.fillText((y>0?'+':'')+y+'m',8,p.y+4);
  }

  // Local ground reference.
  const groundLeft=profileXY(0,0,w,h,margin);
  const groundRight=profileXY(PROFILE_MAX_DISTANCE,0,w,h,margin);
  ctx.strokeStyle='#00cfd5'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(groundLeft.x,groundLeft.y); ctx.lineTo(groundRight.x,groundRight.y); ctx.stroke();
  ctx.lineWidth=1;

  // Draw each selected vertical channel ray from the LiDAR origin.
  for (const ray of (profile.rays || [])) {
    const range = Math.min(PROFILE_MAX_DISTANCE, ray.horizontalRange || ray.distance || PROFILE_MAX_DISTANCE);
    let endpointHeight;

    if (ray.isDetected && ray.heightAboveLocalGround !== null) {
      endpointHeight = ray.heightAboveLocalGround;
    } else {
      // Positive vertical angle points downward.
      endpointHeight = sensorHeight - range * Math.tan(ray.verticalAngle * Math.PI / 180.0);
    }

    const end=profileXY(range,endpointHeight,w,h,margin);

    ctx.save();
    ctx.globalAlpha=ray.isDetected ? 0.30 : 0.22;
    ctx.strokeStyle=ray.isDetected ? '#44d62c' : '#888';
    if (!ray.isDetected) ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(origin.x,origin.y); ctx.lineTo(end.x,end.y); ctx.stroke();
    ctx.restore();

    if (ray.isDetected) {
      ctx.fillStyle=distanceColor(ray.distance);
      ctx.beginPath(); ctx.arc(end.x,end.y,3.0,0,Math.PI*2); ctx.fill();
    }
  }

  // Draw lower-envelope local terrain profile.
  const terrain=(profile.groundProfilePoints || []).filter(p => p.heightAboveLocalGround !== null);
  if (terrain.length) {
    ctx.strokeStyle='#00ffff'; ctx.lineWidth=3;
    ctx.beginPath();
    terrain.forEach((p,i) => {
      const q=profileXY(p.horizontalRange,p.heightAboveLocalGround,w,h,margin);
      if (i===0) ctx.moveTo(q.x,q.y); else ctx.lineTo(q.x,q.y);
    });
    ctx.stroke(); ctx.lineWidth=1;

    for (const p of terrain) {
      const q=profileXY(p.horizontalRange,p.heightAboveLocalGround,w,h,margin);
      ctx.fillStyle='#00ffff'; ctx.beginPath(); ctx.arc(q.x,q.y,3,0,Math.PI*2); ctx.fill();
    }
  }

  ctx.fillStyle='#4cc9f0'; ctx.beginPath(); ctx.arc(origin.x,origin.y,7,0,Math.PI*2); ctx.fill();

  ctx.fillStyle='#eee';
  ctx.fillText('LiDAR origin',origin.x+10,origin.y-8);
  ctx.fillText('forward horizontal range',w/2-55,h-10);
  ctx.fillText('height above local ground',8,18);
  ctx.fillText(
    'selected azimuth: '+(profile.selectedAngle == null ? 'n/a' : profile.selectedAngle.toFixed(1)+'°')
      +' | channels: '+(profile.channelCount || 0)
      +' | hits: '+(profile.hitCount || 0)
      +' | misses: '+(profile.missCount || 0),
    margin,34
  );
  ctx.fillText(
    'approx ground slope: '+(profile.approxGroundSlopeDeg == null ? 'n/a' : profile.approxGroundSlopeDeg.toFixed(1)+'°')
      +' | max uphill: '+(profile.maxUpSlopeDeg == null ? 'n/a' : profile.maxUpSlopeDeg.toFixed(1)+'°')
      +' | max downhill: '+(profile.maxDownSlopeDeg == null ? 'n/a' : profile.maxDownSlopeDeg.toFixed(1)+'°'),
    margin,50
  );
}

function draw(scan) {
  drawTop(scan);
  drawVerticalProfile(scan);
  updateFireControl(scan.fireControl);
  updateAimZero(scan);
  const alignment = scan.bodyAlignment || {};
  const alignmentMoveAD = alignment.moveAD || { command: '', weight: 0.0 };

  const provisionalLines=(scan.provisionalObjects||[]).slice(0,15).map(o =>
    '~'+o.candidateLabel+'? | '+o.candidateMeaning
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | width='+o.estimatedWidth.toFixed(2)+'m'
      +' | aboveGround='+(o.medianHeightAboveLocalGround == null ? 'n/a' : o.medianHeightAboveLocalGround.toFixed(2)+'m')
  );
  const trackLines=(scan.confirmedObjects||[]).map(o =>
    objectLabel(o)+' | '+o.candidateMeaning
      +' | yolo='+(o.recognizedClass || 'pending')
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | width='+o.estimatedWidth.toFixed(2)+'m'
      +' | persist='+o.persistenceHits+'/'+o.historySize
  );
  const priorityLines=(scan.fusionPriorityQueue||[]).slice(0,12).map(o =>
    '#'+o.fusionPriorityRank
      +' | '+objectLabel(o)
      +' | tier='+o.fusionPriorityTier
      +' | stage='+o.trackingStage
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | turretTarget='+o.recommendedTurretBodyRelativeAngleDeg.toFixed(1)+'deg'
  );
  const terrainLines=(scan.terrainSectors||[]).map(s =>
    s.centerAngle.toFixed(0)+'deg | state='+s.state+' | raw='+s.rawState+' | type='+s.hazardType+' | reason='+s.reason+' | range='+s.hazardBoundaryRange.toFixed(1)+'m | votes(B/C/P)='+s.blockedVoteCount+'/'+s.cautionVoteCount+'/'+s.passableVoteCount
  );
  const fusion=scan.lidarVisionFusion || {};
  const fusionLidar=fusion.lidar || {};
  const fusionVision=fusion.vision || {};
  const impactLines=(scan.impactMarkers||[]).slice().reverse().map(m =>
    '#'+m.id
      +' | dist='+(m.distance == null ? 'n/a' : Number(m.distance).toFixed(1)+'m')
      +' | angle='+(m.angle == null ? 'n/a' : Number(m.angle).toFixed(1)+'deg')
      +' | pos='+(m.position && m.position.x != null ? 'x '+Number(m.position.x).toFixed(1)+' / z '+Number(m.position.z).toFixed(1) : 'n/a')
      +' | turret='+(m.turret && m.turret.pitch != null ? 'pitch '+Number(m.turret.pitch).toFixed(2)+' / yaw '+Number(m.turret.yaw || 0).toFixed(2) : 'n/a')
      +' | RF='+((m.lastAction && m.lastAction.turretRF && m.lastAction.turretRF.command) ? m.lastAction.turretRF.command+' '+Number(m.lastAction.turretRF.weight || 0).toFixed(2) : 'STOP')
      +' | QE='+((m.lastAction && m.lastAction.turretQE && m.lastAction.turretQE.command) ? m.lastAction.turretQE.command+' '+Number(m.lastAction.turretQE.weight || 0).toFixed(2) : 'STOP')
      +' | zeroY='+(m.aimZero ? Number(m.aimZero.offsetY || 0).toFixed(3) : '0.000')
      +' | '+(m.objectName || 'impact')
  );
  const fusionLine=fusion.status
    ? 'LiDAR↔YOLO mapping: '+fusion.status
      +' | aligned='+fusion.alignedForFusion
      +' | lidar='+(fusionLidar.candidateLabel || '?')+'?'
      +' track='+(fusionLidar.trackId == null ? 'n/a' : fusionLidar.trackId)
      +' dist='+(fusionLidar.nearestDistance == null ? 'n/a' : Number(fusionLidar.nearestDistance).toFixed(1)+'m')
      +' angle='+(fusionLidar.bodyRelativeAngleErrorDeg == null ? 'n/a' : Number(fusionLidar.bodyRelativeAngleErrorDeg).toFixed(1)+'deg')
      +' | yolo='+(fusion.semanticClass || fusionVision.className || 'pending')
      +' conf='+(fusionVision.confidence == null ? 'n/a' : Number(fusionVision.confidence).toFixed(2))
      +' | attackTarget='+fusion.isAttackTarget
    : 'LiDAR↔YOLO mapping: none';
  statusText.textContent=[
    'simulationTime: '+scan.simulationTime,
    'rawRayCount: '+scan.rawRayCount,
    'rawDetectedPointCount: '+scan.rawDetectedPointCount,
    'frontPointCloudView: disabled for lightweight real-time monitoring',
    'frontVerticalProfile: selectedAngle='+(scan.frontVerticalProfile.selectedAngle == null ? 'n/a' : scan.frontVerticalProfile.selectedAngle+'deg')      +' | channels='+scan.frontVerticalProfile.channelCount      +' | hits='+scan.frontVerticalProfile.hitCount      +' | misses='+scan.frontVerticalProfile.missCount,
    'frontVerticalSlope: approx='+(scan.frontVerticalProfile.approxGroundSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.approxGroundSlopeDeg+'deg')      +' | maxUp='+(scan.frontVerticalProfile.maxUpSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.maxUpSlopeDeg+'deg')      +' | maxDown='+(scan.frontVerticalProfile.maxDownSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.maxDownSlopeDeg+'deg')      +' | maxDrop='+(scan.frontVerticalProfile.maxDrop == null ? 'n/a' : scan.frontVerticalProfile.maxDrop+'m'),
    'localGroundGridCellCount: '+scan.localGroundGridCellCount,
    'terrainDecision: '+scan.terrainDecision.state+' | action='+scan.terrainDecision.recommendedAction+' | reason='+scan.terrainDecision.reason,
    'deadEndDetected: '+scan.terrainDecision.deadEndDetected+' | blockedRatio='+scan.terrainDecision.deadEndBlockedRatio,
    'rawObjectCount: '+scan.rawObjectCount
      +' | suppressedTerrainObjects: '+scan.suppressedTerrainObjectCount
      +' | provisionalObjects: '+scan.provisionalObjectCount
      +' | trackedObjects: '+scan.trackedObjectCount
      +' | confirmedTracks: '+scan.confirmedObjectCount,
    '',
    'Automatic body alignment: '
      +(alignment.enabled ? 'enabled' : 'disabled')
      +' | reason='+(alignment.reason || 'waiting_for_target')
      +' | lockedTrackId='+(alignment.lockedTrackId == null ? 'none' : alignment.lockedTrackId)
      +' | moveAD='+(alignmentMoveAD.command || 'STOP')
      +' | weight='+alignmentMoveAD.weight,
    'Body alignment target: '
      +(alignment.target
        ? objectLabel(alignment.target)
          +' | dist='+alignment.target.nearestDistance.toFixed(1)+'m'
          +' | angleError='+alignment.target.bodyRelativeAngleErrorDeg.toFixed(1)+'deg'
        : 'none'),
    fusionLine,
    'Vision target: '
      +(scan.visionTarget
        ? scan.visionTarget.className+' | conf='+Number(scan.visionTarget.confidence).toFixed(2)
          +' | errorX='+Number(scan.visionTarget.errorX).toFixed(3)
          +' | errorY='+Number(scan.visionTarget.errorY).toFixed(3)
        : 'none'),
    'Fire control: '
      +(scan.fireControl
        ? 'ready='+scan.fireControl.ready
          +' | approved='+scan.fireControl.approved
          +' | nextFire='+scan.fireControl.fireOnNextAction
          +' | reason='+scan.fireControl.reason
        : 'none'),
    'Ballistic pitch: '
      +(scan.fireControl && scan.fireControl.ballisticPitch
        ? 'ready='+scan.fireControl.ballisticPitch.ready
          +' | dist='+(scan.fireControl.ballisticPitch.distance == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.distance).toFixed(1)+'m')
          +' | current='+(scan.fireControl.ballisticPitch.currentPitch == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.currentPitch).toFixed(2))
          +' | target='+(scan.fireControl.ballisticPitch.targetPitch == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.targetPitch).toFixed(2))
          +' | error='+(scan.fireControl.ballisticPitch.pitchError == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.pitchError).toFixed(2))
          +' | RF='+(scan.fireControl.ballisticPitch.turretRF.command || 'STOP')
          +' '+Number(scan.fireControl.ballisticPitch.turretRF.weight || 0).toFixed(2)
          +' | reason='+scan.fireControl.ballisticPitch.reason
        : 'none'),
    'Aim zero: offsetY='+(scan.aimZero ? Number(scan.aimZero.offsetY || 0).toFixed(3) : '0.000')
      +' | rawY='+(scan.fireControl && scan.fireControl.aimError.rawY == null ? 'n/a' : (scan.fireControl ? Number(scan.fireControl.aimError.rawY).toFixed(3) : 'n/a'))
      +' | adjustedY='+(scan.fireControl && scan.fireControl.aimError.adjustedY == null ? 'n/a' : (scan.fireControl ? Number(scan.fireControl.aimError.adjustedY).toFixed(3) : 'n/a')),
    '',
    'YOLO fusion priority queue (BK first, <=50m next):', ...(priorityLines.length?priorityLines:['none']),
    '',
    'Primary fusion target: '
      +(scan.primaryFusionTarget
        ? objectLabel(scan.primaryFusionTarget)+' | dist='+scan.primaryFusionTarget.nearestDistance.toFixed(1)+'m | angle='+scan.primaryFusionTarget.centerAngle.toFixed(1)+'deg | tier='+scan.primaryFusionTarget.fusionPriorityTier
        : 'none'),
    '',
    'Bullet impacts (red X):', ...(impactLines.length?impactLines:['none']),
    '',
    'Provisional LiDAR candidates (~ hollow marker):', ...(provisionalLines.length?provisionalLines:['none']),
    '',
    'Confirmed LiDAR object tracks with YOLO class mapping (solid marker):', ...(trackLines.length?trackLines:['none']),
    '',
    'Terrain sectors:', ...(terrainLines.length?terrainLines:['none']),
    '',
    'Lightweight monitor API: /lidar_monitor_status',
    'Priority queue API: /priority_status',
    'Body alignment API: /body_align_status',
    'Enable: /body_align_enable?enabled=true | Disable: /body_align_enable?enabled=false',
    'Full debug JSON API: /lidar_status'
  ].join('\n');
}
async function refresh(){try{const r=await fetch('/lidar_monitor_status',{cache:'no-store'}); draw(await r.json());}catch(e){statusText.textContent=String(e);}}
async function confirmFire(){
  fireButton.disabled = true;
  fireText.textContent = 'Approving fire...';
  try {
    const r = await fetch('/fire_confirm', {method:'POST'});
    const payload = await r.json();
    updateFireControl(payload.fireControl);
  } catch (e) {
    fireText.textContent = 'Fire approval failed: '+String(e);
  }
}
fireButton.addEventListener('click', confirmFire);
async function postAimZero(action){
  try {
    const r = await fetch('/aim_zero', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})
    });
    const payload = await r.json();
    zeroText.textContent = payload.aimZero
      ? 'Zero Y '+Number(payload.aimZero.offsetY || 0).toFixed(3)
      : 'Zero update failed';
    await refresh();
  } catch (e) {
    zeroText.textContent = 'Zero update failed: '+String(e);
  }
}
document.getElementById('zeroDownButton').addEventListener('click', () => postAimZero('down'));
document.getElementById('zeroUpButton').addEventListener('click', () => postAimZero('up'));
document.getElementById('zeroCaptureButton').addEventListener('click', () => postAimZero('capture'));
document.getElementById('zeroResetButton').addEventListener('click', () => postAimZero('reset'));
setInterval(refresh,200); refresh();
</script>
</body>
</html>
"""


@app.route("/action_debug", methods=["GET"])
def action_debug():
    with state_lock:
        return jsonify(deepcopy(action_debug_state))


@app.route("/get_action", methods=["POST"])
def get_action():
    """
    Tank Challenge control endpoint.

    moveWS:
      STOP while the body is aligning.

    moveAD:
      A = body left turn
      D = body right turn

    turret and firing remain disabled at this stage.
    """
    request_body = request.get_json(silent=True) or {}

    with state_lock:
        alignment = deepcopy(latest_state.get("bodyAlignment", {}))
        decision = deepcopy(latest_state.get("terrainDecision", {}))
        sectors = deepcopy(latest_state.get("terrainSectors", []))
        lidar_target = current_lidar_fusion_target(latest_state)
        vision_target = active_vision_target() if USE_YOLO_FOR_AIM else None

    turret_qe = {"command": "", "weight": 0.0}
    turret_rf = {"command": "", "weight": 0.0}

    if vision_target:
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad, turret_qe, turret_rf = vision_aim_commands(vision_target)
    elif alignment.get("enabled") and alignment.get("target"):
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = deepcopy(alignment.get("moveAD", {"command": "", "weight": 0.0}))
    elif not AUTO_DRIVE_ENABLED:
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = {"command": "", "weight": 0.0}
    elif decision.get("state") == "passable":
        move_ws = {"command": "W", "weight": 0.55}
        move_ad = {"command": "", "weight": 0.0}
    elif decision.get("state") == "caution":
        move_ws = {"command": "W", "weight": 0.25}
        move_ad = {"command": "", "weight": 0.0}
    else:
        left = [s for s in sectors if s["centerAngle"] < -15 and s["state"] == "passable"]
        right = [s for s in sectors if s["centerAngle"] > 15 and s["state"] == "passable"]
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = (
            {"command": "A", "weight": 0.35}
            if left and not right
            else (
                {"command": "D", "weight": 0.35}
                if right
                else {"command": "", "weight": 0.0}
            )
        )

    if not USE_YOLO_FOR_AIM:
        lidar_turret_qe = lidar_turret_yaw_control(lidar_target)
        if lidar_turret_qe.get("command"):
            turret_qe = lidar_turret_qe

    pitch_status = ballistic_pitch_control(lidar_target)
    if pitch_status.get("enabled") and pitch_status.get("turretRF", {}).get("command"):
        turret_rf = deepcopy(pitch_status["turretRF"])

    should_fire = False
    with state_lock:
        fire_status_payload = fire_readiness_status()
        if fire_status_payload.get("fireOnNextAction"):
            should_fire = True
            fire_control_state["approvedUntil"] = 0.0
            fire_control_state["lastFiredAt"] = datetime.now().isoformat(timespec="milliseconds")
            fire_control_state["fireCount"] = int(fire_control_state.get("fireCount", 0) or 0) + 1
            fire_status_payload = fire_readiness_status()

    response_body = {
        "moveWS": move_ws,
        "moveAD": move_ad,
        "turretQE": turret_qe,
        "turretRF": turret_rf,
        "fire": should_fire,
    }

    with state_lock:
        action_debug_state["getActionRequestCount"] += 1
        action_debug_state["lastRequestBody"] = deepcopy(request_body)
        action_debug_state["lastResponse"] = deepcopy(response_body)
        action_debug_state["lastFireStatus"] = deepcopy(fire_status_payload)
        action_debug_state["lastRequestedAt"] = datetime.now().isoformat(timespec="milliseconds")

    print(
        "GET_ACTION | "
        f"moveWS={response_body['moveWS']} | "
        f"moveAD={response_body['moveAD']} | "
        f"turretQE={response_body['turretQE']} | "
        f"turretRF={response_body['turretRF']} | "
        f"fire={response_body['fire']} | "
        f"vision={vision_target.get('className') if vision_target else None} | "
        f"target={alignment.get('target', {}).get('candidateLabel') if alignment.get('target') else None} | "
        f"reason={alignment.get('reason')}"
    )

    return jsonify(response_body)


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    data = request.get_json(force=True, silent=True) or {}
    marker = record_impact(data)
    return jsonify({"status": "OK", "message": "Bullet impact data received", "impact": marker})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    data = request.get_json(force=True, silent=True) or {}
    if nested_position(data) or any(key in data for key in ("angle", "distance", "range")):
        marker = record_impact(data)
        return jsonify({"status": "OK", "message": "Impact-like obstacle data received", "impact": marker})
    return jsonify({"status": "success", "message": "Obstacle data received"})


@app.route("/reset_state", methods=["POST", "GET"])
def reset_state():
    with state_lock:
        reset_runtime_state()
    return jsonify({"status": "success", "message": "v8 temporal state cleared"})


@app.route("/init", methods=["GET"])
def init():
    with state_lock:
        reset_runtime_state()

    return jsonify({
        "startMode": "start", "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True, "detectMode": True, "logMode": True,
        "stereoCameraMode": False, "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000, "destoryObstaclesOnHit": True,
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": "start"})


if __name__ == "__main__":
    print("Open v8.7 automatic body-alignment view: http://127.0.0.1:5000/lidar_view")
    print("Lightweight monitor JSON: http://127.0.0.1:5000/lidar_monitor_status")
    print("Full debug JSON: http://127.0.0.1:5000/lidar_status")
    print("YOLO fusion priority queue JSON: http://127.0.0.1:5000/priority_status")
    print("Automatic body alignment JSON: http://127.0.0.1:5000/body_align_status")
    print("GET_ACTION debug JSON: http://127.0.0.1:5000/action_debug")
    print("Disable body alignment: http://127.0.0.1:5000/body_align_enable?enabled=false")
    print("Front vertical profile JSON: http://127.0.0.1:5000/front_vertical_profile")
    print("Reset temporal state: http://127.0.0.1:5000/reset_state")
    app.run(host="0.0.0.0", port=5000, threaded=True)
