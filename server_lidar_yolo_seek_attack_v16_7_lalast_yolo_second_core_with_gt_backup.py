from __future__ import annotations

"""
Tank Challenge LiDAR-first YOLO Fusion Server v16.6 Top-View Hill/Flat Target Scan
=================================================

Design goals
------------
1. LiDAR remains the realtime primary sensor.
2. /info parses each LiDAR frame once, classifies ground/obstacles once, and
   caches NumPy arrays.
3. /detect always returns LiDAR overlay boxes immediately.
4. YOLO runs asynchronously at a slower configurable interval. Slow image
   inference must not block the LiDAR overlay.
5. YOLO labels are accepted as fused objects only when they match a LiDAR
   obstacle cluster.
6. Fused YOLO boxes are displayed only while fresh and while the turret has not
   rotated too far since the source image was captured.
7. Optional map ground truth compares exact registered object-center distance
   and body-relative bearing against the LiDAR estimate.
8. v16.1 adds YOLO env tuning, lalast.pt class alignment, and unified LiDAR dashboard.
9. Fire is allowed only after a fresh YOLO-fused tank confirmation and aim lock.

Recommended first benchmark
---------------------------
Simulator Properties:
- Mode: Simulation
- Request Port: 5000
- Interval: 0.2
- Y Position: 3
- Channel: 32
- Minimap Channel: 16
- Max Distance: 120
- Lidar Position: Body
- Send Detected Lidar: enabled
- Frame Rate Settings: 60
- Graphics Quality Settings: Medium

Put this file and lalast.pt in the same folder.
"""

import base64
import csv
import html
import json
import os
import uuid
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from math import cos, radians, sin, tan
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR: str | None = None
except Exception as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

app = Flask(__name__)
state_lock = Lock()

# ===========================================================================
# 1. Operator settings
# ===========================================================================
SERVER_PORT = 5000

EXPECTED_LIDAR_Y_POSITION_M = 3.0
EXPECTED_CHANNELS = 32
EXPECTED_MINIMAP_CHANNEL = 16
EXPECTED_MAX_DISTANCE_M = 120.0
EXPECTED_INTERVAL_SEC = 0.2

DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1057
MAX_LIDAR_DISTANCE_M = 120.0


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


# YOLO runtime tuning requested by user. These can be changed without editing
# code by setting environment variables before running the server.
YOLO_CONF = env_float("YOLO_CONF", 0.25)
YOLO_IOU = env_float("YOLO_IOU", 0.50)
YOLO_IMGSZ = env_int("YOLO_IMGSZ", 640)
YOLO_MAX_DET = env_int("YOLO_MAX_DET", 100)
YOLO_AUGMENT = env_bool("YOLO_AUGMENT", False)

# Ground/object separation.
GROUND_ANGLE_BIN_DEG = 2.0
GROUND_RANGE_BIN_M = 2.0
GROUND_HEIGHT_TOLERANCE_M = 0.30
GLOBAL_GROUND_PERCENTILE = 20.0

# Promote vertical stacks to obstacle surfaces.
STACK_ANGLE_BIN_DEG = 1.5
STACK_RANGE_BIN_M = 1.5
STACK_MIN_HEIGHT_SPAN_M = 0.45
STACK_MIN_POINT_COUNT = 2
STACK_EXPAND_ANGLE_BINS = 1
STACK_EXPAND_RANGE_BINS = 2

# v16: object validity from LiDAR vertical geometry.
# A traversable hill changes height smoothly along range. A real object usually
# creates a vertical stack at almost the same azimuth/range. Only the upper/high
# part of such vertical stacks is displayed and used as a valid target candidate.
VALID_OBJECT_STACK_MIN_SPAN_M = 0.60
VALID_OBJECT_STACK_MIN_POINTS = 2
VALID_OBJECT_MIN_ABOVE_STACK_BASE_M = 0.35
VALID_OBJECT_MIN_DISTANCE_M = 3.0
VALID_OBJECT_MAX_DISTANCE_M = 120.0
VALID_OBJECT_MIN_CLUSTER_POINTS = 2
VALID_OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG = 2.5
VALID_OBJECT_CLUSTER_MAX_DISTANCE_GAP_M = 5.0
VALID_OBJECT_CLUSTER_MAX_COUNT = 30
# v16.1: a true object surface in this simulator often appears as a near-vertical
# plane: large height span at almost the same azimuth/range. Smooth hills change
# height over range and fail this verticality test.
VALID_OBJECT_MAX_RANGE_SPAN_M = env_float("VALID_OBJECT_MAX_RANGE_SPAN_M", 1.35)
VALID_OBJECT_MIN_VERTICALITY_RATIO = env_float("VALID_OBJECT_MIN_VERTICALITY_RATIO", 0.85)

# v16.5 hill/object filter. Empty hills form a smooth terrain profile in the
# centerline side view. Objects sitting on that hill create points that rise
# above the local terrain profile at almost the same azimuth/range.
TERRAIN_PROFILE_ANGLE_BIN_DEG = env_float("TERRAIN_PROFILE_ANGLE_BIN_DEG", 2.0)
TERRAIN_PROFILE_RANGE_BIN_M = env_float("TERRAIN_PROFILE_RANGE_BIN_M", 1.0)
TERRAIN_GROUND_RESIDUAL_TOL_M = env_float("TERRAIN_GROUND_RESIDUAL_TOL_M", 0.70)
OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M = env_float("OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M", 0.95)
OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M = env_float("OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M", 1.05)
OBJECT_ON_HILL_MIN_HIGH_POINTS = env_int("OBJECT_ON_HILL_MIN_HIGH_POINTS", 2)
OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO = env_float("OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO", 0.25)
OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO = env_float("OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO", 0.80)

# v16.6: keep flat-ground objects too.  Flat vehicles can hide the ground
# directly below them, so their terrain residual may be weak even though the
# LiDAR stack is clearly object-like.
FLAT_OBJECT_MIN_HEIGHT_SPAN_M = env_float("FLAT_OBJECT_MIN_HEIGHT_SPAN_M", 0.90)
FLAT_OBJECT_MIN_POINTS = env_int("FLAT_OBJECT_MIN_POINTS", 3)
FLAT_OBJECT_MIN_VERTICALITY_RATIO = env_float("FLAT_OBJECT_MIN_VERTICALITY_RATIO", 0.95)
FLAT_OBJECT_MAX_RANGE_SPAN_M = env_float("FLAT_OBJECT_MAX_RANGE_SPAN_M", 1.20)
TARGET_AIM_HEIGHT_RATIO = env_float("TARGET_AIM_HEIGHT_RATIO", 0.45)
TARGET_AIM_MIN_CLEARANCE_M = env_float("TARGET_AIM_MIN_CLEARANCE_M", 0.55)

# Local ground-plane estimation for slope / tilt compensation.
# The simulator payload used so far exposes yaw and turret pitch, but not a
# reliable chassis roll/pitch pair.  Estimate a local road normal from nearby
# LiDAR ground points and use it for screen projection.
GROUND_PLANE_MIN_RANGE_M = 3.0
GROUND_PLANE_MAX_RANGE_M = 32.0
GROUND_PLANE_MAX_SAMPLE_POINTS = 600
GROUND_PLANE_MIN_SAMPLE_POINTS = 20
GROUND_PLANE_RESIDUAL_LIMIT_M = 0.35

# LiDAR-only obstacle clusters.
CLUSTER_MAX_ANGLE_GAP_DEG = 2.5
CLUSTER_MAX_DISTANCE_GAP_M = 5.0
CLUSTER_MIN_POINTS = 2
CLUSTER_MAX_COUNT = 40

# LiDAR point UI budget. This does NOT reduce the points used for recognition.
POINT_RADIUS_PX = 2
POINT_CLASS_NAME = " "
UPDATE_BOX_WHILE_MOVING = False

COLOR_OBSTACLE_NEAR = "#FF2020"   # <= 20 m
COLOR_OBSTACLE_MID = "#FFD23F"    # <= 50 m
COLOR_OBSTACLE_FAR = "#34D058"    # > 50 m
COLOR_SAFE_GROUND = "#45C96B"

overlay_settings: dict[str, Any] = {
    "showLidarPoints": True,
    "showSafeGround": False,
    "obstacleBoxLimit": 260,
    "safeGroundBoxLimit": 16,
    "totalLidarBoxLimit": 300,
    "obstaclePixelCell": 4,
    "safeGroundPixelCell": 28,
}

# Overlay calibration. LiDAR height is 3.0 m; the camera is assumed to be
# around 3.03 m, so cameraOffsetUpM starts at +0.03 m.
calibration: dict[str, Any] = {
    "cameraHorizontalFovDeg": 48.0,
    "cameraVerticalFovDeg": 28.0,
    "cameraOffsetForwardM": 0.0,
    "cameraOffsetRightM": 0.0,
    "cameraOffsetUpM": 0.03,
    "yawOffsetDeg": 0.0,
    "pitchOffsetDeg": 0.0,
    "screenCenterOffsetXPx": 0.0,
    "screenCenterOffsetYPx": 0.0,
    "cameraPoseMode": "same_frame_info",  # same_frame_info | latest_action | auto
    "latestActionFreshnessSec": 0.75,
    "turretYawMode": "absolute",          # absolute | body_plus_relative
    "turretYawSign": 1.0,
    "turretPitchSign": 1.0,

    # off | ground_plane
    # ground_plane estimates local slope from LiDAR ground returns. This is a
    # practical fallback when the simulator does not send chassis roll/pitch.
    "tiltCompensationMode": "ground_plane",
    "tiltSmoothingAlpha": 0.28,
    "maxGroundTiltDeg": 22.0,
    "rollOffsetDeg": 0.0,
}

# YOLO runs slower than LiDAR and never blocks /detect.
BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / os.getenv("YOLO_MODEL_FILE", "lalast.pt")

fusion_settings: dict[str, Any] = {
    "enabled": True,
    "modelPath": str(YOLO_MODEL_PATH),
    "confidence": YOLO_CONF,
    "iou": YOLO_IOU,
    "imageSize": YOLO_IMGSZ,
    "augment": YOLO_AUGMENT,
    "device": "auto",               # auto | cpu | 0
    "yoloIntervalSec": 0.50,        # async cadence for YOLO debug visibility

    # Primary fusion: use projected LiDAR obstacle points inside the YOLO bbox.
    # This is more stable on slopes than selecting a broad angle window.
    "roiFusionEnabled": True,
    "roiExpandRatio": 0.08,
    "roiMinObstaclePoints": 2,
    "roiSurfaceBandM": 3.0,

    # Fallback: use LiDAR-only obstacle clusters when ROI points are sparse.
    "clusterFallbackEnabled": True,
    "maxFusionAngleGapDeg": 12.0,

    # A stale YOLO box should disappear rather than drift away from the object
    # while the turret rotates.
    "maxDisplayAgeSec": 3.00,
    "maxDisplayYawDeltaDeg": 25.0,
    "maxDisplayPitchDeltaDeg": 10.0,
    "maxDisplayPositionDeltaM": 1.20,
    "maxDisplayGroundNormalDeltaDeg": 5.0,

    "showFusedBoxes": True,
    "showUnmatchedYoloBoxes": True,

    # v16.3 display/debug help:
    # If a YOLO box is not LiDAR-fused, still show its body-relative angle and
    # the nearest LiDAR-cluster hint in the label.  This is only a hint; firing
    # still requires a real fused tank confirmation.
    "showYoloOnlyAngleLabel": True,
    "showYoloOnlyLidarHint": True,
    "yoloOnlyHintAngleGateDeg": 20.0,

    # Runtime controls. A 22 MB YOLO model can be slower on CPU; keep the
    # LiDAR path realtime and let YOLO run asynchronously.
    "maxDetections": YOLO_MAX_DET,
    "halfPrecisionAuto": True,

    # best_8s.pt has a generic "tank" class. Keep this as "tank" unless the
    # current scenario guarantees that every visible tank is an enemy.
    "tankDisplayName": "Tank_enemy",

    # Experimental workaround only:
    # The current model sometimes predicts a visible tracked tank as car2.
    # When enabled, a fused car2 box with wide / tall LiDAR ROI geometry is
    # displayed as "tank?" rather than silently claiming a confirmed tank.
    # True tank recognition still requires additional model training.
    "tankCandidateRescueEnabled": False,
    "tankCandidateSourceClasses": "car",
    "tankCandidateDisplayName": "tank?",
    "tankRescueMinWidthM": 2.80,
    "tankRescueMinHeightSpanM": 0.45,
    "tankRescueMinRoiPoints": 4,
    "tankRescueMinBoxAspectRatio": 1.25,
}

# lalast.pt classes.
#
# Embedded checkpoint metadata found in the uploaded model:
#   0 Ally, 1 Enemy, 2 House, 3 Rock, 4 Rock_L, 5 Tank_enemy, 6 Tent, 7 car
#
# MODEL_CLASS_NAMES is only a fallback. During inference, result.names from
# the loaded .pt model takes priority, preventing silent class-ID mismatches.


def current_yolo_model_path() -> str:
    return str(fusion_settings.get("modelPath", YOLO_MODEL_PATH))

MODEL_CLASS_NAMES = {
    # lalast.pt embedded names from checkpoint metadata.
    0: "Ally",
    1: "Enemy",
    2: "House",
    3: "Rock",
    4: "Rock_L",
    5: "Tank_enemy",
    6: "Tent",
    7: "car",
}

CLASS_SEMANTIC = {
    # lalast.pt classes. Keep raw display names aligned with the .pt file,
    # but map semantics for targeting / fusion logic.
    "Ally": "ally",
    "Enemy": "enemy",
    "House": "house",
    "Rock": "rock",
    "Rock_L": "rock",
    "Tank_enemy": "enemy_tank",
    "Tent": "tent",
    "car": "car",

    # Lowercase / common aliases from model.names or older weights.
    "ally": "ally",
    "enemy": "enemy",
    "house": "house",
    "rock": "rock",
    "rock_l": "rock",
    "tank_enemy": "enemy_tank",
    "tent": "tent",

    # Backward-compatible aliases for earlier experiments.
    "Car002": "car",
    "House002": "house",
    "Human003": "human",
    "Rock001": "rock",
    "Tank001": "enemy_tank",
    "Tent001": "tent",
    "car1": "car1",
    "car2": "car2",
    "human": "human",
    "tank": "tank",
    "Tank_ally": "ally_tank",
    "Tank_enemy": "enemy_tank",
    "Tank_001": "enemy_tank",
}

BULKY_SEMANTICS = {
    "tank", "car1", "car2", "car",
    "rock", "house", "ally_tank", "enemy_tank", "tent",
}
THIN_SEMANTICS = {"human", "ally", "enemy"}

FUSED_COLORS = {
    # lalast.pt raw class names.
    "Ally": "#42A5F5",
    "Enemy": "#FF4DB8",
    "House": "#B0BEC5",
    "Rock": "#FF9F1C",
    "Rock_L": "#FFB74D",
    "Tank_enemy": "#FF3030",
    "Tent": "#B388FF",
    "car": "#00C853",
    "Car002": "#00C853",
    "House002": "#B0BEC5",
    "Human003": "#FF4DB8",
    "Rock001": "#FF9F1C",
    "Tank001": "#FF3030",
    "Tent001": "#B388FF",
    "tank": "#FF3030",
    "human": "#FF4DB8",
    "car1": "#00C853",
    "car2": "#00BFA5",
    "tank_candidate": "#FF8C00",

    # Backward-compatible aliases.
    "enemy_tank": "#FF3030",
    "ally_tank": "#00E5FF",
    "rock": "#FF9F1C",
    "tent": "#B388FF",
    "car": "#00C853",
    "enemy": "#FF4DB8",
    "ally": "#42A5F5",
    "unknown": "#FFFFFF",
}



# ===========================================================================
# 1-B. Optional map ground-truth comparison
# ===========================================================================
# A sensor estimate and a map ground-truth value are intentionally kept
# separate:
#   - LiDAR distance: nearest visible surface hit.
#   - GT center distance: exact distance to the registered object's map pivot.
#   - GT approximate surface distance: center distance - radiusM, only when a
#     radius approximation is supplied.
#
# Static objects can be loaded from ground_truth_objects.json.
# Dynamic objects need a live world-position feed from the simulator or from a
# debug endpoint.  A map's initial coordinate is not exact after an enemy tank
# moves.
GROUND_TRUTH_FILE = BASE_DIR / "ground_truth_objects.json"
GT_ERROR_LOG_FILE = BASE_DIR / "gt_error_log.csv"
GT_ACTIVE_MAP_SESSION_FILE = BASE_DIR / "active_map_gt_session.json"

SERVER_STARTED_AT = datetime.now().isoformat(timespec="seconds")
SERVER_SESSION_ID = uuid.uuid4().hex[:12]
SERVER_PROCESS_ID = os.getpid()

ground_truth_settings: dict[str, Any] = {
    "enabled": True,
    "filePath": str(GROUND_TRUTH_FILE),
    "autoLoadFile": True,
    "autoExtractInfo": True,
    "autoExtractObstacleUpdate": True,
    "autoExtractCollision": True,
    "showComparisonInLabel": True,
    "showErrorInLabel": True,
    "showMissingGtInLabel": True,
    "showGtObjectIdInLabel": False,
    "errorLogEnabled": True,
    "errorLogMinIntervalSec": 0.50,

    # Matching thresholds: estimated LiDAR object -> registered GT object.
    "matchMaxAngleGapDeg": 18.0,
    "matchMaxRangeGapM": 35.0,
    "rangeWeight": 0.25,
    "classMismatchPenalty": 5.0,
    "strictClassMatch": False,

    # Dynamic GT objects must be refreshed. Static map objects never expire.
    "dynamicObjectTtlSec": 3.0,

    # World bearing convention. Default assumes +Z is world-forward and +X is
    # world-right when yaw=0. Change only after a one-object calibration test.
    "worldForwardAxis": "+z",  # +z | -z | +x | -x
    "bodyYawSign": 1.0,
    "bodyYawOffsetDeg": 0.0,

    # Optional direct .map loader. The .map file stores exact obstacle pivots.
    "activeMapFile": None,
}

KNOWN_GT_CONTAINER_KEYS = {
    "objects",
    "obstacles",
    "targets",
    "enemies",
    "allies",
    "mapObjects",
    "map_objects",
    "groundTruth",
    "ground_truth",
    "groundTruthObjects",
    "ground_truth_objects",
}

# ===========================================================================
# 2. Cached state
# ===========================================================================
@dataclass(frozen=True)
class FrameCache:
    seq: int
    simulation_time: Any
    pose: dict[str, Any]
    angles: np.ndarray
    vertical_angles: np.ndarray
    distances: np.ndarray
    horizontal_ranges: np.ndarray
    channels: np.ndarray
    xyz: np.ndarray
    ground_mask: np.ndarray
    obstacle_mask: np.ndarray
    valid_object_mask: np.ndarray
    stack_promoted_mask: np.ndarray
    terrain_y: np.ndarray
    height_above_terrain: np.ndarray
    ground_normal: np.ndarray
    ground_plane_debug: dict[str, Any]
    clusters: tuple[dict[str, Any], ...]
    analysis_ms: float
    raw_point_count: int
    detected_hit_count: int


@dataclass(frozen=True)
class VisionJob:
    image_bytes: bytes
    width: int
    height: int
    cache: FrameCache
    turret_state: dict[str, Any]
    submitted_monotonic: float
    submitted_at: str


EMPTY_CACHE = FrameCache(
    seq=0,
    simulation_time=None,
    pose={},
    angles=np.empty(0, dtype=np.float32),
    vertical_angles=np.empty(0, dtype=np.float32),
    distances=np.empty(0, dtype=np.float32),
    horizontal_ranges=np.empty(0, dtype=np.float32),
    channels=np.empty(0, dtype=np.int16),
    xyz=np.empty((0, 3), dtype=np.float32),
    ground_mask=np.empty(0, dtype=bool),
    obstacle_mask=np.empty(0, dtype=bool),
    valid_object_mask=np.empty(0, dtype=bool),
    stack_promoted_mask=np.empty(0, dtype=bool),
    terrain_y=np.empty(0, dtype=np.float32),
    height_above_terrain=np.empty(0, dtype=np.float32),
    ground_normal=np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
    ground_plane_debug={"status": "empty"},
    clusters=(),
    analysis_ms=0.0,
    raw_point_count=0,
    detected_hit_count=0,
)

latest_cache: FrameCache = EMPTY_CACHE
tilt_state: dict[str, Any] = {
    "smoothedGroundNormal": np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
    "updatedAt": None,
}
latest_turret: dict[str, Any] = {
    "x": 0.0,
    "y": 0.0,
    "updatedAt": None,
    "updatedMonotonic": None,
}

latest_player_state: dict[str, Any] = {
    "position": None,
    "updatedAt": None,
    "updatedMonotonic": None,
}

aim_settings: dict[str, Any] = {
    "enabled": True,
    "autoFireEnabled": True,
    "candidateSort": "tank_first_then_nearest",
    "maxCandidateDistanceM": 120.0,
    "minCandidateDistanceM": 3.0,
    "yawDeadbandDeg": 1.2,
    "pitchDeadbandDeg": 2.2,
    "yawCommandWeight": 0.42,
    "pitchCommandWeight": 0.34,
    "yawRightCommand": "E",
    "yawLeftCommand": "Q",
    "pitchUpCommand": "R",
    "pitchDownCommand": "F",
    "turretYawMode": "absolute",  # absolute: turret.x is world yaw; relative: body-relative
    "targetConfirmMaxAgeSec": 3.0,
    "targetYoloAngleGateDeg": 8.0,
    "targetYoloDistanceGateM": 12.0,
    "fireYawGateDeg": 1.8,
    "firePitchGateDeg": 2.8,
    "fireCooldownSec": 0.6,
    "fireOnTankCandidate": False,

    # v16.3 target-scan behavior:
    # 1) If a fresh YOLO-fused tank exists, aim it before nearest unknown LiDAR object.
    # 2) If the turret has looked at a LiDAR object and YOLO still sees nothing,
    #    treat it as hill / unlabeled terrain and skip it briefly.
    "tankPriorityEnabled": True,
    "tankPriorityAngleGateDeg": 10.0,
    "tankPriorityDistanceGateM": 15.0,
    "skipNoYoloAfterDwell": True,
    "noYoloDwellSec": 1.2,
    "nonTankIgnoreSec": 10.0,

    # v16.3: avoid car -> house -> car -> house loops.  The exact cluster key
    # can jitter slightly frame-to-frame, so skip by both exact key and a coarse
    # angle/range sector.  This keeps the scan moving outward to farther objects.
    "useCoarseIgnoreKey": True,
    "ignoreAngleBinDeg": 8.0,
    "ignoreDistanceBinM": 12.0,

    # v16.4 anti-hunt turret control:
    # LiDAR cluster angle can jitter by a few degrees frame-to-frame.  Smooth the
    # target angle and reduce Q/E, R/F command weight near zero error so the
    # turret does not overshoot and bounce left/right on the same object.
    "aimTargetSmoothingEnabled": True,
    "aimTargetSmoothingAlpha": 0.30,
    "aimTargetSmoothingResetSec": 1.00,
    "proportionalAimControl": True,
    "yawSlowdownErrorDeg": 12.0,
    "pitchSlowdownErrorDeg": 10.0,
    "minYawCommandWeight": 0.10,
    "minPitchCommandWeight": 0.08,
    "suppressReverseCommandNearLock": True,
    "yawReverseSuppressDeg": 3.0,
    "pitchReverseSuppressDeg": 4.0,

    # v16.7: 360-degree LiDAR top view + pitch-sweep tank firing.
    "hillObjectMinTopClearanceM": OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M,
    "hillObjectMinClusterHeightM": OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M,
    "flatObjectFallbackEnabled": True,
    "flatObjectMinHeightSpanM": FLAT_OBJECT_MIN_HEIGHT_SPAN_M,
    "flatObjectMinPoints": FLAT_OBJECT_MIN_POINTS,
    "flatObjectMinVerticalityRatio": FLAT_OBJECT_MIN_VERTICALITY_RATIO,
    "flatObjectMaxRangeSpanM": FLAT_OBJECT_MAX_RANGE_SPAN_M,
    "scanYoloFusedObjectsEnabled": True,
    "targetAimHeightRatio": TARGET_AIM_HEIGHT_RATIO,
    "targetAimMinClearanceM": TARGET_AIM_MIN_CLEARANCE_M,

    # v16.7: tank-hit tuning.  The LiDAR/YOLO fusion estimates a geometric aim
    # pitch, then adds a small distance-based lift and cycles a narrow vertical
    # bracket after each shot.  This makes the gun raise/lower around the tank
    # center instead of firing every round at one brittle pitch value.
    "ballisticPitchCompEnabled": True,
    "ballisticPitchBaseOffsetDeg": 0.15,
    "ballisticPitchMaxOffsetDeg": 2.6,
    "ballisticPitchStartDistanceM": 25.0,
    "ballisticPitchFullDistanceM": 120.0,
    "pitchSweepEnabled": True,
    "pitchSweepOnlyConfirmedTank": True,
    "pitchSweepOffsetsDeg": "0,0.6,-0.6,1.2,-1.2,1.8,-1.8,2.4,-2.4",
    "fireYawGateDeg": 1.0,
    "firePitchGateDeg": 1.2,
}

aim_state: dict[str, Any] = {
    "mode": "idle",
    "updatedAt": None,
    "candidateCount": 0,
    "candidates": [],
    "selectedTarget": None,
    "confirmedTarget": None,
    "ignoredCandidateKeys": {},
    "alignedSinceByKey": {},
    "smoothedTargetByKey": {},
    "pitchSweepState": {},
    "lastYawCommandDirection": 0,
    "lastPitchCommandDirection": 0,
    "lastSkippedCandidate": None,
    "checkedCandidateHistory": [],
    "yawErrorDeg": None,
    "pitchErrorDeg": None,
    "action": empty_action() if 'empty_action' in globals() else None,
    "debug": {},
}

fire_state: dict[str, Any] = {
    "fireCount": 0,
    "lastFireAt": None,
    "lastFireMonotonic": None,
    "lastFireTarget": None,
    "lastBlockedReason": None,
}

ground_truth_state: dict[str, Any] = {
    "objects": {},
    "lastLoadAt": None,
    "lastLoadError": None,
    "lastRegisterAt": None,
    "lastComparisonAt": None,
    "lastComparisons": [],
    "payloadDebug": {
        "info": None,
        "get_action": None,
        "update_obstacle": None,
        "collision": None,
    },
    "autoExtractedCount": 0,
    "activeMapFile": None,
    "activeMapTerrainIndex": None,
    "lastMapLoadAt": None,
    "lastMapLoadError": None,
    "lastMapRegisteredCount": 0,
    "errorLogPath": str(GT_ERROR_LOG_FILE),
    "errorLogRowCount": 0,
    "lastLoggedPairAt": {},
    "serverStartedAt": SERVER_STARTED_AT,
    "serverSessionId": SERVER_SESSION_ID,
    "serverProcessId": SERVER_PROCESS_ID,
    "lastClearAt": None,
    "lastClearReason": None,
    "lastClearRemovedCount": 0,
    "lastAutoRestoreAt": None,
    "lastAutoRestoreResult": None,
}

status_state: dict[str, Any] = {
    "infoRequestCount": 0,
    "detectRequestCount": 0,
    "getActionRequestCount": 0,
    "lastInfoProcessingMs": None,
    "lastDetectProcessingMs": None,
    "lastReturnedLidarBoxCount": 0,
    "lastReturnedFusedBoxCount": 0,
    "lastProjectedPointCount": 0,
    "lastImageSize": None,
    "lastPoseSource": None,
    "lastCameraYawDeg": None,
    "lastCameraPitchDeg": None,
    "lastInfoTurretYawDeg": None,
    "lastInfoTurretPitchDeg": None,
    "lastActionTurretYawDeg": None,
    "lastActionTurretPitchDeg": None,
    "lastActionPoseAgeSec": None,
    "lastGroundTiltDeg": None,
    "lastGroundNormal": None,
    "lastInfoUpdatedAt": None,
    "lastDetectUpdatedAt": None,
}

yolo_state: dict[str, Any] = {
    "modelLoaded": False,
    "modelLoadError": ULTRALYTICS_IMPORT_ERROR,
    "modelNames": {},
    "workerBusy": False,
    "pendingJob": False,
    "submittedCount": 0,
    "completedCount": 0,
    "failedCount": 0,
    "replacedPendingJobCount": 0,
    "resolvedDevice": None,
    "resolvedHalfPrecision": None,
    "lastSubmittedMonotonic": None,
    "lastSubmittedAt": None,
    "lastCompletedAt": None,
    "lastInferenceMs": None,
    "lastFusionMs": None,
    "lastResultAgeSec": None,
    "latestYoloDetections": [],
    "latestFusedObjects": [],
    "latestResultMeta": {},
}

_yolo_model: Any = None
_pending_vision_job: VisionJob | None = None
_yolo_event = Event()


# ===========================================================================
# 3. Small helpers
# ===========================================================================
def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def normalize_signed_angle(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def angle_gap_deg(a: float, b: float) -> float:
    return abs(normalize_signed_angle(float(a) - float(b)))


def get_xyz(raw: Any) -> tuple[float, float, float] | None:
    if not isinstance(raw, dict):
        return None
    x = safe_float(raw.get("x"))
    y = safe_float(raw.get("y"))
    z = safe_float(raw.get("z"))
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def obstacle_color(distance_m: float) -> str:
    if distance_m <= 20.0:
        return COLOR_OBSTACLE_NEAR
    if distance_m <= 50.0:
        return COLOR_OBSTACLE_MID
    return COLOR_OBSTACLE_FAR


def empty_action() -> dict[str, Any]:
    return {
        "moveWS": {"command": "", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


def json_copy(value: Any) -> Any:
    """Small JSON-safe deep copy without copying NumPy arrays."""
    if isinstance(value, dict):
        return {key: json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_copy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ===========================================================================
# 4. One-pass LiDAR analysis
# ===========================================================================
def parse_detected_hits(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    raw_points = (
        data.get("lidarPoints")
        or data.get("lidar_points")
        or data.get("lidarData")
        or data.get("lidar")
        or []
    )
    if not isinstance(raw_points, list):
        raw_points = []

    angles: list[float] = []
    vertical_angles: list[float] = []
    distances: list[float] = []
    channels: list[int] = []
    xyz: list[tuple[float, float, float]] = []

    for raw in raw_points:
        if not isinstance(raw, dict):
            continue
        if "isDetected" in raw and not bool(raw.get("isDetected")):
            continue

        distance = safe_float(raw.get("distance", raw.get("range", raw.get("dist"))))
        angle = safe_float(raw.get("angle", raw.get("azimuth", raw.get("horizontalAngle"))))
        vertical_angle = safe_float(
            raw.get("verticalAngle", raw.get("vertical", raw.get("pitch"))),
            0.0,
        )
        position = raw.get("position") or raw.get("worldPosition") or {}
        point_xyz = get_xyz(
            {
                "x": position.get("x", raw.get("x")),
                "y": position.get("y", raw.get("y")),
                "z": position.get("z", raw.get("z")),
            }
        )

        if (
            distance is None
            or angle is None
            or vertical_angle is None
            or point_xyz is None
            or not (0.0 < float(distance) <= MAX_LIDAR_DISTANCE_M)
        ):
            continue

        angles.append(normalize_signed_angle(float(angle)))
        vertical_angles.append(float(vertical_angle))
        distances.append(float(distance))
        channel_raw = raw.get("channelIndex")
        channels.append(int(channel_raw) if channel_raw is not None else -1)
        xyz.append(point_xyz)

    if not distances:
        return {
            "angles": np.empty(0, dtype=np.float32),
            "vertical_angles": np.empty(0, dtype=np.float32),
            "distances": np.empty(0, dtype=np.float32),
            "horizontal_ranges": np.empty(0, dtype=np.float32),
            "channels": np.empty(0, dtype=np.int16),
            "xyz": np.empty((0, 3), dtype=np.float32),
        }, len(raw_points)

    angle_arr = np.asarray(angles, dtype=np.float32)
    vertical_arr = np.asarray(vertical_angles, dtype=np.float32)
    distance_arr = np.asarray(distances, dtype=np.float32)

    return {
        "angles": angle_arr,
        "vertical_angles": vertical_arr,
        "distances": distance_arr,
        "horizontal_ranges": distance_arr * np.cos(np.deg2rad(vertical_arr)),
        "channels": np.asarray(channels, dtype=np.int16),
        "xyz": np.asarray(xyz, dtype=np.float32),
    }, len(raw_points)


def reduced_group_stats(
    keys: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if keys.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_values = values[order]
    unique_keys, start = np.unique(sorted_keys, return_index=True)
    minimum = np.minimum.reduceat(sorted_values, start)
    maximum = np.maximum.reduceat(sorted_values, start)
    counts = np.diff(np.append(start, len(sorted_keys))).astype(np.int32)
    return unique_keys, minimum, maximum, counts


def classify_ground_and_obstacles(
    angles: np.ndarray,
    horizontal_ranges: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if angles.size == 0:
        empty = np.empty(0, dtype=bool)
        return empty, empty, empty

    y = xyz[:, 1]
    global_ground_y = float(np.percentile(y, GLOBAL_GROUND_PERCENTILE))

    angle_index = np.floor((angles + 180.0) / GROUND_ANGLE_BIN_DEG).astype(np.int32)
    range_index = np.floor(horizontal_ranges / GROUND_RANGE_BIN_M).astype(np.int32)
    ground_keys = angle_index.astype(np.int64) * 10000 + range_index.astype(np.int64)

    unique_ground_keys, min_y, _, _ = reduced_group_stats(ground_keys, y)
    lookup_index = np.searchsorted(unique_ground_keys, ground_keys)
    local_ground_y = min_y[lookup_index]
    local_ground_y = np.minimum(local_ground_y, global_ground_y + 0.80)

    stack_angle_index = np.floor((angles + 180.0) / STACK_ANGLE_BIN_DEG).astype(np.int32)
    stack_range_index = np.floor(horizontal_ranges / STACK_RANGE_BIN_M).astype(np.int32)
    stack_keys = stack_angle_index.astype(np.int64) * 10000 + stack_range_index.astype(np.int64)

    unique_stack_keys, stack_min_y, stack_max_y, stack_counts = reduced_group_stats(stack_keys, y)
    raw_stack_keys = unique_stack_keys[
        ((stack_max_y - stack_min_y) >= STACK_MIN_HEIGHT_SPAN_M)
        & (stack_counts >= STACK_MIN_POINT_COUNT)
    ]

    promoted_keys: set[int] = set()
    for packed in raw_stack_keys.tolist():
        a_index = int(packed // 10000)
        r_index = int(packed % 10000)
        for da in range(-STACK_EXPAND_ANGLE_BINS, STACK_EXPAND_ANGLE_BINS + 1):
            for dr in range(-STACK_EXPAND_RANGE_BINS, STACK_EXPAND_RANGE_BINS + 1):
                if r_index + dr >= 0:
                    promoted_keys.add((a_index + da) * 10000 + (r_index + dr))

    if promoted_keys:
        stack_promoted_mask = np.isin(
            stack_keys,
            np.fromiter(promoted_keys, dtype=np.int64),
        )
    else:
        stack_promoted_mask = np.zeros(angles.size, dtype=bool)

    height_above_local_ground = y - local_ground_y
    ground_mask = (
        (np.abs(height_above_local_ground) <= GROUND_HEIGHT_TOLERANCE_M)
        & (~stack_promoted_mask)
    )
    obstacle_mask = ~ground_mask
    return ground_mask, obstacle_mask, stack_promoted_mask



def estimate_terrain_profile_y(
    angles: np.ndarray,
    horizontal_ranges: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a smooth local hill profile y(range) per azimuth sector."""
    if angles.size == 0 or xyz.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, {"status": "empty"}

    y = xyz[:, 1].astype(np.float32)
    terrain_y = y.copy()
    angle_bin = max(0.25, float(TERRAIN_PROFILE_ANGLE_BIN_DEG))
    range_bin = max(0.25, float(TERRAIN_PROFILE_RANGE_BIN_M))
    angle_index = np.floor((angles + 180.0) / angle_bin).astype(np.int32)
    range_index = np.floor(horizontal_ranges / range_bin).astype(np.int32)
    keys = angle_index.astype(np.int64) * 100000 + range_index.astype(np.int64)

    unique_keys, min_y, _, _ = reduced_group_stats(keys, y)
    if unique_keys.size == 0:
        residual = y - terrain_y
        return terrain_y.astype(np.float32), residual.astype(np.float32), {"status": "no_keys"}

    unique_angle_index = (unique_keys // 100000).astype(np.int32)
    unique_range_index = (unique_keys % 100000).astype(np.int32)
    used_profiles = 0
    profile_samples = 0

    for a_index in np.unique(angle_index):
        point_mask = angle_index == a_index
        profile_mask = unique_angle_index == a_index
        if not np.any(point_mask) or not np.any(profile_mask):
            continue
        p_idx = np.flatnonzero(point_mask)
        r_centers = (unique_range_index[profile_mask].astype(np.float64) + 0.5) * range_bin
        p_y = min_y[profile_mask].astype(np.float64)
        order = np.argsort(r_centers)
        r_centers = r_centers[order]
        p_y = p_y[order]
        if r_centers.size >= 2:
            terrain_y[p_idx] = np.interp(
                horizontal_ranges[p_idx].astype(np.float64),
                r_centers,
                p_y,
            ).astype(np.float32)
        else:
            terrain_y[p_idx] = np.float32(p_y[0])
        used_profiles += 1
        profile_samples += int(r_centers.size)

    residual = (y - terrain_y).astype(np.float32)
    return terrain_y.astype(np.float32), residual, {
        "status": "ok",
        "method": "per_azimuth_lower_envelope_range_interp",
        "angleBinDeg": round(angle_bin, 3),
        "rangeBinM": round(range_bin, 3),
        "profileCount": int(used_profiles),
        "profileSampleCount": int(profile_samples),
        "groundResidualTolM": TERRAIN_GROUND_RESIDUAL_TOL_M,
        "objectMinTopClearanceM": OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M,
        "note": "empty hills stay near this profile; objects rise above it",
    }


def _counts_per_key(keys: np.ndarray, mask: np.ndarray, unique_keys: np.ndarray) -> np.ndarray:
    if keys.size == 0 or unique_keys.size == 0:
        return np.zeros(unique_keys.size, dtype=np.int32)
    selected_keys = keys[mask]
    if selected_keys.size == 0:
        return np.zeros(unique_keys.size, dtype=np.int32)
    selected_values = np.ones(selected_keys.size, dtype=np.float32)
    selected_unique, _, _, selected_counts = reduced_group_stats(selected_keys, selected_values)
    result = np.zeros(unique_keys.size, dtype=np.int32)
    positions = np.searchsorted(unique_keys, selected_unique)
    for pos, count in zip(positions.tolist(), selected_counts.tolist()):
        if 0 <= int(pos) < unique_keys.size:
            result[int(pos)] = int(count)
    return result


def compute_valid_object_mask(
    angles: np.ndarray,
    horizontal_ranges: np.ndarray,
    distances: np.ndarray,
    xyz: np.ndarray,
    obstacle_mask: np.ndarray,
    terrain_y: np.ndarray,
    height_above_terrain: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    v16.6: terrain-aware object filter.

    Two valid object modes are used:
      A) object_on_hill: upper points rise above the local hill profile.
      B) flat_object_fallback: a compact vertical LiDAR stack on flat ground,
         useful when the object hides the ground below it and terrain residual
         becomes weak.
    """
    if angles.size == 0:
        return np.empty(0, dtype=bool), {"status": "empty"}

    y = xyz[:, 1]
    if terrain_y.size != angles.size or height_above_terrain.size != angles.size:
        terrain_y = y.astype(np.float32)
        height_above_terrain = np.zeros(angles.size, dtype=np.float32)

    stack_angle_index = np.floor((angles + 180.0) / STACK_ANGLE_BIN_DEG).astype(np.int32)
    stack_range_index = np.floor(horizontal_ranges / STACK_RANGE_BIN_M).astype(np.int32)
    stack_keys = stack_angle_index.astype(np.int64) * 10000 + stack_range_index.astype(np.int64)

    unique_keys, min_y, max_y, counts = reduced_group_stats(stack_keys, y)
    _, min_r, max_r, _ = reduced_group_stats(stack_keys, horizontal_ranges)
    _, _, max_above, _ = reduced_group_stats(stack_keys, height_above_terrain)
    if unique_keys.size == 0:
        return np.zeros(angles.size, dtype=bool), {"status": "no_keys"}

    top_clearance = float(aim_settings.get("hillObjectMinTopClearanceM", OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M))
    cluster_height = float(aim_settings.get("hillObjectMinClusterHeightM", OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M))
    flat_enabled = bool(aim_settings.get("flatObjectFallbackEnabled", True))
    flat_span_min = float(aim_settings.get("flatObjectMinHeightSpanM", FLAT_OBJECT_MIN_HEIGHT_SPAN_M))
    flat_min_points = int(float(aim_settings.get("flatObjectMinPoints", FLAT_OBJECT_MIN_POINTS)))
    flat_vert_min = float(aim_settings.get("flatObjectMinVerticalityRatio", FLAT_OBJECT_MIN_VERTICALITY_RATIO))
    flat_range_max = float(aim_settings.get("flatObjectMaxRangeSpanM", FLAT_OBJECT_MAX_RANGE_SPAN_M))

    high_mask = height_above_terrain >= top_clearance
    ground_like_mask = height_above_terrain <= TERRAIN_GROUND_RESIDUAL_TOL_M
    high_counts = _counts_per_key(stack_keys, high_mask, unique_keys)
    ground_like_counts = _counts_per_key(stack_keys, ground_like_mask, unique_keys)

    lookup = np.searchsorted(unique_keys, stack_keys)
    local_min_y = min_y[lookup]
    local_max_y = max_y[lookup]
    local_count = counts[lookup]
    local_span = local_max_y - local_min_y
    local_range_span = max_r[lookup] - min_r[lookup]
    above_stack_base = y - local_min_y
    verticality_ratio = local_span / np.maximum(0.15, local_range_span)
    local_max_above = max_above[lookup]
    local_high_count = high_counts[lookup]
    local_ground_like_count = ground_like_counts[lookup]
    local_high_ratio = local_high_count / np.maximum(1, local_count)
    local_ground_like_ratio = local_ground_like_count / np.maximum(1, local_count)

    vertical_plane_bin = (
        (local_span >= VALID_OBJECT_STACK_MIN_SPAN_M)
        & (local_count >= VALID_OBJECT_STACK_MIN_POINTS)
        & (local_range_span <= VALID_OBJECT_MAX_RANGE_SPAN_M)
        & (verticality_ratio >= VALID_OBJECT_MIN_VERTICALITY_RATIO)
    )
    object_on_hill_bin = (
        vertical_plane_bin
        & (local_max_above >= cluster_height)
        & (local_high_count >= OBJECT_ON_HILL_MIN_HIGH_POINTS)
        & (local_high_ratio >= OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO)
        & (local_ground_like_ratio <= OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO)
    )

    # Flat-ground object fallback.  This is stricter on range thickness and
    # point count so an empty hill slope does not come back as an object.
    flat_object_bin = (
        flat_enabled
        & (local_span >= flat_span_min)
        & (local_count >= flat_min_points)
        & (local_range_span <= flat_range_max)
        & (verticality_ratio >= flat_vert_min)
        & (above_stack_base >= max(VALID_OBJECT_MIN_ABOVE_STACK_BASE_M, 0.45))
    )

    accepted_bin = object_on_hill_bin | flat_object_bin
    accepted_point = (
        ((height_above_terrain >= top_clearance) & object_on_hill_bin)
        | ((above_stack_base >= max(VALID_OBJECT_MIN_ABOVE_STACK_BASE_M, 0.45)) & flat_object_bin)
    )
    valid = (
        obstacle_mask
        & (distances >= VALID_OBJECT_MIN_DISTANCE_M)
        & (distances <= VALID_OBJECT_MAX_DISTANCE_M)
        & accepted_bin
        & accepted_point
    )

    bin_verticality = (max_y - min_y) / np.maximum(0.15, max_r - min_r)
    vertical_candidate_bin_mask = (
        ((max_y - min_y) >= VALID_OBJECT_STACK_MIN_SPAN_M)
        & (counts >= VALID_OBJECT_STACK_MIN_POINTS)
        & ((max_r - min_r) <= VALID_OBJECT_MAX_RANGE_SPAN_M)
        & (bin_verticality >= VALID_OBJECT_MIN_VERTICALITY_RATIO)
    )
    bin_high_ratio = high_counts / np.maximum(1, counts)
    bin_ground_like_ratio = ground_like_counts / np.maximum(1, counts)
    object_on_hill_candidate_bin_mask = (
        vertical_candidate_bin_mask
        & (max_above >= cluster_height)
        & (high_counts >= OBJECT_ON_HILL_MIN_HIGH_POINTS)
        & (bin_high_ratio >= OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO)
        & (bin_ground_like_ratio <= OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO)
    )
    flat_candidate_bin_mask = (
        flat_enabled
        & ((max_y - min_y) >= flat_span_min)
        & (counts >= flat_min_points)
        & ((max_r - min_r) <= flat_range_max)
        & (bin_verticality >= flat_vert_min)
    )
    return valid, {
        "status": "ok",
        "validPointCount": int(valid.sum()),
        "verticalPlaneBinCount": int(np.sum(vertical_candidate_bin_mask)),
        "objectOnHillBinCount": int(np.sum(object_on_hill_candidate_bin_mask)),
        "flatObjectBinCount": int(np.sum(flat_candidate_bin_mask)),
        "hillRejectedVerticalBinCount": int(max(0, np.sum(vertical_candidate_bin_mask) - np.sum(object_on_hill_candidate_bin_mask) - np.sum(flat_candidate_bin_mask))),
        "terrainGroundResidualTolM": TERRAIN_GROUND_RESIDUAL_TOL_M,
        "objectMinTopClearanceM": top_clearance,
        "objectMinClusterHeightM": cluster_height,
        "flatObjectFallbackEnabled": flat_enabled,
        "flatObjectMinHeightSpanM": flat_span_min,
        "flatObjectMinPoints": flat_min_points,
        "flatObjectMinVerticalityRatio": flat_vert_min,
        "flatObjectMaxRangeSpanM": flat_range_max,
        "maxRangeSpanM": VALID_OBJECT_MAX_RANGE_SPAN_M,
        "minVerticalityRatio": VALID_OBJECT_MIN_VERTICALITY_RATIO,
        "method": "terrain_residual_plus_flat_vertical_stack",
        "note": "hill objects must rise above terrain; flat objects can pass by compact vertical stack geometry",
    }

def make_valid_object_clusters(
    angles: np.ndarray,
    vertical_angles: np.ndarray,
    distances: np.ndarray,
    horizontal_ranges: np.ndarray,
    xyz: np.ndarray,
    valid_object_mask: np.ndarray,
    terrain_y: np.ndarray,
    height_above_terrain: np.ndarray,
    lidar_origin_y: float = EXPECTED_LIDAR_Y_POSITION_M,
) -> tuple[dict[str, Any], ...]:
    indices = np.flatnonzero(valid_object_mask)
    if indices.size == 0:
        return ()
    order = indices[np.argsort(angles[indices])]
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in order.tolist():
        if not current:
            current = [idx]
            continue
        prev = current[-1]
        if (
            abs(float(angles[idx] - angles[prev])) <= VALID_OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG
            and abs(float(distances[idx] - distances[prev])) <= VALID_OBJECT_CLUSTER_MAX_DISTANCE_GAP_M
        ):
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    if current:
        groups.append(current)

    clusters: list[dict[str, Any]] = []
    for cluster_id, group in enumerate(groups):
        if len(group) < VALID_OBJECT_MIN_CLUSTER_POINTS:
            continue
        idx = np.asarray(group, dtype=np.int32)
        group_angles = angles[idx]
        group_distances = distances[idx]
        group_ranges = horizontal_ranges[idx]
        group_vertical = vertical_angles[idx] if vertical_angles.size else np.zeros(idx.size, dtype=np.float32)
        group_xyz = xyz[idx]
        group_terrain_y = terrain_y[idx] if terrain_y.size == angles.size else group_xyz[:, 1]
        group_above = height_above_terrain[idx] if height_above_terrain.size == angles.size else np.zeros(idx.size, dtype=np.float32)
        surface_distance = float(np.percentile(group_distances, 20.0))
        median_distance = float(np.median(group_distances))
        far_distance = float(np.max(group_distances))
        median_range = float(np.median(group_ranges))
        median_angle = float(circular_mean_deg(group_angles))
        angular_span = max(0.0, float(group_angles.max() - group_angles.min()))
        visible_width = 2.0 * median_distance * tan(radians(max(0.5, angular_span) / 2.0))
        terrain_base_y = float(np.median(group_terrain_y)) if group_terrain_y.size else 0.0
        top_y = float(np.max(group_xyz[:, 1])) if group_xyz.size else terrain_base_y
        bottom_y = float(np.min(group_xyz[:, 1])) if group_xyz.size else terrain_base_y
        object_height_above_terrain = float(np.max(group_above)) if group_above.size else max(0.0, top_y - terrain_base_y)
        median_above_terrain = float(np.median(group_above)) if group_above.size else 0.0
        height_span = float(top_y - bottom_y) if group_xyz.size else 0.0
        depth_span = float(group_distances.max() - group_distances.min()) if group_distances.size else 0.0
        object_base_y = min(terrain_base_y, bottom_y)
        total_height_from_base = max(0.0, top_y - object_base_y)
        object_height_for_aim = max(object_height_above_terrain, total_height_from_base, height_span)
        aim_ratio = max(0.05, min(0.95, float(aim_settings.get("targetAimHeightRatio", TARGET_AIM_HEIGHT_RATIO))))
        min_clearance = max(0.0, float(aim_settings.get("targetAimMinClearanceM", TARGET_AIM_MIN_CLEARANCE_M)))
        aim_clearance = max(min_clearance, object_height_for_aim * aim_ratio)
        if object_height_for_aim > 0.01:
            aim_clearance = min(aim_clearance, object_height_for_aim * 0.90)
        aim_point_y = object_base_y + aim_clearance
        aim_pitch = float(np.degrees(np.arctan2(aim_point_y - float(lidar_origin_y), max(0.5, median_range))))
        verticality_ratio = object_height_above_terrain / max(0.15, depth_span)
        key = f"a{round(median_angle / 2.0) * 2:+.0f}_d{round(surface_distance / 5.0) * 5:.0f}"
        clusters.append({
            "clusterId": int(cluster_id),
            "candidateLabel": "OBJ_HILL",
            "candidateKey": key,
            "angleDeg": round(median_angle, 3),
            "distanceM": round(surface_distance, 3),
            "surfaceDistanceM": round(surface_distance, 3),
            "medianDistanceM": round(median_distance, 3),
            "farDistanceM": round(far_distance, 3),
            "horizontalRangeM": round(median_range, 3),
            "aimPitchDeg": round(aim_pitch, 3),
            "pointCount": int(len(group)),
            "visibleWidthM": round(float(visible_width), 3),
            "heightSpanM": round(height_span, 3),
            "objectHeightAboveTerrainM": round(object_height_above_terrain, 3),
            "medianHeightAboveTerrainM": round(median_above_terrain, 3),
            "objectTopYWorldM": round(top_y, 3),
            "objectBottomYWorldM": round(bottom_y, 3),
            "terrainBaseYWorldM": round(terrain_base_y, 3),
            "aimPointYWorldM": round(aim_point_y, 3),
            "aimHeightAboveBaseM": round(aim_clearance, 3),
            "objectBaseYWorldM": round(object_base_y, 3),
            "depthSpanM": round(depth_span, 3),
            "verticalityRatio": round(float(verticality_ratio), 3),
            "objectFilter": "terrain_residual_plus_vertical_plane",
        })
    clusters.sort(key=lambda item: (float(item["distanceM"]), abs(float(item["angleDeg"]))))
    return tuple(clusters[:VALID_OBJECT_CLUSTER_MAX_COUNT])

def normalize_vector(vector: np.ndarray, fallback: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (array / norm).astype(np.float32)


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = normalize_vector(a)
    b_n = normalize_vector(b)
    dot = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def estimate_local_ground_normal(
    xyz: np.ndarray,
    horizontal_ranges: np.ndarray,
    ground_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate a nearby road plane as y = ax + bz + c.

    This does not claim to measure the exact chassis suspension angle.
    It is a stable LiDAR-based approximation that keeps the screen projection
    aligned better while driving on a slope.
    """
    if xyz.size == 0 or ground_mask.size == 0:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {"status": "empty"}

    selected = (
        ground_mask
        & (horizontal_ranges >= GROUND_PLANE_MIN_RANGE_M)
        & (horizontal_ranges <= GROUND_PLANE_MAX_RANGE_M)
    )
    points = xyz[selected]

    if points.shape[0] < GROUND_PLANE_MIN_SAMPLE_POINTS:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {
            "status": "not_enough_ground_points",
            "sampleCount": int(points.shape[0]),
        }

    if points.shape[0] > GROUND_PLANE_MAX_SAMPLE_POINTS:
        step = max(1, points.shape[0] // GROUND_PLANE_MAX_SAMPLE_POINTS)
        points = points[::step][:GROUND_PLANE_MAX_SAMPLE_POINTS]

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)
    design = np.column_stack((x, z, np.ones_like(x)))

    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = np.abs(y - design @ coefficients)
        inliers = residuals <= GROUND_PLANE_RESIDUAL_LIMIT_M

        if int(inliers.sum()) >= GROUND_PLANE_MIN_SAMPLE_POINTS:
            coefficients, *_ = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)
            residuals = np.abs(y[inliers] - design[inliers] @ coefficients)

        a, b, _ = [float(value) for value in coefficients]
        normal = normalize_vector(np.asarray((-a, 1.0, -b), dtype=np.float64))
        if normal[1] < 0:
            normal = -normal

        tilt_deg = vector_angle_deg(normal, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
        max_tilt = float(calibration.get("maxGroundTiltDeg", 22.0))
        if tilt_deg > max_tilt:
            return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {
                "status": "tilt_rejected",
                "sampleCount": int(points.shape[0]),
                "estimatedTiltDeg": round(tilt_deg, 3),
                "maxGroundTiltDeg": round(max_tilt, 3),
            }

        return normal.astype(np.float32), {
            "status": "ok",
            "sampleCount": int(points.shape[0]),
            "estimatedTiltDeg": round(tilt_deg, 3),
            "normal": [round(float(value), 6) for value in normal.tolist()],
            "medianResidualM": round(float(np.median(residuals)), 4) if residuals.size else None,
        }
    except Exception as exc:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {
            "status": "fit_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def smooth_ground_normal(normal: np.ndarray) -> np.ndarray:
    if str(calibration.get("tiltCompensationMode", "ground_plane")) == "off":
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32)

    alpha = float(calibration.get("tiltSmoothingAlpha", 0.28))
    alpha = max(0.01, min(1.0, alpha))

    with state_lock:
        previous = np.asarray(
            tilt_state.get("smoothedGroundNormal", (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        smoothed = normalize_vector((1.0 - alpha) * previous + alpha * normal)
        tilt_state["smoothedGroundNormal"] = smoothed
        tilt_state["updatedAt"] = datetime.now().isoformat(timespec="milliseconds")
        return smoothed.copy()


def pose_position(pose: dict[str, Any]) -> np.ndarray | None:
    raw = get_xyz(pose.get("playerPos"))
    if raw is None:
        raw = get_xyz(pose.get("lidarOrigin"))
    return np.asarray(raw, dtype=np.float32) if raw is not None else None



def now_text() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def canonical_gt_class_name(value: Any) -> str:
    token = str(value or "object").strip().lower().replace(" ", "_")
    aliases = {
        "tank_enemy": "tank",
        "enemy_tank": "tank",
        "tank_ally": "tank",
        "ally_tank": "tank",
        "tank?": "tank",
        "tank_candidate": "tank",
        "person": "human",
        "enemy": "human",
        "ally": "human",
        "rock_l": "rock",
        "vehicle": "car",
        "car1": "car",
        "car2": "car",
        "target": "human",
        "tree": "tree",
        "house": "house",
        "tent": "tent",
    }
    return aliases.get(token, token)


def extract_position_dict(raw: Any) -> dict[str, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        x = safe_float(raw[0])
        y = safe_float(raw[1])
        z = safe_float(raw[2])
        if x is not None and y is not None and z is not None:
            return {"x": float(x), "y": float(y), "z": float(z)}
        return None

    if not isinstance(raw, dict):
        return None

    nested = (
        raw.get("position")
        or raw.get("worldPosition")
        or raw.get("world_position")
        or raw.get("pos")
        or raw.get("location")
    )
    if nested is not None and nested is not raw:
        result = extract_position_dict(nested)
        if result is not None:
            return result

    xyz = get_xyz(raw)
    if xyz is None:
        return None
    return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}


def looks_like_gt_object(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    position = extract_position_dict(raw)
    if position is None:
        return False
    identity_keys = {
        "id", "objectId", "objectID", "name", "objectName", "className",
        "class", "type", "category", "label",
    }
    return any(key in raw for key in identity_keys)


def iter_gt_candidates(raw: Any, allow_direct: bool = True):
    if isinstance(raw, list):
        for item in raw:
            yield from iter_gt_candidates(item, allow_direct=True)
        return

    if not isinstance(raw, dict):
        return

    if allow_direct and looks_like_gt_object(raw):
        yield raw

    for key, value in raw.items():
        if key in KNOWN_GT_CONTAINER_KEYS:
            yield from iter_gt_candidates(value, allow_direct=True)


def gt_object_id(raw: dict[str, Any], fallback_prefix: str) -> str:
    for key in ("id", "objectId", "objectID", "name", "objectName", "label"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    position = extract_position_dict(raw) or {"x": 0.0, "y": 0.0, "z": 0.0}
    class_name = raw.get("className", raw.get("class", raw.get("type", "object")))
    return (
        f"{fallback_prefix}:{class_name}:"
        f"{position['x']:.3f},{position['y']:.3f},{position['z']:.3f}"
    )


def register_gt_object(
    raw: dict[str, Any],
    source: str,
    dynamic_default: bool,
) -> dict[str, Any] | None:
    position = extract_position_dict(raw)
    if position is None:
        return None

    object_id = gt_object_id(raw, source)
    class_name = str(
        raw.get(
            "className",
            raw.get("class", raw.get("type", raw.get("category", raw.get("objectName", "object")))),
        )
    )
    radius = safe_float(
        raw.get("radiusM", raw.get("approxRadiusM", raw.get("radius"))),
        None,
    )
    dynamic = safe_bool(raw.get("dynamic"), dynamic_default)

    record = {
        "id": object_id,
        "className": class_name,
        "canonicalClass": canonical_gt_class_name(class_name),
        "position": position,
        "radiusM": float(radius) if radius is not None and radius >= 0 else None,
        "dynamic": bool(dynamic),
        "source": source,
        "updatedAt": now_text(),
        "updatedMonotonic": monotonic(),
    }
    with state_lock:
        ground_truth_state["objects"][object_id] = record
        ground_truth_state["lastRegisterAt"] = record["updatedAt"]
    return json_copy(record)


def ingest_gt_payload(
    payload: Any,
    source: str,
    dynamic_default: bool,
    allow_direct: bool = True,
) -> list[dict[str, Any]]:
    registered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in iter_gt_candidates(payload, allow_direct=allow_direct):
        record = register_gt_object(candidate, source=source, dynamic_default=dynamic_default)
        if record is None:
            continue
        object_id = str(record["id"])
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        registered.append(record)

    if registered and source.startswith("auto:"):
        with state_lock:
            ground_truth_state["autoExtractedCount"] += len(registered)
    return registered


def record_payload_debug(source: str, payload: Any) -> None:
    if source == "info" and isinstance(payload, dict):
        compact = {
            "updatedAt": now_text(),
            "keys": sorted(str(key) for key in payload.keys()),
            "candidateContainerKeys": sorted(
                str(key) for key in payload.keys() if key in KNOWN_GT_CONTAINER_KEYS
            ),
            "time": payload.get("time"),
        }
    else:
        compact = {
            "updatedAt": now_text(),
            "payload": json_copy(payload),
        }
    with state_lock:
        ground_truth_state["payloadDebug"][source] = compact


def persist_active_map_selection(path: Path) -> None:
    payload = {
        "path": str(path),
        "filename": path.name,
        "savedAt": now_text(),
        "serverSessionId": SERVER_SESSION_ID,
    }
    GT_ACTIVE_MAP_SESSION_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_persisted_map_selection() -> dict[str, Any] | None:
    if not GT_ACTIVE_MAP_SESSION_FILE.exists():
        return None
    try:
        payload = json.loads(GT_ACTIVE_MAP_SESSION_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def clear_gt_objects(
    reason: str = "manual",
    forget_persisted_map: bool = False,
) -> int:
    with state_lock:
        count = len(ground_truth_state["objects"])
        ground_truth_state["objects"] = {}
        ground_truth_state["lastComparisons"] = []
        ground_truth_state["lastClearAt"] = now_text()
        ground_truth_state["lastClearReason"] = str(reason)
        ground_truth_state["lastClearRemovedCount"] = count
        if forget_persisted_map:
            ground_truth_state["activeMapFile"] = None
            ground_truth_state["activeMapTerrainIndex"] = None
            ground_truth_state["lastMapRegisteredCount"] = 0
            ground_truth_settings["activeMapFile"] = None

    if forget_persisted_map and GT_ACTIVE_MAP_SESSION_FILE.exists():
        try:
            GT_ACTIVE_MAP_SESSION_FILE.unlink()
        except Exception:
            pass
    return count


def load_ground_truth_file(path_value: str | None = None, clear_existing: bool = False) -> dict[str, Any]:
    """
    Manual JSON loader.

    Safety rule:
    - An empty ground_truth_objects.json must NOT erase a map that was already
      loaded with /map_gt_load. This prevents an old browser tab containing
      /gt_reload?clearExisting=true from wiping the active map GT state.
    """
    path = Path(path_value or str(ground_truth_settings["filePath"]))

    if not path.exists():
        message = f"Ground-truth file not found: {path}"
        with state_lock:
            ground_truth_state["lastLoadAt"] = now_text()
            ground_truth_state["lastLoadError"] = message
        return {"status": "not_found", "path": str(path), "registeredCount": 0, "message": message}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = list(iter_gt_candidates(payload, allow_direct=False))

        if clear_existing and not candidates:
            with state_lock:
                existing_count = len(ground_truth_state.get("objects", {}))
                active_map = ground_truth_state.get("activeMapFile")
                ground_truth_state["lastLoadAt"] = now_text()
                ground_truth_state["lastLoadError"] = None
            return {
                "status": "protected_empty_file",
                "path": str(path),
                "registeredCount": 0,
                "existingObjectCountPreserved": existing_count,
                "activeMapFilePreserved": active_map,
                "message": (
                    "The JSON file contains no GT objects. Existing map GT was preserved. "
                    "Use /gt_clear?forgetMap=true only when you intentionally want to remove it."
                ),
            }

        if clear_existing:
            clear_gt_objects(reason=f"gt_reload:{path.name}", forget_persisted_map=False)

        records = ingest_gt_payload(
            payload,
            source=f"file:{path.name}",
            dynamic_default=False,
            allow_direct=False,
        )
        with state_lock:
            ground_truth_state["lastLoadAt"] = now_text()
            ground_truth_state["lastLoadError"] = None
        return {"status": "success", "path": str(path), "registeredCount": len(records)}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        with state_lock:
            ground_truth_state["lastLoadAt"] = now_text()
            ground_truth_state["lastLoadError"] = message
        return {"status": "error", "path": str(path), "registeredCount": 0, "message": message}



def infer_map_prefab_class(prefab_name: Any) -> str:
    token = str(prefab_name or "object").strip().lower()
    if "tank" in token:
        return "tank"
    if "rock" in token:
        return "rock"
    if "human" in token or "person" in token or "target" in token:
        return "human"
    if "car" in token or "truck" in token or "vehicle" in token:
        return "car"
    if "tent" in token:
        return "tent"
    if "tree" in token:
        return "tree"
    if "house" in token or "building" in token:
        return "house"
    return "object"


def infer_map_prefab_radius_m(prefab_name: Any, class_name: str) -> float | None:
    # Only an approximate center-to-surface radius. GT center coordinates are
    # exact map pivots; the mesh closest point is not available in Python.
    token = str(prefab_name or "").strip().lower()
    canonical = canonical_gt_class_name(class_name)
    if canonical == "tank":
        return 3.5
    if canonical == "rock":
        return 3.0 if "002" in token else 2.0
    if canonical == "car":
        return 2.4
    if canonical == "human":
        return 0.35
    if canonical == "tent":
        return 3.0
    if canonical == "tree":
        return 0.6
    if canonical == "house":
        return 5.0
    return None


def resolve_local_map_path(path_value: str | None = None, filename: str | None = None) -> Path:
    raw = str(path_value or filename or "").strip()
    if not raw:
        raise ValueError("Use filename=<your_map_file.map> or path=<full_path_to_map_file>.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def load_map_ground_truth(
    path_value: str | None = None,
    filename: str | None = None,
    clear_existing: bool = True,
    persist_selection: bool = True,
) -> dict[str, Any]:
    """
    Load exact static map pivots from a .map JSON file.

    Important:
    - Parse and validate the file BEFORE clearing existing GT.
    - Persist the selected map so a Python-server restart restores it.
    """
    try:
        path = resolve_local_map_path(path_value=path_value, filename=filename)
    except Exception as exc:
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}", "registeredCount": 0}

    if not path.exists():
        message = f"Map file not found: {path}"
        with state_lock:
            ground_truth_state["lastMapLoadAt"] = now_text()
            ground_truth_state["lastMapLoadError"] = message
            ground_truth_state["lastMapRegisteredCount"] = 0
        return {"status": "not_found", "path": str(path), "registeredCount": 0, "message": message}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        obstacles = payload.get("obstacles", [])
        if not isinstance(obstacles, list):
            raise ValueError("The map JSON does not contain an obstacles list.")

        prepared: list[dict[str, Any]] = []
        for index, obstacle in enumerate(obstacles):
            if not isinstance(obstacle, dict):
                continue
            prefab_name = str(obstacle.get("prefabName", obstacle.get("name", f"map_object_{index:04d}")))
            class_name = infer_map_prefab_class(prefab_name)
            radius = infer_map_prefab_radius_m(prefab_name, class_name)
            raw = {
                "id": prefab_name,
                "className": class_name,
                "position": obstacle.get("position"),
                "radiusM": radius,
                "dynamic": False,
                "prefabName": prefab_name,
                "rotation": obstacle.get("rotation"),
            }
            if extract_position_dict(raw) is not None:
                prepared.append(raw)

        if not prepared:
            message = "The map file was parsed, but no obstacle position records were found."
            with state_lock:
                ground_truth_state["lastMapLoadAt"] = now_text()
                ground_truth_state["lastMapLoadError"] = message
                ground_truth_state["lastMapRegisteredCount"] = 0
            return {
                "status": "no_obstacles",
                "path": str(path),
                "registeredCount": 0,
                "message": message,
            }

        if clear_existing:
            clear_gt_objects(reason=f"map_gt_load:{path.name}", forget_persisted_map=False)

        records: list[dict[str, Any]] = []
        for raw in prepared:
            record = register_gt_object(raw, source=f"map:{path.name}", dynamic_default=False)
            if record is not None:
                records.append(record)

        with state_lock:
            ground_truth_settings["activeMapFile"] = str(path)
            ground_truth_state["activeMapFile"] = str(path)
            ground_truth_state["activeMapTerrainIndex"] = payload.get("terrainIndex")
            ground_truth_state["lastMapLoadAt"] = now_text()
            ground_truth_state["lastMapLoadError"] = None
            ground_truth_state["lastMapRegisteredCount"] = len(records)

        if persist_selection:
            persist_active_map_selection(path)

        return {
            "status": "success",
            "path": str(path),
            "terrainIndex": payload.get("terrainIndex"),
            "registeredCount": len(records),
            "classCounts": count_registered_gt_classes(),
            "serverSessionId": SERVER_SESSION_ID,
            "serverProcessId": SERVER_PROCESS_ID,
            "persistedSelectionFile": str(GT_ACTIVE_MAP_SESSION_FILE),
            "note": (
                "Loaded exact obstacle map pivots and persisted the selected map. "
                "LiDAR still measures nearest visible surfaces, so compare LiDAR with GTc "
                "and optionally the approximate surface distance."
            ),
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        with state_lock:
            ground_truth_state["lastMapLoadAt"] = now_text()
            ground_truth_state["lastMapLoadError"] = message
            ground_truth_state["lastMapRegisteredCount"] = 0
        return {"status": "error", "path": str(path), "registeredCount": 0, "message": message}


def restore_persisted_map_ground_truth(force: bool = False) -> dict[str, Any]:
    selection = read_persisted_map_selection()
    if not selection:
        return {"status": "no_persisted_map"}

    with state_lock:
        existing_count = len(ground_truth_state.get("objects", {}))
    if existing_count > 0 and not force:
        return {"status": "already_loaded", "registeredCount": existing_count}

    path = str(selection.get("path", "")).strip()
    if not path:
        return {"status": "invalid_persisted_map", "selection": selection}

    result = load_map_ground_truth(
        path_value=path,
        clear_existing=True,
        persist_selection=False,
    )
    with state_lock:
        ground_truth_state["lastAutoRestoreAt"] = now_text()
        ground_truth_state["lastAutoRestoreResult"] = json_copy(result)
    return result


def ensure_map_gt_available() -> dict[str, Any] | None:
    with state_lock:
        existing_count = len(ground_truth_state.get("objects", {}))
    if existing_count > 0:
        return None
    return restore_persisted_map_ground_truth(force=False)


def count_registered_gt_classes() -> dict[str, int]:
    with state_lock:
        records = [json_copy(item) for item in ground_truth_state.get("objects", {}).values()]
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("canonicalClass", "object"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def error_log_header() -> list[str]:
    return [
        "loggedAt",
        "simulationTime",
        "frameSeq",
        "estimatedClass",
        "gtObjectId",
        "gtClass",
        "lidarDistanceM",
        "gtCenterDistanceM",
        "gtApproxSurfaceDistanceM",
        "distanceErrorToCenterM",
        "distanceErrorToApproxSurfaceM",
        "lidarBodyRelativeAngleDeg",
        "gtBodyRelativeAngleDeg",
        "angleErrorDeg",
        "classConsistent",
        "source",
    ]


def append_gt_error_log(
    comparison: dict[str, Any],
    cache: FrameCache,
) -> None:
    if not bool(ground_truth_settings.get("errorLogEnabled", True)):
        return

    gt = comparison.get("groundTruth") or {}
    object_id = str(gt.get("id", "unknown"))
    estimated_class = str(comparison.get("estimatedClass", "object"))
    pair_key = f"{estimated_class}::{object_id}"
    now_mono = monotonic()
    min_interval = max(0.0, float(ground_truth_settings.get("errorLogMinIntervalSec", 0.50)))

    with state_lock:
        previous = ground_truth_state.get("lastLoggedPairAt", {}).get(pair_key)
        if previous is not None and now_mono - float(previous) < min_interval:
            return
        ground_truth_state.setdefault("lastLoggedPairAt", {})[pair_key] = now_mono

    row = {
        "loggedAt": now_text(),
        "simulationTime": cache.simulation_time,
        "frameSeq": cache.seq,
        "estimatedClass": estimated_class,
        "gtObjectId": object_id,
        "gtClass": gt.get("className"),
        "lidarDistanceM": gt.get("lidarDistanceM"),
        "gtCenterDistanceM": gt.get("centerHorizontalDistanceM"),
        "gtApproxSurfaceDistanceM": gt.get("approxSurfaceDistanceM"),
        "distanceErrorToCenterM": gt.get("distanceErrorToCenterM"),
        "distanceErrorToApproxSurfaceM": gt.get("distanceErrorToApproxSurfaceM"),
        "lidarBodyRelativeAngleDeg": gt.get("lidarBodyRelativeAngleDeg"),
        "gtBodyRelativeAngleDeg": gt.get("bodyRelativeAngleDeg"),
        "angleErrorDeg": gt.get("angleErrorDeg"),
        "classConsistent": gt.get("classConsistent"),
        "source": gt.get("source"),
    }

    path = Path(str(ground_truth_state.get("errorLogPath", GT_ERROR_LOG_FILE)))
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_log_header())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    with state_lock:
        ground_truth_state["errorLogRowCount"] = int(ground_truth_state.get("errorLogRowCount", 0)) + 1


def gt_diagnosis(cache: FrameCache) -> dict[str, Any]:
    ensure_map_gt_available()
    with state_lock:
        registered_count = len(ground_truth_state.get("objects", {}))
        active_map = ground_truth_state.get("activeMapFile")
        map_error = ground_truth_state.get("lastMapLoadError")
    player = current_player_position(cache)
    active_count = len(active_gt_metrics(cache))

    if registered_count == 0:
        return {
            "code": "NO_GT_OBJECTS",
            "message": (
                "No map ground-truth objects are registered. "
                "Load the .map file with /map_gt_load?filename=YOUR_MAP.map or register coordinates manually."
            ),
            "activeMapFile": active_map,
            "lastMapLoadError": map_error,
        }
    if player is None:
        return {
            "code": "WAIT_PLAYER_POSITION",
            "message": "GT objects exist, but the player world position has not arrived yet. Start or restart the simulator.",
            "activeMapFile": active_map,
        }
    if active_count == 0:
        return {
            "code": "NO_ACTIVE_GT_OBJECTS",
            "message": "GT records exist, but none are active. Refresh dynamic object coordinates or reload the map.",
            "activeMapFile": active_map,
        }
    return {
        "code": "READY",
        "message": "Ground-truth comparison is ready. Look at /gt_dashboard or /gt_status.",
        "activeMapFile": active_map,
        "registeredCount": registered_count,
        "activeMetricCount": active_count,
    }


def decorate_missing_gt_label(copy_obj: dict[str, Any], reason: str) -> dict[str, Any]:
    if not bool(ground_truth_settings.get("showComparisonInLabel", True)):
        return copy_obj
    if not bool(ground_truth_settings.get("showMissingGtInLabel", True)):
        return copy_obj
    base_label = str(copy_obj.get("className", copy_obj.get("semanticClass", "object")))
    copy_obj["className"] = f"{base_label} | GT:{reason}"
    return copy_obj


def gt_forward_basis(axis: str) -> tuple[np.ndarray, np.ndarray]:
    token = str(axis or "+z").strip().lower()
    forward_by_axis = {
        "+z": np.asarray((0.0, 1.0), dtype=np.float64),
        "-z": np.asarray((0.0, -1.0), dtype=np.float64),
        "+x": np.asarray((1.0, 0.0), dtype=np.float64),
        "-x": np.asarray((-1.0, 0.0), dtype=np.float64),
    }
    forward = forward_by_axis.get(token, forward_by_axis["+z"])
    # Right basis in the horizontal XZ plane. With +Z forward, +X is right.
    right = np.asarray((forward[1], -forward[0]), dtype=np.float64)
    return forward, right


def gt_world_bearing_deg(dx: float, dz: float) -> float:
    forward, right = gt_forward_basis(str(ground_truth_settings.get("worldForwardAxis", "+z")))
    vector = np.asarray((float(dx), float(dz)), dtype=np.float64)
    forward_component = float(np.dot(vector, forward))
    right_component = float(np.dot(vector, right))
    return normalize_signed_angle(np.degrees(np.arctan2(right_component, forward_component)))


def current_player_position(cache: FrameCache) -> np.ndarray | None:
    position = pose_position(cache.pose)
    if position is not None:
        return position
    with state_lock:
        raw = ground_truth_state.get("latestPlayerPosition")
    if isinstance(raw, list) and len(raw) == 3:
        return np.asarray(raw, dtype=np.float32)
    return None


def gt_object_is_active(record: dict[str, Any]) -> bool:
    if not bool(record.get("dynamic", False)):
        return True
    updated = record.get("updatedMonotonic")
    if updated is None:
        return False
    ttl = max(0.1, float(ground_truth_settings.get("dynamicObjectTtlSec", 3.0)))
    return monotonic() - float(updated) <= ttl


def gt_metrics_for_record(record: dict[str, Any], cache: FrameCache) -> dict[str, Any] | None:
    player = current_player_position(cache)
    position = extract_position_dict(record.get("position"))
    if player is None or position is None:
        return None

    target = np.asarray((position["x"], position["y"], position["z"]), dtype=np.float64)
    delta = target - player.astype(np.float64)
    dx, dy, dz = [float(value) for value in delta.tolist()]
    horizontal = float(np.hypot(dx, dz))
    distance_3d = float(np.linalg.norm(delta))

    world_bearing = gt_world_bearing_deg(dx, dz)
    body_yaw = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
    relative_angle = normalize_signed_angle(
        float(ground_truth_settings.get("bodyYawSign", 1.0)) * (world_bearing - body_yaw)
        + float(ground_truth_settings.get("bodyYawOffsetDeg", 0.0))
    )

    radius = safe_float(record.get("radiusM"), None)
    approximate_surface = (
        max(0.0, horizontal - float(radius))
        if radius is not None and radius >= 0
        else None
    )

    return {
        "id": str(record["id"]),
        "className": str(record.get("className", "object")),
        "canonicalClass": str(record.get("canonicalClass", canonical_gt_class_name(record.get("className")))),
        "position": json_copy(position),
        "source": str(record.get("source", "unknown")),
        "dynamic": bool(record.get("dynamic", False)),
        "active": gt_object_is_active(record),
        "radiusM": float(radius) if radius is not None else None,
        "centerHorizontalDistanceM": round(horizontal, 4),
        "centerDistance3dM": round(distance_3d, 4),
        "approxSurfaceDistanceM": round(approximate_surface, 4) if approximate_surface is not None else None,
        "worldBearingDeg": round(world_bearing, 4),
        "bodyRelativeAngleDeg": round(relative_angle, 4),
    }


def active_gt_metrics(cache: FrameCache) -> list[dict[str, Any]]:
    with state_lock:
        records = [json_copy(item) for item in ground_truth_state["objects"].values()]
    metrics: list[dict[str, Any]] = []
    for record in records:
        if not gt_object_is_active(record):
            continue
        item = gt_metrics_for_record(record, cache)
        if item is not None:
            metrics.append(item)
    metrics.sort(key=lambda item: float(item["centerHorizontalDistanceM"]))
    return metrics


def estimated_class_for_gt_match(obj: dict[str, Any]) -> str:
    return canonical_gt_class_name(
        obj.get("originalRawClassName", obj.get("rawClassName", obj.get("semanticClass", "object")))
    )


def distance_reference_for_gt(gt: dict[str, Any]) -> float:
    surface = safe_float(gt.get("approxSurfaceDistanceM"), None)
    if surface is not None:
        return float(surface)
    return float(gt["centerHorizontalDistanceM"])


def attach_ground_truth_comparisons(
    fused_objects: list[dict[str, Any]],
    cache: FrameCache,
) -> list[dict[str, Any]]:
    ensure_map_gt_available()
    if not bool(ground_truth_settings.get("enabled", True)):
        return fused_objects

    available = {str(item["id"]): item for item in active_gt_metrics(cache)}
    comparisons: list[dict[str, Any]] = []
    decorated: list[dict[str, Any]] = []

    for obj in fused_objects:
        copy_obj = dict(obj)
        lidar_distance = safe_float(copy_obj.get("distance"), None)
        lidar_angle = safe_float(copy_obj.get("lidarBodyAngleDeg"), None)
        if lidar_distance is None or lidar_angle is None:
            decorated.append(decorate_missing_gt_label(copy_obj, "no-lidar"))
            continue
        if not available:
            decorated.append(decorate_missing_gt_label(copy_obj, "no-map"))
            continue

        estimated_class = estimated_class_for_gt_match(copy_obj)
        ranked: list[tuple[float, float, float, str, dict[str, Any], bool]] = []

        for gt_id, gt in available.items():
            angle_gap = angle_gap_deg(float(lidar_angle), float(gt["bodyRelativeAngleDeg"]))
            reference_distance = distance_reference_for_gt(gt)
            range_gap = abs(float(lidar_distance) - reference_distance)
            class_consistent = estimated_class == canonical_gt_class_name(gt.get("className"))

            if angle_gap > float(ground_truth_settings.get("matchMaxAngleGapDeg", 18.0)):
                continue
            if range_gap > float(ground_truth_settings.get("matchMaxRangeGapM", 35.0)):
                continue
            if bool(ground_truth_settings.get("strictClassMatch", False)) and not class_consistent:
                continue

            score = (
                angle_gap
                + float(ground_truth_settings.get("rangeWeight", 0.25)) * range_gap
                + (0.0 if class_consistent else float(ground_truth_settings.get("classMismatchPenalty", 5.0)))
            )
            ranked.append((score, angle_gap, range_gap, gt_id, gt, class_consistent))

        if not ranked:
            decorated.append(decorate_missing_gt_label(copy_obj, "no-match"))
            continue

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        _, angle_gap, range_gap, gt_id, gt, class_consistent = ranked[0]
        available.pop(gt_id, None)

        center_distance = float(gt["centerHorizontalDistanceM"])
        center_error = float(lidar_distance) - center_distance
        surface_distance = safe_float(gt.get("approxSurfaceDistanceM"), None)
        surface_error = (
            float(lidar_distance) - float(surface_distance)
            if surface_distance is not None
            else None
        )
        angle_error = normalize_signed_angle(float(lidar_angle) - float(gt["bodyRelativeAngleDeg"]))

        match = {
            **gt,
            "lidarDistanceM": round(float(lidar_distance), 4),
            "lidarBodyRelativeAngleDeg": round(float(lidar_angle), 4),
            "distanceErrorToCenterM": round(center_error, 4),
            "distanceErrorToApproxSurfaceM": round(surface_error, 4) if surface_error is not None else None,
            "angleErrorDeg": round(angle_error, 4),
            "matchAngleGapDeg": round(float(angle_gap), 4),
            "matchRangeGapM": round(float(range_gap), 4),
            "classConsistent": bool(class_consistent),
        }
        copy_obj["groundTruth"] = match

        if bool(ground_truth_settings.get("showComparisonInLabel", True)):
            base_label = str(copy_obj.get("className", copy_obj.get("semanticClass", "object")))
            gt_label = (
                f"GTc:{center_distance:.1f}m,"
                f"{float(gt['bodyRelativeAngleDeg']):+.1f}deg"
            )
            if bool(ground_truth_settings.get("showErrorInLabel", True)):
                gt_label += f" | e:{center_error:+.1f}m,{angle_error:+.1f}deg"
            if bool(ground_truth_settings.get("showGtObjectIdInLabel", False)):
                gt_label += f" | id:{gt_id}"
            copy_obj["className"] = f"{base_label} | {gt_label}"

        decorated.append(copy_obj)
        comparisons.append(
            {
                "estimatedClass": estimated_class,
                "displayClassName": copy_obj.get("className"),
                "groundTruth": match,
            }
        )
        append_gt_error_log(comparisons[-1], cache)

    with state_lock:
        ground_truth_state["lastComparisonAt"] = now_text()
        ground_truth_state["lastComparisons"] = json_copy(comparisons)
    return decorated


def make_obstacle_clusters(
    angles: np.ndarray,
    distances: np.ndarray,
    obstacle_mask: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    indices = np.flatnonzero(obstacle_mask)
    if indices.size == 0:
        return ()

    order = indices[np.argsort(angles[indices])]
    groups: list[list[int]] = []
    current: list[int] = []

    for idx in order.tolist():
        if not current:
            current = [idx]
            continue

        previous = current[-1]
        angle_gap = abs(float(angles[idx] - angles[previous]))
        distance_gap = abs(float(distances[idx] - distances[previous]))

        if angle_gap <= CLUSTER_MAX_ANGLE_GAP_DEG and distance_gap <= CLUSTER_MAX_DISTANCE_GAP_M:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

    if current:
        groups.append(current)

    clusters: list[dict[str, Any]] = []
    for cluster_id, group in enumerate(groups):
        if len(group) < CLUSTER_MIN_POINTS:
            continue

        group_idx = np.asarray(group, dtype=np.int32)
        group_angles = angles[group_idx]
        group_distances = distances[group_idx]
        median_distance = float(np.median(group_distances))
        angular_span = max(0.0, float(group_angles.max() - group_angles.min()))
        visible_width = 2.0 * median_distance * tan(radians(max(0.5, angular_span) / 2.0))

        candidate_label = "BK?" if visible_width >= 2.2 or len(group) >= 8 else "TH?"
        clusters.append(
            {
                "clusterId": int(cluster_id),
                "candidateLabel": candidate_label,
                "angleDeg": round(float(np.median(group_angles)), 3),
                "distanceM": round(median_distance, 3),
                "pointCount": len(group),
                "visibleWidthM": round(visible_width, 3),
            }
        )

    clusters.sort(
        key=lambda item: (
            0 if item["candidateLabel"] == "BK?" else 1,
            item["distanceM"],
        )
    )
    return tuple(clusters[:CLUSTER_MAX_COUNT])


def make_pose_subset(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": data.get("time"),
        "lidarOrigin": data.get("lidarOrigin", {}),
        "playerPos": data.get("playerPos", {}),
        "playerBodyX": data.get("playerBodyX", 0.0),
        "playerBodyY": data.get("playerBodyY"),
        "playerBodyZ": data.get("playerBodyZ"),
        "playerTurretX": data.get("playerTurretX", 0.0),
        "playerTurretY": data.get("playerTurretY", 0.0),
        "availableInfoKeys": sorted(str(key) for key in data.keys()),
    }


def build_frame_cache(data: dict[str, Any], seq: int) -> FrameCache:
    started = monotonic()
    arrays, raw_point_count = parse_detected_hits(data)
    ground_mask, obstacle_mask, stack_mask = classify_ground_and_obstacles(
        angles=arrays["angles"],
        horizontal_ranges=arrays["horizontal_ranges"],
        xyz=arrays["xyz"],
    )
    terrain_y, height_above_terrain, terrain_profile_debug = estimate_terrain_profile_y(
        angles=arrays["angles"],
        horizontal_ranges=arrays["horizontal_ranges"],
        xyz=arrays["xyz"],
    )
    valid_object_mask, object_filter_debug = compute_valid_object_mask(
        angles=arrays["angles"],
        horizontal_ranges=arrays["horizontal_ranges"],
        distances=arrays["distances"],
        xyz=arrays["xyz"],
        obstacle_mask=obstacle_mask,
        terrain_y=terrain_y,
        height_above_terrain=height_above_terrain,
    )

    estimated_normal, ground_plane_debug = estimate_local_ground_normal(
        xyz=arrays["xyz"],
        horizontal_ranges=arrays["horizontal_ranges"],
        ground_mask=ground_mask,
    )
    ground_normal = smooth_ground_normal(estimated_normal)

    pose_subset = make_pose_subset(data)
    lidar_origin_raw = get_xyz(pose_subset.get("lidarOrigin"))
    lidar_origin_y = float(lidar_origin_raw[1]) if lidar_origin_raw is not None else EXPECTED_LIDAR_Y_POSITION_M
    clusters = make_valid_object_clusters(
        angles=arrays["angles"],
        vertical_angles=arrays["vertical_angles"],
        distances=arrays["distances"],
        horizontal_ranges=arrays["horizontal_ranges"],
        xyz=arrays["xyz"],
        valid_object_mask=valid_object_mask,
        terrain_y=terrain_y,
        height_above_terrain=height_above_terrain,
        lidar_origin_y=lidar_origin_y,
    )

    ground_plane_debug = {
        **ground_plane_debug,
        "smoothedNormal": [round(float(value), 6) for value in ground_normal.tolist()],
        "smoothedTiltDeg": round(
            vector_angle_deg(ground_normal, np.asarray((0.0, 1.0, 0.0), dtype=np.float32)),
            3,
        ),
        "terrainProfile": terrain_profile_debug,
        "objectFilter": object_filter_debug,
    }

    return FrameCache(
        seq=seq,
        simulation_time=data.get("time"),
        pose=pose_subset,
        angles=arrays["angles"],
        vertical_angles=arrays["vertical_angles"],
        distances=arrays["distances"],
        horizontal_ranges=arrays["horizontal_ranges"],
        channels=arrays["channels"],
        xyz=arrays["xyz"],
        ground_mask=ground_mask,
        obstacle_mask=obstacle_mask,
        valid_object_mask=valid_object_mask,
        stack_promoted_mask=stack_mask,
        terrain_y=terrain_y,
        height_above_terrain=height_above_terrain,
        ground_normal=ground_normal,
        ground_plane_debug=ground_plane_debug,
        clusters=clusters,
        analysis_ms=round((monotonic() - started) * 1000.0, 2),
        raw_point_count=raw_point_count,
        detected_hit_count=int(arrays["distances"].size),
    )


# ===========================================================================
# 5. Camera projection
# ===========================================================================
def camera_angles(
    pose: dict[str, Any],
    turret_state: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    body_yaw = safe_float(pose.get("playerBodyX"), 0.0) or 0.0
    info_yaw = safe_float(pose.get("playerTurretX"), 0.0) or 0.0
    info_pitch = safe_float(pose.get("playerTurretY"), 0.0) or 0.0

    action_yaw = safe_float(turret_state.get("x"))
    action_pitch = safe_float(turret_state.get("y"))
    updated_monotonic = turret_state.get("updatedMonotonic")

    action_age_sec: float | None = None
    if updated_monotonic is not None:
        try:
            action_age_sec = max(0.0, monotonic() - float(updated_monotonic))
        except (TypeError, ValueError):
            action_age_sec = None

    freshness_limit = float(calibration.get("latestActionFreshnessSec", 0.75))
    action_is_fresh = (
        action_yaw is not None
        and action_pitch is not None
        and action_age_sec is not None
        and action_age_sec <= freshness_limit
    )

    pose_mode = str(calibration.get("cameraPoseMode", "same_frame_info")).strip().lower()
    if pose_mode == "latest_action" and action_is_fresh:
        raw_yaw = float(action_yaw)
        raw_pitch = float(action_pitch)
        pose_source = "latest_action"
    elif pose_mode == "auto" and action_is_fresh:
        raw_yaw = float(action_yaw)
        raw_pitch = float(action_pitch)
        pose_source = "latest_action_auto"
    else:
        raw_yaw = float(info_yaw)
        raw_pitch = float(info_pitch)
        pose_source = "same_frame_info"

    if str(calibration.get("turretYawMode", "absolute")) == "body_plus_relative":
        raw_yaw = body_yaw + raw_yaw

    yaw = (
        float(calibration.get("turretYawSign", 1.0)) * raw_yaw
        + float(calibration.get("yawOffsetDeg", 0.0))
    )
    pitch = (
        float(calibration.get("turretPitchSign", 1.0)) * raw_pitch
        + float(calibration.get("pitchOffsetDeg", 0.0))
    )

    debug = {
        "poseSource": pose_source,
        "cameraYawDeg": round(normalize_signed_angle(yaw), 3),
        "cameraPitchDeg": round(float(pitch), 3),
        "bodyYawDeg": round(float(body_yaw), 3),
        "infoTurretYawDeg": round(float(info_yaw), 3),
        "infoTurretPitchDeg": round(float(info_pitch), 3),
        "actionTurretYawDeg": round(float(action_yaw), 3) if action_yaw is not None else None,
        "actionTurretPitchDeg": round(float(action_pitch), 3) if action_pitch is not None else None,
        "actionPoseAgeSec": round(float(action_age_sec), 3) if action_age_sec is not None else None,
        "actionPoseFresh": bool(action_is_fresh),
    }
    return normalize_signed_angle(yaw), float(pitch), debug


def camera_basis(
    yaw_deg: float,
    pitch_deg: float,
    ground_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the camera frame.

    Flat mode:
      Same yaw/pitch basis used in previous versions.

    Ground-plane mode:
      Use the estimated nearby road normal as the camera up direction.
      This approximates chassis pitch and roll while climbing or leaning.
    """
    yaw = radians(yaw_deg)
    pitch = radians(pitch_deg)

    if str(calibration.get("tiltCompensationMode", "ground_plane")) == "off":
        base_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    else:
        base_up = normalize_vector(
            np.asarray(ground_normal if ground_normal is not None else (0.0, 1.0, 0.0)),
        )

    yaw_forward = np.asarray((sin(yaw), 0.0, cos(yaw)), dtype=np.float32)
    base_forward = yaw_forward - base_up * float(np.dot(yaw_forward, base_up))
    base_forward = normalize_vector(base_forward, fallback=(sin(yaw), 0.0, cos(yaw)))
    base_right = normalize_vector(np.cross(base_up, base_forward), fallback=(cos(yaw), 0.0, -sin(yaw)))
    base_up = normalize_vector(np.cross(base_forward, base_right))

    # Positive pitch looks upward, preserving the sign convention used earlier.
    forward = normalize_vector(base_forward * cos(pitch) + base_up * sin(pitch))
    up = normalize_vector(base_up * cos(pitch) - base_forward * sin(pitch))
    right = base_right

    # Optional final roll trim for manual calibration.
    roll = radians(float(calibration.get("rollOffsetDeg", 0.0)))
    if abs(roll) > 1e-9:
        right, up = (
            normalize_vector(right * cos(roll) + up * sin(roll)),
            normalize_vector(up * cos(roll) - right * sin(roll)),
        )

    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def camera_origin(
    pose: dict[str, Any],
    yaw_deg: float,
    pitch_deg: float,
    ground_normal: np.ndarray | None = None,
) -> np.ndarray | None:
    origin_raw = get_xyz(pose.get("lidarOrigin"))
    if origin_raw is None:
        origin_raw = get_xyz(pose.get("playerPos"))
    if origin_raw is None:
        return None

    right, up, forward = camera_basis(yaw_deg, pitch_deg, ground_normal)
    origin = np.asarray(origin_raw, dtype=np.float32)
    origin = origin + right * float(calibration.get("cameraOffsetRightM", 0.0))
    origin = origin + up * float(calibration.get("cameraOffsetUpM", 0.0))
    origin = origin + forward * float(calibration.get("cameraOffsetForwardM", 0.0))
    return origin


def project_cached_points(
    cache: FrameCache,
    turret_state: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    if cache.xyz.size == 0:
        return {
            "source_index": np.empty(0, dtype=np.int32),
            "x": np.empty(0, dtype=np.int32),
            "y": np.empty(0, dtype=np.int32),
        }

    yaw_deg, pitch_deg, _ = camera_angles(cache.pose, turret_state)
    origin = camera_origin(cache.pose, yaw_deg, pitch_deg, cache.ground_normal)
    if origin is None:
        return {
            "source_index": np.empty(0, dtype=np.int32),
            "x": np.empty(0, dtype=np.int32),
            "y": np.empty(0, dtype=np.int32),
        }

    right, up, forward = camera_basis(yaw_deg, pitch_deg, cache.ground_normal)
    delta = cache.xyz - origin

    x_cam = delta @ right
    y_cam = delta @ up
    z_cam = delta @ forward

    forward_mask = z_cam > 0.05
    if not np.any(forward_mask):
        return {
            "source_index": np.empty(0, dtype=np.int32),
            "x": np.empty(0, dtype=np.int32),
            "y": np.empty(0, dtype=np.int32),
        }

    source_index = np.flatnonzero(forward_mask)
    x_cam = x_cam[forward_mask]
    y_cam = y_cam[forward_mask]
    z_cam = z_cam[forward_mask]

    hfov = float(calibration.get("cameraHorizontalFovDeg", 48.0))
    vfov = float(calibration.get("cameraVerticalFovDeg", 28.0))
    fx = image_width / (2.0 * tan(radians(hfov / 2.0)))
    fy = image_height / (2.0 * tan(radians(vfov / 2.0)))
    cx = image_width / 2.0 + float(calibration.get("screenCenterOffsetXPx", 0.0))
    cy = image_height / 2.0 + float(calibration.get("screenCenterOffsetYPx", 0.0))

    x_px = np.rint(cx + fx * (x_cam / z_cam)).astype(np.int32)
    y_px = np.rint(cy - fy * (y_cam / z_cam)).astype(np.int32)
    inside = (
        (x_px >= 0)
        & (x_px < image_width)
        & (y_px >= 0)
        & (y_px < image_height)
    )

    return {
        "source_index": source_index[inside],
        "x": x_px[inside],
        "y": y_px[inside],
    }


def choose_nearest_per_pixel_cell(
    source_index: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    distances: np.ndarray,
    cell_size: int,
    limit: int,
) -> np.ndarray:
    if source_index.size == 0 or limit <= 0:
        return np.empty(0, dtype=np.int32)

    order = np.argsort(distances[source_index], kind="mergesort")
    src_sorted = source_index[order]
    x_sorted = x[order]
    y_sorted = y[order]

    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    cell_size = max(1, int(cell_size))

    for src, x_px, y_px in zip(src_sorted.tolist(), x_sorted.tolist(), y_sorted.tolist()):
        cell = (int(x_px) // cell_size, int(y_px) // cell_size)
        if cell in seen:
            continue
        seen.add(cell)
        selected.append(int(src))
        if len(selected) >= limit:
            break

    return np.asarray(selected, dtype=np.int32)


def make_lidar_box(x_px: int, y_px: int, color: str) -> dict[str, Any]:
    radius = POINT_RADIUS_PX
    return {
        "className": POINT_CLASS_NAME,
        "bbox": [
            float(max(0, x_px - radius)),
            float(max(0, y_px - radius)),
            float(x_px + radius),
            float(y_px + radius),
        ],
        "confidence": 1.0,
        "color": color,
        "filled": True,
        "updateBoxWhileMoving": UPDATE_BOX_WHILE_MOVING,
    }


def render_lidar_overlay_boxes(
    cache: FrameCache,
    turret_state: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], int]:
    if not bool(overlay_settings.get("showLidarPoints", True)):
        return [], 0

    projected = project_cached_points(cache, turret_state, width, height)
    projected_source = projected["source_index"]
    if projected_source.size == 0:
        return [], 0

    x_lookup = {int(src): int(x) for src, x in zip(projected_source.tolist(), projected["x"].tolist())}
    y_lookup = {int(src): int(y) for src, y in zip(projected_source.tolist(), projected["y"].tolist())}

    projected_obstacles = projected_source[cache.valid_object_mask[projected_source]]
    projected_ground = projected_source[cache.ground_mask[projected_source]]

    obstacle_selected = choose_nearest_per_pixel_cell(
        source_index=projected_obstacles,
        x=np.asarray([x_lookup[int(src)] for src in projected_obstacles], dtype=np.int32),
        y=np.asarray([y_lookup[int(src)] for src in projected_obstacles], dtype=np.int32),
        distances=cache.distances,
        cell_size=int(overlay_settings["obstaclePixelCell"]),
        limit=int(overlay_settings["obstacleBoxLimit"]),
    )

    if bool(overlay_settings.get("showSafeGround", False)):
        ground_selected = choose_nearest_per_pixel_cell(
            source_index=projected_ground,
            x=np.asarray([x_lookup[int(src)] for src in projected_ground], dtype=np.int32),
            y=np.asarray([y_lookup[int(src)] for src in projected_ground], dtype=np.int32),
            distances=cache.distances,
            cell_size=int(overlay_settings["safeGroundPixelCell"]),
            limit=int(overlay_settings["safeGroundBoxLimit"]),
        )
    else:
        ground_selected = np.empty(0, dtype=np.int32)

    boxes: list[dict[str, Any]] = []
    for src in obstacle_selected.tolist():
        boxes.append(
            make_lidar_box(
                x_lookup[src],
                y_lookup[src],
                obstacle_color(float(cache.distances[src])),
            )
        )

    for src in ground_selected.tolist():
        boxes.append(make_lidar_box(x_lookup[src], y_lookup[src], COLOR_SAFE_GROUND))

    return boxes[: int(overlay_settings["totalLidarBoxLimit"])], int(projected_source.size)


# ===========================================================================
# 6. Image reading and asynchronous YOLO
# ===========================================================================
def image_header_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    if image_bytes and Image is not None:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            pass
    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def should_capture_yolo_image() -> bool:
    """
    Return True only when a new asynchronous YOLO frame is due.

    Most /detect calls only need the image width and height for LiDAR screen
    projection. Avoid copying the entire camera JPEG on every /detect request.
    """
    if not bool(fusion_settings.get("enabled", True)):
        return False

    now = monotonic()
    min_interval = max(0.05, float(fusion_settings.get("yoloIntervalSec", 0.50)))

    with state_lock:
        last_submitted = yolo_state.get("lastSubmittedMonotonic")
        if last_submitted is not None and now - float(last_submitted) < min_interval:
            return False
        # Capture a fresh frame even when one older pending frame exists.
        # maybe_submit_yolo_job replaces the stale pending frame.
        return True


def image_header_size_from_stream(stream: Any) -> tuple[int, int]:
    if stream is not None and Image is not None:
        try:
            stream.seek(0)
            with Image.open(stream) as image:
                width, height = image.size
            stream.seek(0)
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            try:
                stream.seek(0)
            except Exception:
                pass
    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def read_detect_image(capture_bytes: bool) -> tuple[bytes, int, int]:
    image_file = request.files.get("image")
    if image_file is not None:
        stream = image_file.stream
        width, height = image_header_size_from_stream(stream)
        if capture_bytes:
            try:
                stream.seek(0)
                return stream.read(), width, height
            except Exception:
                return b"", width, height
        return b"", width, height

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_value = payload.get("image")
        if isinstance(image_value, str) and image_value:
            if "," in image_value and "base64" in image_value:
                image_value = image_value.split(",", 1)[1]
            try:
                # Base64 JSON transport already requires decoding the payload,
                # even when YOLO is not due. Multipart image upload is preferred.
                image_bytes = base64.b64decode(image_value)
                width, height = image_header_size_from_bytes(image_bytes)
                return image_bytes if capture_bytes else b"", width, height
            except Exception:
                pass

    return b"", DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def get_yolo_model() -> Any:
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {ULTRALYTICS_IMPORT_ERROR}")

    model_path = Path(current_yolo_model_path())
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model file not found: {model_path}. "
            "Default is lalast.pt. Put lalast.pt next to this Python file "
            "or set YOLO_MODEL_FILE before running."
        )

    model = YOLO(str(model_path))
    native_names = getattr(model, "names", {}) or {}
    native_names = {
        int(class_id): str(class_name)
        for class_id, class_name in dict(native_names).items()
    }

    with state_lock:
        _yolo_model = model
        yolo_state["modelLoaded"] = True
        yolo_state["modelLoadError"] = None
        yolo_state["modelNames"] = native_names
    return model


def semantic_from_class_id(class_id: int, fallback_name: str | None = None) -> tuple[str, str]:
    # Prefer the names embedded in the loaded YOLO model. This prevents a
    # class-ID mismatch when switching between the older 12-class weights and
    # Tank_combine.pt, which contains 8 classes.
    raw_name = str(fallback_name) if fallback_name is not None else MODEL_CLASS_NAMES.get(class_id, str(class_id))
    semantic = CLASS_SEMANTIC.get(raw_name, raw_name)
    return raw_name, semantic


def camera_relative_angle_from_bbox(bbox: list[float], image_width: int) -> float:
    x1, _, x2, _ = bbox
    center_x = (float(x1) + float(x2)) / 2.0
    normalized = center_x / max(1.0, float(image_width)) - 0.5
    return normalize_signed_angle(normalized * float(calibration["cameraHorizontalFovDeg"]))


def yolo_body_angle_from_bbox(
    bbox: list[float],
    image_width: int,
    cache: FrameCache,
    turret_state: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    camera_yaw_world, _, pose_debug = camera_angles(cache.pose, turret_state)
    body_yaw_world = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
    relative_camera_angle = camera_relative_angle_from_bbox(bbox, image_width)
    body_relative_angle = normalize_signed_angle(
        camera_yaw_world + relative_camera_angle - body_yaw_world
    )
    return body_relative_angle, relative_camera_angle, pose_debug


def resolved_yolo_runtime() -> tuple[str, bool]:
    requested = str(fusion_settings.get("device", "auto")).strip().lower()
    if requested != "auto":
        resolved_device = requested
    elif torch is not None and bool(torch.cuda.is_available()):
        resolved_device = "0"
    else:
        resolved_device = "cpu"

    half = (
        bool(fusion_settings.get("halfPrecisionAuto", True))
        and resolved_device not in {"cpu", "mps"}
        and torch is not None
        and bool(torch.cuda.is_available())
    )

    with state_lock:
        yolo_state["resolvedDevice"] = resolved_device
        yolo_state["resolvedHalfPrecision"] = bool(half)

    return resolved_device, bool(half)


def run_yolo_image_bytes(image_bytes: bytes) -> list[dict[str, Any]]:
    """Run lalast.pt using the same YOLO path as Second.py.

    Second.py works because it feeds a PIL RGB image directly to
    model.predict(...), then reads results[0].boxes.data.  Keep the surrounding
    async fusion server unchanged, but use that proven detection core here.
    """
    if not image_bytes:
        return []
    if Image is None:
        raise RuntimeError("Pillow is required for in-memory YOLO inference.")

    model = get_yolo_model()

    with Image.open(BytesIO(image_bytes)) as image:
        pil_img = image.convert("RGB")

        predict_kwargs: dict[str, Any] = {
            "source": pil_img,
            "conf": float(fusion_settings.get("confidence", YOLO_CONF)),
            "iou": float(fusion_settings.get("iou", YOLO_IOU)),
            "imgsz": int(fusion_settings.get("imageSize", YOLO_IMGSZ)),
            "max_det": int(fusion_settings.get("maxDetections", YOLO_MAX_DET)),
            "augment": False,
            "verbose": False,
        }

        # Match Second.py's lightweight inference path.  Do not force device or
        # half precision here; Ultralytics will choose the safe default.  This
        # avoids a server-only mismatch where the model loads but returns zero
        # boxes while Second.py detects correctly.
        if torch is not None:
            with torch.inference_mode():
                results = model.predict(**predict_kwargs)
        else:
            results = model.predict(**predict_kwargs)

    detections: list[dict[str, Any]] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes.data.detach().cpu().numpy()
    result_names = getattr(results[0], "names", None) or getattr(model, "names", {}) or {}

    for box in boxes:
        x1, y1, x2, y2, confidence, class_value = box[:6]
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h

        # Same small-box filtering as Second.py.  This removes gun-barrel / tiny
        # speck false positives while preserving visible vehicles and tanks.
        if area < 1800:
            continue
        if w < 35 or h < 25:
            continue

        class_id = int(class_value)
        fallback_name = str(result_names.get(class_id, MODEL_CLASS_NAMES.get(class_id, class_id)))
        raw_name, semantic = semantic_from_class_id(class_id, fallback_name)

        detections.append(
            {
                "classId": class_id,
                "rawClassName": raw_name,
                "semanticClass": semantic,
                "confidence": round(float(confidence), 4),
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                "yoloCore": "Second.py_PIL_predict",
            }
        )

    return detections

def circular_mean_deg(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    radians_values = np.deg2rad(values.astype(np.float64))
    return normalize_signed_angle(
        np.rad2deg(
            np.arctan2(
                np.mean(np.sin(radians_values)),
                np.mean(np.cos(radians_values)),
            )
        )
    )


def expand_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    dx = width * max(0.0, float(expand_ratio))
    dy = height * max(0.0, float(expand_ratio))
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(image_width - 1), x2 + dx),
        min(float(image_height - 1), y2 + dy),
    )


def summarize_projected_lidar_roi(
    bbox: list[float],
    cache: FrameCache,
    turret_state: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """
    Fuse in screen space first:
    - project the cached LiDAR points into the same camera image as YOLO
    - keep obstacle points inside an expanded YOLO bbox
    - use a near-surface band so background terrain does not dominate distance

    This avoids the old failure mode where a wide angle window mixed the road,
    a rock, and a tank into one pseudo-object on hilly terrain.
    """
    if not bool(fusion_settings.get("roiFusionEnabled", True)):
        return None

    projected = project_cached_points(cache, turret_state, image_width, image_height)
    source_index = projected["source_index"]
    if source_index.size == 0:
        return None

    x1, y1, x2, y2 = expand_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        expand_ratio=float(fusion_settings.get("roiExpandRatio", 0.08)),
    )

    x_px = projected["x"]
    y_px = projected["y"]
    inside = (
        (x_px >= x1)
        & (x_px <= x2)
        & (y_px >= y1)
        & (y_px <= y2)
    )
    if not np.any(inside):
        return None

    roi_source = source_index[inside]
    valid_source = roi_source[cache.valid_object_mask[roi_source]] if cache.valid_object_mask.size else np.empty(0, dtype=np.int32)
    raw_obstacle_source = roi_source[cache.obstacle_mask[roi_source]]

    min_points = max(1, int(fusion_settings.get("roiMinObstaclePoints", 2)))
    obstacle_source = valid_source if valid_source.size >= min_points else raw_obstacle_source
    if obstacle_source.size < min_points:
        return None

    obstacle_distances = cache.distances[obstacle_source]
    # Robust near-surface seed: avoid a single noisy near point.
    nearest_seed = float(np.percentile(obstacle_distances, 20.0))
    surface_band = max(0.5, float(fusion_settings.get("roiSurfaceBandM", 3.0)))
    surface_source = obstacle_source[
        obstacle_distances <= nearest_seed + surface_band
    ]
    if surface_source.size < min_points:
        surface_source = obstacle_source

    surface_distances = cache.distances[surface_source]
    surface_angles = cache.angles[surface_source]
    surface_xyz = cache.xyz[surface_source]
    surface_terrain_y = cache.terrain_y[surface_source] if cache.terrain_y.size == cache.distances.size else surface_xyz[:, 1]
    surface_above = cache.height_above_terrain[surface_source] if cache.height_above_terrain.size == cache.distances.size else np.zeros(surface_source.size, dtype=np.float32)
    surface_distance = float(np.percentile(surface_distances, 20.0))
    median_distance = float(np.median(surface_distances))
    far_distance = float(np.max(surface_distances))
    mean_angle = float(circular_mean_deg(surface_angles))

    if surface_angles.size >= 2:
        angular_span = float(np.max(surface_angles) - np.min(surface_angles))
    else:
        angular_span = 0.0
    visible_width = 2.0 * median_distance * tan(radians(max(0.0, angular_span) / 2.0))
    height_span = float(np.max(surface_xyz[:, 1]) - np.min(surface_xyz[:, 1])) if surface_xyz.size else 0.0
    terrain_base_y = float(np.median(surface_terrain_y)) if surface_terrain_y.size else 0.0
    top_y = float(np.max(surface_xyz[:, 1])) if surface_xyz.size else terrain_base_y
    object_height = float(np.max(surface_above)) if surface_above.size else max(0.0, top_y - terrain_base_y)
    bottom_y = float(np.min(surface_xyz[:, 1])) if surface_xyz.size else terrain_base_y
    object_base_y = min(terrain_base_y, bottom_y)
    total_height_from_base = max(0.0, top_y - object_base_y)
    object_height_for_aim = max(object_height, total_height_from_base, height_span)
    aim_ratio = max(0.05, min(0.95, float(aim_settings.get("targetAimHeightRatio", TARGET_AIM_HEIGHT_RATIO))))
    min_clearance = max(0.0, float(aim_settings.get("targetAimMinClearanceM", TARGET_AIM_MIN_CLEARANCE_M)))
    aim_clearance = max(min_clearance, object_height_for_aim * aim_ratio)
    if object_height_for_aim > 0.01:
        aim_clearance = min(aim_clearance, object_height_for_aim * 0.90)
    aim_point_y = object_base_y + aim_clearance
    origin_raw = get_xyz(cache.pose.get("lidarOrigin"))
    lidar_origin_y = float(origin_raw[1]) if origin_raw is not None else EXPECTED_LIDAR_Y_POSITION_M
    aim_pitch = float(np.degrees(np.arctan2(aim_point_y - lidar_origin_y, max(0.5, float(np.median(cache.horizontal_ranges[surface_source])) if cache.horizontal_ranges.size else median_distance))))

    return {
        "clusterId": "ROI",
        "candidateLabel": "ROI_OBJ" if valid_source.size >= min_points else "ROI_RAW",
        "angleDeg": round(mean_angle, 3),
        "distanceM": round(surface_distance, 3),
        "surfaceDistanceM": round(surface_distance, 3),
        "medianDistanceM": round(median_distance, 3),
        "farDistanceM": round(far_distance, 3),
        "pointCount": int(surface_source.size),
        "roiObstaclePointCount": int(raw_obstacle_source.size),
        "roiValidObjectPointCount": int(valid_source.size),
        "angularSpanDeg": round(angular_span, 3),
        "visibleWidthM": round(visible_width, 3),
        "heightSpanM": round(height_span, 3),
        "objectHeightAboveTerrainM": round(object_height, 3),
        "objectTopYWorldM": round(top_y, 3),
        "objectBottomYWorldM": round(bottom_y, 3),
        "objectBaseYWorldM": round(object_base_y, 3),
        "terrainBaseYWorldM": round(terrain_base_y, 3),
        "aimPointYWorldM": round(aim_point_y, 3),
        "aimHeightAboveBaseM": round(aim_clearance, 3),
        "aimPitchDeg": round(aim_pitch, 3),
        "fusionMethod": "pixel_roi_object_on_hill" if valid_source.size >= min_points else "pixel_roi_raw_obstacle",
    }


def split_csv_tokens(value: Any) -> set[str]:
    return {
        token.strip().lower()
        for token in str(value or "").split(",")
        if token.strip()
    }


def maybe_make_tank_candidate(
    detection: dict[str, Any],
    matched_lidar: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any] | None]:
    """
    Experimental post-processing only.

    It does NOT prove that a car2 box is a tank. It marks a conservative
    LiDAR-supported "tank?" candidate so the user can keep testing while
    collecting retraining images.
    """
    raw_name = str(detection.get("rawClassName", "object"))
    semantic = str(detection.get("semanticClass", raw_name))

    if not bool(fusion_settings.get("tankCandidateRescueEnabled", False)):
        return raw_name, semantic, None
    if matched_lidar is None:
        return raw_name, semantic, None
    if raw_name.lower() not in split_csv_tokens(fusion_settings.get("tankCandidateSourceClasses", "car2")):
        return raw_name, semantic, None

    bbox = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    aspect_ratio = max(0.0, (x2 - x1) / max(1.0, y2 - y1))

    visible_width = float(matched_lidar.get("visibleWidthM", 0.0) or 0.0)
    height_span = float(matched_lidar.get("heightSpanM", 0.0) or 0.0)
    roi_points = int(matched_lidar.get("roiObstaclePointCount", matched_lidar.get("pointCount", 0)) or 0)

    passed = (
        visible_width >= float(fusion_settings.get("tankRescueMinWidthM", 2.80))
        and height_span >= float(fusion_settings.get("tankRescueMinHeightSpanM", 0.45))
        and roi_points >= int(fusion_settings.get("tankRescueMinRoiPoints", 4))
        and aspect_ratio >= float(fusion_settings.get("tankRescueMinBoxAspectRatio", 1.25))
    )
    if not passed:
        return raw_name, semantic, None

    debug = {
        "sourceClass": raw_name,
        "visibleWidthM": round(visible_width, 3),
        "heightSpanM": round(height_span, 3),
        "roiObstaclePointCount": roi_points,
        "bboxAspectRatio": round(aspect_ratio, 3),
        "note": "experimental LiDAR-supported candidate; retrain YOLO for confirmed tank class",
    }
    return str(fusion_settings.get("tankCandidateDisplayName", "tank?")), "tank_candidate", debug


def format_fused_label(
    raw_class_name: str,
    matched_lidar: dict[str, Any] | None,
) -> str:
    """
    Compact simulator UI label.

    Default example:
      tank | 42.6m | +13.2deg

    Scenario-only alias:
      /fusion_update?tankDisplayName=Tank_enemy

    Angle convention:
      0deg   = body forward
      +deg   = body right
      -deg   = body left
    """
    display_name = str(raw_class_name)
    if str(raw_class_name).lower() == "tank":
        display_name = str(fusion_settings.get("tankDisplayName", "tank"))

    if matched_lidar is None:
        return display_name

    distance_m = float(matched_lidar["distanceM"])
    angle_deg = float(matched_lidar["angleDeg"])
    height_value = safe_float(matched_lidar.get("objectHeightAboveTerrainM"), None)
    if height_value is not None:
        return f"{display_name} | {distance_m:.1f}m | {angle_deg:+.1f}deg | h:{float(height_value):.1f}m"
    return f"{display_name} | {distance_m:.1f}m | {angle_deg:+.1f}deg"


def cluster_geometry_penalty(semantic: str, cluster: dict[str, Any]) -> float:
    candidate = str(cluster.get("candidateLabel", ""))
    if candidate in {"OBJ", "VOBJ", "OBJ_HILL", "ROI_OBJ"}:
        return 0.0
    if semantic in BULKY_SEMANTICS and candidate != "BK?":
        return 3.0
    if semantic in THIN_SEMANTICS and candidate != "TH?":
        return 2.0
    return 0.0


def nearest_lidar_hint_for_yolo(
    body_angle: float,
    cache: FrameCache,
) -> dict[str, Any] | None:
    """
    Debug-only hint for YOLO-only boxes.

    This does not make the object fused and is never used as fire evidence.
    It only restores useful distance/angle text when YOLO detected something
    but pixel ROI / strict cluster fusion did not match.
    """
    if not bool(fusion_settings.get("showYoloOnlyLidarHint", True)):
        return None
    max_gap = float(fusion_settings.get("yoloOnlyHintAngleGateDeg", 20.0))
    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for cluster in cache.clusters:
        gap = angle_gap_deg(float(body_angle), float(cluster.get("angleDeg", 0.0)))
        if gap > max_gap:
            continue
        ranked.append((gap, float(cluster.get("distanceM", 9999.0)), cluster))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    gap, _, cluster = ranked[0]
    return {
        **json_copy(cluster),
        "hintAngleGapDeg": round(float(gap), 3),
        "note": "YOLO-only nearest LiDAR hint; not a fused object",
    }


def fuse_yolo_to_lidar(
    detections: list[dict[str, Any]],
    cache: FrameCache,
    turret_state: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    clusters = list(cache.clusters)
    available_cluster_ids = {int(cluster["clusterId"]) for cluster in clusters}
    fused: list[dict[str, Any]] = []

    # High-confidence YOLO detections reserve LiDAR evidence first.
    for det in sorted(detections, key=lambda item: -float(item["confidence"])):
        body_angle, relative_camera_angle, pose_debug = yolo_body_angle_from_bbox(
            bbox=list(det["bbox"]),
            image_width=image_width,
            cache=cache,
            turret_state=turret_state,
        )

        # 1) Primary method: projected obstacle points inside the YOLO bbox.
        matched_lidar = summarize_projected_lidar_roi(
            bbox=list(det["bbox"]),
            cache=cache,
            turret_state=turret_state,
            image_width=image_width,
            image_height=image_height,
        )
        fusion_method = "pixel_roi" if matched_lidar is not None else None
        angle_gap: float | None = None

        if matched_lidar is not None:
            angle_gap = angle_gap_deg(body_angle, float(matched_lidar["angleDeg"]))

        # 2) Fallback method: nearest LiDAR obstacle cluster in body-angle space.
        if matched_lidar is None and bool(fusion_settings.get("clusterFallbackEnabled", True)):
            ranked: list[tuple[float, float, dict[str, Any]]] = []
            for cluster in clusters:
                cluster_id = int(cluster["clusterId"])
                if cluster_id not in available_cluster_ids:
                    continue

                gap = angle_gap_deg(body_angle, float(cluster["angleDeg"]))
                if gap > float(fusion_settings["maxFusionAngleGapDeg"]):
                    continue

                score = gap + cluster_geometry_penalty(str(det["semanticClass"]), cluster)
                ranked.append((score, gap, cluster))

            if ranked:
                ranked.sort(key=lambda item: (item[0], item[1], float(item[2]["distanceM"])))
                _, angle_gap, cluster = ranked[0]
                available_cluster_ids.discard(int(cluster["clusterId"]))
                matched_lidar = {
                    **cluster,
                    "fusionMethod": "angle_cluster_fallback",
                }
                fusion_method = "angle_cluster_fallback"

        fusion_matched = matched_lidar is not None
        nearest_lidar_hint = None if fusion_matched else nearest_lidar_hint_for_yolo(body_angle, cache)
        raw_class_name = str(det["rawClassName"])
        display_class_name, display_semantic, tank_candidate_debug = maybe_make_tank_candidate(
            detection=det,
            matched_lidar=matched_lidar,
        )

        fused.append(
            {
                **det,
                "originalRawClassName": raw_class_name,
                "originalSemanticClass": str(det["semanticClass"]),
                "className": format_fused_label(display_class_name, matched_lidar),
                "semanticClass": display_semantic,
                "color": FUSED_COLORS.get(display_semantic, FUSED_COLORS["unknown"]),
                "tankCandidateRescue": tank_candidate_debug,
                "filled": False,
                "updateBoxWhileMoving": False,
                "fusionMatched": fusion_matched,
                "fusionMethod": fusion_method,
                "cameraRelativeAngleDeg": round(relative_camera_angle, 3),
                "lidarBodyAngleDeg": (
                    round(float(matched_lidar["angleDeg"]), 3)
                    if matched_lidar is not None
                    else round(body_angle, 3)
                ),
                "fusionAngleGapDeg": (
                    round(float(angle_gap), 3)
                    if angle_gap is not None
                    else None
                ),
                "lidarCluster": json_copy(matched_lidar) if matched_lidar is not None else None,
                "nearestLidarHint": json_copy(nearest_lidar_hint) if nearest_lidar_hint is not None else None,
                "distance": (
                    float(matched_lidar["distanceM"])
                    if matched_lidar is not None
                    else None
                ),
                "sourceFrameSeq": cache.seq,
                "sourceSimulationTime": cache.simulation_time,
                "sourceCameraYawDeg": pose_debug["cameraYawDeg"],
            }
        )

    fused.sort(
        key=lambda item: (
            not bool(item["fusionMatched"]),
            float(item["distance"]) if item["distance"] is not None else 9999.0,
        )
    )
    return fused


def yolo_worker_loop() -> None:
    global _pending_vision_job

    while True:
        _yolo_event.wait()

        with state_lock:
            job = _pending_vision_job
            _pending_vision_job = None
            yolo_state["pendingJob"] = False
            if job is None:
                _yolo_event.clear()
                continue
            yolo_state["workerBusy"] = True

        inference_started = monotonic()
        try:
            detections = run_yolo_image_bytes(job.image_bytes)
            inference_ms = round((monotonic() - inference_started) * 1000.0, 2)

            fusion_started = monotonic()
            fused_objects = fuse_yolo_to_lidar(
                detections=detections,
                cache=job.cache,
                turret_state=job.turret_state,
                image_width=job.width,
                image_height=job.height,
            )
            fused_objects = attach_ground_truth_comparisons(fused_objects, job.cache)
            fusion_ms = round((monotonic() - fusion_started) * 1000.0, 2)

            completed_at = datetime.now().isoformat(timespec="milliseconds")
            with state_lock:
                yolo_state["latestYoloDetections"] = json_copy(detections)
                yolo_state["latestFusedObjects"] = json_copy(fused_objects)
                yolo_state["latestResultMeta"] = {
                    "sourceFrameSeq": job.cache.seq,
                    "sourceSimulationTime": job.cache.simulation_time,
                    "sourceCameraYawDeg": (
                        fused_objects[0]["sourceCameraYawDeg"]
                        if fused_objects
                        else camera_angles(job.cache.pose, job.turret_state)[2]["cameraYawDeg"]
                    ),
                    "sourceCameraPitchDeg": camera_angles(job.cache.pose, job.turret_state)[2]["cameraPitchDeg"],
                    "sourcePlayerPos": (
                        pose_position(job.cache.pose).tolist()
                        if pose_position(job.cache.pose) is not None
                        else None
                    ),
                    "sourceGroundNormal": job.cache.ground_normal.tolist(),
                    "submittedAt": job.submitted_at,
                    "completedAt": completed_at,
                    "completedMonotonic": monotonic(),
                    "imageSize": [job.width, job.height],
                }
                yolo_state["completedCount"] += 1
                yolo_state["lastCompletedAt"] = completed_at
                yolo_state["lastInferenceMs"] = inference_ms
                yolo_state["lastFusionMs"] = fusion_ms
                yolo_state["modelLoadError"] = None
        except Exception as exc:
            with state_lock:
                yolo_state["failedCount"] += 1
                yolo_state["modelLoadError"] = f"{type(exc).__name__}: {exc}"
        finally:
            with state_lock:
                yolo_state["workerBusy"] = False
                # If a newer frame was queued while this job ran, process it.
                if _pending_vision_job is None:
                    _yolo_event.clear()


def maybe_submit_yolo_job(
    image_bytes: bytes,
    width: int,
    height: int,
    cache: FrameCache,
    turret_state: dict[str, Any],
) -> bool:
    global _pending_vision_job

    if not bool(fusion_settings.get("enabled", True)) or not image_bytes:
        return False

    now = monotonic()
    min_interval = max(0.05, float(fusion_settings.get("yoloIntervalSec", 0.50)))

    with state_lock:
        last_submitted = yolo_state.get("lastSubmittedMonotonic")
        if last_submitted is not None and now - float(last_submitted) < min_interval:
            return False

        # Keep exactly one pending LATEST image. When YOLO is still processing
        # an older frame, replace the queued frame instead of accumulating lag.
        if _pending_vision_job is not None:
            yolo_state["replacedPendingJobCount"] += 1

        submitted_at = datetime.now().isoformat(timespec="milliseconds")
        _pending_vision_job = VisionJob(
            image_bytes=image_bytes,
            width=int(width),
            height=int(height),
            cache=cache,
            turret_state=dict(turret_state),
            submitted_monotonic=now,
            submitted_at=submitted_at,
        )
        yolo_state["pendingJob"] = True
        yolo_state["submittedCount"] += 1
        yolo_state["lastSubmittedMonotonic"] = now
        yolo_state["lastSubmittedAt"] = submitted_at
        _yolo_event.set()
        return True


def current_fused_boxes(
    cache: FrameCache,
    turret_state: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(fusion_settings.get("showFusedBoxes", True)):
        return []

    with state_lock:
        objects = json_copy(yolo_state.get("latestFusedObjects", []))
        meta = json_copy(yolo_state.get("latestResultMeta", {}))

    completed_monotonic = meta.get("completedMonotonic")
    if completed_monotonic is None:
        return []

    result_age = max(0.0, monotonic() - float(completed_monotonic))
    with state_lock:
        yolo_state["lastResultAgeSec"] = round(result_age, 3)

    if result_age > float(fusion_settings.get("maxDisplayAgeSec", 1.20)):
        return []

    current_camera_yaw, current_camera_pitch, _ = camera_angles(cache.pose, turret_state)
    source_camera_yaw = safe_float(meta.get("sourceCameraYawDeg"))
    source_camera_pitch = safe_float(meta.get("sourceCameraPitchDeg"))
    if source_camera_yaw is None:
        return []

    if angle_gap_deg(current_camera_yaw, source_camera_yaw) > float(
        fusion_settings.get("maxDisplayYawDeltaDeg", 5.0)
    ):
        return []

    if source_camera_pitch is not None and abs(float(current_camera_pitch) - float(source_camera_pitch)) > float(
        fusion_settings.get("maxDisplayPitchDeltaDeg", 4.0)
    ):
        return []

    current_position = pose_position(cache.pose)
    source_position_raw = meta.get("sourcePlayerPos")
    if current_position is not None and isinstance(source_position_raw, list) and len(source_position_raw) == 3:
        source_position = np.asarray(source_position_raw, dtype=np.float32)
        if float(np.linalg.norm(current_position - source_position)) > float(
            fusion_settings.get("maxDisplayPositionDeltaM", 1.20)
        ):
            return []

    source_normal_raw = meta.get("sourceGroundNormal")
    if isinstance(source_normal_raw, list) and len(source_normal_raw) == 3:
        if vector_angle_deg(cache.ground_normal, np.asarray(source_normal_raw, dtype=np.float32)) > float(
            fusion_settings.get("maxDisplayGroundNormalDeltaDeg", 5.0)
        ):
            return []

    boxes: list[dict[str, Any]] = []
    for obj in objects:
        matched = bool(obj.get("fusionMatched", False))
        if not matched and not bool(fusion_settings.get("showUnmatchedYoloBoxes", False)):
            continue

        label = str(obj.get("className", obj.get("semanticClass", "object")))
        color = str(obj.get("color", "#FFFFFF"))
        if matched:
            # Defensive label repair: keep distance/angle visible even if a
            # later label decoration changed className.
            distance = safe_float(obj.get("distance"))
            angle = safe_float(obj.get("lidarBodyAngleDeg"))
            if distance is not None and angle is not None and "m |" not in label:
                height = safe_float((obj.get("lidarCluster") or {}).get("objectHeightAboveTerrainM"))
                if height is not None:
                    label = f"{label} | {float(distance):.1f}m | {float(angle):+.1f}deg | h:{float(height):.1f}m"
                else:
                    label = f"{label} | {float(distance):.1f}m | {float(angle):+.1f}deg"
        else:
            # Yellow YOLO-only boxes prove that image sensing works even before
            # LiDAR angle matching is calibrated.  Show angle and nearest LiDAR
            # hint for debugging, but do not treat it as fused distance.
            if bool(fusion_settings.get("showYoloOnlyAngleLabel", True)):
                angle = safe_float(obj.get("lidarBodyAngleDeg"), safe_float(obj.get("cameraRelativeAngleDeg")))
                hint = obj.get("nearestLidarHint") or {}
                hint_distance = safe_float(hint.get("distanceM")) if isinstance(hint, dict) else None
                hint_angle = safe_float(hint.get("angleDeg")) if isinstance(hint, dict) else None
                hint_height = safe_float(hint.get("objectHeightAboveTerrainM")) if isinstance(hint, dict) else None
                if hint_distance is not None and hint_angle is not None:
                    if hint_height is not None:
                        label = f"YOLO? {label} | hint {float(hint_distance):.1f}m | {float(hint_angle):+.1f}deg | h:{float(hint_height):.1f}m"
                    else:
                        label = f"YOLO? {label} | hint {float(hint_distance):.1f}m | {float(hint_angle):+.1f}deg"
                elif angle is not None:
                    label = f"YOLO? {label} | no LiDAR | {float(angle):+.1f}deg"
                else:
                    label = f"YOLO? {label} | no LiDAR"
            else:
                label = f"YOLO? {label}"
            color = "#FFFF00"

        boxes.append(
            {
                "className": label,
                "bbox": obj.get("bbox", []),
                "confidence": float(obj.get("confidence", 1.0)),
                "color": color,
                "filled": False,
                "updateBoxWhileMoving": False,
            }
        )
    return boxes


# ===========================================================================
# 7. API routes
# ===========================================================================
@app.route("/recommended_settings", methods=["GET"])
def recommended_settings():
    return jsonify(
        {
            "simulatorProperties": {
                "mode": "Simulation",
                "requestPort": 5000,
                "intervalSec": 0.2,
                "lidarYPositionM": 3.0,
                "channel": 32,
                "minimapChannel": 16,
                "maxDistanceM": 120,
                "lidarPosition": "Body",
                "sendDetectedLidar": True,
                "frameRate": 60,
                "graphicsQuality": "Medium",
            },
            "fusionDefaults": {
                "model": "lalast.pt",
                "modelClasses": ["Ally", "Enemy", "House", "Rock", "Rock_L", "Tank_enemy", "Tent", "car"],
                "imageSize": 512,
                "yoloIntervalSec": 0.70,
                "labelFormat": "Tank_enemy | distance_m | body_relative_angle_deg",
                "tiltCompensationMode": "ground_plane",
                "presets": [
                    "balanced",
                    "cpu_light",
                    "tank_accuracy",
                    "tank_candidate_test",
                ],
                "genericTankNote": (
                    "lalast.pt detects Tank_enemy directly. The fire logic maps Tank_enemy to enemy_tank."
                ),
            },
        }
    )


@app.route("/tilt_status", methods=["GET"])
def tilt_status():
    with state_lock:
        cache = latest_cache
        tilt_snapshot = {
            "updatedAt": tilt_state.get("updatedAt"),
            "smoothedGroundNormal": np.asarray(
                tilt_state.get("smoothedGroundNormal", (0.0, 1.0, 0.0))
            ).tolist(),
        }
    return jsonify(
        {
            "calibration": dict(calibration),
            "frameSeq": cache.seq,
            "groundPlane": json_copy(cache.ground_plane_debug),
            "groundNormal": cache.ground_normal.tolist(),
            "tiltState": tilt_snapshot,
            "rawBodyTiltFields": {
                "playerBodyY": cache.pose.get("playerBodyY"),
                "playerBodyZ": cache.pose.get("playerBodyZ"),
            },
            "availableInfoKeys": json_copy(cache.pose.get("availableInfoKeys", [])),
            "note": (
                "ground_plane mode estimates local road tilt from LiDAR. "
                "It approximates chassis lean when explicit body roll/pitch is unavailable."
            ),
        }
    )


@app.route("/fusion_preset", methods=["GET"])
def fusion_preset():
    mode = str(request.args.get("mode", "balanced")).strip().lower()

    if mode == "balanced":
        fusion_settings.update(
            {
                "confidence": YOLO_CONF,
                "iou": YOLO_IOU,
                "imageSize": YOLO_IMGSZ,
                "maxDetections": YOLO_MAX_DET,
                "augment": YOLO_AUGMENT,
                "yoloIntervalSec": 0.70,
                "showUnmatchedYoloBoxes": True,
                "tankCandidateRescueEnabled": False,
            }
        )
    elif mode == "cpu_light":
        fusion_settings.update(
            {
                "confidence": 0.28,
                "imageSize": 416,
                "yoloIntervalSec": 0.95,
                "showUnmatchedYoloBoxes": True,
                "tankCandidateRescueEnabled": False,
            }
        )
    elif mode == "tank_accuracy":
        fusion_settings.update(
            {
                "confidence": 0.22,
                "imageSize": 640,
                "yoloIntervalSec": 0.85,
                "showUnmatchedYoloBoxes": True,
                "tankCandidateRescueEnabled": False,
            }
        )
    elif mode == "tank_candidate_test":
        fusion_settings.update(
            {
                "confidence": 0.22,
                "imageSize": 640,
                "yoloIntervalSec": 0.85,
                "showUnmatchedYoloBoxes": True,
                "tankCandidateRescueEnabled": True,
            }
        )
    else:
        return jsonify(
            {
                "status": "error",
                "message": "Use balanced, cpu_light, tank_accuracy, or tank_candidate_test.",
            }
        ), 400

    return jsonify({"status": "success", "mode": mode, "fusion": dict(fusion_settings)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "server": "Tank Challenge LiDAR-first YOLO Fusion v16.7 360 Pitch-Sweep Tank Fire"})


@app.route("/info", methods=["POST"])
def info():
    global latest_cache

    started = monotonic()
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "No JSON received"}), 400

    record_payload_debug("info", data)
    if bool(ground_truth_settings.get("autoExtractInfo", True)):
        ingest_gt_payload(
            data,
            source="auto:info",
            dynamic_default=True,
            allow_direct=False,
        )

    with state_lock:
        next_seq = latest_cache.seq + 1

    cache = build_frame_cache(data, next_seq)
    total_ms = round((monotonic() - started) * 1000.0, 2)

    with state_lock:
        latest_cache = cache
        status_state["infoRequestCount"] += 1
        status_state["lastInfoProcessingMs"] = total_ms
        status_state["lastInfoUpdatedAt"] = datetime.now().isoformat(timespec="milliseconds")

    return jsonify({"status": "success", "control": ""})


@app.route("/detect", methods=["POST"])
def detect():
    started = monotonic()
    capture_yolo_frame = should_capture_yolo_image()
    image_bytes, width, height = read_detect_image(capture_yolo_frame)

    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)

    _, _, pose_debug = camera_angles(cache.pose, turret_state)
    lidar_boxes, projected_count = render_lidar_overlay_boxes(cache, turret_state, width, height)
    maybe_submit_yolo_job(image_bytes, width, height, cache, turret_state)
    fused_boxes = current_fused_boxes(cache, turret_state)

    response_boxes = fused_boxes + lidar_boxes
    processing_ms = round((monotonic() - started) * 1000.0, 2)

    with state_lock:
        status_state["detectRequestCount"] += 1
        status_state["lastDetectProcessingMs"] = processing_ms
        status_state["lastReturnedLidarBoxCount"] = len(lidar_boxes)
        status_state["lastReturnedFusedBoxCount"] = len(fused_boxes)
        status_state["lastProjectedPointCount"] = projected_count
        status_state["lastImageSize"] = [width, height]
        status_state["lastPoseSource"] = pose_debug["poseSource"]
        status_state["lastCameraYawDeg"] = pose_debug["cameraYawDeg"]
        status_state["lastCameraPitchDeg"] = pose_debug["cameraPitchDeg"]
        status_state["lastInfoTurretYawDeg"] = pose_debug["infoTurretYawDeg"]
        status_state["lastInfoTurretPitchDeg"] = pose_debug["infoTurretPitchDeg"]
        status_state["lastActionTurretYawDeg"] = pose_debug["actionTurretYawDeg"]
        status_state["lastActionTurretPitchDeg"] = pose_debug["actionTurretPitchDeg"]
        status_state["lastActionPoseAgeSec"] = pose_debug["actionPoseAgeSec"]
        status_state["lastGroundTiltDeg"] = cache.ground_plane_debug.get("smoothedTiltDeg")
        status_state["lastGroundNormal"] = cache.ground_normal.tolist()
        status_state["lastDetectUpdatedAt"] = datetime.now().isoformat(timespec="milliseconds")

    return jsonify(response_boxes)



def current_turret_body_yaw_deg(cache: FrameCache, turret_state: dict[str, Any]) -> float:
    turret_yaw = safe_float(turret_state.get("x"), 0.0) or 0.0
    if str(aim_settings.get("turretYawMode", "absolute")).lower() == "relative":
        return normalize_signed_angle(turret_yaw)
    body_yaw = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
    return normalize_signed_angle(turret_yaw - body_yaw)


def current_turret_pitch_deg(turret_state: dict[str, Any]) -> float:
    return safe_float(turret_state.get("y"), 0.0) or 0.0


def cleanup_ignored_candidates() -> None:
    now = monotonic()
    ttl = float(aim_settings.get("nonTankIgnoreSec", 3.0))
    ignored = aim_state.setdefault("ignoredCandidateKeys", {})
    expired = [key for key, value in ignored.items() if now - float(value) > ttl]
    for key in expired:
        ignored.pop(key, None)


def fresh_yolo_objects() -> list[dict[str, Any]]:
    with state_lock:
        objects = json_copy(yolo_state.get("latestFusedObjects", []))
        meta = json_copy(yolo_state.get("latestResultMeta", {}))
    completed = meta.get("completedMonotonic")
    if completed is None:
        return []
    age = monotonic() - float(completed)
    if age > float(aim_settings.get("targetConfirmMaxAgeSec", 2.0)):
        return []
    return objects if isinstance(objects, list) else []


def is_tank_semantic(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if token in {"tank", "enemy_tank", "tank_enemy", "tank001", "tank_001"}:
        return True
    if token in {"tank?", "tank_candidate"}:
        return bool(aim_settings.get("fireOnTankCandidate", False))
    return False


def classify_selected_candidate_with_yolo(candidate: dict[str, Any]) -> dict[str, Any] | None:
    objects = fresh_yolo_objects()
    if not objects:
        return None
    target_angle = float(candidate.get("angleDeg", 0.0))
    target_distance = float(candidate.get("distanceM", 9999.0))
    angle_gate = float(aim_settings.get("targetYoloAngleGateDeg", 8.0))
    distance_gate = float(aim_settings.get("targetYoloDistanceGateM", 12.0))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for obj in objects:
        angle = safe_float(obj.get("lidarBodyAngleDeg"))
        distance = safe_float(obj.get("distance"))
        if angle is None or distance is None:
            cluster = obj.get("lidarCluster") or {}
            angle = safe_float(cluster.get("angleDeg")) if angle is None else angle
            distance = safe_float(cluster.get("distanceM")) if distance is None else distance
        if angle is None or distance is None:
            continue
        ag = angle_gap_deg(float(angle), target_angle)
        dg = abs(float(distance) - target_distance)
        if ag <= angle_gate and dg <= distance_gate:
            ranked.append((ag + dg * 0.1, obj))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidateKey", candidate.get("clusterId", "unknown")))


def candidate_coarse_key(candidate: dict[str, Any]) -> str:
    """
    Stable skip key for the same visual object.

    The original candidateKey is precise, so the same car/house/rock can come
    back as a new key when the median LiDAR angle/range jitters by a little.
    Coarse angle/range bins prevent car -> house -> car -> house loops while
    still allowing a farther target at a different range bin to be checked.
    """
    angle = safe_float(candidate.get("angleDeg"), 0.0) or 0.0
    distance = safe_float(candidate.get("distanceM"), 9999.0) or 9999.0
    angle_bin = max(1.0, float(aim_settings.get("ignoreAngleBinDeg", 8.0)))
    distance_bin = max(1.0, float(aim_settings.get("ignoreDistanceBinM", 12.0)))
    a = round(float(angle) / angle_bin) * angle_bin
    d = round(float(distance) / distance_bin) * distance_bin
    return f"sector_a{a:+.1f}_d{d:.1f}"


def candidate_ignore_keys(candidate: dict[str, Any]) -> list[str]:
    keys = [candidate_key(candidate)]
    if bool(aim_settings.get("useCoarseIgnoreKey", True)):
        keys.append(candidate_coarse_key(candidate))
    return list(dict.fromkeys(str(key) for key in keys if str(key)))


def candidate_is_ignored(candidate: dict[str, Any]) -> bool:
    ignored = aim_state.setdefault("ignoredCandidateKeys", {})
    return any(key in ignored for key in candidate_ignore_keys(candidate))


def mark_candidate_ignored(candidate: dict[str, Any], reason: str, now_text_value: str, extra: dict[str, Any] | None = None) -> None:
    now_mono = monotonic()
    ignored = aim_state.setdefault("ignoredCandidateKeys", {})
    keys = candidate_ignore_keys(candidate)
    for key in keys:
        ignored[key] = now_mono
    entry = {
        "keys": keys,
        "primaryKey": keys[0] if keys else candidate_key(candidate),
        "coarseKey": candidate_coarse_key(candidate),
        "reason": reason,
        "at": now_text_value,
        "candidate": json_copy(candidate),
    }
    if extra:
        entry.update(json_copy(extra))
    aim_state["lastSkippedCandidate"] = entry
    history = aim_state.setdefault("checkedCandidateHistory", [])
    history.append(entry)
    del history[:-20]



def cleanup_smoothed_targets() -> None:
    now_mono = monotonic()
    ttl = max(2.0, float(aim_settings.get("aimTargetSmoothingResetSec", 1.0)) * 5.0)
    smoothed = aim_state.setdefault("smoothedTargetByKey", {})
    expired = [
        key for key, value in smoothed.items()
        if now_mono - float(value.get("updatedMonotonic", 0.0)) > ttl
    ]
    for key in expired:
        smoothed.pop(key, None)


def candidate_control_key(candidate: dict[str, Any]) -> str:
    # Use the same coarse key as ignore logic so one physical object keeps one
    # stable aiming filter even when the exact cluster median jitters a little.
    if bool(aim_settings.get("useCoarseIgnoreKey", True)):
        return candidate_coarse_key(candidate)
    return candidate_key(candidate)


def smooth_target_angles(
    candidate: dict[str, Any],
    raw_target_yaw: float,
    raw_target_pitch: float,
) -> tuple[float, float, dict[str, Any]]:
    if not bool(aim_settings.get("aimTargetSmoothingEnabled", True)):
        return float(raw_target_yaw), float(raw_target_pitch), {
            "enabled": False,
            "rawTargetYawDeg": round(float(raw_target_yaw), 3),
            "rawTargetPitchDeg": round(float(raw_target_pitch), 3),
        }

    cleanup_smoothed_targets()
    key = candidate_control_key(candidate)
    now_mono = monotonic()
    alpha = max(0.01, min(1.0, float(aim_settings.get("aimTargetSmoothingAlpha", 0.30))))
    reset_sec = max(0.05, float(aim_settings.get("aimTargetSmoothingResetSec", 1.0)))
    smoothed = aim_state.setdefault("smoothedTargetByKey", {})
    previous = smoothed.get(key)

    initialized = True
    if previous is None or now_mono - float(previous.get("updatedMonotonic", 0.0)) > reset_sec:
        yaw = normalize_signed_angle(float(raw_target_yaw))
        pitch = float(raw_target_pitch)
        initialized = False
    else:
        prev_yaw = float(previous.get("yawDeg", raw_target_yaw))
        prev_pitch = float(previous.get("pitchDeg", raw_target_pitch))
        yaw = normalize_signed_angle(
            prev_yaw + normalize_signed_angle(float(raw_target_yaw) - prev_yaw) * alpha
        )
        pitch = prev_pitch + (float(raw_target_pitch) - prev_pitch) * alpha

    smoothed[key] = {
        "yawDeg": yaw,
        "pitchDeg": pitch,
        "rawYawDeg": normalize_signed_angle(float(raw_target_yaw)),
        "rawPitchDeg": float(raw_target_pitch),
        "updatedMonotonic": now_mono,
    }
    return yaw, pitch, {
        "enabled": True,
        "key": key,
        "alpha": round(alpha, 3),
        "initializedFromPrevious": initialized,
        "rawTargetYawDeg": round(normalize_signed_angle(float(raw_target_yaw)), 3),
        "rawTargetPitchDeg": round(float(raw_target_pitch), 3),
        "smoothedTargetYawDeg": round(float(yaw), 3),
        "smoothedTargetPitchDeg": round(float(pitch), 3),
    }


def scaled_command_weight(
    error_abs_deg: float,
    base_weight: float,
    slowdown_error_deg: float,
    min_weight: float,
) -> float:
    if not bool(aim_settings.get("proportionalAimControl", True)):
        return round(float(base_weight), 3)
    slowdown = max(0.1, float(slowdown_error_deg))
    scale = min(1.0, max(0.0, float(error_abs_deg) / slowdown))
    return round(max(float(min_weight), min(float(base_weight), float(base_weight) * scale)), 3)


def command_direction(error_deg: float) -> int:
    if error_deg > 0:
        return 1
    if error_deg < 0:
        return -1
    return 0


def apply_anti_hunt_turret_control(
    action: dict[str, Any],
    yaw_error: float,
    pitch_error: float,
) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "proportionalAimControl": bool(aim_settings.get("proportionalAimControl", True)),
        "suppressReverseCommandNearLock": bool(aim_settings.get("suppressReverseCommandNearLock", True)),
    }

    yaw_abs = abs(float(yaw_error))
    yaw_deadband = float(aim_settings.get("yawDeadbandDeg", 1.2))
    yaw_dir = command_direction(float(yaw_error))
    last_yaw_dir = int(aim_state.get("lastYawCommandDirection", 0) or 0)
    yaw_suppressed = False

    if yaw_abs > yaw_deadband and yaw_dir != 0:
        if (
            bool(aim_settings.get("suppressReverseCommandNearLock", True))
            and last_yaw_dir != 0
            and yaw_dir != last_yaw_dir
            and yaw_abs <= float(aim_settings.get("yawReverseSuppressDeg", 3.0))
        ):
            # The error crossed zero by a small amount.  Do not immediately hit
            # the opposite key; let the turret settle for one frame.
            aim_state["lastYawCommandDirection"] = 0
            yaw_suppressed = True
        else:
            weight = scaled_command_weight(
                yaw_abs,
                float(aim_settings.get("yawCommandWeight", 0.42)),
                float(aim_settings.get("yawSlowdownErrorDeg", 12.0)),
                float(aim_settings.get("minYawCommandWeight", 0.10)),
            )
            action["turretQE"] = {
                "command": str(aim_settings.get("yawRightCommand", "E") if yaw_dir > 0 else aim_settings.get("yawLeftCommand", "Q")),
                "weight": weight,
            }
            aim_state["lastYawCommandDirection"] = yaw_dir
    else:
        aim_state["lastYawCommandDirection"] = 0

    pitch_abs = abs(float(pitch_error))
    pitch_deadband = float(aim_settings.get("pitchDeadbandDeg", 2.2))
    pitch_dir = command_direction(float(pitch_error))
    last_pitch_dir = int(aim_state.get("lastPitchCommandDirection", 0) or 0)
    pitch_suppressed = False

    if pitch_abs > pitch_deadband and pitch_dir != 0:
        if (
            bool(aim_settings.get("suppressReverseCommandNearLock", True))
            and last_pitch_dir != 0
            and pitch_dir != last_pitch_dir
            and pitch_abs <= float(aim_settings.get("pitchReverseSuppressDeg", 4.0))
        ):
            aim_state["lastPitchCommandDirection"] = 0
            pitch_suppressed = True
        else:
            weight = scaled_command_weight(
                pitch_abs,
                float(aim_settings.get("pitchCommandWeight", 0.34)),
                float(aim_settings.get("pitchSlowdownErrorDeg", 10.0)),
                float(aim_settings.get("minPitchCommandWeight", 0.08)),
            )
            action["turretRF"] = {
                "command": str(aim_settings.get("pitchUpCommand", "R") if pitch_dir > 0 else aim_settings.get("pitchDownCommand", "F")),
                "weight": weight,
            }
            aim_state["lastPitchCommandDirection"] = pitch_dir
    else:
        aim_state["lastPitchCommandDirection"] = 0

    debug.update({
        "yawAbsErrorDeg": round(yaw_abs, 3),
        "yawDeadbandDeg": round(yaw_deadband, 3),
        "yawDirection": yaw_dir,
        "yawReverseSuppressed": yaw_suppressed,
        "pitchAbsErrorDeg": round(pitch_abs, 3),
        "pitchDeadbandDeg": round(pitch_deadband, 3),
        "pitchDirection": pitch_dir,
        "pitchReverseSuppressed": pitch_suppressed,
        "commandedTurretQE": json_copy(action.get("turretQE")),
        "commandedTurretRF": json_copy(action.get("turretRF")),
    })
    return debug


def yolo_fused_scan_candidates() -> list[dict[str, Any]]:
    """Return fresh YOLO+LiDAR fused objects as scan candidates.

    This lets flat-ground objects participate in nearest-first scan even when
    the terrain-only LiDAR clustering is conservative.
    """
    if not bool(aim_settings.get("scanYoloFusedObjectsEnabled", True)):
        return []
    candidates: list[dict[str, Any]] = []
    for obj in fresh_yolo_objects():
        if not bool(obj.get("fusionMatched", False)):
            continue
        cluster = obj.get("lidarCluster") or {}
        angle = safe_float(obj.get("lidarBodyAngleDeg"), None)
        distance = safe_float(obj.get("distance"), None)
        if angle is None:
            angle = safe_float(cluster.get("angleDeg"), None)
        if distance is None:
            distance = safe_float(cluster.get("distanceM"), None)
        if angle is None or distance is None:
            continue
        raw_name = str(obj.get("originalRawClassName", obj.get("rawClassName", obj.get("semanticClass", "object"))))
        semantic = str(obj.get("semanticClass", obj.get("originalSemanticClass", raw_name)))
        label = f"YOBJ_{semantic}"
        key = f"YOBJ_{semantic}_a{round(float(angle) / 2.0) * 2:+.0f}_d{round(float(distance) / 5.0) * 5:.0f}"
        candidates.append({
            "clusterId": key,
            "candidateLabel": label,
            "candidateKey": key,
            "angleDeg": round(float(angle), 3),
            "distanceM": round(float(distance), 3),
            "surfaceDistanceM": round(float(distance), 3),
            "aimPitchDeg": round(float(cluster.get("aimPitchDeg", 0.0) or 0.0), 3),
            "pointCount": int(cluster.get("pointCount", 0) or 0),
            "visibleWidthM": cluster.get("visibleWidthM"),
            "heightSpanM": cluster.get("heightSpanM"),
            "objectHeightAboveTerrainM": cluster.get("objectHeightAboveTerrainM"),
            "objectTopYWorldM": cluster.get("objectTopYWorldM"),
            "terrainBaseYWorldM": cluster.get("terrainBaseYWorldM"),
            "aimPointYWorldM": cluster.get("aimPointYWorldM"),
            "aimHeightAboveBaseM": cluster.get("aimHeightAboveBaseM"),
            "depthSpanM": cluster.get("depthSpanM"),
            "verticalityRatio": cluster.get("verticalityRatio"),
            "selectionReason": "nearest_fresh_yolo_fused_object",
            "source": "yolo_fused_scan",
            "yoloClassName": raw_name,
            "yoloSemanticClass": semantic,
            "yoloConfidence": obj.get("confidence"),
        })
    return candidates


def select_nearest_candidate(cache: FrameCache) -> dict[str, Any] | None:
    cleanup_ignored_candidates()
    max_dist = float(aim_settings.get("maxCandidateDistanceM", 120.0))
    min_dist = float(aim_settings.get("minCandidateDistanceM", 3.0))

    # LiDAR clusters + fresh YOLO-fused objects are sorted nearest-first.  This
    # keeps the scan moving outward and allows flat-ground objects to be checked.
    scan_candidates = [json_copy(item) for item in cache.clusters] + yolo_fused_scan_candidates()
    scan_candidates.sort(key=lambda item: (float(item.get("distanceM", 9999.0)), abs(float(item.get("angleDeg", 0.0)))))
    for candidate in scan_candidates:
        dist = float(candidate.get("distanceM", 9999.0))
        if not (min_dist <= dist <= max_dist):
            continue
        if candidate_is_ignored(candidate):
            continue
        chosen = json_copy(candidate)
        chosen["selectionReason"] = "nearest_unchecked_lidar_candidate"
        chosen["ignoreKeys"] = candidate_ignore_keys(chosen)
        return chosen
    return None


def select_fresh_tank_candidate(cache: FrameCache) -> dict[str, Any] | None:
    """
    v16.3: If YOLO already has a fresh LiDAR-fused tank, do not keep staring at
    the nearest hill/rock.  Promote the tank-matched LiDAR cluster first.
    """
    if not bool(aim_settings.get("tankPriorityEnabled", True)):
        return None

    cleanup_ignored_candidates()
    objects = fresh_yolo_objects()
    if not objects:
        return None

    angle_gate = float(aim_settings.get("tankPriorityAngleGateDeg", 10.0))
    distance_gate = float(aim_settings.get("tankPriorityDistanceGateM", 15.0))

    tank_targets: list[tuple[float, float, dict[str, Any]]] = []
    for obj in objects:
        if not bool(obj.get("fusionMatched", False)):
            continue
        semantic = str(obj.get("semanticClass", obj.get("originalSemanticClass", "")))
        raw_name = str(obj.get("originalRawClassName", obj.get("rawClassName", semantic)))
        if not (is_tank_semantic(semantic) or is_tank_semantic(raw_name)):
            continue

        angle = safe_float(obj.get("lidarBodyAngleDeg"))
        distance = safe_float(obj.get("distance"))
        cluster = obj.get("lidarCluster") or {}
        if angle is None:
            angle = safe_float(cluster.get("angleDeg"))
        if distance is None:
            distance = safe_float(cluster.get("distanceM"))
        if angle is None or distance is None:
            continue
        tank_targets.append((float(distance), float(angle), obj))

    if not tank_targets:
        return None

    tank_targets.sort(key=lambda item: item[0])
    tank_distance, tank_angle, tank_obj = tank_targets[0]

    ranked: list[tuple[float, dict[str, Any]]] = []
    for cluster in cache.clusters:
        # A fresh confirmed tank overrides previous terrain/non-tank skips.
        cluster_angle = float(cluster.get("angleDeg", 0.0))
        cluster_distance = float(cluster.get("distanceM", 9999.0))
        ag = angle_gap_deg(cluster_angle, tank_angle)
        dg = abs(cluster_distance - tank_distance)
        if ag <= angle_gate and dg <= distance_gate:
            ranked.append((ag + dg * 0.1, cluster))

    if ranked:
        ranked.sort(key=lambda item: item[0])
        chosen = json_copy(ranked[0][1])
        chosen["selectionReason"] = "fresh_yolo_tank_priority"
        chosen["priorityYoloTank"] = {
            "className": tank_obj.get("className"),
            "semanticClass": tank_obj.get("semanticClass"),
            "distanceM": round(tank_distance, 3),
            "angleDeg": round(tank_angle, 3),
        }
        return chosen

    # Pixel-ROI fusion can produce a LiDAR-supported tank even when it does not
    # map cleanly to one of the precomputed vertical clusters. Use it as a safe
    # fallback so the turret still turns toward the confirmed tank.
    key = f"YTANK_a{round(tank_angle / 2.0) * 2:+.0f}_d{round(tank_distance / 5.0) * 5:.0f}"
    cluster = tank_obj.get("lidarCluster") or {}
    return {
        "clusterId": key,
        "candidateLabel": "YTANK",
        "candidateKey": key,
        "angleDeg": round(tank_angle, 3),
        "distanceM": round(tank_distance, 3),
        "aimPitchDeg": round(float(cluster.get("aimPitchDeg", 0.0) or 0.0), 3),
        "pointCount": int(cluster.get("pointCount", 0) or 0),
        "visibleWidthM": cluster.get("visibleWidthM"),
        "heightSpanM": cluster.get("heightSpanM"),
        "depthSpanM": cluster.get("depthSpanM"),
        "verticalityRatio": cluster.get("verticalityRatio"),
        "selectionReason": "fresh_yolo_tank_priority_roi_fallback",
        "priorityYoloTank": {
            "className": tank_obj.get("className"),
            "semanticClass": tank_obj.get("semanticClass"),
            "distanceM": round(tank_distance, 3),
            "angleDeg": round(tank_angle, 3),
        },
    }



def candidate_distance_for_aim(candidate: dict[str, Any]) -> float:
    for key in ("horizontalRangeM", "surfaceDistanceM", "distanceM", "medianDistanceM"):
        value = safe_float(candidate.get(key), None)
        if value is not None and value > 0:
            return float(value)
    return 0.0


def ballistic_pitch_offset_deg(distance_m: float) -> float:
    if not bool(aim_settings.get("ballisticPitchCompEnabled", True)):
        return 0.0
    base = float(aim_settings.get("ballisticPitchBaseOffsetDeg", 0.15))
    max_offset = float(aim_settings.get("ballisticPitchMaxOffsetDeg", 2.6))
    start_d = float(aim_settings.get("ballisticPitchStartDistanceM", 25.0))
    full_d = max(start_d + 1.0, float(aim_settings.get("ballisticPitchFullDistanceM", 120.0)))
    t = max(0.0, min(1.0, (float(distance_m) - start_d) / (full_d - start_d)))
    # Smooth quadratic ramp: small close-range correction, stronger far-range lift.
    return base + max_offset * (t * t)


def parse_pitch_sweep_offsets() -> list[float]:
    raw = str(aim_settings.get("pitchSweepOffsetsDeg", "0,0.6,-0.6,1.2,-1.2"))
    offsets: list[float] = []
    for token in raw.split(','):
        value = safe_float(token.strip(), None)
        if value is not None and abs(float(value)) <= 10.0:
            offsets.append(float(value))
    return offsets or [0.0]


def candidate_is_yolo_tank_priority(candidate: dict[str, Any]) -> bool:
    if candidate.get("priorityYoloTank") is not None:
        return True
    label = str(candidate.get("candidateLabel", "")).strip().lower()
    return label in {"ytank", "tank", "tank001"}


def pitch_sweep_offset_deg(candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not bool(aim_settings.get("pitchSweepEnabled", True)):
        return 0.0, {"enabled": False}
    if bool(aim_settings.get("pitchSweepOnlyConfirmedTank", True)) and not candidate_is_yolo_tank_priority(candidate):
        return 0.0, {"enabled": True, "active": False, "reason": "not_yolo_tank_priority"}

    offsets = parse_pitch_sweep_offsets()
    key = candidate_coarse_key(candidate)
    state = aim_state.setdefault("pitchSweepState", {})
    current = state.get(key)
    if not isinstance(current, dict):
        current = {"index": 0, "shotCount": 0, "updatedAt": now_text()}
        state[key] = current
    index = int(current.get("index", 0)) % len(offsets)
    return float(offsets[index]), {
        "enabled": True,
        "active": True,
        "key": key,
        "index": index,
        "offsetDeg": round(float(offsets[index]), 3),
        "offsetsDeg": offsets,
        "shotCount": int(current.get("shotCount", 0)),
    }


def advance_pitch_sweep_after_fire(candidate: dict[str, Any]) -> None:
    if not bool(aim_settings.get("pitchSweepEnabled", True)):
        return
    offsets = parse_pitch_sweep_offsets()
    key = candidate_coarse_key(candidate)
    state = aim_state.setdefault("pitchSweepState", {})
    current = state.get(key)
    if not isinstance(current, dict):
        current = {"index": 0, "shotCount": 0}
    current["shotCount"] = int(current.get("shotCount", 0)) + 1
    current["index"] = (int(current.get("index", 0)) + 1) % max(1, len(offsets))
    current["updatedAt"] = now_text()
    state[key] = current


def build_seek_attack_action(cache: FrameCache, turret_state: dict[str, Any]) -> dict[str, Any]:
    action = empty_action()
    if not bool(aim_settings.get("enabled", True)):
        with state_lock:
            aim_state.update({"mode": "disabled", "updatedAt": now_text(), "action": action})
        return action

    candidate = select_fresh_tank_candidate(cache)
    if candidate is None:
        candidate = select_nearest_candidate(cache)
    debug_scan_candidates = [json_copy(item) for item in cache.clusters] + yolo_fused_scan_candidates()
    debug_scan_candidates.sort(key=lambda item: (float(item.get("distanceM", 9999.0)), abs(float(item.get("angleDeg", 0.0)))))
    candidates = debug_scan_candidates[:10]
    now_t = now_text()

    if candidate is None:
        with state_lock:
            aim_state.update({
                "mode": "search_no_valid_object",
                "updatedAt": now_t,
                "candidateCount": len(cache.clusters),
                "candidates": candidates,
                "selectedTarget": None,
                "confirmedTarget": None,
                "yawErrorDeg": None,
                "pitchErrorDeg": None,
                "action": action,
                "debug": {"reason": "no vertical-stack high-Y object"},
            })
            fire_state["lastBlockedReason"] = "no_valid_object"
        return action

    current_yaw = current_turret_body_yaw_deg(cache, turret_state)
    current_pitch = current_turret_pitch_deg(turret_state)
    raw_target_yaw = float(candidate.get("angleDeg", 0.0))
    base_target_pitch = float(candidate.get("aimPitchDeg", 0.0))
    aim_distance_m = candidate_distance_for_aim(candidate)
    ballistic_offset = ballistic_pitch_offset_deg(aim_distance_m)
    pitch_sweep_offset, pitch_sweep_debug = pitch_sweep_offset_deg(candidate)
    raw_target_pitch = base_target_pitch + ballistic_offset + pitch_sweep_offset
    target_yaw, target_pitch, target_smoothing_debug = smooth_target_angles(
        candidate,
        raw_target_yaw,
        raw_target_pitch,
    )
    yaw_error = normalize_signed_angle(target_yaw - current_yaw)
    pitch_error = target_pitch - current_pitch

    aim_control_debug = apply_anti_hunt_turret_control(action, yaw_error, pitch_error)

    yolo_match = classify_selected_candidate_with_yolo(candidate)
    confirmed = None
    yolo_semantic = None
    if yolo_match is not None:
        yolo_semantic = str(yolo_match.get("semanticClass", yolo_match.get("originalSemanticClass", "")))
        raw_name = str(yolo_match.get("originalRawClassName", yolo_match.get("rawClassName", yolo_semantic)))
        is_tank = is_tank_semantic(yolo_semantic) or is_tank_semantic(raw_name)
        if is_tank:
            confirmed = json_copy(yolo_match)
        else:
            # This candidate was looked at and YOLO says it is not a tank; skip it briefly.
            mark_candidate_ignored(candidate, f"non_tank_yolo:{yolo_semantic}", now_t)
            for key in candidate_ignore_keys(candidate):
                aim_state.setdefault("alignedSinceByKey", {}).pop(key, None)

    aligned = (
        abs(yaw_error) <= float(aim_settings.get("fireYawGateDeg", 1.5))
        and abs(pitch_error) <= float(aim_settings.get("firePitchGateDeg", 3.0))
    )
    candidate_key_value = candidate_key(candidate)
    candidate_track_key = candidate_coarse_key(candidate) if bool(aim_settings.get("useCoarseIgnoreKey", True)) else candidate_key_value
    aligned_since_by_key = aim_state.setdefault("alignedSinceByKey", {})
    skipped_after_no_yolo_dwell = False
    no_yolo_dwell_age_sec: float | None = None

    if bool(aim_settings.get("skipNoYoloAfterDwell", True)) and confirmed is None and yolo_match is None:
        if aligned:
            now_mono = monotonic()
            first_aligned = aligned_since_by_key.get(candidate_track_key)
            if first_aligned is None:
                aligned_since_by_key[candidate_track_key] = now_mono
                no_yolo_dwell_age_sec = 0.0
            else:
                no_yolo_dwell_age_sec = max(0.0, now_mono - float(first_aligned))
                dwell_sec = float(aim_settings.get("noYoloDwellSec", 1.2))
                if no_yolo_dwell_age_sec >= dwell_sec:
                    mark_candidate_ignored(
                        candidate,
                        "no_yolo_after_dwell",
                        now_t,
                        {"dwellSec": round(no_yolo_dwell_age_sec, 3)},
                    )
                    for key in candidate_ignore_keys(candidate) + [candidate_track_key]:
                        aligned_since_by_key.pop(key, None)
                    skipped_after_no_yolo_dwell = True
        else:
            aligned_since_by_key.pop(candidate_track_key, None)
    else:
        aligned_since_by_key.pop(candidate_track_key, None)

    can_fire = False
    blocked_reason = None
    if skipped_after_no_yolo_dwell:
        blocked_reason = "no_yolo_after_dwell_skipped"
    elif not bool(aim_settings.get("autoFireEnabled", True)):
        blocked_reason = "auto_fire_disabled"
    elif confirmed is None:
        blocked_reason = f"not_confirmed_tank:{yolo_semantic or 'no_yolo'}"
    elif not aligned:
        blocked_reason = "not_aligned"
    else:
        last_fire = fire_state.get("lastFireMonotonic")
        cooldown = float(aim_settings.get("fireCooldownSec", 1.0))
        if last_fire is not None and monotonic() - float(last_fire) < cooldown:
            blocked_reason = "cooldown"
        else:
            can_fire = True

    if can_fire:
        action["fire"] = True
        with state_lock:
            fire_state["fireCount"] = int(fire_state.get("fireCount", 0)) + 1
            fire_state["lastFireAt"] = now_t
            fire_state["lastFireMonotonic"] = monotonic()
            fire_state["lastFireTarget"] = json_copy(candidate)
            fire_state["lastBlockedReason"] = None
            advance_pitch_sweep_after_fire(candidate)
    else:
        with state_lock:
            fire_state["lastBlockedReason"] = blocked_reason

    with state_lock:
        aim_state.update({
            "mode": "confirmed_tank" if confirmed is not None else "seeking_nearest_object",
            "updatedAt": now_t,
            "candidateCount": len(cache.clusters),
            "candidates": candidates,
            "selectedTarget": json_copy(candidate),
            "confirmedTarget": confirmed,
            "yawErrorDeg": round(float(yaw_error), 3),
            "pitchErrorDeg": round(float(pitch_error), 3),
            "action": json_copy(action),
            "debug": {
                "currentTurretBodyYawDeg": round(float(current_yaw), 3),
                "currentTurretPitchDeg": round(float(current_pitch), 3),
                "rawTargetYawDeg": round(float(raw_target_yaw), 3),
                "baseTargetPitchDeg": round(float(base_target_pitch), 3),
                "ballisticPitchOffsetDeg": round(float(ballistic_offset), 3),
                "pitchSweepOffsetDeg": round(float(pitch_sweep_offset), 3),
                "rawTargetPitchDeg": round(float(raw_target_pitch), 3),
                "targetYawDeg": round(float(target_yaw), 3),
                "targetPitchDeg": round(float(target_pitch), 3),
                "aimDistanceM": round(float(aim_distance_m), 3),
                "pitchSweep": json_copy(pitch_sweep_debug),
                "targetSmoothing": json_copy(target_smoothing_debug),
                "aimControl": json_copy(aim_control_debug),
                "candidateKey": candidate_key_value,
                "candidateTrackKey": candidate_track_key,
                "candidateIgnoreKeys": candidate_ignore_keys(candidate),
                "selectionReason": candidate.get("selectionReason", "nearest_lidar_candidate"),
                "yoloSemanticNearTarget": yolo_semantic,
                "aligned": aligned,
                "noYoloDwellAgeSec": round(float(no_yolo_dwell_age_sec), 3) if no_yolo_dwell_age_sec is not None else None,
                "skippedAfterNoYoloDwell": skipped_after_no_yolo_dwell,
                "blockedReason": blocked_reason,
            },
        })
    return action

@app.route("/get_action", methods=["POST"])
def get_action():
    body = request.get_json(silent=True) or {}
    turret = body.get("turret", {}) or {}
    position = extract_position_dict(body.get("position") or {})

    record_payload_debug("get_action", body)

    with state_lock:
        latest_turret["x"] = safe_float(turret.get("x"), 0.0) or 0.0
        latest_turret["y"] = safe_float(turret.get("y"), 0.0) or 0.0
        latest_turret["updatedAt"] = datetime.now().isoformat(timespec="milliseconds")
        latest_turret["updatedMonotonic"] = monotonic()
        if position is not None:
            latest_player_state["position"] = [position["x"], position["y"], position["z"]]
            latest_player_state["updatedAt"] = now_text()
            latest_player_state["updatedMonotonic"] = monotonic()
            ground_truth_state["latestPlayerPosition"] = list(latest_player_state["position"])
        status_state["getActionRequestCount"] += 1

    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)

    action = build_seek_attack_action(cache, turret_state)
    return jsonify(action)


@app.route("/lidar_status", methods=["GET"])
def lidar_status():
    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)
    return jsonify(
        {
            "frameSeq": cache.seq,
            "simulationTime": cache.simulation_time,
            "rawPointCount": cache.raw_point_count,
            "detectedHitCount": cache.detected_hit_count,
            "groundPointCount": int(cache.ground_mask.sum()),
            "obstaclePointCount": int(cache.obstacle_mask.sum()),
            "validObjectPointCount": int(cache.valid_object_mask.sum()),
            "verticalStackPromotedPointCount": int(cache.stack_promoted_mask.sum()),
            "terrainLikePointCount": int(np.sum(cache.height_above_terrain <= TERRAIN_GROUND_RESIDUAL_TOL_M)) if cache.height_above_terrain.size else 0,
            "aboveTerrainPointCount": int(np.sum(cache.height_above_terrain >= float(aim_settings.get("hillObjectMinTopClearanceM", OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M)))) if cache.height_above_terrain.size else 0,
            "terrainProfile": json_copy(cache.ground_plane_debug.get("terrainProfile", {})),
            "objectFilter": json_copy(cache.ground_plane_debug.get("objectFilter", {})),
            "clusters": json_copy(cache.clusters),
            "turretBodyYawDeg": current_turret_body_yaw_deg(cache, turret_state),
            "turretPitchDeg": current_turret_pitch_deg(turret_state),
        }
    )


@app.route("/aim_status", methods=["GET"])
def aim_status():
    with state_lock:
        return jsonify({"settings": dict(aim_settings), "state": json_copy(aim_state)})


@app.route("/fire_status", methods=["GET"])
def fire_status():
    with state_lock:
        return jsonify(json_copy(fire_state))


@app.route("/action_debug", methods=["GET"])
def action_debug():
    with state_lock:
        cache = latest_cache
        return jsonify(
            {
                "frameSeq": cache.seq,
                "aim": json_copy(aim_state),
                "fire": json_copy(fire_state),
                "latestTurret": json_copy(latest_turret),
                "latestPlayer": json_copy(latest_player_state),
                "validClusters": json_copy(cache.clusters),
                "lastYoloFusedObjects": json_copy(yolo_state.get("latestFusedObjects", [])),
            }
        )


@app.route("/aim_update", methods=["GET", "POST"])
def aim_update():
    numeric_ranges = {
        "maxCandidateDistanceM": (1.0, 200.0),
        "minCandidateDistanceM": (0.0, 50.0),
        "yawDeadbandDeg": (0.1, 20.0),
        "pitchDeadbandDeg": (0.1, 20.0),
        "yawCommandWeight": (0.05, 1.0),
        "pitchCommandWeight": (0.05, 1.0),
        "targetConfirmMaxAgeSec": (0.1, 10.0),
        "targetYoloAngleGateDeg": (0.5, 45.0),
        "targetYoloDistanceGateM": (0.5, 50.0),
        "fireYawGateDeg": (0.1, 10.0),
        "firePitchGateDeg": (0.1, 10.0),
        "fireCooldownSec": (0.1, 10.0),
        "tankPriorityAngleGateDeg": (0.5, 45.0),
        "tankPriorityDistanceGateM": (0.5, 80.0),
        "noYoloDwellSec": (0.1, 10.0),
        "nonTankIgnoreSec": (0.5, 30.0),
        "hillObjectMinTopClearanceM": (0.1, 5.0),
        "hillObjectMinClusterHeightM": (0.1, 6.0),
        "flatObjectMinHeightSpanM": (0.1, 6.0),
        "flatObjectMinPoints": (1.0, 20.0),
        "flatObjectMinVerticalityRatio": (0.1, 10.0),
        "flatObjectMaxRangeSpanM": (0.2, 10.0),
        "targetAimHeightRatio": (0.05, 0.95),
        "targetAimMinClearanceM": (0.0, 5.0),
        "ignoreAngleBinDeg": (1.0, 45.0),
        "ignoreDistanceBinM": (1.0, 80.0),
        "aimTargetSmoothingAlpha": (0.01, 1.0),
        "aimTargetSmoothingResetSec": (0.05, 5.0),
        "yawSlowdownErrorDeg": (0.5, 45.0),
        "pitchSlowdownErrorDeg": (0.5, 45.0),
        "minYawCommandWeight": (0.01, 1.0),
        "minPitchCommandWeight": (0.01, 1.0),
        "yawReverseSuppressDeg": (0.0, 20.0),
        "pitchReverseSuppressDeg": (0.0, 20.0),
        "ballisticPitchBaseOffsetDeg": (-5.0, 5.0),
        "ballisticPitchMaxOffsetDeg": (0.0, 8.0),
        "ballisticPitchStartDistanceM": (0.0, 150.0),
        "ballisticPitchFullDistanceM": (1.0, 200.0),
    }
    bool_keys = {"enabled", "autoFireEnabled", "fireOnTankCandidate", "tankPriorityEnabled", "skipNoYoloAfterDwell", "useCoarseIgnoreKey", "aimTargetSmoothingEnabled", "proportionalAimControl", "suppressReverseCommandNearLock", "flatObjectFallbackEnabled", "scanYoloFusedObjectsEnabled", "ballisticPitchCompEnabled", "pitchSweepEnabled", "pitchSweepOnlyConfirmedTank"}
    string_keys = {"yawRightCommand", "yawLeftCommand", "pitchUpCommand", "pitchDownCommand", "turretYawMode", "pitchSweepOffsetsDeg"}
    for key, (lo, hi) in numeric_ranges.items():
        if key in request.args:
            value = safe_float(request.args.get(key))
            if value is not None:
                aim_settings[key] = max(lo, min(hi, float(value)))
    for key in bool_keys:
        if key in request.args:
            aim_settings[key] = safe_bool(request.args.get(key), bool(aim_settings.get(key)))
    for key in string_keys:
        if key in request.args:
            aim_settings[key] = str(request.args.get(key)).strip()
    return jsonify({"status": "success", "aim": dict(aim_settings)})


def lidar_point_color(cache: FrameCache, idx: int) -> str:
    if cache.valid_object_mask.size and bool(cache.valid_object_mask[idx]):
        return "#ff4d4d"  # object above hill profile
    if cache.height_above_terrain.size and float(cache.height_above_terrain[idx]) >= float(aim_settings.get("hillObjectMinTopClearanceM", OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M)):
        return "#ff9f1c"  # high above terrain but not a full valid object cluster
    if cache.height_above_terrain.size and float(cache.height_above_terrain[idx]) <= TERRAIN_GROUND_RESIDUAL_TOL_M:
        return "#45c96b"  # hill/ground profile
    if cache.obstacle_mask.size and bool(cache.obstacle_mask[idx]):
        return "#ffd23f"
    return "#888888"


def svg_top_lidar(cache: FrameCache, aim_snapshot: dict[str, Any] | None = None, width: int = 980, height: int = 820) -> str:
    """Full 360-degree local polar LiDAR view. 0deg is gun/LiDAR forward, +deg is right."""
    aim_snapshot = aim_snapshot or {}
    cx, cy = width / 2.0, height / 2.0 + 20.0
    max_r = MAX_LIDAR_DISTANCE_M
    scale = min((width - 96) / (2.0 * max_r), (height - 120) / (2.0 * max_r))
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart topchart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#111' stroke='#444'/>")

    # Full 360 rings.
    for rr in [20, 40, 60, 80, 100, 120]:
        rad = rr * scale
        parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{rad:.1f}' fill='none' stroke='#333'/>")
        parts.append(f"<text x='{cx+6:.1f}' y='{cy-rad+13:.1f}' fill='#777' font-size='11'>{rr}m</text>")

    # Full 360 spokes. 0deg is forward/up, +90 right, 180 rear, -90 left.
    for a in [-180, -135, -90, -45, 0, 45, 90, 135]:
        x = cx + max_r * scale * sin(radians(a))
        y = cy - max_r * scale * cos(radians(a))
        parts.append(f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{x:.1f}' y2='{y:.1f}' stroke='#252525'/>")
        label = '+180/-180°' if abs(a) == 180 else f'{a:+d}°'
        parts.append(f"<text x='{x:.1f}' y='{y-4:.1f}' fill='#aaa' font-size='11' text-anchor='middle'>{label}</text>")

    # Raw points, downsampled only for display. Recognition still uses all points.
    idx = np.arange(cache.distances.size, dtype=np.int32)
    if idx.size > 5200:
        idx = idx[np.linspace(0, idx.size - 1, 5200).astype(np.int32)]
    for i in idx.tolist():
        dist = float(cache.horizontal_ranges[i] if cache.horizontal_ranges.size else cache.distances[i])
        if not (0.0 < dist <= max_r):
            continue
        a = normalize_signed_angle(float(cache.angles[i]))
        x = cx + dist * scale * sin(radians(a))
        y = cy - dist * scale * cos(radians(a))
        c = lidar_point_color(cache, i)
        is_obj = cache.valid_object_mask.size and bool(cache.valid_object_mask[i])
        r = 2.4 if is_obj else 1.25
        opacity = 0.86 if is_obj else 0.34
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{c}' opacity='{opacity:.2f}'/>")

    selected = aim_snapshot.get("selectedTarget") or {}
    confirmed = aim_snapshot.get("confirmedTarget") or {}
    selected_key = str(selected.get("candidateKey", ""))

    # Draw recognized LiDAR object candidates over the raw points.
    for item in cache.clusters[:80]:
        dist = float(item.get("distanceM", 0.0) or 0.0)
        a = normalize_signed_angle(float(item.get("angleDeg", 0.0) or 0.0))
        if not (0.0 < dist <= max_r):
            continue
        x = cx + dist * scale * sin(radians(a))
        y = cy - dist * scale * cos(radians(a))
        key = str(item.get("candidateKey", ""))
        is_selected = key == selected_key
        color = '#00e5ff' if is_selected else '#ff4d4d'
        radius = 13 if is_selected else 7
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='none' stroke='{color}' stroke-width='3'/>")
        h = safe_float(item.get("objectHeightAboveTerrainM"), None)
        pitch = safe_float(item.get("aimPitchDeg"), None)
        htxt = f" h{h:.1f}m" if h is not None else ""
        ptxt = f" p{pitch:+.1f}°" if pitch is not None else ""
        parts.append(f"<text x='{x+9:.1f}' y='{y-7:.1f}' fill='{color}' font-size='12'>{html.escape(str(item.get('candidateLabel','OBJ')))} {dist:.1f}m {a:+.1f}°{htxt}{ptxt}</text>")

    # Confirmed YOLO tank marker if present.
    if isinstance(confirmed, dict) and confirmed:
        angle = safe_float(confirmed.get("lidarBodyAngleDeg"), None)
        dist = safe_float(confirmed.get("distance"), None)
        if angle is not None and dist is not None:
            a = normalize_signed_angle(float(angle))
            x = cx + float(dist) * scale * sin(radians(a))
            y = cy - float(dist) * scale * cos(radians(a))
            parts.append(f"<text x='{x+14:.1f}' y='{y+20:.1f}' fill='#00ffff' font-size='14'>CONFIRMED TANK</text>")
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='17' fill='none' stroke='#00ffff' stroke-width='3'/>")

    # Own tank center and gun-forward line.
    parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='11' fill='#45d9ff'/>")
    parts.append(f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{cx:.1f}' y2='{cy-42:.1f}' stroke='#00ffff' stroke-width='3'/>")
    parts.append("<text x='12' y='20' fill='#eee' font-size='14'>360° Top LiDAR: 0°=gun/LiDAR forward, +right, 180°=rear | green=hill/ground, red=object, cyan=selected/confirmed</text>")
    parts.append("<text x='12' y='40' fill='#aaa' font-size='12'>This is display downsampling only. Object recognition still uses the full cached LiDAR frame.</text>")
    parts.append("</svg>")
    return "".join(parts)

def svg_front_lidar(cache: FrameCache, width: int = 820, height: int = 360) -> str:
    margin_l, margin_r, margin_t, margin_b = 48, 16, 22, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    min_a, max_a = -60.0, 60.0
    min_v, max_v = -22.5, 22.5
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#151515' stroke='#444'/>")
    for a in [-60, -30, 0, 30, 60]:
        x = margin_l + (a - min_a) / (max_a - min_a) * plot_w
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#333'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-10}' fill='#bbb' font-size='11' text-anchor='middle'>{a:+d}°</text>")
    for v in [-22.5, -10, 0, 10, 22.5]:
        y = margin_t + (max_v - v) / (max_v - min_v) * plot_h
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#333'/>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#bbb' font-size='11'>{v:+.1f}°</text>")
    idx = np.flatnonzero((cache.angles >= min_a) & (cache.angles <= max_a))
    if idx.size > 1400:
        idx = idx[np.linspace(0, idx.size - 1, 1400).astype(np.int32)]
    for i in idx.tolist():
        a = float(cache.angles[i]); v = float(cache.vertical_angles[i])
        x = margin_l + (a - min_a) / (max_a - min_a) * plot_w
        y = margin_t + (max_v - v) / (max_v - min_v) * plot_h
        c = lidar_point_color(cache, i)
        r = 2.2 if cache.valid_object_mask.size and cache.valid_object_mask[i] else 1.4
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{c}' opacity='0.88'/>")
    parts.append("<text x='410' y='16' fill='#eee' font-size='13' text-anchor='middle'>Raw front LiDAR: angle × vertical channel</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_side_profile(cache: FrameCache, width: int = 820, height: int = 360) -> str:
    margin_l, margin_r, margin_t, margin_b = 48, 16, 22, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_r = MAX_LIDAR_DISTANCE_M
    min_y, max_y = -5.0, 15.0
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#151515' stroke='#444'/>")
    for rr in [0, 20, 40, 60, 80, 100, 120]:
        x = margin_l + rr / max_r * plot_w
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#333'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-10}' fill='#bbb' font-size='11' text-anchor='middle'>{rr}m</text>")
    for yy in [-5, 0, 5, 10, 15]:
        y = margin_t + (max_y - yy) / (max_y - min_y) * plot_h
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#333'/>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#bbb' font-size='11'>{yy:+d}m</text>")
    origin = cache.pose.get("lidarOrigin", {}) if isinstance(cache.pose, dict) else {}
    origin_y = safe_float(origin.get("y"), EXPECTED_LIDAR_Y_POSITION_M) or EXPECTED_LIDAR_Y_POSITION_M
    if cache.angles.size:
        selected_az = float(cache.angles[np.argmin(np.abs(cache.angles))])
        idx = np.flatnonzero(np.abs(cache.angles - selected_az) <= 0.75)
    else:
        selected_az = 0.0; idx = np.empty(0, dtype=np.int32)
    for i in idx.tolist():
        rr = float(cache.horizontal_ranges[i])
        yy = float(cache.xyz[i, 1] - origin_y)
        if not (0 <= rr <= max_r and min_y <= yy <= max_y):
            continue
        x = margin_l + rr / max_r * plot_w
        y = margin_t + (max_y - yy) / (max_y - min_y) * plot_h
        c = lidar_point_color(cache, i)
        r = 2.4 if cache.valid_object_mask.size and cache.valid_object_mask[i] else 1.5
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{c}' opacity='0.9'/>")
    if idx.size and cache.terrain_y.size == cache.xyz.shape[0]:
        terrain_points = []
        for i in idx.tolist():
            rr = float(cache.horizontal_ranges[i])
            yy = float(cache.terrain_y[i] - origin_y)
            if 0 <= rr <= max_r and min_y <= yy <= max_y:
                x = margin_l + rr / max_r * plot_w
                y = margin_t + (max_y - yy) / (max_y - min_y) * plot_h
                terrain_points.append((rr, x, y))
        terrain_points.sort(key=lambda item: item[0])
        if len(terrain_points) >= 2:
            d = " ".join(("M" if k == 0 else "L") + f" {x:.1f} {y:.1f}" for k, (_, x, y) in enumerate(terrain_points))
            parts.append(f"<path d='{d}' fill='none' stroke='#45c96b' stroke-width='1.5' opacity='0.75'/>")
    y0 = margin_t + (max_y - 0) / (max_y - min_y) * plot_h
    parts.append(f"<line x1='{margin_l}' y1='{y0:.1f}' x2='{margin_l+plot_w}' y2='{y0:.1f}' stroke='#00d4d4'/>")
    parts.append(f"<text x='410' y='16' fill='#eee' font-size='13' text-anchor='middle'>Side profile: azimuth {selected_az:+.1f}° | green=hill profile, red=object above hill</text>")
    parts.append("</svg>")
    return "".join(parts)


@app.route("/lidar_360", methods=["GET"])
def lidar_360_view():
    """Dedicated full-screen 360-degree LiDAR monitor."""
    with state_lock:
        cache = latest_cache
        aim_snapshot = json_copy(aim_state)

    page = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta http-equiv='refresh' content='0.35'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>360 LiDAR Monitor</title>
<style>
html, body {{
    width:100%;
    height:100%;
    margin:0;
    overflow:hidden;
    background:#080808;
    color:#eee;
    font-family:Arial,sans-serif;
}}
.viewer {{
    width:100vw;
    height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
}}
.viewer svg {{
    width:100%;
    height:100%;
    max-width:100vw;
    max-height:100vh;
}}
.status {{
    position:fixed;
    right:12px;
    bottom:10px;
    padding:7px 10px;
    border:1px solid #444;
    border-radius:5px;
    background:rgba(0,0,0,.72);
    color:#bbb;
    font-size:12px;
}}
a {{ color:#6fdcff; }}
</style>
</head>
<body>
<div class='viewer'>{svg_top_lidar(cache, aim_snapshot, width=1200, height=1000)}</div>
<div class='status'>
frame {cache.seq} · detected {cache.detected_hit_count} · objects {len(cache.clusters)}
· <a href='/lidar_view'>dashboard</a>
</div>
</body>
</html>"""
    return page


@app.route("/dashboard", methods=["GET"])
@app.route("/lidar_view", methods=["GET"])
def lidar_view():
    with state_lock:
        cache = latest_cache
        aim_snapshot = json_copy(aim_state)
        fire_snapshot = json_copy(fire_state)
        yolo_snapshot = json_copy(yolo_state)
        fusion_snapshot = dict(fusion_settings)
    rows = []
    for item in cache.clusters[:20]:
        selected = aim_snapshot.get("selectedTarget") or {}
        is_sel = str(selected.get("candidateKey")) == str(item.get("candidateKey"))
        rows.append(
            f"<tr class='{ 'sel' if is_sel else '' }'>"
            f"<td>{html.escape(str(item.get('candidateKey')))}</td>"
            f"<td>{html.escape(str(item.get('candidateLabel')))}</td>"
            f"<td>{float(item.get('distanceM', 0)):.1f}</td>"
            f"<td>{float(item.get('angleDeg', 0)):+.1f}</td>"
            f"<td>{float(item.get('aimPitchDeg', 0)):+.1f}</td>"
            f"<td>{item.get('pointCount')}</td>"
            f"<td>{item.get('heightSpanM')}</td>"
            f"<td>{item.get('objectHeightAboveTerrainM', '-')}</td>"
            f"<td>{item.get('surfaceDistanceM', item.get('distanceM', '-'))}</td>"
            f"<td>{item.get('terrainBaseYWorldM', '-')}</td>"
            f"<td>{item.get('objectTopYWorldM', '-')}</td>"
            f"<td>{item.get('aimPointYWorldM', '-')}</td>"
            f"<td>{item.get('aimHeightAboveBaseM', '-')}</td>"
            f"<td>{item.get('depthSpanM', '-')}</td>"
            f"<td>{item.get('verticalityRatio', '-')}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='15'>No object-above-hill LiDAR clusters yet.</td></tr>")

    names = yolo_snapshot.get("modelNames") or MODEL_CLASS_NAMES
    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='0.7'>
<title>v16.7 YOLO DEBUG lalast.pt + LiDAR fusion dashboard</title>
<style>
body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:14px; }}
h1 {{ margin: 0 0 8px 0; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
.card {{ background:#1d1d1d; border:1px solid #444; padding:10px; margin:10px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} th,td {{ border:1px solid #555; padding:5px; text-align:right; }}
th:first-child,td:first-child, th:nth-child(2),td:nth-child(2) {{ text-align:left; }} .sel {{ background:#3a3000; color:#ffd34d; }}
.good {{ color:#6fe36f; }} .warn {{ color:#ffce54; }} .bad {{ color:#ff7777; }}
.chart {{ width:100%; max-height:360px; }} .topchart {{ max-height:820px; }} code {{ color:#9cdcfe; }}
</style></head><body>
<h1>LiDAR 360° top-view object scan + YOLO fusion + pitch-sweep tank fire v16.7</h1>
<div>{svg_top_lidar(cache, aim_snapshot)}</div>
<div class='grid'>
<div>{svg_front_lidar(cache)}</div>
<div>{svg_side_profile(cache)}</div>
</div>
<div class='card'>
<b>Frame:</b> {cache.seq} / time={cache.simulation_time}<br>
<b>Points:</b> raw={cache.raw_point_count}, detected={cache.detected_hit_count}, ground={int(cache.ground_mask.sum())}, obstacle={int(cache.obstacle_mask.sum())}, <span class='good'>validObject={int(cache.valid_object_mask.sum())}</span><br>
<b>Terrain profile:</b> {html.escape(str(cache.ground_plane_debug.get('terrainProfile', {})))}<br>
<b>Object filter:</b> {html.escape(str(cache.ground_plane_debug.get('objectFilter', {})))}<br>
<b>YOLO:</b> model={html.escape(str(fusion_snapshot.get('modelPath')))} / loaded={yolo_snapshot.get('modelLoaded')} / names={html.escape(str(names))} / conf={fusion_snapshot.get('confidence')} / iou={fusion_snapshot.get('iou')} / imgsz={fusion_snapshot.get('imageSize')} / max_det={fusion_snapshot.get('maxDetections')} / augment={fusion_snapshot.get('augment')}<br>
<b>Aim:</b> mode={aim_snapshot.get('mode')} / yawErr={aim_snapshot.get('yawErrorDeg')} / pitchErr={aim_snapshot.get('pitchErrorDeg')}<br>
<b>Fire:</b> count={fire_snapshot.get('fireCount')} / blocked={fire_snapshot.get('lastBlockedReason')}<br>
<b>YOLO debug:</b> submitted={yolo_snapshot.get('submittedCount')} / completed={yolo_snapshot.get('completedCount')} / failed={yolo_snapshot.get('failedCount')} / det={len(yolo_snapshot.get('latestYoloDetections', []))} / fused={len(yolo_snapshot.get('latestFusedObjects', []))} / error={html.escape(str(yolo_snapshot.get('modelLoadError')))}
</div>
<table><thead><tr><th>key</th><th>type</th><th>distance m</th><th>angle deg</th><th>pitch deg</th><th>points</th><th>height span m</th><th>obj above hill m</th><th>surface dist m</th><th>terrain Y</th><th>top Y</th><th>aim Y</th><th>aim h</th><th>depth span m</th><th>verticality</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class='card'>
<b>Links:</b>
<a href='/lidar_360'>full-screen 360 LiDAR</a> |
<a href='/lidar_status'>lidar_status</a> |
<a href='/aim_status'>aim_status</a> |
<a href='/fire_status'>fire_status</a> |
<a href='/action_debug'>action_debug</a> |
<a href='/fusion_status'>fusion_status</a> |
<a href='/yolo_preload'>yolo_preload</a><br>
360° top-view target scan: LiDAR points are shown around the full vehicle. Objects are checked nearest-first; confirmed YOLO Tank001 gets priority. Green=ground/hill, yellow=obstacle, red=LiDAR object, cyan=selected/confirmed target.
</div>
</body></html>"""
    return page


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)
        counters = dict(status_state)
        yolo_snapshot = json_copy(yolo_state)

    return jsonify(
        {
            "server": "Tank Challenge LiDAR-first YOLO Fusion v16.7 360 Pitch-Sweep Tank Fire",
            "purpose": "realtime LiDAR primary pipeline with asynchronous YOLO semantic fusion",
            "recommendedSimulatorProperties": {
                "intervalSec": EXPECTED_INTERVAL_SEC,
                "lidarYPositionM": EXPECTED_LIDAR_Y_POSITION_M,
                "channel": EXPECTED_CHANNELS,
                "minimapChannel": EXPECTED_MINIMAP_CHANNEL,
                "maxDistanceM": EXPECTED_MAX_DISTANCE_M,
                "frameRate": 60,
            },
            "frame": {
                "seq": cache.seq,
                "simulationTime": cache.simulation_time,
                "analysisMs": cache.analysis_ms,
                "rawPointCount": cache.raw_point_count,
                "detectedHitCount": cache.detected_hit_count,
                "groundPointCount": int(cache.ground_mask.sum()),
                "obstaclePointCount": int(cache.obstacle_mask.sum()),
                "validObjectPointCount": int(cache.valid_object_mask.sum()),
                "verticalStackPromotedPointCount": int(cache.stack_promoted_mask.sum()),
                "terrainLikePointCount": int(np.sum(cache.height_above_terrain <= TERRAIN_GROUND_RESIDUAL_TOL_M)) if cache.height_above_terrain.size else 0,
                "aboveTerrainPointCount": int(np.sum(cache.height_above_terrain >= float(aim_settings.get("hillObjectMinTopClearanceM", OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M)))) if cache.height_above_terrain.size else 0,
                "groundNormal": cache.ground_normal.tolist(),
                "groundPlane": json_copy(cache.ground_plane_debug),
                "availableInfoKeys": json_copy(cache.pose.get("availableInfoKeys", [])),
                "rawBodyTiltFields": {
                    "playerBodyY": cache.pose.get("playerBodyY"),
                    "playerBodyZ": cache.pose.get("playerBodyZ"),
                },
                "clusters": json_copy(cache.clusters),
            },
            "turret": turret_state,
            "calibration": dict(calibration),
            "overlay": dict(overlay_settings),
            "fusion": dict(fusion_settings),
            "yolo": yolo_snapshot,
            "groundTruth": {
                "settings": dict(ground_truth_settings),
                "registeredObjectCount": len(ground_truth_state.get("objects", {})),
                "lastLoadAt": ground_truth_state.get("lastLoadAt"),
                "lastLoadError": ground_truth_state.get("lastLoadError"),
                "lastComparisonAt": ground_truth_state.get("lastComparisonAt"),
                "lastComparisons": json_copy(ground_truth_state.get("lastComparisons", [])),
            },
            "aim": {
                "settings": dict(aim_settings),
                "state": json_copy(aim_state),
            },
            "fire": json_copy(fire_state),
            "performance": counters,
        }
    )


@app.route("/fusion_status", methods=["GET"])
def fusion_status():
    with state_lock:
        cache = latest_cache
        yolo_snapshot = json_copy(yolo_state)

    return jsonify(
        {
            "frameSeq": cache.seq,
            "simulationTime": cache.simulation_time,
            "lidarClusters": json_copy(cache.clusters),
            "settings": dict(fusion_settings),
            "yolo": yolo_snapshot,
        }
    )


@app.route("/yolo_tuning", methods=["GET"])
def yolo_tuning():
    return jsonify(
        {
            "status": "success",
            "modelPath": current_yolo_model_path(),
            "requestedTuning": {
                "YOLO_CONF": YOLO_CONF,
                "YOLO_IOU": YOLO_IOU,
                "YOLO_IMGSZ": YOLO_IMGSZ,
                "YOLO_MAX_DET": YOLO_MAX_DET,
                "YOLO_AUGMENT": YOLO_AUGMENT,
            },
            "activeFusionSettings": {
                "confidence": fusion_settings.get("confidence"),
                "iou": fusion_settings.get("iou"),
                "imageSize": fusion_settings.get("imageSize"),
                "maxDetections": fusion_settings.get("maxDetections"),
                "augment": fusion_settings.get("augment"),
            },
            "modelClassNamesFallback": MODEL_CLASS_NAMES,
            "semanticMapForCurrentModel": {
                "Ally": "ally",
                "Enemy": "enemy",
                "House": "house",
                "Rock": "rock",
                "Rock_L": "rock",
                "Tank_enemy": "enemy_tank",
                "Tent": "tent",
                "car": "car",
            },
        }
    )


@app.route("/fusion_debug", methods=["GET"])
def fusion_debug():
    with state_lock:
        cache = latest_cache
        yolo_snapshot = json_copy(yolo_state)
        counters = dict(status_state)

    detections = list(yolo_snapshot.get("latestYoloDetections", []))
    fused_objects = list(yolo_snapshot.get("latestFusedObjects", []))
    matched_count = sum(1 for obj in fused_objects if bool(obj.get("fusionMatched", False)))

    if yolo_snapshot.get("modelLoadError"):
        likely_reason = "YOLO model import/load/inference error. Read yolo.modelLoadError."
    elif int(yolo_snapshot.get("submittedCount", 0)) == 0:
        likely_reason = "No YOLO image job submitted. Confirm Detect Mode and multipart image input."
    elif int(yolo_snapshot.get("completedCount", 0)) == 0:
        likely_reason = "YOLO job has not completed yet. Wait briefly or inspect failedCount/modelLoadError."
    elif not detections:
        likely_reason = "YOLO ran but detected no objects. Lower confidence or verify the .pt model/classes."
    elif matched_count == 0:
        likely_reason = "YOLO works, but LiDAR-cluster fusion did not match. Yellow YOLO? boxes should be visible; tune angle calibration/gap."
    else:
        likely_reason = "Fusion is working. Matched colored boxes with LiDAR distance should be visible."

    return jsonify(
        {
            "likelyReason": likely_reason,
            "lidar": {
                "frameSeq": cache.seq,
                "clusterCount": len(cache.clusters),
                "clusters": json_copy(cache.clusters),
            },
            "yolo": yolo_snapshot,
            "counts": {
                "latestYoloDetectionCount": len(detections),
                "latestFusedObjectCount": len(fused_objects),
                "matchedFusionCount": matched_count,
                "lastReturnedFusedBoxCount": counters.get("lastReturnedFusedBoxCount"),
            },
            "firstActions": [
                "Open /yolo_preload once before starting the simulator.",
                "Keep turret still for 3 seconds during the first test.",
                "Yellow YOLO? box = image sensing works but LiDAR match needs tuning.",
                "Colored box with distance = YOLO and LiDAR fusion matched.",
            ],
        }
    )


@app.route("/yolo_preload", methods=["GET"])
def yolo_preload():
    try:
        get_yolo_model()
        return jsonify(
            {
                "status": "success",
                "message": "YOLO model loaded",
                "modelPath": current_yolo_model_path(),
                "modelNames": json_copy(yolo_state.get("modelNames", {})),
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "modelPath": current_yolo_model_path(),
            }
        ), 500


@app.route("/calibration_status", methods=["GET"])
def calibration_status():
    return jsonify({"status": "success", "calibration": dict(calibration)})


@app.route("/calibration_update", methods=["GET", "POST"])
def calibration_update():
    numeric_fields = {
        "cameraHorizontalFovDeg": (10.0, 160.0),
        "cameraVerticalFovDeg": (5.0, 120.0),
        "cameraOffsetForwardM": (-10.0, 10.0),
        "cameraOffsetRightM": (-10.0, 10.0),
        "cameraOffsetUpM": (-10.0, 10.0),
        "yawOffsetDeg": (-180.0, 180.0),
        "pitchOffsetDeg": (-90.0, 90.0),
        "screenCenterOffsetXPx": (-2000.0, 2000.0),
        "screenCenterOffsetYPx": (-2000.0, 2000.0),
        "turretYawSign": (-1.0, 1.0),
        "turretPitchSign": (-1.0, 1.0),
        "latestActionFreshnessSec": (0.05, 5.0),
        "tiltSmoothingAlpha": (0.01, 1.0),
        "maxGroundTiltDeg": (0.0, 45.0),
        "rollOffsetDeg": (-45.0, 45.0),
    }
    for key, (minimum, maximum) in numeric_fields.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            calibration[key] = max(minimum, min(maximum, value))

    if "turretYawMode" in request.args:
        value = str(request.args.get("turretYawMode", "absolute")).strip().lower()
        if value in {"absolute", "body_plus_relative"}:
            calibration["turretYawMode"] = value

    if "cameraPoseMode" in request.args:
        value = str(request.args.get("cameraPoseMode", "same_frame_info")).strip().lower()
        if value in {"same_frame_info", "latest_action", "auto"}:
            calibration["cameraPoseMode"] = value

    if "tiltCompensationMode" in request.args:
        value = str(request.args.get("tiltCompensationMode", "ground_plane")).strip().lower()
        if value in {"off", "ground_plane"}:
            calibration["tiltCompensationMode"] = value

    return jsonify({"status": "success", "calibration": dict(calibration)})


@app.route("/overlay_update", methods=["GET", "POST"])
def overlay_update():
    integer_fields = {
        "obstacleBoxLimit": (0, 2000),
        "safeGroundBoxLimit": (0, 1000),
        "totalLidarBoxLimit": (0, 3000),
        "obstaclePixelCell": (1, 50),
        "safeGroundPixelCell": (1, 100),
    }
    for key, (minimum, maximum) in integer_fields.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            overlay_settings[key] = int(max(minimum, min(maximum, round(value))))

    for key in {"showLidarPoints", "showSafeGround"}:
        if key in request.args:
            overlay_settings[key] = str(request.args.get(key)).strip().lower() in {
                "1", "true", "yes", "on"
            }

    return jsonify({"status": "success", "overlay": dict(overlay_settings)})


@app.route("/fusion_update", methods=["GET", "POST"])
def fusion_update():
    numeric_fields = {
        "confidence": (0.05, 0.95),
        "iou": (0.10, 0.95),
        "imageSize": (160, 1280),
        "yoloIntervalSec": (0.10, 10.0),
        "maxFusionAngleGapDeg": (1.0, 45.0),
        "maxDisplayAgeSec": (0.10, 10.0),
        "maxDisplayYawDeltaDeg": (1.0, 45.0),
        "roiExpandRatio": (0.0, 0.50),
        "roiMinObstaclePoints": (1, 100),
        "roiSurfaceBandM": (0.5, 10.0),
        "maxDisplayPitchDeltaDeg": (0.5, 45.0),
        "maxDisplayPositionDeltaM": (0.1, 20.0),
        "maxDisplayGroundNormalDeltaDeg": (0.5, 45.0),
        "maxDetections": (1, 100),
        "tankRescueMinWidthM": (0.5, 20.0),
        "tankRescueMinHeightSpanM": (0.1, 10.0),
        "tankRescueMinRoiPoints": (1, 100),
        "tankRescueMinBoxAspectRatio": (0.1, 10.0),
        "yoloOnlyHintAngleGateDeg": (1.0, 45.0),
    }
    for key, (minimum, maximum) in numeric_fields.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            value = max(minimum, min(maximum, value))
            fusion_settings[key] = (
                int(round(value))
                if key in {"imageSize", "roiMinObstaclePoints", "maxDetections", "tankRescueMinRoiPoints"}
                else float(value)
            )

    for key in {
        "enabled",
        "augment",
        "roiFusionEnabled",
        "clusterFallbackEnabled",
        "showFusedBoxes",
        "showUnmatchedYoloBoxes",
        "halfPrecisionAuto",
        "tankCandidateRescueEnabled",
        "showYoloOnlyAngleLabel",
        "showYoloOnlyLidarHint",
    }:
        if key in request.args:
            fusion_settings[key] = str(request.args.get(key)).strip().lower() in {
                "1", "true", "yes", "on"
            }

    if "device" in request.args:
        fusion_settings["device"] = str(request.args.get("device", "auto")).strip()

    if "tankDisplayName" in request.args:
        candidate = str(request.args.get("tankDisplayName", "tank")).strip()
        fusion_settings["tankDisplayName"] = candidate[:40] if candidate else "tank"

    if "tankCandidateDisplayName" in request.args:
        candidate = str(request.args.get("tankCandidateDisplayName", "tank?")).strip()
        fusion_settings["tankCandidateDisplayName"] = candidate[:40] if candidate else "tank?"

    if "tankCandidateSourceClasses" in request.args:
        fusion_settings["tankCandidateSourceClasses"] = str(
            request.args.get("tankCandidateSourceClasses", "car2")
        ).strip()[:200]

    return jsonify({"status": "success", "fusion": dict(fusion_settings)})


@app.route("/reset_state", methods=["GET", "POST"])
def reset_state():
    global latest_cache, _pending_vision_job

    with state_lock:
        latest_cache = EMPTY_CACHE
        _pending_vision_job = None
        tilt_state["smoothedGroundNormal"] = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        tilt_state["updatedAt"] = None
        _yolo_event.clear()

        latest_turret["x"] = 0.0
        latest_turret["y"] = 0.0
        latest_turret["updatedAt"] = None
        latest_turret["updatedMonotonic"] = None

        aim_state["smoothedTargetByKey"] = {}
        aim_state["lastYawCommandDirection"] = 0
        aim_state["lastPitchCommandDirection"] = 0

        for key in list(status_state):
            if key.endswith("Count"):
                status_state[key] = 0
            elif key.startswith("last"):
                status_state[key] = None

        yolo_state["workerBusy"] = False
        yolo_state["pendingJob"] = False
        yolo_state["replacedPendingJobCount"] = 0
        yolo_state["latestYoloDetections"] = []
        yolo_state["latestFusedObjects"] = []
        yolo_state["latestResultMeta"] = {}

        aim_state["mode"] = "idle"
        aim_state["candidateCount"] = 0
        aim_state["candidates"] = []
        aim_state["selectedTarget"] = None
        aim_state["confirmedTarget"] = None
        aim_state["ignoredCandidateKeys"] = {}
        aim_state["alignedSinceByKey"] = {}
        aim_state["lastSkippedCandidate"] = None
        aim_state["checkedCandidateHistory"] = []
        aim_state["yawErrorDeg"] = None
        aim_state["pitchErrorDeg"] = None
        aim_state["action"] = empty_action()
        aim_state["debug"] = {}
        fire_state["lastBlockedReason"] = None

    return jsonify({"status": "success"})



@app.route("/gt_register", methods=["GET", "POST"])
def gt_register():
    if request.method == "GET":
        raw: dict[str, Any] = {
            "id": request.args.get("objectId", request.args.get("id")),
            "className": request.args.get("className", request.args.get("class", "object")),
            "position": {
                "x": request.args.get("x"),
                "y": request.args.get("y"),
                "z": request.args.get("z"),
            },
            "radiusM": request.args.get("radiusM"),
            "dynamic": request.args.get("dynamic"),
        }
        record = register_gt_object(raw, source="manual:get", dynamic_default=False)
        if record is None:
            return jsonify(
                {
                    "status": "error",
                    "message": "Use objectId, className, x, y, z. radiusM and dynamic are optional.",
                }
            ), 400
        return jsonify({"status": "success", "registered": [record]})

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"status": "error", "message": "Send JSON object or {'objects': [...]}."}), 400

    records = ingest_gt_payload(payload, source="manual:post", dynamic_default=False, allow_direct=True)
    return jsonify({"status": "success", "registeredCount": len(records), "registered": records})


@app.route("/gt_status", methods=["GET"])
def gt_status():
    with state_lock:
        cache = latest_cache
        snapshot = {
            "lastLoadAt": ground_truth_state.get("lastLoadAt"),
            "lastLoadError": ground_truth_state.get("lastLoadError"),
            "lastRegisterAt": ground_truth_state.get("lastRegisterAt"),
            "lastComparisonAt": ground_truth_state.get("lastComparisonAt"),
            "lastComparisons": json_copy(ground_truth_state.get("lastComparisons", [])),
            "autoExtractedCount": ground_truth_state.get("autoExtractedCount", 0),
            "registeredObjects": json_copy(list(ground_truth_state.get("objects", {}).values())),
        }

    return jsonify(
        {
            "serverProcessId": SERVER_PROCESS_ID,
            "serverSessionId": SERVER_SESSION_ID,
            "serverStartedAt": SERVER_STARTED_AT,
            "persistedMapSelection": read_persisted_map_selection(),
            "settings": dict(ground_truth_settings),
            "frameSeq": cache.seq,
            "simulationTime": cache.simulation_time,
            **snapshot,
            "diagnosis": gt_diagnosis(cache),
            "registeredClassCounts": count_registered_gt_classes(),
            "activeMapFile": ground_truth_state.get("activeMapFile"),
            "activeMapTerrainIndex": ground_truth_state.get("activeMapTerrainIndex"),
            "lastMapLoadAt": ground_truth_state.get("lastMapLoadAt"),
            "lastMapLoadError": ground_truth_state.get("lastMapLoadError"),
            "lastMapRegisteredCount": ground_truth_state.get("lastMapRegisteredCount", 0),
            "errorLogPath": ground_truth_state.get("errorLogPath"),
            "errorLogRowCount": ground_truth_state.get("errorLogRowCount", 0),
            "lastClearAt": ground_truth_state.get("lastClearAt"),
            "lastClearReason": ground_truth_state.get("lastClearReason"),
            "lastClearRemovedCount": ground_truth_state.get("lastClearRemovedCount", 0),
            "lastAutoRestoreAt": ground_truth_state.get("lastAutoRestoreAt"),
            "lastAutoRestoreResult": ground_truth_state.get("lastAutoRestoreResult"),
            "activeObjectMetrics": active_gt_metrics(cache),
            "notes": [
                "GTc is the exact map-pivot center distance.",
                "LiDAR range is a nearest visible surface distance.",
                "When radiusM is provided, approxSurfaceDistanceM is only an approximation.",
                "Moving objects require a live position refresh; an initial map coordinate becomes stale.",
            ],
        }
    )


@app.route("/gt_reload", methods=["GET", "POST"])
def gt_reload():
    clear_existing = safe_bool(request.args.get("clearExisting"), False)
    return jsonify(load_ground_truth_file(clear_existing=clear_existing))


@app.route("/gt_clear", methods=["GET", "POST"])
def gt_clear():
    forget_map = safe_bool(request.args.get("forgetMap"), False)
    removed = clear_gt_objects(
        reason="gt_clear",
        forget_persisted_map=forget_map,
    )
    return jsonify(
        {
            "status": "success",
            "removedCount": removed,
            "forgetMap": forget_map,
            "note": (
                "Use forgetMap=true only when you intentionally want to remove "
                "the persisted active-map selection."
            ),
        }
    )


@app.route("/gt_update", methods=["GET"])
def gt_update():
    numeric_limits: dict[str, tuple[float, float]] = {
        "matchMaxAngleGapDeg": (0.1, 180.0),
        "matchMaxRangeGapM": (0.1, 500.0),
        "rangeWeight": (0.0, 100.0),
        "classMismatchPenalty": (0.0, 100.0),
        "dynamicObjectTtlSec": (0.1, 3600.0),
        "errorLogMinIntervalSec": (0.0, 60.0),
        "bodyYawSign": (-1.0, 1.0),
        "bodyYawOffsetDeg": (-360.0, 360.0),
    }
    for key, (low, high) in numeric_limits.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            ground_truth_settings[key] = max(low, min(high, float(value)))

    for key in {
        "enabled",
        "autoLoadFile",
        "autoExtractInfo",
        "autoExtractObstacleUpdate",
        "autoExtractCollision",
        "showComparisonInLabel",
        "showErrorInLabel",
        "showMissingGtInLabel",
        "showGtObjectIdInLabel",
        "errorLogEnabled",
        "strictClassMatch",
    }:
        if key in request.args:
            ground_truth_settings[key] = safe_bool(request.args.get(key))

    if "worldForwardAxis" in request.args:
        axis = str(request.args.get("worldForwardAxis", "+z")).strip().lower()
        if axis in {"+z", "-z", "+x", "-x"}:
            ground_truth_settings["worldForwardAxis"] = axis

    return jsonify({"status": "success", "groundTruth": dict(ground_truth_settings)})


@app.route("/map_gt_load", methods=["GET", "POST"])
def map_gt_load():
    clear_existing = safe_bool(request.args.get("clearExisting"), True)
    return jsonify(
        load_map_ground_truth(
            path_value=request.args.get("path"),
            filename=request.args.get("filename"),
            clear_existing=clear_existing,
        )
    )


@app.route("/gt_restore_last_map", methods=["GET", "POST"])
def gt_restore_last_map():
    force = safe_bool(request.args.get("force"), True)
    return jsonify(restore_persisted_map_ground_truth(force=force))


@app.route("/gt_state_debug", methods=["GET"])
def gt_state_debug():
    ensure_map_gt_available()
    class_counts = count_registered_gt_classes()
    with state_lock:
        return jsonify(
            {
                "serverProcessId": SERVER_PROCESS_ID,
                "serverSessionId": SERVER_SESSION_ID,
                "serverStartedAt": SERVER_STARTED_AT,
                "registeredObjectCount": len(ground_truth_state.get("objects", {})),
                "registeredClassCounts": class_counts,
                "activeMapFile": ground_truth_state.get("activeMapFile"),
                "activeMapTerrainIndex": ground_truth_state.get("activeMapTerrainIndex"),
                "lastMapLoadAt": ground_truth_state.get("lastMapLoadAt"),
                "lastMapLoadError": ground_truth_state.get("lastMapLoadError"),
                "lastMapRegisteredCount": ground_truth_state.get("lastMapRegisteredCount", 0),
                "lastClearAt": ground_truth_state.get("lastClearAt"),
                "lastClearReason": ground_truth_state.get("lastClearReason"),
                "lastClearRemovedCount": ground_truth_state.get("lastClearRemovedCount", 0),
                "lastAutoRestoreAt": ground_truth_state.get("lastAutoRestoreAt"),
                "lastAutoRestoreResult": ground_truth_state.get("lastAutoRestoreResult"),
                "persistedMapSelectionFile": str(GT_ACTIVE_MAP_SESSION_FILE),
                "persistedMapSelection": read_persisted_map_selection(),
            }
        )


@app.route("/map_gt_list", methods=["GET"])
def map_gt_list():
    maps = sorted(path.name for path in BASE_DIR.glob("*.map") if path.is_file())
    return jsonify(
        {
            "baseDir": str(BASE_DIR),
            "mapFiles": maps,
            "usage": "Open /map_gt_load?filename=YOUR_FILE.map&clearExisting=true",
        }
    )


@app.route("/gt_error_log_reset", methods=["GET", "POST"])
def gt_error_log_reset():
    path = Path(str(ground_truth_state.get("errorLogPath", GT_ERROR_LOG_FILE)))
    removed = False
    if path.exists():
        path.unlink()
        removed = True
    with state_lock:
        ground_truth_state["errorLogRowCount"] = 0
        ground_truth_state["lastLoggedPairAt"] = {}
    return jsonify({"status": "success", "removedExistingFile": removed, "path": str(path)})


@app.route("/gt_diagnose", methods=["GET"])
def gt_diagnose():
    with state_lock:
        cache = latest_cache
    return jsonify(gt_diagnosis(cache))


@app.route("/gt_dashboard", methods=["GET"])
def gt_dashboard():
    with state_lock:
        cache = latest_cache
        comparisons = json_copy(ground_truth_state.get("lastComparisons", []))
        active_map = ground_truth_state.get("activeMapFile")
        map_error = ground_truth_state.get("lastMapLoadError")
        row_count = ground_truth_state.get("errorLogRowCount", 0)
        last_clear_at = ground_truth_state.get("lastClearAt")
        last_clear_reason = ground_truth_state.get("lastClearReason")
        last_clear_removed = ground_truth_state.get("lastClearRemovedCount", 0)
    diagnosis = gt_diagnosis(cache)

    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return html.escape(str(value))

    rows: list[str] = []
    for item in comparisons:
        gt = item.get("groundTruth") or {}
        rows.append(
            "<tr>"
            f"<td>{cell(item.get('estimatedClass'))}</td>"
            f"<td>{cell(gt.get('id'))}</td>"
            f"<td>{cell(gt.get('lidarDistanceM'))}</td>"
            f"<td>{cell(gt.get('centerHorizontalDistanceM'))}</td>"
            f"<td>{cell(gt.get('approxSurfaceDistanceM'))}</td>"
            f"<td>{cell(gt.get('distanceErrorToCenterM'))}</td>"
            f"<td>{cell(gt.get('distanceErrorToApproxSurfaceM'))}</td>"
            f"<td>{cell(gt.get('lidarBodyRelativeAngleDeg'))}</td>"
            f"<td>{cell(gt.get('bodyRelativeAngleDeg'))}</td>"
            f"<td>{cell(gt.get('angleErrorDeg'))}</td>"
            f"<td>{cell(gt.get('classConsistent'))}</td>"
            "</tr>"
        )

    if not rows:
        rows.append("<tr><td colspan='11'>No matched comparison yet.</td></tr>")

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="1">
<title>Tank Challenge GT Error Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }}
h1 {{ margin-bottom: 6px; }}
.card {{ background: #1d1d1d; border: 1px solid #444; padding: 12px; margin: 10px 0; }}
.ready {{ color: #6fe36f; }} .warn {{ color: #ffce54; }} .bad {{ color: #ff7777; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #555; padding: 6px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
code {{ color: #9cdcfe; }}
</style>
</head>
<body>
<h1>LiDAR + YOLO vs Map Ground Truth</h1>
<div class="card">
<b>Diagnosis:</b> <span class="{'ready' if diagnosis.get('code') == 'READY' else 'warn'}">{html.escape(str(diagnosis.get('code')))}</span><br>
{html.escape(str(diagnosis.get('message')))}<br>
<b>Server PID:</b> {cell(SERVER_PROCESS_ID)} &nbsp; <b>Session:</b> {cell(SERVER_SESSION_ID)} &nbsp; <b>Started:</b> {cell(SERVER_STARTED_AT)}<br>
<b>Active map:</b> {cell(active_map)}<br>
<b>Map load error:</b> {cell(map_error)}<br>
<b>Last clear:</b> {cell(last_clear_at)} / {cell(last_clear_reason)} / removed={cell(last_clear_removed)}<br>
<b>Error-log rows:</b> {cell(row_count)}
</div>
<div class="card">
<b>Meaning</b><br>
LiDAR = nearest visible surface range. GTc = exact map-pivot center range.
GTs = approximate surface range when radiusM is known.
</div>
<table>
<thead>
<tr>
<th>YOLO class</th><th>GT object</th><th>LiDAR m</th><th>GTc m</th><th>GTs m</th>
<th>Err→GTc m</th><th>Err→GTs m</th><th>LiDAR deg</th><th>GT deg</th><th>Angle err deg</th><th>Class OK</th>
</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>"""
    return page


@app.route("/payload_debug", methods=["GET"])
def payload_debug():
    with state_lock:
        return jsonify(
            {
                "payloadDebug": json_copy(ground_truth_state.get("payloadDebug", {})),
                "note": (
                    "Inspect update_obstacle and info payloads. If the simulator does not send "
                    "object world coordinates, load static coordinates from ground_truth_objects.json "
                    "or provide a live debug feed for moving targets."
                ),
            }
        )


# Simulator compatibility endpoints.
@app.route("/init", methods=["GET"])
def init():
    return jsonify(
        {
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
            "enemyTracking": False,
            "saveSnapshot": False,
            "saveLog": False,
            "saveLidarData": False,
            "lux": 30000,
            "destoryObstaclesOnHit": True,
        }
    )


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": ""})


@app.route("/stereo_image", methods=["POST"])
def stereo_image():
    return jsonify({"result": "success"})


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    return jsonify({"status": "OK"})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    payload = request.get_json(silent=True) or {}
    record_payload_debug("update_obstacle", payload)
    records: list[dict[str, Any]] = []
    if bool(ground_truth_settings.get("autoExtractObstacleUpdate", True)):
        records = ingest_gt_payload(
            payload,
            source="auto:update_obstacle",
            dynamic_default=False,
            allow_direct=True,
        )
    return jsonify({"status": "success", "groundTruthRegisteredCount": len(records)})


@app.route("/collision", methods=["POST"])
def collision():
    payload = request.get_json(silent=True) or {}
    record_payload_debug("collision", payload)
    records: list[dict[str, Any]] = []
    if bool(ground_truth_settings.get("autoExtractCollision", True)):
        records = ingest_gt_payload(
            payload,
            source="auto:collision",
            dynamic_default=False,
            allow_direct=True,
        )
    return jsonify({"status": "success", "groundTruthRegisteredCount": len(records)})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    data = request.get_json(silent=True) or {}
    return jsonify({"status": "OK", "destination": data.get("destination")})


if bool(ground_truth_settings.get("autoLoadFile", True)):
    load_ground_truth_file()

# Restore the last map automatically after a Python-server restart.
restore_persisted_map_ground_truth(force=False)

Thread(target=yolo_worker_loop, daemon=True, name="yolo-worker").start()


if __name__ == "__main__":
    print("=" * 80)
    print("Tank Challenge LiDAR-first YOLO Fusion Server v16.6 Top-View Hill/Flat Target Scan")
    print("LiDAR overlay: realtime cached path")
    print("YOLO fusion: asynchronous slower path")
    print("Fusion method: object-above-hill LiDAR ROI first, angle-cluster fallback second")
    print("Tilt compensation: local LiDAR ground-plane estimate")
    print("YOLO queue: one latest pending frame; stale queued frame replacement enabled")
    print("Anti-hunt aim: smoothed target angle + proportional turret command")
    print("Label format: tank | distance | body-relative angle")
    print(f"YOLO model expected at: {YOLO_MODEL_PATH}")
    print("Health       : http://127.0.0.1:5000/health")
    print("Status       : http://127.0.0.1:5000/status")
    print("Fusion status: http://127.0.0.1:5000/fusion_status")
    print("Fusion debug : http://127.0.0.1:5000/fusion_debug")
    print("Tilt status  : http://127.0.0.1:5000/tilt_status")
    print("Fusion preset: http://127.0.0.1:5000/fusion_preset?mode=balanced")
    print("GT status    : http://127.0.0.1:5000/gt_status")
    print("Web LiDAR    : http://127.0.0.1:5000/lidar_view  (360 top view)")
    print("360 LiDAR    : http://127.0.0.1:5000/lidar_360  (full screen)")
    print("LiDAR JSON   : http://127.0.0.1:5000/lidar_status")
    print("Aim status   : http://127.0.0.1:5000/aim_status")
    print("Fire status  : http://127.0.0.1:5000/fire_status")
    print("Action debug : http://127.0.0.1:5000/action_debug")
    print("GT dashboard : http://127.0.0.1:5000/gt_dashboard")
    print("GT state dbg : http://127.0.0.1:5000/gt_state_debug")
    print("GT restore   : http://127.0.0.1:5000/gt_restore_last_map")
    print("Map GT list  : http://127.0.0.1:5000/map_gt_list")
    print("Map GT load  : http://127.0.0.1:5000/map_gt_load?filename=YOUR_MAP.map")
    print("Payload debug: http://127.0.0.1:5000/payload_debug")
    print("YOLO preload : http://127.0.0.1:5000/yolo_preload")
    print("Recommended  : Interval=0.2, Y=3, Channel=32, Minimap=16, Range=120, FPS=60")
    print("=" * 80)
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True, debug=False)
