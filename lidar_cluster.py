from __future__ import annotations

"""
Tank Challenge LiDAR-first YOLO Fusion Server v16.8 YOLO.pt HillMap Settings
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
8. v16.1 adds YOLO env tuning, YOLO.pt class alignment, and unified LiDAR dashboard.
9. Fire is allowed only after a fresh YOLO-fused tank confirmation and aim lock.

Recommended first benchmark
---------------------------
Simulator Properties:
- Mode: Simulation
- Request Port: 5000
- Interval: 0.5
- Y Position: 3
- Channel: 32
- Minimap Channel: 16
- Max Distance: 120
- Lidar Position: Body
- Send Detected Lidar: enabled
- Frame Rate Settings: 120
- Graphics Quality Settings: Ultra

Put this file, YOLO.pt, and hill_map_height.csv in the same folder.
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
EXPECTED_INTERVAL_SEC = 0.5

DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1057
MAX_LIDAR_DISTANCE_M = 120.0

# LiDAR-only extraction mode. Ground is removed first, then only clusters with
# vehicle-like width/height/depth geometry are shown in the simulator/web view.
GROUND_FILTER_ENABLED = True
LIDAR_VEHICLE_EXTRACTION_ENABLED = True
VEHICLE_MIN_WIDTH_M = 0.7
VEHICLE_MAX_WIDTH_M = 8.0
VEHICLE_MIN_HEIGHT_M = 0.6
VEHICLE_MAX_HEIGHT_M = 4.5
VEHICLE_MAX_DEPTH_M = 8.0
VEHICLE_MIN_POINTS = 2
VEHICLE_CLUSTER_ANGLE_MARGIN_DEG = 2.0
VEHICLE_CLUSTER_DISTANCE_MARGIN_M = 3.0
VEHICLE_RECOVERY_MIN_ABOVE_TERRAIN_M = 0.10
TANK_LIKE_MIN_WIDTH_M = 0.6
TANK_LIKE_MAX_WIDTH_M = 3.05
TANK_LIKE_MIN_WIDTH_DEPTH_RATIO = 1.10
TANK_LIKE_MIN_HEIGHT_M = 0.25
TANK_LIKE_MAX_HEIGHT_M = 3.8
TANK_CONTOUR_MIN_CHANNELS = 3
TANK_CONTOUR_MIN_WIDTH_VARIATION = 0.20
TANK_CONTOUR_MIN_GRADUAL_STEPS = 2
TANK_CONTOUR_STRONG_VARIATION = 0.70



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
SLIDING_CLUSTER_MERGE_ENABLED = True
SLIDING_CLUSTER_MAX_ANGLE_GAP_DEG = 6.0
SLIDING_CLUSTER_MAX_DISTANCE_GAP_M = 4.5
SLIDING_CLUSTER_MAX_WORLD_GAP_M = 6.0
OBJECT_TRACK_MAX_MATCH_DISTANCE_M = 5.0
OBJECT_TRACK_MAX_AGE_SEC = 2.0
OBJECT_TRACK_POSITION_ALPHA = 0.45
OBJECT_TRACK_VELOCITY_ALPHA = 0.30
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


# Front LiDAR web-view display setting.
# In this simulator payload, the verticalAngle sign appears inverted relative
# to the camera image in /lidar_view.  Keep recognition unchanged and flip only
# the SVG display so the front silhouette matches what the camera sees.
front_view_settings: dict[str, Any] = {
    "flipVerticalDisplay": True,
}

overlay_settings: dict[str, Any] = {
    "showLidarPoints": True,
    "showSafeGround": False,

    # Simulator-screen LiDAR overlay mode for /detect.
    # valid_only      : only points that passed the object filter.
    # valid_plus_high : valid object points plus high-above-terrain object hits.
    # all_obstacles   : every non-ground obstacle hit. Useful for debugging, heavier/noisier.
    "simLidarPointMode": "vehicle_clusters",
    "showLidarClusterBoxes": True,
    "clusterBoxLimit": 12,
    "clusterBoxMinPoints": 2,
    "clusterBoxAngleGateDeg": 4.0,
    "clusterBoxDistanceGateM": 4.0,
    "objectPointRadiusPx": 3,

    "obstacleBoxLimit": 650,
    "safeGroundBoxLimit": 16,
    "totalLidarBoxLimit": 850,
    "obstaclePixelCell": 2,
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
YOLO_MODEL_PATH = BASE_DIR / os.getenv("YOLO_MODEL_FILE", "YOLO.pt")

# Optional precomputed terrain-height map.
# hill_map_height.csv columns: x,z,y,...  The map is used only for terrain base
# height estimation; LiDAR points remain the source for object distance/angle.
HILL_MAP_HEIGHT_FILE = BASE_DIR / os.getenv("HILL_MAP_HEIGHT_FILE", "hill_map_height.csv")
HILL_MAP_HEIGHT_ENABLED = env_bool("HILL_MAP_HEIGHT_ENABLED", True)
_hill_map_height_grid: dict[tuple[int, int], float] = {}
hill_map_height_state: dict[str, Any] = {
    "enabled": HILL_MAP_HEIGHT_ENABLED,
    "filePath": str(HILL_MAP_HEIGHT_FILE),
    "loaded": False,
    "status": "not_loaded",
    "rowCount": 0,
    "gridCount": 0,
    "lastLoadError": None,
    "lastLoadAt": None,
    "lastApplyDebug": None,
}


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

# YOLO.pt classes.
#
# Embedded checkpoint metadata found in the uploaded model:
#   0 Ally, 1 Enemy, 2 House, 3 Rock, 4 Rock_L, 5 Tank_enemy, 6 Tent, 7 car
#
# MODEL_CLASS_NAMES is only a fallback. During inference, result.names from
# the loaded .pt model takes priority, preventing silent class-ID mismatches.


def current_yolo_model_path() -> str:
    return str(fusion_settings.get("modelPath", YOLO_MODEL_PATH))

MODEL_CLASS_NAMES = {
    # YOLO.pt/lalast.pt embedded names from checkpoint metadata.
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
    # YOLO.pt classes. Keep raw display names aligned with the .pt file,
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
    # YOLO.pt raw class names.
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


# Web-controlled .map auto comparison/cycling.
# - cycle mode: rotate active .map every interval while /lidar_view refreshes.
# - best mode: score every .map against current LiDAR clusters and load the best match.
map_cycle_settings: dict[str, Any] = {
    "enabled": False,
    "mode": "cycle",                 # cycle | best
    "intervalSec": 6.0,
    "currentIndex": -1,
    "currentMapFile": None,
    "lastSwitchAt": None,
    "lastSwitchMonotonic": None,
    "lastError": None,
    "lastBestScan": None,
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
    "source": None,
    "updatedAt": None,
    "updatedMonotonic": None,
}

object_tracks: dict[int, dict[str, Any]] = {}
next_object_track_id = 1

def empty_action() -> dict[str, Any]:
    return {
        "moveWS": {"command": "", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
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
    "targetConfirmMaxAgeSec": 6.0,
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
    "pitchSweepOffsetsDeg": "0,0.3,-0.3,0.6,-0.6,0.9,-0.9",
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



def json_copy(value: Any) -> Any:
    """Small JSON-safe deep copy without copying NumPy arrays."""
    if isinstance(value, dict):
        return {key: json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_copy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value



def round_float(value: Any, digits: int = 3, default: float | None = None) -> float | None:
    number = safe_float(value, default)
    if number is None:
        return default
    return round(float(number), int(digits))


def xyz_to_dict(values: Any, digits: int = 3) -> dict[str, float] | None:
    """Return a compact JSON-safe world coordinate dict."""
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return {
            "x": round(float(arr[0]), digits),
            "y": round(float(arr[1]), digits),
            "z": round(float(arr[2]), digits),
        }
    except Exception:
        return None


def xyz_bounds_to_dict(points: np.ndarray, digits: int = 3) -> dict[str, dict[str, float]] | None:
    """Compact world-space AABB used only for object summaries, not raw point transfer."""
    if points is None or np.asarray(points).size == 0:
        return None
    try:
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(arr), axis=1)
        arr = arr[finite]
        if arr.size == 0:
            return None
        return {"min": xyz_to_dict(np.min(arr, axis=0), digits), "max": xyz_to_dict(np.max(arr, axis=0), digits)}
    except Exception:
        return None


def compact_world_geometry(
    points: np.ndarray,
    surface_points: np.ndarray | None,
    aim_y: float,
) -> dict[str, Any]:
    """Build the only world-coordinate payload needed by the fire team.

    The raw LiDAR point cloud stays inside this server.  Downstream modules get
    only center/surface/aim coordinates plus a small bounding box.
    """
    if points is None or np.asarray(points).size == 0:
        return {
            "worldCenter": None,
            "surfaceCenterWorld": None,
            "aimPointWorld": None,
            "worldBounds": None,
        }

    all_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_all = np.all(np.isfinite(all_points), axis=1)
    all_points = all_points[finite_all]
    if all_points.size == 0:
        return {
            "worldCenter": None,
            "surfaceCenterWorld": None,
            "aimPointWorld": None,
            "worldBounds": None,
        }

    if surface_points is None or np.asarray(surface_points).size == 0:
        surface = all_points
    else:
        surface = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
        surface = surface[np.all(np.isfinite(surface), axis=1)]
        if surface.size == 0:
            surface = all_points

    center = np.median(all_points, axis=0)
    surface_center = np.median(surface, axis=0)
    aim_point = np.asarray((surface_center[0], float(aim_y), surface_center[2]), dtype=np.float64)
    return {
        "worldCenter": xyz_to_dict(center),
        "surfaceCenterWorld": xyz_to_dict(surface_center),
        "aimPointWorld": xyz_to_dict(aim_point),
        "worldBounds": xyz_bounds_to_dict(all_points),
    }



def load_hill_map_height(force: bool = False) -> dict[str, Any]:
    """Load hill_map_height.csv into a 1 m x/z lookup table.

    The CSV is optional. If it is present, object height-above-terrain is
    computed from this prebuilt map where possible. If not, the existing LiDAR
    lower-envelope terrain profile is used unchanged.
    """
    global _hill_map_height_grid

    if hill_map_height_state.get("loaded") and not force:
        return dict(hill_map_height_state)

    if not bool(hill_map_height_state.get("enabled", True)):
        hill_map_height_state.update({
            "loaded": False,
            "status": "disabled",
            "lastLoadAt": now_text(),
            "lastLoadError": None,
        })
        return dict(hill_map_height_state)

    path = Path(str(hill_map_height_state.get("filePath", HILL_MAP_HEIGHT_FILE)))
    if not path.exists():
        _hill_map_height_grid = {}
        hill_map_height_state.update({
            "loaded": False,
            "status": "missing",
            "rowCount": 0,
            "gridCount": 0,
            "lastLoadAt": now_text(),
            "lastLoadError": f"not found: {path}",
        })
        return dict(hill_map_height_state)

    grid: dict[tuple[int, int], float] = {}
    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                x = safe_float(row.get("x"), None)
                z = safe_float(row.get("z"), None)
                y = safe_float(row.get("y"), None)
                if x is None or y is None or z is None:
                    continue
                # hill_map_height.csv is a 1 m grid in simulator world x/z.
                grid[(int(round(float(x))), int(round(float(z))))] = float(y)
                row_count += 1
        _hill_map_height_grid = grid
        hill_map_height_state.update({
            "loaded": True,
            "status": "success",
            "rowCount": row_count,
            "gridCount": len(grid),
            "lastLoadAt": now_text(),
            "lastLoadError": None,
        })
    except Exception as exc:
        _hill_map_height_grid = {}
        hill_map_height_state.update({
            "loaded": False,
            "status": "error",
            "rowCount": 0,
            "gridCount": 0,
            "lastLoadAt": now_text(),
            "lastLoadError": f"{type(exc).__name__}: {exc}",
        })
    return dict(hill_map_height_state)


def apply_hill_map_height_to_terrain(
    xyz: np.ndarray,
    fallback_terrain_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply prebuilt terrain y values to LiDAR hit world positions when available."""
    if xyz.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, {"enabled": bool(hill_map_height_state.get("enabled", True)), "status": "empty"}

    if not hill_map_height_state.get("loaded"):
        load_hill_map_height(force=False)

    fallback = np.asarray(fallback_terrain_y, dtype=np.float32)
    if fallback.size != xyz.shape[0]:
        fallback = xyz[:, 1].astype(np.float32)

    grid = _hill_map_height_grid
    if not bool(hill_map_height_state.get("enabled", True)) or not grid:
        residual = (xyz[:, 1].astype(np.float32) - fallback).astype(np.float32)
        debug = {
            "enabled": bool(hill_map_height_state.get("enabled", True)),
            "status": hill_map_height_state.get("status", "not_loaded"),
            "filePath": hill_map_height_state.get("filePath"),
            "matchedPointCount": 0,
            "totalPointCount": int(xyz.shape[0]),
            "used": False,
            "note": "hill_map_height.csv not available; using LiDAR lower-envelope terrain profile",
        }
        hill_map_height_state["lastApplyDebug"] = json_copy(debug)
        return fallback.astype(np.float32), residual, debug

    terrain = fallback.copy().astype(np.float32)
    matched = np.zeros(xyz.shape[0], dtype=bool)
    map_y = np.empty(xyz.shape[0], dtype=np.float32)

    # Exact nearest 1 m grid lookup first. If the rounded cell is absent, try
    # the closest of the four floor/ceil neighbor cells.
    for i, (x, _, z) in enumerate(xyz.astype(np.float64)):
        xr = int(round(float(x)))
        zr = int(round(float(z)))
        value = grid.get((xr, zr))
        if value is None:
            xf = int(np.floor(float(x)))
            xc = int(np.ceil(float(x)))
            zf = int(np.floor(float(z)))
            zc = int(np.ceil(float(z)))
            best_key = None
            best_d2 = None
            for xx in {xf, xc}:
                for zz in {zf, zc}:
                    if (xx, zz) not in grid:
                        continue
                    d2 = (float(x) - xx) ** 2 + (float(z) - zz) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_key = (xx, zz)
            if best_key is not None:
                value = grid.get(best_key)
        if value is not None:
            map_y[i] = float(value)
            matched[i] = True

    if np.any(matched):
        terrain[matched] = map_y[matched]

    residual = (xyz[:, 1].astype(np.float32) - terrain).astype(np.float32)
    matched_count = int(np.sum(matched))
    debug = {
        "enabled": True,
        "status": "applied" if matched_count else "loaded_no_point_match",
        "filePath": hill_map_height_state.get("filePath"),
        "loadedGridCount": int(hill_map_height_state.get("gridCount", len(grid))),
        "matchedPointCount": matched_count,
        "totalPointCount": int(xyz.shape[0]),
        "matchedRatio": round(matched_count / max(1, int(xyz.shape[0])), 4),
        "used": bool(matched_count > 0),
        "note": "matched LiDAR world x/z cells use CSV y as terrain base; unmatched cells keep LiDAR profile",
    }
    hill_map_height_state["lastApplyDebug"] = json_copy(debug)
    return terrain.astype(np.float32), residual.astype(np.float32), debug


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


def merge_cluster_groups_sliding_window(
    groups: list[list[int]],
    angles: np.ndarray,
    distances: np.ndarray,
    xyz: np.ndarray,
) -> tuple[list[list[int]], list[int]]:
    """Merge split object fragments with a circular angular sliding window."""
    if not SLIDING_CLUSTER_MERGE_ENABLED or len(groups) < 2:
        return groups, [1] * len(groups)

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    summaries: list[dict[str, Any]] = []
    for group in groups:
        idx = np.asarray(group, dtype=np.int32)
        group_angles = angles[idx].astype(np.float64)
        group_distances = distances[idx].astype(np.float64)
        group_xyz = xyz[idx].astype(np.float64)
        summaries.append(
            {
                "angles": group_angles,
                "distance": float(np.median(group_distances)),
                "worldX": float(np.median(group_xyz[:, 0])),
                "worldZ": float(np.median(group_xyz[:, 2])),
            }
        )

    # Compare every nearby angular fragment. Circular angle math also joins
    # fragments split across the -180/+180 boundary.
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            left_summary = summaries[left]
            right_summary = summaries[right]
            pairwise_angle_gap = np.abs(
                (
                    left_summary["angles"][:, None]
                    - right_summary["angles"][None, :]
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            minimum_angle_gap = float(np.min(pairwise_angle_gap))
            if minimum_angle_gap > SLIDING_CLUSTER_MAX_ANGLE_GAP_DEG:
                continue

            distance_gap = abs(
                float(left_summary["distance"])
                - float(right_summary["distance"])
            )
            if distance_gap > SLIDING_CLUSTER_MAX_DISTANCE_GAP_M:
                continue

            world_gap = float(
                np.hypot(
                    float(left_summary["worldX"])
                    - float(right_summary["worldX"]),
                    float(left_summary["worldZ"])
                    - float(right_summary["worldZ"]),
                )
            )
            if world_gap > SLIDING_CLUSTER_MAX_WORLD_GAP_M:
                continue
            union(left, right)

    merged: dict[int, list[int]] = {}
    fragment_counts: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        root = find(group_index)
        merged.setdefault(root, []).extend(group)
        fragment_counts[root] = fragment_counts.get(root, 0) + 1

    ordered_roots = sorted(
        merged,
        key=lambda root: float(circular_mean_deg(angles[np.asarray(merged[root], dtype=np.int32)])),
    )
    merged_groups = [
        sorted(merged[root], key=lambda index: float(angles[index]))
        for root in ordered_roots
    ]
    merged_counts = [fragment_counts[root] for root in ordered_roots]
    return merged_groups, merged_counts


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

    groups, sliding_fragment_counts = merge_cluster_groups_sliding_window(
        groups=groups,
        angles=angles,
        distances=distances,
        xyz=xyz,
    )

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
        relative_group_angles = (
            (group_angles.astype(np.float64) - median_angle + 180.0) % 360.0
        ) - 180.0
        angular_span = max(
            0.0,
            float(relative_group_angles.max() - relative_group_angles.min()),
        )
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
        near_surface_mask = group_distances <= surface_distance + max(0.75, float(fusion_settings.get("roiSurfaceBandM", 3.0)))
        surface_xyz_for_world = group_xyz[near_surface_mask] if np.any(near_surface_mask) else group_xyz
        world_geometry = compact_world_geometry(group_xyz, surface_xyz_for_world, aim_point_y)
        verticality_ratio = object_height_above_terrain / max(0.15, depth_span)
        vehicle_like = bool(len(group) >= VEHICLE_MIN_POINTS)
        width_depth_ratio = float(visible_width) / max(0.25, float(depth_span))
        contour_angle_gate = max(
            2.0,
            float(angular_span) * 0.5 + VEHICLE_CLUSTER_ANGLE_MARGIN_DEG,
        )
        contour_angle_delta = np.abs(
            ((angles.astype(np.float32) - median_angle + 180.0) % 360.0) - 180.0
        )
        contour_mask = (
            (contour_angle_delta <= contour_angle_gate)
            & (
                distances
                >= max(0.0, surface_distance - VEHICLE_CLUSTER_DISTANCE_MARGIN_M)
            )
            & (
                distances
                <= far_distance + VEHICLE_CLUSTER_DISTANCE_MARGIN_M
            )
            & (
                height_above_terrain
                >= VEHICLE_RECOVERY_MIN_ABOVE_TERRAIN_M
            )
        )
        contour_angles = angles[contour_mask]
        contour_vertical = vertical_angles[contour_mask]
        contour_rows: list[tuple[float, float]] = []
        rounded_channels = np.round(contour_vertical.astype(np.float64), 3)
        for channel_value in np.unique(rounded_channels):
            channel_angles = contour_angles[rounded_channels == channel_value]
            if channel_angles.size < 2:
                continue
            channel_center = float(circular_mean_deg(channel_angles))
            relative_channel_angles = (
                (channel_angles.astype(np.float64) - channel_center + 180.0)
                % 360.0
            ) - 180.0
            channel_span_deg = float(
                np.max(relative_channel_angles)
                - np.min(relative_channel_angles)
            )
            channel_width_m = 2.0 * median_distance * tan(
                radians(max(0.5, channel_span_deg) / 2.0)
            )
            contour_rows.append((float(channel_value), float(channel_width_m)))
        contour_rows.sort(key=lambda item: item[0])
        contour_widths = [width for _, width in contour_rows]
        contour_channel_count = len(contour_widths)
        if contour_widths:
            ordered_widths = np.asarray(contour_widths, dtype=np.float64)
            contour_max_width = float(np.percentile(ordered_widths, 75.0))
            contour_min_width = float(np.percentile(ordered_widths, 25.0))
            contour_width_variation = (
                contour_max_width - contour_min_width
            ) / max(0.25, contour_max_width)
            adjacent_deltas = np.abs(np.diff(ordered_widths))
            meaningful_step = max(0.35, contour_max_width * 0.12)
            contour_gradual_steps = int(np.sum(adjacent_deltas >= meaningful_step))
        else:
            contour_max_width = 0.0
            contour_min_width = 0.0
            contour_width_variation = 0.0
            contour_gradual_steps = 0
        tank_like = bool(
            vehicle_like
            and contour_channel_count >= TANK_CONTOUR_MIN_CHANNELS
            and contour_width_variation >= TANK_CONTOUR_MIN_WIDTH_VARIATION
            and (
                contour_gradual_steps >= TANK_CONTOUR_MIN_GRADUAL_STEPS
                or (
                    contour_channel_count >= 4
                    and contour_width_variation >= TANK_CONTOUR_STRONG_VARIATION
                    and contour_gradual_steps >= 1
                )
            )
        )
        key = f"a{round(median_angle / 2.0) * 2:+.0f}_d{round(surface_distance / 5.0) * 5:.0f}"
        clusters.append({
            "clusterId": int(cluster_id),
            "slidingMergedFragmentCount": int(sliding_fragment_counts[cluster_id]),
            "candidateLabel": "TANK_LIKE" if tank_like else "OBSTACLE",
            "vehicleLike": vehicle_like,
            "tankLike": tank_like,
            "shapeClass": "tank_like" if tank_like else "obstacle",
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
            "worldCenter": world_geometry.get("worldCenter"),
            "surfaceCenterWorld": world_geometry.get("surfaceCenterWorld"),
            "aimPointWorld": world_geometry.get("aimPointWorld"),
            "worldBounds": world_geometry.get("worldBounds"),
            "depthSpanM": round(depth_span, 3),
            "verticalityRatio": round(float(verticality_ratio), 3),
            "widthDepthRatio": round(width_depth_ratio, 3),
            "contourChannelCount": int(contour_channel_count),
            "contourMinWidthM": round(contour_min_width, 3),
            "contourMaxWidthM": round(contour_max_width, 3),
            "contourWidthVariation": round(contour_width_variation, 3),
            "contourGradualSteps": int(contour_gradual_steps),
            "objectFilter": "terrain_residual_plus_vertical_plane",
        })
    if LIDAR_VEHICLE_EXTRACTION_ENABLED:
        clusters = [item for item in clusters if bool(item.get("vehicleLike", False))]
    clusters.sort(key=lambda item: (float(item["distanceM"]), abs(float(item["angleDeg"]))))
    return tuple(clusters[:VALID_OBJECT_CLUSTER_MAX_COUNT])


def merge_split_world_clusters(
    clusters: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Suppress duplicate fragments that belong to one nearby world object."""
    kept: list[dict[str, Any]] = []
    ranked = sorted(
        (json_copy(item) for item in clusters),
        key=lambda item: (
            0 if bool(item.get("tankLike", False)) else 1,
            -int(item.get("pointCount", 0)),
        ),
    )
    for candidate in ranked:
        center = candidate.get("surfaceCenterWorld") or candidate.get("worldCenter") or {}
        cx = safe_float(center.get("x"), None) if isinstance(center, dict) else None
        cz = safe_float(center.get("z"), None) if isinstance(center, dict) else None
        duplicate = None
        for existing in kept:
            other = existing.get("surfaceCenterWorld") or existing.get("worldCenter") or {}
            ox = safe_float(other.get("x"), None) if isinstance(other, dict) else None
            oz = safe_float(other.get("z"), None) if isinstance(other, dict) else None
            if None in (cx, cz, ox, oz):
                continue
            world_gap = float(np.hypot(float(cx) - float(ox), float(cz) - float(oz)))
            angle_gap = angle_gap_deg(
                float(candidate.get("angleDeg", 0.0)),
                float(existing.get("angleDeg", 0.0)),
            )
            range_gap = abs(
                float(candidate.get("distanceM", 0.0))
                - float(existing.get("distanceM", 0.0))
            )
            if world_gap <= 3.5 and angle_gap <= 6.0 and range_gap <= 6.0:
                duplicate = existing
                break
        if duplicate is None:
            candidate["mergedFragmentCount"] = 1
            kept.append(candidate)
        else:
            duplicate["mergedFragmentCount"] = int(
                duplicate.get("mergedFragmentCount", 1)
            ) + 1
            duplicate["pointCount"] = int(duplicate.get("pointCount", 0)) + int(
                candidate.get("pointCount", 0)
            )
    kept.sort(key=lambda item: (float(item["distanceM"]), abs(float(item["angleDeg"]))))
    return tuple(kept)


def assign_persistent_object_ids(
    clusters: tuple[dict[str, Any], ...],
    simulation_time: Any,
) -> tuple[dict[str, Any], ...]:
    """Track merged objects in world X/Z and assign stable IDs across frames."""
    global next_object_track_id

    now_mono = monotonic()
    sim_time = safe_float(simulation_time, None)
    observations: list[dict[str, Any]] = []
    output = [json_copy(cluster) for cluster in clusters]
    for index, cluster in enumerate(clusters):
        center = cluster.get("surfaceCenterWorld") or cluster.get("worldCenter") or {}
        x = safe_float(center.get("x"), None) if isinstance(center, dict) else None
        y = safe_float(center.get("y"), None) if isinstance(center, dict) else None
        z = safe_float(center.get("z"), None) if isinstance(center, dict) else None
        if x is None or y is None or z is None:
            continue
        observations.append(
            {
                "index": index,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "tankLike": bool(cluster.get("tankLike", False)),
            }
        )

    with state_lock:
        for track_id in list(object_tracks):
            age = now_mono - float(object_tracks[track_id].get("lastMonotonic", now_mono))
            if age > OBJECT_TRACK_MAX_AGE_SEC:
                object_tracks.pop(track_id, None)

        pair_candidates: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_id, track in object_tracks.items():
                last_sim_time = safe_float(track.get("lastSimulationTime"), None)
                dt = (
                    float(sim_time) - float(last_sim_time)
                    if sim_time is not None and last_sim_time is not None
                    else now_mono - float(track.get("lastMonotonic", now_mono))
                )
                dt = max(0.0, min(OBJECT_TRACK_MAX_AGE_SEC, float(dt)))
                predicted_x = float(track["x"]) + float(track.get("vx", 0.0)) * dt
                predicted_z = float(track["z"]) + float(track.get("vz", 0.0)) * dt
                world_gap = float(
                    np.hypot(
                        observation["x"] - predicted_x,
                        observation["z"] - predicted_z,
                    )
                )
                if world_gap > OBJECT_TRACK_MAX_MATCH_DISTANCE_M:
                    continue
                class_penalty = (
                    0.75
                    if bool(track.get("tankLike", False))
                    != bool(observation["tankLike"])
                    else 0.0
                )
                pair_candidates.append(
                    (world_gap + class_penalty, observation_index, track_id)
                )

        observation_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, observation_index, track_id in sorted(pair_candidates):
            if observation_index in observation_to_track or track_id in used_tracks:
                continue
            observation_to_track[observation_index] = track_id
            used_tracks.add(track_id)

        for observation_index, observation in enumerate(observations):
            track_id = observation_to_track.get(observation_index)
            if track_id is None:
                track_id = next_object_track_id
                next_object_track_id += 1
                object_tracks[track_id] = {
                    "x": observation["x"],
                    "y": observation["y"],
                    "z": observation["z"],
                    "vx": 0.0,
                    "vz": 0.0,
                    "tankLike": observation["tankLike"],
                    "hits": 0,
                    "firstMonotonic": now_mono,
                    "lastMonotonic": now_mono,
                    "lastSimulationTime": sim_time,
                }

            track = object_tracks[track_id]
            previous_x = float(track["x"])
            previous_y = float(track["y"])
            previous_z = float(track["z"])
            previous_sim_time = safe_float(track.get("lastSimulationTime"), None)
            dt = (
                float(sim_time) - float(previous_sim_time)
                if sim_time is not None and previous_sim_time is not None
                else now_mono - float(track.get("lastMonotonic", now_mono))
            )
            dt = max(1e-3, float(dt))
            measured_vx = (observation["x"] - previous_x) / dt
            measured_vz = (observation["z"] - previous_z) / dt
            position_alpha = OBJECT_TRACK_POSITION_ALPHA
            velocity_alpha = OBJECT_TRACK_VELOCITY_ALPHA
            track["x"] = (1.0 - position_alpha) * previous_x + position_alpha * observation["x"]
            track["y"] = (1.0 - position_alpha) * previous_y + position_alpha * observation["y"]
            track["z"] = (1.0 - position_alpha) * previous_z + position_alpha * observation["z"]
            track["vx"] = (1.0 - velocity_alpha) * float(track.get("vx", 0.0)) + velocity_alpha * measured_vx
            track["vz"] = (1.0 - velocity_alpha) * float(track.get("vz", 0.0)) + velocity_alpha * measured_vz
            track["tankLike"] = observation["tankLike"]
            track["hits"] = int(track.get("hits", 0)) + 1
            track["lastMonotonic"] = now_mono
            track["lastSimulationTime"] = sim_time

            enriched = output[int(observation["index"])]
            enriched["objectId"] = f"OBJ-{track_id:03d}"
            enriched["trackHits"] = int(track["hits"])
            enriched["trackedWorldCenter"] = {
                "x": round(float(track["x"]), 3),
                "y": round(float(track["y"]), 3),
                "z": round(float(track["z"]), 3),
            }
            enriched["worldVelocityMps"] = {
                "x": round(float(track["vx"]), 3),
                "z": round(float(track["vz"]), 3),
            }
            enriched["worldSpeedMps"] = round(
                float(np.hypot(float(track["vx"]), float(track["vz"]))),
                3,
            )

    return tuple(output)


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
        "playerPos": (
            data.get("playerPos")
            or data.get("playerPosition")
            or data.get("position")
            or {}
        ),
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
    terrain_y, height_above_terrain, hill_map_debug = apply_hill_map_height_to_terrain(
        xyz=arrays["xyz"],
        fallback_terrain_y=terrain_y,
    )
    terrain_profile_debug = {
        **terrain_profile_debug,
        "hillMapHeightCsv": hill_map_debug,
    }
    valid_object_mask, object_filter_debug = compute_valid_object_mask(
        angles=arrays["angles"],
        horizontal_ranges=arrays["horizontal_ranges"],
        distances=arrays["distances"],
        xyz=arrays["xyz"],
        obstacle_mask=obstacle_mask,
        terrain_y=terrain_y,
        height_above_terrain=height_above_terrain,
    )

    if not GROUND_FILTER_ENABLED:
        point_count = arrays["distances"].size
        ground_mask = np.zeros(point_count, dtype=bool)
        obstacle_mask = np.ones(point_count, dtype=bool)
        valid_object_mask = np.ones(point_count, dtype=bool)
        stack_mask = np.ones(point_count, dtype=bool)
        object_filter_debug = {
            **object_filter_debug,
            "enabled": False,
            "method": "disabled_raw_lidar_passthrough",
            "validPointCount": int(point_count),
            "note": "ground filtering disabled; every detected LiDAR point is retained",
        }

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
    clusters = merge_split_world_clusters(clusters)
    clusters = assign_persistent_object_ids(clusters, data.get("time"))

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


def make_lidar_box(x_px: int, y_px: int, color: str, radius_px: int | None = None, label: str | None = None) -> dict[str, Any]:
    radius = int(radius_px if radius_px is not None else POINT_RADIUS_PX)
    radius = max(1, min(12, radius))
    return {
        "className": POINT_CLASS_NAME if label is None else str(label),
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


def lidar_vehicle_cluster_mask(cache: FrameCache) -> np.ndarray:
    """Recover the full LiDAR body around vehicle-like seed clusters.

    Valid-object points usually cover only the roof/upper body because the
    ground filter is intentionally conservative. Once a vehicle-like seed is
    confirmed, recover nearby non-ground returns from the same angle/range
    volume so the lower body is not discarded. This remains LiDAR-only.
    """
    result = np.zeros(cache.distances.size, dtype=bool)
    if cache.valid_object_mask.size != cache.distances.size:
        return result

    for cluster in cache.clusters:
        if not bool(cluster.get("vehicleLike", False)):
            continue
        angle = safe_float(cluster.get("angleDeg"), None)
        near_distance = safe_float(
            cluster.get("surfaceDistanceM", cluster.get("distanceM")),
            None,
        )
        far_distance = safe_float(cluster.get("farDistanceM"), near_distance)
        width = safe_float(cluster.get("visibleWidthM"), 0.0) or 0.0
        if angle is None or near_distance is None:
            continue

        estimated_span = np.degrees(
            2.0
            * np.arctan2(
                max(0.25, float(width)) * 0.5,
                max(0.5, float(near_distance)),
            )
        )
        angle_gate = max(
            2.0,
            float(estimated_span) * 0.5 + VEHICLE_CLUSTER_ANGLE_MARGIN_DEG,
        )
        distance_min = max(
            0.0,
            float(near_distance) - VEHICLE_CLUSTER_DISTANCE_MARGIN_M,
        )
        distance_max = (
            float(far_distance if far_distance is not None else near_distance)
            + VEHICLE_CLUSTER_DISTANCE_MARGIN_M
        )
        angle_delta = np.abs(
            (
                (cache.angles.astype(np.float32) - float(angle) + 180.0)
                % 360.0
            )
            - 180.0
        )
        geometry_gate = (
            (angle_delta <= angle_gate)
            & (cache.distances >= distance_min)
            & (cache.distances <= distance_max)
        )
        if (
            cache.obstacle_mask.size == cache.distances.size
            and cache.height_above_terrain.size == cache.distances.size
        ):
            recovery_source = cache.obstacle_mask | (
                cache.height_above_terrain
                >= VEHICLE_RECOVERY_MIN_ABOVE_TERRAIN_M
            )
        elif cache.obstacle_mask.size == cache.distances.size:
            recovery_source = cache.obstacle_mask
        else:
            recovery_source = cache.valid_object_mask
        result |= recovery_source & geometry_gate
    return result


def lidar_object_overlay_mask(cache: FrameCache) -> np.ndarray:
    """Return the exact object-point mask used by simulator and web views."""
    mode = str(overlay_settings.get("simLidarPointMode", "valid_plus_high")).strip().lower()
    if cache.valid_object_mask.size == cache.distances.size:
        valid_mask = cache.valid_object_mask.copy()
    else:
        valid_mask = np.zeros(cache.distances.size, dtype=bool)

    if mode == "vehicle_clusters":
        return lidar_vehicle_cluster_mask(cache)
    if mode == "all_obstacles" and cache.obstacle_mask.size == cache.distances.size:
        return cache.obstacle_mask.copy()
    if mode == "valid_plus_high":
        result = valid_mask.copy()
        if (
            cache.height_above_terrain.size == cache.distances.size
            and cache.obstacle_mask.size == cache.distances.size
        ):
            high_threshold = max(
                0.35,
                float(
                    aim_settings.get(
                        "hillObjectMinTopClearanceM",
                        OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M,
                    )
                )
                * 0.65,
            )
            result |= cache.obstacle_mask & (
                cache.height_above_terrain >= high_threshold
            )
        return result
    return valid_mask


def render_lidar_overlay_boxes(
    cache: FrameCache,
    turret_state: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return simulator-screen LiDAR overlay boxes for /detect.

    v16.16 keeps the object-recognition path unchanged but makes the simulator
    overlay easier to debug:
      - valid_plus_high shows the LiDAR hits that actually land on objects,
        including high-above-terrain hits that are not yet a full cluster.
      - optional cluster boxes draw one visible rectangle around each LiDAR
        object candidate, so the user can confirm that LiDAR points are being
        projected onto the same camera image as YOLO.
    """
    if not bool(overlay_settings.get("showLidarPoints", True)):
        return [], 0

    projected = project_cached_points(cache, turret_state, width, height)
    projected_source = projected["source_index"]
    if projected_source.size == 0:
        return [], 0

    x_lookup = {int(src): int(x) for src, x in zip(projected_source.tolist(), projected["x"].tolist())}
    y_lookup = {int(src): int(y) for src, y in zip(projected_source.tolist(), projected["y"].tolist())}

    object_overlay_mask = lidar_object_overlay_mask(cache)

    projected_obstacles = projected_source[object_overlay_mask[projected_source]] if object_overlay_mask.size else np.empty(0, dtype=np.int32)
    projected_ground = projected_source[cache.ground_mask[projected_source]] if cache.ground_mask.size else np.empty(0, dtype=np.int32)

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
    point_radius = int(overlay_settings.get("objectPointRadiusPx", POINT_RADIUS_PX))
    for src in obstacle_selected.tolist():
        color = "#ff3b30" if (cache.valid_object_mask.size and bool(cache.valid_object_mask[src])) else obstacle_color(float(cache.distances[src]))
        boxes.append(
            make_lidar_box(
                x_lookup[src],
                y_lookup[src],
                color,
                radius_px=point_radius,
            )
        )

    for src in ground_selected.tolist():
        boxes.append(make_lidar_box(x_lookup[src], y_lookup[src], COLOR_SAFE_GROUND, radius_px=POINT_RADIUS_PX))

    # Draw compact LiDAR cluster rectangles on the simulator image.  These are
    # debug boxes, not YOLO boxes; firing still uses the cached LiDAR/YOLO logic.
    if bool(overlay_settings.get("showLidarClusterBoxes", True)) and cache.clusters:
        angle_gate = max(0.5, float(overlay_settings.get("clusterBoxAngleGateDeg", 4.0)))
        dist_gate = max(0.5, float(overlay_settings.get("clusterBoxDistanceGateM", 4.0)))
        min_points = max(1, int(overlay_settings.get("clusterBoxMinPoints", 2)))
        limit = max(0, int(overlay_settings.get("clusterBoxLimit", 12)))
        cluster_count = 0
        for cluster in cache.clusters[:limit]:
            c_angle = float(cluster.get("angleDeg", 0.0) or 0.0)
            c_dist = float(cluster.get("surfaceDistanceM", cluster.get("distanceM", 0.0)) or 0.0)
            srcs = []
            for src in projected_source.tolist():
                if not object_overlay_mask.size or not bool(object_overlay_mask[int(src)]):
                    continue
                if angle_gap_deg(float(cache.angles[int(src)]), c_angle) > angle_gate:
                    continue
                if abs(float(cache.distances[int(src)]) - c_dist) > dist_gate:
                    continue
                srcs.append(int(src))
            if len(srcs) < min_points:
                continue
            xs = [x_lookup[src] for src in srcs]
            ys = [y_lookup[src] for src in srcs]
            pad = 10
            shape_label = str(cluster.get("candidateLabel", "LIDAR"))
            object_id = str(cluster.get("objectId", ""))
            label = f"{shape_label} {object_id} {float(cluster.get('distanceM', 0.0)):.1f}m {float(cluster.get('angleDeg', 0.0)):+.1f}deg"
            cluster_color = "#00E5FF" if bool(cluster.get("tankLike", False)) else "#FFAE35"
            boxes.append({
                "className": label,
                "bbox": [
                    float(max(0, min(xs) - pad)),
                    float(max(0, min(ys) - pad)),
                    float(min(width - 1, max(xs) + pad)),
                    float(min(height - 1, max(ys) + pad)),
                ],
                "confidence": 1.0,
                "color": cluster_color,
                "filled": False,
                "updateBoxWhileMoving": False,
            })
            cluster_count += 1
            if cluster_count >= limit:
                break

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
            "Default is YOLO.pt. Put YOLO.pt next to this Python file "
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
    """Run YOLO.pt using the same YOLO path as Second.py.

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
    world_geometry = compact_world_geometry(surface_xyz, surface_xyz, aim_point_y)

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
        "worldCenter": world_geometry.get("worldCenter"),
        "surfaceCenterWorld": world_geometry.get("surfaceCenterWorld"),
        "aimPointWorld": world_geometry.get("aimPointWorld"),
        "worldBounds": world_geometry.get("worldBounds"),
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
                "intervalSec": 0.5,
                "lidarYPositionM": 3.0,
                "channel": 32,
                "minimapChannel": 16,
                "maxDistanceM": 120,
                "lidarPosition": "Body",
                "sendDetectedLidar": True,
                "frameRate": 120,
                "graphicsQuality": "Ultra",
            },
            "fusionDefaults": {
                "model": "YOLO.pt",
                "modelClasses": ["Ally", "Enemy", "House", "Rock", "Rock_L", "Tank_enemy", "Tent", "car"],
                "imageSize": 640,
                "yoloIntervalSec": 0.50,
                "labelFormat": "Tank_enemy | distance_m | body_relative_angle_deg",
                "tiltCompensationMode": "ground_plane",
                "presets": [
                    "balanced",
                    "cpu_light",
                    "tank_accuracy",
                    "tank_candidate_test",
                ],
                "genericTankNote": (
                    "YOLO.pt detects Tank_enemy directly. The fire logic maps Tank_enemy to enemy_tank."
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
    return jsonify({"status": "ok", "server": "Tank Challenge LiDAR-first YOLO Fusion v16.16 Easy Map Switch + Sim LiDAR Overlay"})


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

    player_position = extract_position_dict(
        data.get("playerPos")
        or data.get("playerPosition")
        or data.get("position")
        or {}
    )
    player_position_source = "info.playerPos"
    if player_position is None:
        lidar_position = extract_position_dict(data.get("lidarOrigin") or {})
        if lidar_position is not None:
            player_position = {
                "x": float(lidar_position["x"]),
                "y": float(lidar_position["y"]) - EXPECTED_LIDAR_Y_POSITION_M,
                "z": float(lidar_position["z"]),
            }
            player_position_source = "info.lidarOrigin_minus_mount_height"

    with state_lock:
        latest_cache = cache
        if player_position is not None:
            latest_player_state["position"] = [
                player_position["x"],
                player_position["y"],
                player_position["z"],
            ]
            latest_player_state["source"] = player_position_source
            latest_player_state["updatedAt"] = now_text()
            latest_player_state["updatedMonotonic"] = monotonic()
            ground_truth_state["latestPlayerPosition"] = list(
                latest_player_state["position"]
            )
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
            "worldCenter": cluster.get("worldCenter"),
            "surfaceCenterWorld": cluster.get("surfaceCenterWorld"),
            "aimPointWorld": cluster.get("aimPointWorld"),
            "worldBounds": cluster.get("worldBounds"),
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
        "worldCenter": cluster.get("worldCenter"),
        "surfaceCenterWorld": cluster.get("surfaceCenterWorld"),
        "aimPointWorld": cluster.get("aimPointWorld"),
        "worldBounds": cluster.get("worldBounds"),
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
    raw = str(aim_settings.get("pitchSweepOffsetsDeg", "0,0.3,-0.3,0.6,-0.6,0.9,-0.9"))
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
            latest_player_state["source"] = "get_action.position"
            latest_player_state["updatedAt"] = now_text()
            latest_player_state["updatedMonotonic"] = monotonic()
            ground_truth_state["latestPlayerPosition"] = list(latest_player_state["position"])
        status_state["getActionRequestCount"] += 1

    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)

    action = build_seek_attack_action(cache, turret_state)
    return jsonify(action)


@app.route("/hill_map_height_status", methods=["GET"])
def hill_map_height_status():
    if request.args.get("reload") is not None:
        load_hill_map_height(force=True)
    return jsonify({"status": json_copy(hill_map_height_state)})


@app.route("/front_view_update", methods=["GET", "POST"])
def front_view_update():
    if "flipVerticalDisplay" in request.args:
        front_view_settings["flipVerticalDisplay"] = safe_bool(
            request.args.get("flipVerticalDisplay"),
            bool(front_view_settings.get("flipVerticalDisplay", True)),
        )
    return jsonify({"status": "success", "frontView": dict(front_view_settings)})


@app.route("/lidar_status", methods=["GET"])
def lidar_status():
    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)
        player_state = json_copy(latest_player_state)
        track_snapshot = json_copy(object_tracks)
    return jsonify(
        {
            "frameSeq": cache.seq,
            "groundFilterEnabled": GROUND_FILTER_ENABLED,
            "lidarVehicleExtractionEnabled": LIDAR_VEHICLE_EXTRACTION_ENABLED,
            "vehicleExtractionThresholds": {
                "geometryFilterEnabled": False,
                "minPoints": VEHICLE_MIN_POINTS,
            },
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
            "hillMapHeight": json_copy(hill_map_height_state),
            "objectFilter": json_copy(cache.ground_plane_debug.get("objectFilter", {})),
            "clusters": json_copy(cache.clusters),
            "objectTracks": track_snapshot,
            "playerPosition": player_state,
            "playerPosFromInfo": json_copy(cache.pose.get("playerPos", {})),
            "lidarOrigin": json_copy(cache.pose.get("lidarOrigin", {})),
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


def yolo_result_age_sec() -> float | None:
    with state_lock:
        meta = json_copy(yolo_state.get("latestResultMeta", {}))
    completed = meta.get("completedMonotonic")
    if completed is None:
        return None
    try:
        return round(max(0.0, monotonic() - float(completed)), 3)
    except Exception:
        return None


def object_target_id(obj: dict[str, Any], cluster: dict[str, Any]) -> str:
    semantic = str(obj.get("semanticClass", obj.get("originalSemanticClass", "object")))
    raw_name = str(obj.get("originalRawClassName", obj.get("rawClassName", semantic)))
    key = str(cluster.get("candidateKey", cluster.get("clusterId", "unknown")))
    return f"{raw_name}:{semantic}:{key}"


def make_fire_team_target(obj: dict[str, Any], cache: FrameCache, turret_state: dict[str, Any]) -> dict[str, Any] | None:
    """Build the compact target contract for the fire team.

    This intentionally does not expose raw LiDAR points.  The fire side only
    receives the best LiDAR/YOLO-fused object coordinate summary.
    """
    if not bool(obj.get("fusionMatched", False)):
        return None
    cluster = obj.get("lidarCluster") or {}
    if not isinstance(cluster, dict) or not cluster:
        return None

    angle = safe_float(obj.get("lidarBodyAngleDeg"), safe_float(cluster.get("angleDeg"), None))
    distance = safe_float(obj.get("distance"), safe_float(cluster.get("distanceM"), None))
    pitch = safe_float(cluster.get("aimPitchDeg"), None)
    if angle is None or distance is None:
        return None

    raw_name = str(obj.get("originalRawClassName", obj.get("rawClassName", obj.get("semanticClass", "object"))))
    semantic = str(obj.get("semanticClass", obj.get("originalSemanticClass", raw_name)))
    current_yaw = current_turret_body_yaw_deg(cache, turret_state)
    current_pitch = current_turret_pitch_deg(turret_state)
    yaw_error = normalize_signed_angle(float(angle) - float(current_yaw))
    pitch_error = (float(pitch) - float(current_pitch)) if pitch is not None else None
    aim_point = cluster.get("aimPointWorld") or None
    center = cluster.get("worldCenter") or None
    surface_center = cluster.get("surfaceCenterWorld") or None

    target = {
        "targetId": object_target_id(obj, cluster),
        "source": "YOLO_LIDAR_FUSED",
        "frameSeq": cache.seq,
        "simulationTime": cache.simulation_time,
        "freshAgeSec": yolo_result_age_sec(),
        "className": raw_name,
        "semanticClass": semantic,
        "confidence": round_float(obj.get("confidence"), 4, None),
        "isTank": bool(is_tank_semantic(semantic) or is_tank_semantic(raw_name)),
        "isFireCandidate": bool(is_tank_semantic(semantic) or is_tank_semantic(raw_name)),
        "distanceM": round_float(distance),
        "surfaceDistanceM": round_float(cluster.get("surfaceDistanceM", distance)),
        "medianDistanceM": round_float(cluster.get("medianDistanceM"), 3, None),
        "farDistanceM": round_float(cluster.get("farDistanceM"), 3, None),
        "bodyYawDeg": round_float(angle),
        "aimPitchDeg": round_float(pitch, 3, None),
        "turretYawErrorDeg": round_float(yaw_error),
        "turretPitchErrorDeg": round_float(pitch_error, 3, None),
        "world": {
            "aimPoint": json_copy(aim_point),
            "center": json_copy(center),
            "surfaceCenter": json_copy(surface_center),
            "bounds": json_copy(cluster.get("worldBounds")),
        },
        "geometry": {
            "pointCount": int(cluster.get("pointCount", 0) or 0),
            "visibleWidthM": round_float(cluster.get("visibleWidthM"), 3, None),
            "heightSpanM": round_float(cluster.get("heightSpanM"), 3, None),
            "objectHeightAboveTerrainM": round_float(cluster.get("objectHeightAboveTerrainM"), 3, None),
            "aimHeightAboveBaseM": round_float(cluster.get("aimHeightAboveBaseM"), 3, None),
            "terrainBaseYWorldM": round_float(cluster.get("terrainBaseYWorldM"), 3, None),
            "objectTopYWorldM": round_float(cluster.get("objectTopYWorldM"), 3, None),
            "depthSpanM": round_float(cluster.get("depthSpanM"), 3, None),
            "verticalityRatio": round_float(cluster.get("verticalityRatio"), 3, None),
        },
        "quality": {
            "fusionMethod": obj.get("fusionMethod", cluster.get("fusionMethod")),
            "fusionAngleGapDeg": round_float(obj.get("fusionAngleGapDeg"), 3, None),
            "roiObstaclePointCount": int(cluster.get("roiObstaclePointCount", cluster.get("pointCount", 0)) or 0),
            "roiValidObjectPointCount": int(cluster.get("roiValidObjectPointCount", 0) or 0),
        },
    }

    # If an older fused object lacks the new coordinate fields, still expose a
    # deterministic fallback from angle/distance. New frames will replace it with
    # real LiDAR world coordinates.
    if target["world"]["aimPoint"] is None:
        origin = get_xyz(cache.pose.get("lidarOrigin")) or get_xyz(cache.pose.get("playerPos"))
        if origin is not None:
            body_yaw = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
            world_yaw = radians(float(body_yaw) + float(angle))
            x = float(origin[0]) + float(distance) * sin(world_yaw)
            z = float(origin[2]) + float(distance) * cos(world_yaw)
            y = safe_float(cluster.get("aimPointYWorldM"), origin[1]) or float(origin[1])
            fallback = xyz_to_dict((x, y, z))
            target["world"]["aimPoint"] = fallback
            target["world"]["surfaceCenter"] = fallback
            target["quality"]["worldCoordinateFallback"] = "angle_distance_from_lidar_origin"
    return target


def build_fire_team_targets(
    cache: FrameCache,
    turret_state: dict[str, Any],
    tank_only: bool = True,
    max_targets: int = 8,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for obj in fresh_yolo_objects():
        target = make_fire_team_target(obj, cache, turret_state)
        if target is None:
            continue
        if tank_only and not bool(target.get("isTank", False)):
            continue
        targets.append(target)
    targets.sort(key=lambda item: (not bool(item.get("isFireCandidate")), float(item.get("distanceM") or 9999.0)))
    return targets[: max(1, int(max_targets))]


@app.route("/fire_targets", methods=["GET"])
@app.route("/shooting_targets", methods=["GET"])
def fire_targets():
    tank_only = safe_bool(request.args.get("tankOnly"), True)
    max_targets = int(safe_float(request.args.get("limit"), 8) or 8)
    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)
        aim_snapshot = json_copy(aim_state)
        fire_snapshot = json_copy(fire_state)
    targets = build_fire_team_targets(cache, turret_state, tank_only=tank_only, max_targets=max_targets)
    return jsonify({
        "status": "success",
        "contract": "fire_team_targets_v1_compact_world_coordinates",
        "description": "Only LiDAR/YOLO-fused object summaries are returned; raw LiDAR points are not exposed.",
        "frameSeq": cache.seq,
        "simulationTime": cache.simulation_time,
        "targetCount": len(targets),
        "tankOnly": bool(tank_only),
        "targets": targets,
        "selectedTarget": aim_snapshot.get("selectedTarget"),
        "confirmedTarget": aim_snapshot.get("confirmedTarget"),
        "fire": fire_snapshot,
        "recommendedConsumerFields": [
            "targets[].world.aimPoint",
            "targets[].distanceM",
            "targets[].bodyYawDeg",
            "targets[].aimPitchDeg",
            "targets[].quality.fusionMethod",
        ],
    })


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
        color = '#00e5ff' if is_selected or bool(item.get("tankLike", False)) else '#ffae35'
        radius = 13 if is_selected else 7
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='none' stroke='{color}' stroke-width='3'/>")
        h = safe_float(item.get("objectHeightAboveTerrainM"), None)
        pitch = safe_float(item.get("aimPitchDeg"), None)
        htxt = f" h{h:.1f}m" if h is not None else ""
        ptxt = f" p{pitch:+.1f}°" if pitch is not None else ""
        parts.append(f"<text x='{x+9:.1f}' y='{y-7:.1f}' fill='{color}' font-size='12'>{html.escape(str(item.get('candidateLabel','OBJ')))} {html.escape(str(item.get('objectId','')))} {dist:.1f}m {a:+.1f}°{htxt}{ptxt}</text>")

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

def world_xz_from_point(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        x = safe_float(raw.get("x"), None)
        z = safe_float(raw.get("z"), None)
        if x is not None and z is not None:
            return float(x), float(z)
    return None


def cluster_world_point(item: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("aimPointWorld", "surfaceCenterWorld", "worldCenter"):
        point = world_xz_from_point(item.get(key))
        if point is not None:
            return point
    return None


def world_xyz_dict_to_array(raw: Any) -> np.ndarray | None:
    """Convert {'x','y','z'} to a NumPy vector for compact GT/LiDAR comparison."""
    xyz = get_xyz(raw) if isinstance(raw, dict) else None
    if xyz is None:
        return None
    return np.asarray(xyz, dtype=np.float64)


def cluster_world_point_dict(item: dict[str, Any]) -> dict[str, float] | None:
    """Preferred LiDAR-measured world point for map comparison.

    surfaceCenterWorld is preferred because .map coordinates are object pivots,
    while LiDAR sees the nearest visible surface. aimPointWorld is better for
    shooting, but surfaceCenterWorld is better for sensor-vs-map diagnostics.
    """
    for key in ("surfaceCenterWorld", "aimPointWorld", "worldCenter"):
        value = item.get(key)
        if isinstance(value, dict) and world_xz_from_point(value) is not None:
            return value
    return None



def local_map_files() -> list[Path]:
    """Return local .map files sorted by name."""
    return sorted(path for path in BASE_DIR.glob("*.map") if path.is_file())


def summarize_lidar_gt_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact automatic error statistics for the web/API panels."""
    matched = [item for item in matches if bool(item.get("matched"))]
    all_items = list(matches)

    def values(key: str, only_matched: bool = True) -> list[float]:
        source = matched if only_matched else all_items
        out: list[float] = []
        for item in source:
            value = safe_float((item.get("error") or {}).get(key), None)
            if value is not None:
                out.append(float(value))
        return out

    def stat_pack(key: str, only_abs: bool = False) -> dict[str, Any]:
        raw = values(key, only_matched=True)
        if only_abs:
            raw = [abs(v) for v in raw]
        if not raw:
            return {"count": 0, "mean": None, "median": None, "max": None, "min": None}
        arr = np.asarray(raw, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "min": round(float(np.min(arr)), 4),
        }

    dx = values("dxM", only_matched=True)
    dy = values("dyM", only_matched=True)
    dz = values("dzM", only_matched=True)
    bias = {
        "dxM": round(float(np.mean(dx)), 4) if dx else None,
        "dyM": round(float(np.mean(dy)), 4) if dy else None,
        "dzM": round(float(np.mean(dz)), 4) if dz else None,
    }

    # Lower is better. Penalize maps that match only a few objects.
    xz_median = stat_pack("worldErrorXZM").get("median")
    angle_median = stat_pack("angleErrorDeg", only_abs=True).get("median")
    matched_count = len(matched)
    total_count = max(1, len(all_items))
    coverage = matched_count / total_count
    if xz_median is None:
        score = 999999.0
    else:
        score = float(xz_median) + 0.25 * float(angle_median or 0.0) + 10.0 * (1.0 - coverage)

    return {
        "matchedCount": int(matched_count),
        "comparedLidarCount": int(len(all_items)),
        "matchCoverage": round(float(coverage), 4),
        "scoreLowerIsBetter": round(float(score), 4),
        "worldErrorXZ": stat_pack("worldErrorXZM"),
        "worldError3D": stat_pack("worldError3DM"),
        "absAngleErrorDeg": stat_pack("angleErrorDeg", only_abs=True),
        "distanceErrorToGtCenterM": stat_pack("distanceErrorToGtCenterM"),
        "distanceErrorToApproxSurfaceM": stat_pack("distanceErrorToApproxSurfaceM"),
        "meanBiasLidarMinusGt": bias,
        "interpretation": (
            "worldErrorXZ compares .map pivot/center to LiDAR measured surface point. "
            "distanceErrorToApproxSurface is often fairer for large tanks/rocks."
        ),
    }


def format_compare_stats_html(stats: dict[str, Any]) -> str:
    """Small summary line for /lidar_view."""
    wxz = stats.get("worldErrorXZ") or {}
    w3d = stats.get("worldError3D") or {}
    ang = stats.get("absAngleErrorDeg") or {}
    surf = stats.get("distanceErrorToApproxSurfaceM") or {}
    bias = stats.get("meanBiasLidarMinusGt") or {}

    def fmt(value: Any, digits: int = 2) -> str:
        number = safe_float(value, None)
        return "-" if number is None else f"{float(number):.{digits}f}"

    return (
        f"matched={stats.get('matchedCount', 0)}/{stats.get('comparedLidarCount', 0)} "
        f"coverage={fmt(100.0 * float(stats.get('matchCoverage', 0.0) or 0.0), 1)}% | "
        f"XZ err mean/median/max={fmt(wxz.get('mean'))}/{fmt(wxz.get('median'))}/{fmt(wxz.get('max'))}m | "
        f"3D err median={fmt(w3d.get('median'))}m | "
        f"abs angle median={fmt(ang.get('median'))}° | "
        f"surface dist median={fmt(surf.get('median'))}m | "
        f"bias dx/dy/dz={fmt(bias.get('dxM'))}/{fmt(bias.get('dyM'))}/{fmt(bias.get('dzM'))}m | "
        f"score={fmt(stats.get('scoreLowerIsBetter'))}"
    )


def maybe_rotate_active_map_gt(cache: FrameCache | None = None) -> dict[str, Any]:
    """Rotate or auto-select .map files while the web dashboard refreshes."""
    with state_lock:
        enabled = bool(map_cycle_settings.get("enabled", False))
        mode = str(map_cycle_settings.get("mode", "cycle")).strip().lower()
        interval = max(1.0, float(map_cycle_settings.get("intervalSec", 6.0) or 6.0))
        last = map_cycle_settings.get("lastSwitchMonotonic")

    maps = local_map_files()
    if not enabled:
        return {"enabled": False, "mode": mode, "mapFiles": [m.name for m in maps]}
    if not maps:
        with state_lock:
            map_cycle_settings["lastError"] = "No .map files found next to this Python file."
        return {"enabled": True, "status": "no_map_files", "mapFiles": []}

    now = monotonic()
    if last is not None and now - float(last) < interval:
        with state_lock:
            return {**json_copy(map_cycle_settings), "mapFiles": [m.name for m in maps], "status": "waiting_interval"}

    if mode == "best" and cache is not None:
        result = score_all_local_maps_against_lidar(cache, apply_best=True, max_items=40, max_match_world_error_m=18.0)
        with state_lock:
            map_cycle_settings["lastSwitchMonotonic"] = now
            map_cycle_settings["lastSwitchAt"] = now_text()
            map_cycle_settings["lastBestScan"] = json_copy(result)
            best = result.get("best") or {}
            map_cycle_settings["currentMapFile"] = best.get("filename")
            map_cycle_settings["lastError"] = None if result.get("status") == "success" else result.get("message")
        return {**json_copy(map_cycle_settings), "status": "best_scored", "mapFiles": [m.name for m in maps]}

    with state_lock:
        current_index = int(map_cycle_settings.get("currentIndex", -1) or -1)
        next_index = (current_index + 1) % len(maps)
        map_cycle_settings["currentIndex"] = next_index
        map_cycle_settings["currentMapFile"] = maps[next_index].name
        map_cycle_settings["lastSwitchMonotonic"] = now
        map_cycle_settings["lastSwitchAt"] = now_text()
        map_cycle_settings["lastError"] = None

    result = load_map_ground_truth(filename=maps[next_index].name, clear_existing=True, persist_selection=False)
    if result.get("status") != "success":
        with state_lock:
            map_cycle_settings["lastError"] = result.get("message", result.get("status"))
    return {**json_copy(map_cycle_settings), "loadResult": json_copy(result), "status": "rotated", "mapFiles": [m.name for m in maps]}

def ensure_default_map_gt_available() -> dict[str, Any]:
    """Make the web comparison usable with minimum manual steps.

    Priority:
      1) already-registered GT objects
      2) persisted map restored by /map_gt_load
      3) NewMap.map next to this script
      4) exactly one .map file next to this script

    If multiple .map files exist and none was loaded, the user must choose one
    with /map_gt_load?filename=YOUR.map&clearExisting=true.
    """
    ensure_map_gt_available()
    with state_lock:
        existing_count = len(ground_truth_state.get("objects", {}))
    if existing_count > 0:
        return {"status": "already_loaded", "registeredCount": existing_count}

    preferred = BASE_DIR / "NewMap.map"
    if preferred.exists():
        return load_map_ground_truth(filename=preferred.name, clear_existing=True, persist_selection=True)

    maps = sorted(path for path in BASE_DIR.glob("*.map") if path.is_file())
    if len(maps) == 1:
        return load_map_ground_truth(filename=maps[0].name, clear_existing=True, persist_selection=True)

    return {
        "status": "not_loaded",
        "registeredCount": 0,
        "mapFiles": [path.name for path in maps],
        "message": "Load a map with /map_gt_load?filename=YOUR.map&clearExisting=true",
    }


def build_lidar_gt_comparisons(
    cache: FrameCache,
    max_items: int = 40,
    max_match_world_error_m: float = 10.0,
) -> dict[str, Any]:
    """Compare LiDAR-measured object world coordinates with .map object pivots.

    This is intentionally dashboard/debug data. It is not used to aim or fire;
    aiming remains LiDAR-first so it does not cheat with map coordinates.
    """
    cycle_result = maybe_rotate_active_map_gt(cache)
    load_result = ensure_default_map_gt_available()
    gt_items = active_gt_metrics(cache)
    lidar_items = [json_copy(item) for item in cache.clusters[: max(1, int(max_items))]]

    matches: list[dict[str, Any]] = []
    unmatched_lidar: list[dict[str, Any]] = []
    used_gt_ids: set[str] = set()

    for cluster in lidar_items:
        lidar_point_dict = cluster_world_point_dict(cluster)
        lidar_point = world_xyz_dict_to_array(lidar_point_dict)
        if lidar_point is None:
            unmatched_lidar.append({"cluster": cluster, "reason": "no_lidar_world_point"})
            continue

        ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        lidar_angle = safe_float(cluster.get("angleDeg"), None)
        lidar_distance = safe_float(cluster.get("distanceM"), None)
        for gt in gt_items:
            # One map object may be paired with only one LiDAR cluster.
            if str(gt.get("id")) in used_gt_ids:
                continue
            gt_pos = world_xyz_dict_to_array(gt.get("position"))
            if gt_pos is None:
                continue
            dx = float(lidar_point[0] - gt_pos[0])
            dy = float(lidar_point[1] - gt_pos[1])
            dz = float(lidar_point[2] - gt_pos[2])
            world_error_xz = float(np.hypot(dx, dz))
            world_error_3d = float(np.linalg.norm(lidar_point - gt_pos))

            gt_angle = safe_float(gt.get("bodyRelativeAngleDeg"), None)
            angle_error = normalize_signed_angle(float(lidar_angle) - float(gt_angle)) if lidar_angle is not None and gt_angle is not None else None
            angle_gap = abs(float(angle_error)) if angle_error is not None else 9999.0

            gt_center_distance = safe_float(gt.get("centerHorizontalDistanceM"), None)
            gt_surface_distance = safe_float(gt.get("approxSurfaceDistanceM"), None)
            dist_err_center = float(lidar_distance) - float(gt_center_distance) if lidar_distance is not None and gt_center_distance is not None else None
            dist_err_surface = float(lidar_distance) - float(gt_surface_distance) if lidar_distance is not None and gt_surface_distance is not None else None

            # World coordinate error should dominate. Angle/distance only break ties.
            score = world_error_xz + 0.03 * angle_gap
            if dist_err_surface is not None:
                score += 0.02 * abs(float(dist_err_surface))
            elif dist_err_center is not None:
                score += 0.02 * abs(float(dist_err_center))

            detail = {
                "score": round(score, 4),
                "worldErrorXZM": round(world_error_xz, 4),
                "worldError3DM": round(world_error_3d, 4),
                "dxM": round(dx, 4),
                "dyM": round(dy, 4),
                "dzM": round(dz, 4),
                "angleErrorDeg": round(float(angle_error), 4) if angle_error is not None else None,
                "distanceErrorToGtCenterM": round(float(dist_err_center), 4) if dist_err_center is not None else None,
                "distanceErrorToApproxSurfaceM": round(float(dist_err_surface), 4) if dist_err_surface is not None else None,
            }
            ranked.append((score, gt, detail))

        if not ranked:
            unmatched_lidar.append({"cluster": cluster, "reason": "no_gt_candidates"})
            continue

        ranked.sort(key=lambda item: item[0])
        _, best_gt, detail = ranked[0]
        matched = float(detail["worldErrorXZM"]) <= float(max_match_world_error_m)
        if matched:
            used_gt_ids.add(str(best_gt.get("id")))
        else:
            unmatched_lidar.append({
                "cluster": cluster,
                "nearestGt": best_gt,
                "nearestError": detail,
                "reason": "nearest_gt_too_far",
            })

        matches.append({
            "matched": bool(matched),
            "lidar": {
                "candidateKey": cluster.get("candidateKey"),
                "candidateLabel": cluster.get("candidateLabel"),
                "distanceM": cluster.get("distanceM"),
                "angleDeg": cluster.get("angleDeg"),
                "aimPitchDeg": cluster.get("aimPitchDeg"),
                "pointCount": cluster.get("pointCount"),
                "worldPointUsed": json_copy(lidar_point_dict),
                "worldCenter": json_copy(cluster.get("worldCenter")),
                "surfaceCenterWorld": json_copy(cluster.get("surfaceCenterWorld")),
                "aimPointWorld": json_copy(cluster.get("aimPointWorld")),
                "worldBounds": json_copy(cluster.get("worldBounds")),
            },
            "mapGt": json_copy(best_gt),
            "error": detail,
        })

    unmatched_gt = [json_copy(item) for item in gt_items if str(item.get("id")) not in used_gt_ids]
    matches.sort(key=lambda item: (not bool(item.get("matched")), float((item.get("error") or {}).get("worldErrorXZM", 9999.0))))

    return {
        "status": "success" if gt_items else "no_gt_objects",
        "contract": "lidar_vs_map_gt_world_coordinate_compare_v1",
        "frameSeq": cache.seq,
        "simulationTime": cache.simulation_time,
        "mapLoad": json_copy(load_result),
        "activeMapFile": ground_truth_state.get("activeMapFile"),
        "gtCount": len(gt_items),
        "lidarClusterCount": len(lidar_items),
        "matchedCount": int(sum(1 for item in matches if item.get("matched"))),
        "maxMatchWorldErrorM": round(float(max_match_world_error_m), 3),
        "stats": summarize_lidar_gt_matches(matches),
        "mapCycle": json_copy(cycle_result),
        "matches": matches,
        "unmatchedLidar": unmatched_lidar,
        "unmatchedGt": unmatched_gt,
        "note": ".map position is the object pivot/center. LiDAR worldPointUsed is usually the visible surface/aim point, so a few meters of offset can be normal for tanks/rocks.",
    }



def score_all_local_maps_against_lidar(
    cache: FrameCache,
    apply_best: bool = True,
    max_items: int = 40,
    max_match_world_error_m: float = 18.0,
) -> dict[str, Any]:
    """Load each local .map, compare against current LiDAR clusters, and rank by error.

    This is for debugging/calibration only. It does not drive the tank or use
    map coordinates for firing. If apply_best=true, the lowest-score map remains
    loaded as the active GT map for /lidar_view.
    """
    maps = local_map_files()
    if not maps:
        return {"status": "no_map_files", "baseDir": str(BASE_DIR), "results": []}

    with state_lock:
        previous_enabled = bool(map_cycle_settings.get("enabled", False))
        previous_active = ground_truth_state.get("activeMapFile")

    results: list[dict[str, Any]] = []
    try:
        with state_lock:
            map_cycle_settings["enabled"] = False

        for path in maps:
            load_result = load_map_ground_truth(filename=path.name, clear_existing=True, persist_selection=False)
            compare = build_lidar_gt_comparisons(
                cache,
                max_items=max_items,
                max_match_world_error_m=max_match_world_error_m,
            )
            stats = compare.get("stats") or summarize_lidar_gt_matches(compare.get("matches") or [])
            results.append({
                "filename": path.name,
                "path": str(path),
                "loadStatus": load_result.get("status"),
                "gtCount": compare.get("gtCount"),
                "lidarClusterCount": compare.get("lidarClusterCount"),
                "matchedCount": compare.get("matchedCount"),
                "scoreLowerIsBetter": stats.get("scoreLowerIsBetter"),
                "stats": json_copy(stats),
            })

        results.sort(key=lambda item: (
            float(item.get("scoreLowerIsBetter") if item.get("scoreLowerIsBetter") is not None else 999999.0),
            -int(item.get("matchedCount") or 0),
            str(item.get("filename")),
        ))
        best = results[0] if results else None

        if apply_best and best is not None:
            load_map_ground_truth(filename=str(best["filename"]), clear_existing=True, persist_selection=True)
        elif previous_active:
            load_map_ground_truth(path_value=str(previous_active), clear_existing=True, persist_selection=False)

        return {
            "status": "success",
            "baseDir": str(BASE_DIR),
            "applyBest": bool(apply_best),
            "best": json_copy(best),
            "results": json_copy(results),
            "note": (
                "Scores compare the current live LiDAR clusters against each .map file. "
                "The simulator itself is not switched by this route; it only switches the Python-side GT file."
            ),
        }
    finally:
        with state_lock:
            map_cycle_settings["enabled"] = previous_enabled

def gt_xz_from_item(item: dict[str, Any]) -> tuple[float, float] | None:
    pos = item.get("position") if isinstance(item, dict) else None
    return world_xz_from_point(pos) if isinstance(pos, dict) else None


def svg_world_gt_lidar_compare(compare: dict[str, Any], cache: FrameCache, width: int = 980, height: int = 560) -> str:
    """World X/Z map overlay: .map GT pivots vs LiDAR measured object points."""
    matches = compare.get("matches", []) if isinstance(compare, dict) else []
    unmatched_gt = compare.get("unmatchedGt", []) if isinstance(compare, dict) else []
    player_raw = get_xyz(cache.pose.get("playerPos")) or get_xyz(cache.pose.get("lidarOrigin"))
    player_xz = (float(player_raw[0]), float(player_raw[2])) if player_raw is not None else (0.0, 0.0)

    points: list[tuple[float, float]] = [player_xz]
    for item in matches:
        lidar_point = ((item.get("lidar") or {}).get("worldPointUsed") or {})
        p = world_xz_from_point(lidar_point)
        if p is not None:
            points.append(p)
        gt_p = gt_xz_from_item(item.get("mapGt") or {})
        if gt_p is not None:
            points.append(gt_p)
    for gt in unmatched_gt:
        p = gt_xz_from_item(gt)
        if p is not None:
            points.append(p)

    pad = 18.0
    if len(points) <= 1:
        min_x, max_x = player_xz[0] - 80.0, player_xz[0] + 80.0
        min_z, max_z = player_xz[1] - 80.0, player_xz[1] + 80.0
    else:
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        min_x, max_x = min(xs) - pad, max(xs) + pad
        min_z, max_z = min(zs) - pad, max(zs) + pad
        span = max(max_x - min_x, max_z - min_z, 30.0)
        mid_x, mid_z = (min_x + max_x) / 2.0, (min_z + max_z) / 2.0
        min_x, max_x = mid_x - span / 2.0, mid_x + span / 2.0
        min_z, max_z = mid_z - span / 2.0, mid_z + span / 2.0

    margin_l, margin_r, margin_t, margin_b = 54, 18, 40, 46
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def sx(x: float) -> float:
        return margin_l + (x - min_x) / max(1e-6, max_x - min_x) * plot_w

    def sy(z: float) -> float:
        return margin_t + (max_z - z) / max(1e-6, max_z - min_z) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart gtchart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#111318' stroke='#444'/>")
    for t in np.linspace(0.0, 1.0, 6):
        x = margin_l + t * plot_w
        y = margin_t + t * plot_h
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#2a2d35'/>")
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#2a2d35'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-14}' fill='#aaa' font-size='11' text-anchor='middle'>x {min_x + t*(max_x-min_x):.0f}</text>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#aaa' font-size='11'>z {max_z - t*(max_z-min_z):.0f}</text>")

    px, py = sx(player_xz[0]), sy(player_xz[1])
    body_yaw = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
    parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='8' fill='#45d9ff'/>")
    parts.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{px + 32*sin(radians(float(body_yaw))):.1f}' y2='{py - 32*cos(radians(float(body_yaw))):.1f}' stroke='#00ffff' stroke-width='3'/>")
    parts.append(f"<text x='{px+10:.1f}' y='{py-10:.1f}' fill='#45d9ff' font-size='12'>PLAYER</text>")

    # Draw matched pairs: GT square, LiDAR circle, line between them.
    for item in matches[:80]:
        gt = item.get("mapGt") or {}
        lidar = item.get("lidar") or {}
        err = item.get("error") or {}
        gp = gt_xz_from_item(gt)
        lp = world_xz_from_point(lidar.get("worldPointUsed") or {})
        if gp is None or lp is None:
            continue
        gx, gy = sx(gp[0]), sy(gp[1])
        lx, ly = sx(lp[0]), sy(lp[1])
        matched = bool(item.get("matched"))
        color = '#56d364' if matched else '#ff9f1c'
        parts.append(f"<line x1='{gx:.1f}' y1='{gy:.1f}' x2='{lx:.1f}' y2='{ly:.1f}' stroke='{color}' stroke-width='1.8' opacity='0.75'/>")
        parts.append(f"<rect x='{gx-5:.1f}' y='{gy-5:.1f}' width='10' height='10' fill='none' stroke='#b388ff' stroke-width='3'/>")
        parts.append(f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='7' fill='none' stroke='{color}' stroke-width='3'/>")
        label = f"{gt.get('id','GT')} ↔ {lidar.get('candidateKey','LiDAR')} e:{float(err.get('worldErrorXZM',0) or 0):.1f}m"
        parts.append(f"<text x='{lx+9:.1f}' y='{ly-7:.1f}' fill='{color}' font-size='12'>{html.escape(label)}</text>")

    # Draw unmatched GT pivots faintly.
    matched_gt_ids = {str((item.get("mapGt") or {}).get("id")) for item in matches if item.get("matched")}
    for gt in unmatched_gt[:80]:
        if str(gt.get("id")) in matched_gt_ids:
            continue
        gp = gt_xz_from_item(gt)
        if gp is None:
            continue
        gx, gy = sx(gp[0]), sy(gp[1])
        parts.append(f"<rect x='{gx-4:.1f}' y='{gy-4:.1f}' width='8' height='8' fill='none' stroke='#777' stroke-width='2' opacity='0.65'/>")
        parts.append(f"<text x='{gx+7:.1f}' y='{gy-5:.1f}' fill='#999' font-size='11'>{html.escape(str(gt.get('id','GT')))}</text>")

    parts.append("<text x='12' y='20' fill='#eee' font-size='14'>.map GT vs LiDAR world coordinates: purple square=.map pivot/center, circle=LiDAR measured surface/aim point, line=coordinate error</text>")
    parts.append("<text x='12' y='37' fill='#aaa' font-size='12'>Green line = within match gate, orange = nearest GT is too far. LiDAR sees surfaces, so tank/rock center offsets are normal.</text>")
    if not matches and not unmatched_gt:
        msg = html.escape(str((compare.get('mapLoad') or {}).get('message', 'No GT map loaded or no LiDAR clusters yet.')))
        parts.append(f"<text x='{width/2:.1f}' y='{height/2:.1f}' fill='#aaa' font-size='16' text-anchor='middle'>{msg}</text>")
    parts.append("</svg>")
    return "".join(parts)


def render_gt_lidar_compare_table(compare: dict[str, Any]) -> str:
    def cell(value: Any, digits: int = 3) -> str:
        number = safe_float(value, None)
        if number is None:
            return "-"
        return f"{float(number):.{digits}f}"

    rows: list[str] = []
    for item in (compare.get("matches") or [])[:40]:
        lidar = item.get("lidar") or {}
        gt = item.get("mapGt") or {}
        err = item.get("error") or {}
        lp = lidar.get("worldPointUsed") or {}
        gp = gt.get("position") or {}
        cls = "good" if item.get("matched") else "warn"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{'OK' if item.get('matched') else 'NEAR'}</td>"
            f"<td>{html.escape(str(lidar.get('candidateKey')))}</td>"
            f"<td>{html.escape(str(gt.get('id')))}</td>"
            f"<td>{html.escape(str(gt.get('className')))}</td>"
            f"<td>{cell(lp.get('x'))}</td><td>{cell(lp.get('y'))}</td><td>{cell(lp.get('z'))}</td>"
            f"<td>{cell(gp.get('x'))}</td><td>{cell(gp.get('y'))}</td><td>{cell(gp.get('z'))}</td>"
            f"<td>{cell(err.get('worldErrorXZM'))}</td>"
            f"<td>{cell(err.get('distanceErrorToGtCenterM'))}</td>"
            f"<td>{cell(err.get('distanceErrorToApproxSurfaceM'))}</td>"
            f"<td>{cell(err.get('angleErrorDeg'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='14'>No LiDAR↔map comparison yet. Put NewMap.map next to this script or open /map_gt_load?filename=NewMap.map&clearExisting=true.</td></tr>")
    return "".join(rows)


@app.route("/gt_lidar_compare", methods=["GET"])
def gt_lidar_compare():
    with state_lock:
        cache = latest_cache
    max_items = int(safe_float(request.args.get("limit"), 40) or 40)
    max_error = float(safe_float(request.args.get("maxWorldErrorM"), 10.0) or 10.0)
    result = build_lidar_gt_comparisons(cache, max_items=max_items, max_match_world_error_m=max_error)
    return jsonify(result)


def svg_world_lidar_objects(cache: FrameCache, aim_snapshot: dict[str, Any] | None = None, width: int = 980, height: int = 500) -> str:
    """World X/Z map view for object summaries, not raw point cloud transfer."""
    aim_snapshot = aim_snapshot or {}
    selected = aim_snapshot.get("selectedTarget") or {}
    selected_key = str(selected.get("candidateKey", ""))
    player_raw = get_xyz(cache.pose.get("playerPos")) or get_xyz(cache.pose.get("lidarOrigin"))
    player_xz = (float(player_raw[0]), float(player_raw[2])) if player_raw is not None else (0.0, 0.0)

    object_points: list[tuple[float, float, dict[str, Any]]] = []
    for item in cache.clusters[:120]:
        point = cluster_world_point(item)
        if point is not None:
            object_points.append((point[0], point[1], item))

    xs = [player_xz[0]] + [p[0] for p in object_points]
    zs = [player_xz[1]] + [p[1] for p in object_points]
    pad = 15.0
    if len(xs) <= 1:
        min_x, max_x = player_xz[0] - 60.0, player_xz[0] + 60.0
        min_z, max_z = player_xz[1] - 60.0, player_xz[1] + 60.0
    else:
        min_x, max_x = min(xs) - pad, max(xs) + pad
        min_z, max_z = min(zs) - pad, max(zs) + pad
        span = max(max_x - min_x, max_z - min_z, 30.0)
        mid_x, mid_z = (min_x + max_x) / 2.0, (min_z + max_z) / 2.0
        min_x, max_x = mid_x - span / 2.0, mid_x + span / 2.0
        min_z, max_z = mid_z - span / 2.0, mid_z + span / 2.0

    margin_l, margin_r, margin_t, margin_b = 54, 18, 34, 44
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def sx(x: float) -> float:
        return margin_l + (x - min_x) / max(1e-6, (max_x - min_x)) * plot_w

    def sy(z: float) -> float:
        # world +Z goes upward on the map
        return margin_t + (max_z - z) / max(1e-6, (max_z - min_z)) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart worldchart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#121212' stroke='#444'/>")
    for t in np.linspace(0.0, 1.0, 6):
        x = margin_l + t * plot_w
        z_val = min_z + t * (max_z - min_z)
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#282828'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-14}' fill='#aaa' font-size='11' text-anchor='middle'>x {min_x + t*(max_x-min_x):.0f}</text>")
        y = margin_t + t * plot_h
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#282828'/>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#aaa' font-size='11'>z {max_z - t*(max_z-min_z):.0f}</text>")

    # player and body forward axis
    px, py = sx(player_xz[0]), sy(player_xz[1])
    body_yaw = safe_float(cache.pose.get("playerBodyX"), 0.0) or 0.0
    fx = px + 30.0 * sin(radians(float(body_yaw)))
    fy = py - 30.0 * cos(radians(float(body_yaw)))
    parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='8' fill='#45d9ff'/>")
    parts.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{fx:.1f}' y2='{fy:.1f}' stroke='#00ffff' stroke-width='3'/>")
    parts.append(f"<text x='{px+10:.1f}' y='{py-10:.1f}' fill='#45d9ff' font-size='12'>PLAYER x:{player_xz[0]:.1f}, z:{player_xz[1]:.1f}</text>")

    for x_val, z_val, item in object_points:
        x, y = sx(x_val), sy(z_val)
        key = str(item.get("candidateKey", ""))
        is_selected = key == selected_key
        color = '#00e5ff' if is_selected else '#ff4d4d'
        r = 11 if is_selected else 7
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='none' stroke='{color}' stroke-width='3'/>")
        label = f"{item.get('candidateLabel','OBJ')} {item.get('objectId','')} x:{x_val:.1f} z:{z_val:.1f} d:{float(item.get('distanceM',0) or 0):.1f}m"
        parts.append(f"<text x='{x+9:.1f}' y='{y-7:.1f}' fill='{color}' font-size='12'>{html.escape(label)}</text>")

    parts.append("<text x='12' y='20' fill='#eee' font-size='14'>World X/Z object map: circles are LiDAR object summaries; labels show world coordinates used for firing handoff.</text>")
    if not object_points:
        parts.append(f"<text x='{width/2:.1f}' y='{height/2:.1f}' fill='#aaa' font-size='16' text-anchor='middle'>No world-coordinate object cluster yet</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_front_lidar(cache: FrameCache, width: int = 820, height: int = 360) -> str:
    margin_l, margin_r, margin_t, margin_b = 48, 16, 22, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    min_a, max_a = -60.0, 60.0
    min_v, max_v = -22.5, 22.5

    def sy_raw(vangle: float) -> float:
        if bool(front_view_settings.get("flipVerticalDisplay", True)):
            return margin_t + (float(vangle) - min_v) / (max_v - min_v) * plot_h
        return margin_t + (max_v - float(vangle)) / (max_v - min_v) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#151515' stroke='#444'/>")
    for a in [-60, -30, 0, 30, 60]:
        x = margin_l + (a - min_a) / (max_a - min_a) * plot_w
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#333'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-10}' fill='#bbb' font-size='11' text-anchor='middle'>{a:+d}°</text>")
    for v in [-22.5, -10, 0, 10, 22.5]:
        y = sy_raw(v)
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#333'/>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#bbb' font-size='11'>{v:+.1f}°</text>")
    idx = np.flatnonzero((cache.angles >= min_a) & (cache.angles <= max_a))
    if idx.size > 1400:
        idx = idx[np.linspace(0, idx.size - 1, 1400).astype(np.int32)]
    for i in idx.tolist():
        a = float(cache.angles[i]); v = float(cache.vertical_angles[i])
        x = margin_l + (a - min_a) / (max_a - min_a) * plot_w
        y = sy_raw(v)
        c = lidar_point_color(cache, i)
        r = 2.2 if cache.valid_object_mask.size and cache.valid_object_mask[i] else 1.4
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{c}' opacity='0.88'/>")
    parts.append("<text x='410' y='16' fill='#eee' font-size='13' text-anchor='middle'>Raw front LiDAR: angle × vertical channel (vertical display flipped)</text>")
    parts.append("</svg>")
    return "".join(parts)



def front_object_group_indices(cache: FrameCache, cluster: dict[str, Any]) -> np.ndarray:
    """Return the LiDAR points that make one object cluster visible in the front view.

    The original cache.clusters are compact summaries, not raw point lists.
    For display only, reconstruct a small point group around each cluster's
    angle/range.  Recognition and firing logic remain unchanged.
    """
    if cache.angles.size == 0 or cache.distances.size == 0:
        return np.empty(0, dtype=np.int32)

    angle = safe_float(cluster.get("angleDeg"), None)
    distance = safe_float(cluster.get("surfaceDistanceM", cluster.get("distanceM")), None)
    if angle is None or distance is None:
        return np.empty(0, dtype=np.int32)

    visible_width = safe_float(cluster.get("visibleWidthM"), None)
    if visible_width is not None and float(distance) > 0.5:
        estimated_span = np.degrees(2.0 * np.arctan2(float(visible_width) * 0.5, max(0.5, float(distance))))
    else:
        estimated_span = 3.0
    angle_window = max(2.0, min(14.0, float(estimated_span) * 0.5 + 1.75))

    depth_span = safe_float(cluster.get("depthSpanM"), 0.0) or 0.0
    distance_window = max(2.2, min(12.0, float(depth_span) + 2.5))

    angle_delta = np.abs(((cache.angles.astype(np.float32) - float(angle) + 180.0) % 360.0) - 180.0)
    base_mask = (angle_delta <= angle_window) & (np.abs(cache.distances - float(distance)) <= distance_window)

    overlay_mask = lidar_object_overlay_mask(cache)
    if overlay_mask.size == cache.angles.size:
        idx = np.flatnonzero(base_mask & overlay_mask)
        if idx.size >= 1:
            return idx.astype(np.int32)
    if cache.obstacle_mask.size == cache.angles.size:
        idx = np.flatnonzero(base_mask & cache.obstacle_mask)
        if idx.size >= 1:
            return idx.astype(np.int32)
    return np.flatnonzero(base_mask).astype(np.int32)


def svg_front_object_silhouettes(
    cache: FrameCache,
    aim_snapshot: dict[str, Any] | None = None,
    width: int = 980,
    height: int = 520,
) -> str:
    """Front-view object-only LiDAR silhouette.

    X axis: body-relative LiDAR azimuth, +right / -left.
    Y axis: vertical LiDAR channel angle, +up / -down.
    This isolates valid object clusters so the operator can see how each object
    is actually sampled from the front, without the full ground cloud hiding it.
    """
    aim_snapshot = aim_snapshot or {}
    margin_l, margin_r, margin_t, margin_b = 58, 24, 34, 42
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    min_a, max_a = -70.0, 70.0
    min_v, max_v = -22.5, 22.5

    def sx(angle: float) -> float:
        return margin_l + (float(angle) - min_a) / (max_a - min_a) * plot_w

    def sy(vangle: float) -> float:
        if bool(front_view_settings.get("flipVerticalDisplay", True)):
            return margin_t + (float(vangle) - min_v) / (max_v - min_v) * plot_h
        return margin_t + (max_v - float(vangle)) / (max_v - min_v) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart frontobjectchart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#141414' stroke='#444'/>")

    for a in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
        x = sx(a)
        stroke = '#444' if a == 0 else '#2f2f2f'
        sw = 1.8 if a == 0 else 1.0
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='{stroke}' stroke-width='{sw}'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-13}' fill='#bbb' font-size='11' text-anchor='middle'>{a:+d}°</text>")
    for v in [-22.5, -15, -10, -5, 0, 5, 10, 15, 22.5]:
        y = sy(v)
        stroke = '#444' if v == 0 else '#2f2f2f'
        sw = 1.8 if v == 0 else 1.0
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='{stroke}' stroke-width='{sw}'/>")
        parts.append(f"<text x='10' y='{y+4:.1f}' fill='#bbb' font-size='11'>{v:+.1f}°</text>")

    # Faint raw valid-object points first, then cluster boxes.
    object_idx = np.flatnonzero(cache.valid_object_mask) if cache.valid_object_mask.size == cache.angles.size else np.empty(0, dtype=np.int32)
    front_obj_idx = object_idx[(cache.angles[object_idx] >= min_a) & (cache.angles[object_idx] <= max_a)] if object_idx.size else object_idx
    if front_obj_idx.size > 2600:
        front_obj_idx = front_obj_idx[np.linspace(0, front_obj_idx.size - 1, 2600).astype(np.int32)]
    for i in front_obj_idx.tolist():
        a = float(cache.angles[i]); v = float(cache.vertical_angles[i])
        if min_v <= v <= max_v:
            parts.append(f"<circle cx='{sx(a):.1f}' cy='{sy(v):.1f}' r='1.6' fill='#ff4d4d' opacity='0.38'/>")

    selected = aim_snapshot.get("selectedTarget") or {}
    selected_key = str(selected.get("candidateKey", ""))
    drawn_count = 0
    for cluster in cache.clusters[:40]:
        angle = safe_float(cluster.get("angleDeg"), None)
        if angle is None or not (min_a <= float(angle) <= max_a):
            continue
        idx = front_object_group_indices(cache, cluster)
        idx = idx[(cache.angles[idx] >= min_a) & (cache.angles[idx] <= max_a)] if idx.size else idx
        if idx.size == 0:
            continue
        a_vals = cache.angles[idx].astype(np.float64)
        v_vals = cache.vertical_angles[idx].astype(np.float64)
        d_vals = cache.distances[idx].astype(np.float64)
        a1, a2 = max(min_a, float(np.min(a_vals)) - 0.8), min(max_a, float(np.max(a_vals)) + 0.8)
        v1, v2 = max(min_v, float(np.min(v_vals)) - 1.0), min(max_v, float(np.max(v_vals)) + 1.0)
        x1, x2 = sx(a1), sx(a2)
        y_top, y_bot = sy(v2), sy(v1)
        key = str(cluster.get("candidateKey", ""))
        is_selected = key == selected_key
        color = '#00e5ff' if is_selected else '#ff4d4d'
        stroke_w = 3.0 if is_selected else 2.0
        fill_opacity = 0.10 if is_selected else 0.045
        parts.append(f"<rect x='{x1:.1f}' y='{y_top:.1f}' width='{max(4.0, x2-x1):.1f}' height='{max(4.0, y_bot-y_top):.1f}' fill='{color}' fill-opacity='{fill_opacity:.3f}' stroke='{color}' stroke-width='{stroke_w}'/>")
        # Draw the cluster's own points stronger on top of the rectangle.
        if idx.size > 280:
            idx_draw = idx[np.linspace(0, idx.size - 1, 280).astype(np.int32)]
        else:
            idx_draw = idx
        for i in idx_draw.tolist():
            a = float(cache.angles[i]); v = float(cache.vertical_angles[i])
            if min_v <= v <= max_v:
                c = '#00e5ff' if is_selected else lidar_point_color(cache, i)
                parts.append(f"<circle cx='{sx(a):.1f}' cy='{sy(v):.1f}' r='2.2' fill='{c}' opacity='0.90'/>")
        aim_pitch = safe_float(cluster.get("aimPitchDeg"), None)
        if aim_pitch is not None and min_v <= float(aim_pitch) <= max_v:
            ax, ay = sx(float(angle)), sy(float(aim_pitch))
            parts.append(f"<line x1='{ax-7:.1f}' y1='{ay:.1f}' x2='{ax+7:.1f}' y2='{ay:.1f}' stroke='#ffff66' stroke-width='2'/>")
            parts.append(f"<line x1='{ax:.1f}' y1='{ay-7:.1f}' x2='{ax:.1f}' y2='{ay+7:.1f}' stroke='#ffff66' stroke-width='2'/>")
        center = cluster.get("surfaceCenterWorld") or cluster.get("worldCenter") or {}
        dist = safe_float(cluster.get("distanceM"), 0.0) or 0.0
        h = safe_float(cluster.get("objectHeightAboveTerrainM"), None)
        label = f"{cluster.get('candidateLabel','OBJ')} {cluster.get('objectId','')} {key} d:{float(dist):.1f}m a:{float(angle):+.1f}°"
        if h is not None:
            label += f" h:{float(h):.1f}m"
        if isinstance(center, dict) and center.get('x') is not None and center.get('z') is not None:
            label += f" | x:{center.get('x')} z:{center.get('z')}"
        label_x = min(width - 12, max(margin_l + 4, x2 + 8))
        label_y = max(margin_t + 14, y_top - 5)
        parts.append(f"<text x='{label_x:.1f}' y='{label_y:.1f}' fill='{color}' font-size='12'>{html.escape(label)}</text>")
        drawn_count += 1

    parts.append("<text x='12' y='20' fill='#eee' font-size='14'>Front object silhouettes: LiDAR valid-object clusters only | X=body angle, Y=vertical LiDAR channel, display flipped to match camera, yellow cross=aim pitch</text>")
    parts.append("<text x='12' y='38' fill='#aaa' font-size='12'>This view keeps the existing raw front view, but flips only the display Y axis so the object silhouette matches the simulator camera.</text>")
    if drawn_count == 0:
        parts.append(f"<text x='{width/2:.1f}' y='{height/2:.1f}' fill='#aaa' font-size='16' text-anchor='middle'>No front-facing LiDAR object cluster in ±70° yet</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_object_cluster_closeups(
    cache: FrameCache,
    max_clusters: int = 8,
    width: int = 1200,
    panel_width: int = 292,
    panel_height: int = 250,
) -> str:
    """Auto-zoom each LiDAR object so sparse far targets remain readable."""
    visible: list[tuple[dict[str, Any], np.ndarray]] = []
    for cluster in cache.clusters[:max_clusters]:
        idx = front_object_group_indices(cache, cluster)
        if idx.size:
            visible.append((cluster, idx))

    columns = 4
    rows = max(1, int(np.ceil(len(visible) / columns)))
    height = 34 + rows * panel_height
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart closeupchart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#111' stroke='#444'/>")
    parts.append("<text x='12' y='22' fill='#eee' font-size='14'>Object close-ups: each panel is independently auto-zoomed</text>")

    if not visible:
        parts.append(f"<text x='{width/2:.1f}' y='{height/2:.1f}' fill='#aaa' font-size='16' text-anchor='middle'>No LiDAR object cluster to magnify</text>")
        parts.append("</svg>")
        return "".join(parts)

    for panel_index, (cluster, idx) in enumerate(visible):
        col = panel_index % columns
        row = panel_index // columns
        ox = 8 + col * panel_width
        oy = 34 + row * panel_height
        plot_x, plot_y = ox + 34, oy + 42
        plot_w, plot_h = panel_width - 46, panel_height - 62

        angle_values = cache.angles[idx].astype(np.float64)
        vertical_values = cache.vertical_angles[idx].astype(np.float64)
        distance_values = cache.distances[idx].astype(np.float64)
        center_angle = safe_float(cluster.get("angleDeg"), float(np.median(angle_values))) or 0.0
        angle_span = max(6.0, float(np.ptp(angle_values)) + 4.0)
        min_a, max_a = center_angle - angle_span / 2.0, center_angle + angle_span / 2.0
        min_v = min(-2.0, float(np.min(vertical_values)) - 2.0)
        max_v = max(2.0, float(np.max(vertical_values)) + 2.0)
        if max_v - min_v < 10.0:
            middle_v = (min_v + max_v) / 2.0
            min_v, max_v = middle_v - 5.0, middle_v + 5.0

        def sx(value: float) -> float:
            return plot_x + (float(value) - min_a) / max(1e-6, max_a - min_a) * plot_w

        def sy(value: float) -> float:
            if bool(front_view_settings.get("flipVerticalDisplay", True)):
                return plot_y + (float(value) - min_v) / max(1e-6, max_v - min_v) * plot_h
            return plot_y + (max_v - float(value)) / max(1e-6, max_v - min_v) * plot_h

        parts.append(f"<rect x='{ox:.1f}' y='{oy:.1f}' width='{panel_width-8:.1f}' height='{panel_height-8:.1f}' fill='#171717' stroke='#555'/>")
        for fraction in (0.0, 0.5, 1.0):
            gx, gy = plot_x + fraction * plot_w, plot_y + fraction * plot_h
            parts.append(f"<line x1='{gx:.1f}' y1='{plot_y:.1f}' x2='{gx:.1f}' y2='{plot_y+plot_h:.1f}' stroke='#303030'/>")
            parts.append(f"<line x1='{plot_x:.1f}' y1='{gy:.1f}' x2='{plot_x+plot_w:.1f}' y2='{gy:.1f}' stroke='#303030'/>")

        channel_count = int(np.unique(np.round(vertical_values, 4)).size)
        shape_label = str(cluster.get("candidateLabel", "OBSTACLE"))
        key_label = str(cluster.get("candidateKey", ""))
        label = html.escape(f"{shape_label} {cluster.get('objectId','')} {key_label}")
        distance = safe_float(cluster.get("distanceM"), float(np.median(distance_values))) or 0.0
        panel_color = "#00e5ff" if bool(cluster.get("tankLike", False)) else "#ffae35"
        parts.append(f"<text x='{ox+8:.1f}' y='{oy+17:.1f}' fill='{panel_color}' font-size='13'>{label} | {distance:.1f}m | pts {idx.size} | channels {channel_count}</text>")
        parts.append(f"<text x='{ox+8:.1f}' y='{oy+33:.1f}' fill='#aaa' font-size='11'>angle {min_a:+.1f}..{max_a:+.1f}° | vertical {min_v:+.1f}..{max_v:+.1f}°</text>")

        for i in idx.tolist():
            color = panel_color if bool(cache.valid_object_mask[i]) else "#888888"
            parts.append(f"<circle cx='{sx(float(cache.angles[i])):.1f}' cy='{sy(float(cache.vertical_angles[i])):.1f}' r='5.0' fill='{color}' opacity='0.92'/>")

    parts.append("</svg>")
    return "".join(parts)


def svg_side_profile(cache: FrameCache, width: int = 820, height: int = 360) -> str:
    margin_l, margin_r, margin_t, margin_b = 48, 16, 22, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_r = MAX_LIDAR_DISTANCE_M
    # The simulator's playerPos.y is effectively the vehicle/ground pivot,
    # not the visual center of the tank body. Keep this view tight around that
    # pivot so the configured ~3 m LiDAR mount height is easy to verify.
    min_y, max_y = -2.0, 8.0
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#151515' stroke='#444'/>")
    for rr in [0, 20, 40, 60, 80, 100, 120]:
        x = margin_l + rr / max_r * plot_w
        parts.append(f"<line x1='{x:.1f}' y1='{margin_t}' x2='{x:.1f}' y2='{margin_t+plot_h}' stroke='#333'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-10}' fill='#bbb' font-size='11' text-anchor='middle'>{rr}m</text>")
    for yy in [-2, 0, 2, 4, 6, 8]:
        y = margin_t + (max_y - yy) / (max_y - min_y) * plot_h
        parts.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{margin_l+plot_w}' y2='{y:.1f}' stroke='#333'/>")
        parts.append(f"<text x='8' y='{y+4:.1f}' fill='#bbb' font-size='11'>{yy:+d}m</text>")
    origin = cache.pose.get("lidarOrigin", {}) if isinstance(cache.pose, dict) else {}
    player = cache.pose.get("playerPos", {}) if isinstance(cache.pose, dict) else {}
    origin_y = safe_float(origin.get("y"), EXPECTED_LIDAR_Y_POSITION_M) or EXPECTED_LIDAR_Y_POSITION_M
    player_y = safe_float(player.get("y"), None)
    reference_y = float(player_y) if player_y is not None else float(origin_y) - EXPECTED_LIDAR_Y_POSITION_M
    lidar_relative_y = float(origin_y) - reference_y
    if cache.angles.size:
        selected_az = float(cache.angles[np.argmin(np.abs(cache.angles))])
        idx = np.flatnonzero(np.abs(cache.angles - selected_az) <= 0.75)
    else:
        selected_az = 0.0; idx = np.empty(0, dtype=np.int32)
    for i in idx.tolist():
        rr = float(cache.horizontal_ranges[i])
        yy = float(cache.xyz[i, 1] - reference_y)
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
            yy = float(cache.terrain_y[i] - reference_y)
            if 0 <= rr <= max_r and min_y <= yy <= max_y:
                x = margin_l + rr / max_r * plot_w
                y = margin_t + (max_y - yy) / (max_y - min_y) * plot_h
                terrain_points.append((rr, x, y))
        terrain_points.sort(key=lambda item: item[0])
        if len(terrain_points) >= 2:
            d = " ".join(("M" if k == 0 else "L") + f" {x:.1f} {y:.1f}" for k, (_, x, y) in enumerate(terrain_points))
            parts.append(f"<path d='{d}' fill='none' stroke='#45c96b' stroke-width='1.5' opacity='0.75'/>")
    body_y = margin_t + (max_y - 0.0) / (max_y - min_y) * plot_h
    lidar_y = margin_t + (max_y - lidar_relative_y) / (max_y - min_y) * plot_h
    parts.append(f"<line x1='{margin_l}' y1='{body_y:.1f}' x2='{margin_l+plot_w}' y2='{body_y:.1f}' stroke='#00d4d4' stroke-width='2'/>")
    parts.append(f"<text x='{margin_l+6}' y='{body_y-5:.1f}' fill='#00d4d4' font-size='11'>PLAYER/GROUND PIVOT Y=0</text>")
    if min_y <= lidar_relative_y <= max_y:
        parts.append(f"<line x1='{margin_l}' y1='{lidar_y:.1f}' x2='{margin_l+plot_w}' y2='{lidar_y:.1f}' stroke='#ffd23f' stroke-dasharray='6 4'/>")
        mount_delta = lidar_relative_y - EXPECTED_LIDAR_Y_POSITION_M
        parts.append(f"<text x='{margin_l+170}' y='{lidar_y-5:.1f}' fill='#ffd23f' font-size='11'>LiDAR +{lidar_relative_y:.2f}m (setting {EXPECTED_LIDAR_Y_POSITION_M:.1f}m, delta {mount_delta:+.2f}m)</text>")
    parts.append(f"<text x='410' y='16' fill='#eee' font-size='13' text-anchor='middle'>Side profile: azimuth {selected_az:+.1f}° | green=hill profile, red=object above hill</text>")
    parts.append("</svg>")
    return "".join(parts)


def lidar_view_snapshot() -> tuple[FrameCache, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with state_lock:
        cache = latest_cache
        aim_snapshot = json_copy(aim_state)
        fire_snapshot = json_copy(fire_state)
        yolo_snapshot = json_copy(yolo_state)
        fusion_snapshot = dict(fusion_settings)
        turret_state = dict(latest_turret)
    fire_targets_snapshot = build_fire_team_targets(cache, turret_state, tank_only=True, max_targets=8)
    return cache, aim_snapshot, fire_snapshot, yolo_snapshot, fusion_snapshot, fire_targets_snapshot




def render_map_switcher_controls() -> str:
    """Static .map switcher controls.

    v16.16 keeps these controls outside #live and makes .map application one-click:
    - selecting from the dropdown can immediately load the map,
    - a local path can be pasted when the map file is not next to the script,
    - a .map file can be uploaded from the browser into the Python server folder.
    """
    with state_lock:
        active_map_file = ground_truth_state.get("activeMapFile") or ground_truth_settings.get("activeMapFile")
        cycle_snapshot = json_copy(map_cycle_settings)
        yolo_loaded = yolo_state.get("modelLoaded")
    local_map_names = [path.name for path in local_map_files()]
    active_map_name = Path(str(active_map_file)).name if active_map_file else ""
    if not active_map_name and cycle_snapshot.get("currentMapFile"):
        active_map_name = str(cycle_snapshot.get("currentMapFile"))
    map_options = "".join(
        f"<option value='{html.escape(name, quote=True)}' {'selected' if name == active_map_name else ''}>{html.escape(name)}</option>"
        for name in local_map_names
    )
    if not map_options:
        map_options = "<option value=''>No .map files next to this Python file</option>"
    active_text = html.escape(str(active_map_file or 'not loaded'))
    return f"""
<div class='card sticky-switcher' id='mapSwitcherCard'>
  <b>Web .map switcher:</b>
  <form id='mapSwitchForm' action='/map_gt_select' method='get' onsubmit='return loadSelectedMapFromForm(event)'>
    <select id='mapSelect' name='filename' onchange='return loadSelectedMapFromForm(event)'>{map_options}</select>
    <input type='hidden' name='clearExisting' value='true'>
    <button type='submit'>Load selected map now</button>
    <button type='button' onclick='reloadSelectedMapNow()'>Reload</button>
    <button type='button' onclick='cycleNextMapNow()'>Cycle next now</button>
    <button type='button' onclick='startMapCycle()'>Start cycle</button>
    <button type='button' onclick='startAutoBestMap()'>Start auto-best</button>
    <button type='button' onclick='stopMapCycle()'>Stop</button>
  </form>
  <div style='margin-top:6px;'>
    <input id='mapPathInput' placeholder='optional full path, e.g. C:\\Users\\...\\your.map' style='min-width:430px;'>
    <button type='button' onclick='loadMapPathNow()'>Load pasted path</button>
    <button type='button' onclick='applySimDebugSetup()'>Apply YOLO+LiDAR sim debug preset</button>
    <button type='button' onclick='preloadYoloNow()'>Preload YOLO</button>
  </div>
  <form id='mapUploadForm' enctype='multipart/form-data' onsubmit='return uploadMapFile(event)' style='margin-top:6px;'>
    <input type='file' id='mapUploadFile' name='mapfile' accept='.map'>
    <button type='submit'>Upload .map to server + load</button>
  </form>
  <div id='mapApplyStatus' class='muted'>Ready. Select a .map to apply it to Python-side GT comparison.</div>
  <div class='muted'>Active GT map: {active_text}</div>
  <div class='muted'>Cycle: enabled={cycle_snapshot.get('enabled')} / mode={html.escape(str(cycle_snapshot.get('mode')))} / interval={cycle_snapshot.get('intervalSec')}s / current={html.escape(str(cycle_snapshot.get('currentMapFile')))} / lastSwitch={html.escape(str(cycle_snapshot.get('lastSwitchAt')))}</div>
  <div class='muted'>YOLO loaded={yolo_loaded}. Map files visible to the Python server: {html.escape(str(local_map_names))}. This changes only Python-side GT comparison, not the simulator terrain/map.</div>
</div>
"""



@app.route("/map_switcher_fragment", methods=["GET"])
def map_switcher_fragment():
    response = app.response_class(render_map_switcher_controls(), mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    return response

def render_lidar_view_body(
    cache: FrameCache,
    aim_snapshot: dict[str, Any],
    fire_snapshot: dict[str, Any],
    yolo_snapshot: dict[str, Any],
    fusion_snapshot: dict[str, Any],
    fire_targets_snapshot: list[dict[str, Any]],
    include_map_switcher: bool = False,
) -> str:
    rows = []
    selected = aim_snapshot.get("selectedTarget") or {}
    for item in cache.clusters[:30]:
        is_sel = str(selected.get("candidateKey")) == str(item.get("candidateKey"))
        center = item.get("worldCenter") or {}
        aim_world = item.get("aimPointWorld") or {}
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
            f"<td>{center.get('x', '-')}</td><td>{center.get('y', '-')}</td><td>{center.get('z', '-')}</td>"
            f"<td>{aim_world.get('x', '-')}</td><td>{aim_world.get('y', item.get('aimPointYWorldM', '-'))}</td><td>{aim_world.get('z', '-')}</td>"
            f"<td>{item.get('terrainBaseYWorldM', '-')}</td>"
            f"<td>{item.get('objectTopYWorldM', '-')}</td>"
            f"<td>{item.get('depthSpanM', '-')}</td>"
            f"<td>{item.get('verticalityRatio', '-')}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='19'>No object-above-hill LiDAR clusters yet.</td></tr>")

    target_rows = []
    for target in fire_targets_snapshot:
        world = target.get("world") or {}
        aim_point = world.get("aimPoint") or {}
        q = target.get("quality") or {}
        target_rows.append(
            "<tr>"
            f"<td>{html.escape(str(target.get('targetId')))}</td>"
            f"<td>{html.escape(str(target.get('className')))}</td>"
            f"<td>{float(target.get('confidence') or 0):.3f}</td>"
            f"<td>{float(target.get('distanceM') or 0):.1f}</td>"
            f"<td>{float(target.get('bodyYawDeg') or 0):+.1f}</td>"
            f"<td>{float(target.get('aimPitchDeg') or 0):+.1f}</td>"
            f"<td>{aim_point.get('x', '-')}</td><td>{aim_point.get('y', '-')}</td><td>{aim_point.get('z', '-')}</td>"
            f"<td>{html.escape(str(q.get('fusionMethod')))}</td>"
            "</tr>"
        )
    if not target_rows:
        target_rows.append("<tr><td colspan='10'>No fresh confirmed YOLO+LiDAR tank target yet. Check /fire_targets?tankOnly=false to see all fused objects.</td></tr>")

    names = yolo_snapshot.get("modelNames") or MODEL_CLASS_NAMES
    gt_compare = build_lidar_gt_comparisons(cache, max_items=40, max_match_world_error_m=10.0)
    gt_rows = render_gt_lidar_compare_table(gt_compare)
    gt_stats_html = format_compare_stats_html(gt_compare.get("stats") or {})
    gt_map_file = gt_compare.get("activeMapFile") or (gt_compare.get("mapLoad") or {}).get("path") or "not loaded"
    map_cycle_snapshot = json_copy(map_cycle_settings)
    local_map_names = [path.name for path in local_map_files()]
    active_map_name = Path(str(gt_map_file)).name if gt_map_file and str(gt_map_file) != 'not loaded' else ''
    if not active_map_name and map_cycle_snapshot.get('currentMapFile'):
        active_map_name = str(map_cycle_snapshot.get('currentMapFile'))
    map_options = ''.join(
        f"<option value='{html.escape(name, quote=True)}' {'selected' if name == active_map_name else ''}>{html.escape(name)}</option>"
        for name in local_map_names
    )
    if not map_options:
        map_options = "<option value=''>No .map files next to this Python file</option>"
    player_position = current_player_position(cache)
    player_position_text = (
        f"x={float(player_position[0]):.3f}, "
        f"y={float(player_position[1]):.3f}, "
        f"z={float(player_position[2]):.3f}"
        if player_position is not None
        else "not received"
    )
    return f"""
<h1>LiDAR object scan + world-coordinate fire handoff + GT compare v16.16</h1>
<div class='muted'>Live panel updates with fetch(); the whole browser page is not meta-refreshed anymore.</div>
<div>{svg_top_lidar(cache, aim_snapshot)}</div>
<div>{svg_world_lidar_objects(cache, aim_snapshot)}</div>
<div>{svg_world_gt_lidar_compare(gt_compare, cache)}</div>
<div class='grid'>
<div>{svg_front_lidar(cache)}</div>
<div>{svg_side_profile(cache)}</div>
</div>
<div>{svg_front_object_silhouettes(cache, aim_snapshot)}</div>
<div>{svg_object_cluster_closeups(cache)}</div>
<div class='card'>
<b>Frame:</b> {cache.seq} / time={cache.simulation_time}<br>
<b>My tank position:</b> {player_position_text}<br>
<b>LiDAR origin:</b> {html.escape(str(cache.pose.get('lidarOrigin', {})))}<br>
<b>Points:</b> raw={cache.raw_point_count}, detected={cache.detected_hit_count}, ground={int(cache.ground_mask.sum())}, obstacle={int(cache.obstacle_mask.sum())}, <span class='good'>validObject={int(cache.valid_object_mask.sum())}</span><br>
<b>Terrain profile:</b> {html.escape(str(cache.ground_plane_debug.get('terrainProfile', {})))}<br>
<b>Object filter:</b> {html.escape(str(cache.ground_plane_debug.get('objectFilter', {})))}<br>
<b>YOLO:</b> model={html.escape(str(fusion_snapshot.get('modelPath')))} / loaded={yolo_snapshot.get('modelLoaded')} / names={html.escape(str(names))} / conf={fusion_snapshot.get('confidence')} / iou={fusion_snapshot.get('iou')} / imgsz={fusion_snapshot.get('imageSize')} / max_det={fusion_snapshot.get('maxDetections')} / augment={fusion_snapshot.get('augment')}<br>
<b>Aim:</b> mode={aim_snapshot.get('mode')} / yawErr={aim_snapshot.get('yawErrorDeg')} / pitchErr={aim_snapshot.get('pitchErrorDeg')}<br>
<b>Fire:</b> count={fire_snapshot.get('fireCount')} / blocked={fire_snapshot.get('lastBlockedReason')}<br>
<b>YOLO debug:</b> submitted={yolo_snapshot.get('submittedCount')} / completed={yolo_snapshot.get('completedCount')} / failed={yolo_snapshot.get('failedCount')} / det={len(yolo_snapshot.get('latestYoloDetections', []))} / fused={len(yolo_snapshot.get('latestFusedObjects', []))} / error={html.escape(str(yolo_snapshot.get('modelLoadError')))}
</div>
<div class='card'>
<h2>Fire-team compact target handoff</h2>
<table><thead><tr><th>targetId</th><th>class</th><th>conf</th><th>dist m</th><th>yaw deg</th><th>pitch deg</th><th>aim x</th><th>aim y</th><th>aim z</th><th>fusion</th></tr></thead><tbody>{''.join(target_rows)}</tbody></table>
<code>GET /fire_targets</code> returns this as JSON without raw LiDAR points.
</div>
<div class='card'>
<h2>.map GT ↔ LiDAR measured coordinate comparison</h2>
<div class='muted'>Active map: {html.escape(str(gt_map_file))} / GT objects={gt_compare.get('gtCount')} / LiDAR clusters={gt_compare.get('lidarClusterCount')} / matched={gt_compare.get('matchedCount')} / gate={gt_compare.get('maxMatchWorldErrorM')}m</div>
<div class='card'><b>Auto error summary:</b> {html.escape(str(gt_stats_html))}</div>
<div class='muted'>Map cycle: enabled={map_cycle_snapshot.get('enabled')} / mode={html.escape(str(map_cycle_snapshot.get('mode')))} / interval={map_cycle_snapshot.get('intervalSec')}s / current={html.escape(str(map_cycle_snapshot.get('currentMapFile')))} / lastSwitch={html.escape(str(map_cycle_snapshot.get('lastSwitchAt')))}</div>
{render_map_switcher_controls() if include_map_switcher else ""}
<table><thead><tr><th>match</th><th>LiDAR key</th><th>.map id</th><th>.map class</th><th>LiDAR x</th><th>LiDAR y</th><th>LiDAR z</th><th>GT x</th><th>GT y</th><th>GT z</th><th>XZ err m</th><th>dist err center m</th><th>dist err surface m</th><th>angle err deg</th></tr></thead><tbody>{gt_rows}</tbody></table>
<code>GET /gt_lidar_compare</code> returns this comparison as JSON. <code>GET /map_auto_best</code> scores every local .map and loads the best one.
</div>
<h2>LiDAR object clusters with world coordinates</h2>
<table><thead><tr><th>key</th><th>type</th><th>distance m</th><th>angle deg</th><th>pitch deg</th><th>points</th><th>height span m</th><th>obj above hill m</th><th>surface dist m</th><th>center x</th><th>center y</th><th>center z</th><th>aim x</th><th>aim y</th><th>aim z</th><th>terrain Y</th><th>top Y</th><th>depth span m</th><th>verticality</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class='card'>
<b>Links:</b>
<a href='/fire_targets'>fire_targets</a> |
<a href='/fire_targets?tankOnly=false'>all fused targets</a> |
<a href='/gt_lidar_compare'>gt_lidar_compare</a> |
<a href='/map_gt_load?filename=NewMap.map&clearExisting=true'>load NewMap.map</a> |
<a href='/map_gt_list'>map_gt_list</a> |
<a href='/map_cycle_update?enabled=true&mode=cycle&intervalSec=6&reset=true'>start map cycle</a> |
<a href='/map_cycle_update?enabled=true&mode=best&intervalSec=8&reset=true'>start auto-best</a> |
<a href='/map_cycle_update?enabled=false'>stop map cycle</a> |
<a href='/map_auto_best'>score all maps</a> |
<a href='/lidar_status'>lidar_status</a> |
<a href='/aim_status'>aim_status</a> |
<a href='/fire_status'>fire_status</a> |
<a href='/action_debug'>action_debug</a> |
<a href='/fusion_status'>fusion_status</a> |
<a href='/yolo_preload'>yolo_preload</a><br>
Top-view is local polar LiDAR. World map uses actual X/Z object summary coordinates. Green=ground/hill, yellow=obstacle, red=LiDAR object, cyan=selected/confirmed target.
</div>
"""


@app.route("/lidar_view_fragment", methods=["GET"])
def lidar_view_fragment():
    map_name = request.args.get("map") or request.args.get("filename")
    if map_name:
        load_map_ground_truth(filename=map_name, clear_existing=True, persist_selection=True)
        with state_lock:
            map_cycle_settings["enabled"] = False
    cache, aim_snapshot, fire_snapshot, yolo_snapshot, fusion_snapshot, fire_targets_snapshot = lidar_view_snapshot()
    response = app.response_class(
        render_lidar_view_body(cache, aim_snapshot, fire_snapshot, yolo_snapshot, fusion_snapshot, fire_targets_snapshot, include_map_switcher=False),
        mimetype="text/html",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/dashboard", methods=["GET"])
@app.route("/lidar_view", methods=["GET"])
def lidar_view():
    map_name = request.args.get("map") or request.args.get("filename")
    if map_name:
        load_map_ground_truth(filename=map_name, clear_existing=True, persist_selection=True)
        with state_lock:
            map_cycle_settings["enabled"] = False
    cache, aim_snapshot, fire_snapshot, yolo_snapshot, fusion_snapshot, fire_targets_snapshot = lidar_view_snapshot()
    body = render_lidar_view_body(cache, aim_snapshot, fire_snapshot, yolo_snapshot, fusion_snapshot, fire_targets_snapshot, include_map_switcher=False)
    controls = render_map_switcher_controls()
    page = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>v16.16 YOLO.pt + easy map switch + sim LiDAR overlay</title>
<style>
body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:14px; }}
h1 {{ margin: 0 0 8px 0; }} h2 {{ margin: 12px 0 6px 0; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
.card {{ background:#1d1d1d; border:1px solid #444; padding:10px; margin:10px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} th,td {{ border:1px solid #555; padding:5px; text-align:right; }}
th:first-child,td:first-child, th:nth-child(2),td:nth-child(2) {{ text-align:left; }} .sel {{ background:#3a3000; color:#ffd34d; }}
.good {{ color:#6fe36f; }} .warn {{ color:#ffce54; }} .bad {{ color:#ff7777; }} .muted {{ color:#aaa; font-size:12px; margin-bottom:6px; }}
.chart {{ width:100%; max-height:360px; }} .frontobjectchart {{ max-height:520px; }} .closeupchart {{ max-height:none; }} .topchart {{ max-height:820px; }} .worldchart {{ max-height:500px; }} .gtchart {{ max-height:560px; }} code {{ color:#9cdcfe; }} a {{ color:#8cc8ff; }}
.sticky-switcher {{ position: sticky; top: 0; z-index: 30; box-shadow: 0 4px 12px rgba(0,0,0,0.35); }}
#mapSwitcherCard select {{ min-width: 300px; }}
</style></head><body>
<div id='map-controls'>{controls}</div>
<div id='live'>{body}</div>
<script>
let liveRefreshEnabled = true;
let mapSwitcherPauseUntil = 0;
function pauseMapSwitcher(ms = 5000) {{ mapSwitcherPauseUntil = Date.now() + ms; }}
function mapSwitcherIsActive() {{
  const card = document.getElementById('mapSwitcherCard');
  if (!card) return false;
  return Date.now() < mapSwitcherPauseUntil || card.contains(document.activeElement);
}}
document.addEventListener('focusin', (e) => {{ if (e.target.closest && e.target.closest('#mapSwitcherCard')) pauseMapSwitcher(10000); }});
document.addEventListener('pointerdown', (e) => {{ if (e.target.closest && e.target.closest('#mapSwitcherCard')) pauseMapSwitcher(10000); }});
document.addEventListener('mouseover', (e) => {{ if (e.target.closest && e.target.closest('#mapSwitcherCard')) pauseMapSwitcher(2500); }});
async function refreshMapControls() {{
  try {{
    const res = await fetch('/map_switcher_fragment?ts=' + Date.now(), {{cache: 'no-store'}});
    if (res.ok && !mapSwitcherIsActive()) {{
      document.getElementById('map-controls').innerHTML = await res.text();
    }}
  }} catch (err) {{ console.warn('map switcher refresh failed', err); }}
}}
async function refreshLivePanel() {{
  if (!liveRefreshEnabled || mapSwitcherIsActive()) return;
  try {{
    const res = await fetch('/lidar_view_fragment?ts=' + Date.now(), {{cache: 'no-store'}});
    if (res.ok) {{
      document.getElementById('live').innerHTML = await res.text();
    }}
  }} catch (err) {{
    console.warn('lidar_view refresh failed', err);
  }}
}}
async function apiGet(url) {{
  const res = await fetch(url + (url.includes('?') ? '&' : '?') + 'ts=' + Date.now(), {{cache: 'no-store'}});
  const text = await res.text();
  if (!res.ok) throw new Error(text || res.statusText);
  try {{ return JSON.parse(text); }} catch (e) {{ return text; }}
}}
function setMapStatus(msg, cls='muted') {{
  const el = document.getElementById('mapApplyStatus');
  if (el) {{ el.className = cls; el.textContent = msg; }}
}}
async function loadSelectedMapFromForm(event) {{
  if (event) event.preventDefault();
  const sel = document.getElementById('mapSelect');
  if (!sel || !sel.value) return false;
  pauseMapSwitcher(2500);
  setMapStatus('Loading ' + sel.value + ' ...');
  try {{
    const out = await apiGet('/map_gt_select?filename=' + encodeURIComponent(sel.value) + '&clearExisting=true&stopCycle=true');
    setMapStatus('Loaded: ' + (out.loadedMap || sel.value), 'good');
  }} catch (err) {{
    setMapStatus('Map load failed: ' + err.message, 'bad');
  }}
  await refreshMapControls();
  await refreshLivePanel();
  return false;
}}
async function reloadSelectedMapNow() {{ return loadSelectedMapFromForm(null); }}
async function loadMapPathNow() {{
  const input = document.getElementById('mapPathInput');
  if (!input || !input.value.trim()) return false;
  pauseMapSwitcher(3000);
  setMapStatus('Loading path ' + input.value.trim() + ' ...');
  try {{
    const out = await apiGet('/map_gt_select?path=' + encodeURIComponent(input.value.trim()) + '&clearExisting=true&stopCycle=true');
    setMapStatus('Loaded path: ' + (out.loadedMap || input.value.trim()), 'good');
  }} catch (err) {{
    setMapStatus('Path load failed: ' + err.message, 'bad');
  }}
  await refreshMapControls();
  await refreshLivePanel();
  return false;
}}
async function uploadMapFile(event) {{
  if (event) event.preventDefault();
  const fileInput = document.getElementById('mapUploadFile');
  if (!fileInput || !fileInput.files || !fileInput.files[0]) return false;
  pauseMapSwitcher(5000);
  const form = new FormData();
  form.append('mapfile', fileInput.files[0]);
  setMapStatus('Uploading and loading ' + fileInput.files[0].name + ' ...');
  try {{
    const res = await fetch('/map_upload_load?ts=' + Date.now(), {{method:'POST', body:form, cache:'no-store'}});
    const text = await res.text();
    if (!res.ok) throw new Error(text || res.statusText);
    let out; try {{ out = JSON.parse(text); }} catch(e) {{ out = {{}}; }}
    setMapStatus('Uploaded + loaded: ' + (out.loadedMap || fileInput.files[0].name), 'good');
  }} catch (err) {{
    setMapStatus('Upload failed: ' + err.message, 'bad');
  }}
  await refreshMapControls();
  await refreshLivePanel();
  return false;
}}
async function applySimDebugSetup() {{
  pauseMapSwitcher(2500);
  setMapStatus('Applying YOLO + object LiDAR overlay preset ...');
  try {{
    await apiGet('/sim_debug_setup?mode=tank_object_overlay');
    setMapStatus('YOLO + object LiDAR overlay preset applied. Check simulator /detect overlay.', 'good');
  }} catch (err) {{
    setMapStatus('Preset failed: ' + err.message, 'bad');
  }}
  await refreshMapControls();
  await refreshLivePanel();
}}
async function preloadYoloNow() {{
  pauseMapSwitcher(5000);
  setMapStatus('Preloading YOLO model ...');
  try {{
    await apiGet('/yolo_preload');
    setMapStatus('YOLO model preloaded.', 'good');
  }} catch (err) {{
    setMapStatus('YOLO preload failed: ' + err.message, 'bad');
  }}
  await refreshMapControls();
  await refreshLivePanel();
}}
async function cycleNextMapNow() {{
  pauseMapSwitcher(800);
  await apiGet('/map_cycle_next?mode=cycle');
  await refreshMapControls();
  await refreshLivePanel();
}}
async function startMapCycle() {{
  pauseMapSwitcher(800);
  await apiGet('/map_cycle_update?enabled=true&mode=cycle&intervalSec=6&reset=true');
  await refreshMapControls();
  await refreshLivePanel();
}}
async function startAutoBestMap() {{
  pauseMapSwitcher(800);
  await apiGet('/map_cycle_update?enabled=true&mode=best&intervalSec=8&reset=true');
  await refreshMapControls();
  await refreshLivePanel();
}}
async function stopMapCycle() {{
  pauseMapSwitcher(800);
  await apiGet('/map_cycle_update?enabled=false');
  await refreshMapControls();
  await refreshLivePanel();
}}
setInterval(refreshLivePanel, 900);
</script>
</body></html>"""
    response = app.response_class(page, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        cache = latest_cache
        turret_state = dict(latest_turret)
        player_state = json_copy(latest_player_state)
        counters = dict(status_state)
        yolo_snapshot = json_copy(yolo_state)

    return jsonify(
        {
            "server": "Tank Challenge LiDAR-first YOLO Fusion v16.16 Easy Map Switch + Sim LiDAR Overlay",
            "purpose": "realtime LiDAR primary pipeline with asynchronous YOLO semantic fusion",
            "recommendedSimulatorProperties": {
                "intervalSec": EXPECTED_INTERVAL_SEC,
                "lidarYPositionM": EXPECTED_LIDAR_Y_POSITION_M,
                "channel": EXPECTED_CHANNELS,
                "minimapChannel": EXPECTED_MINIMAP_CHANNEL,
                "maxDistanceM": EXPECTED_MAX_DISTANCE_M,
                "lidarPosition": "Body",
                "sendDetectedLidar": True,
                "frameRate": 120,
                "graphicsQuality": "Ultra",
            },
            "frame": {
                "seq": cache.seq,
                "groundFilterEnabled": GROUND_FILTER_ENABLED,
                "lidarVehicleExtractionEnabled": LIDAR_VEHICLE_EXTRACTION_ENABLED,
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
                "playerPosition": player_state,
                "playerPosFromInfo": json_copy(cache.pose.get("playerPos", {})),
                "lidarOrigin": json_copy(cache.pose.get("lidarOrigin", {})),
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
        "obstacleBoxLimit": (0, 3000),
        "safeGroundBoxLimit": (0, 1000),
        "totalLidarBoxLimit": (0, 4000),
        "obstaclePixelCell": (1, 50),
        "safeGroundPixelCell": (1, 100),
        "objectPointRadiusPx": (1, 12),
        "clusterBoxLimit": (0, 80),
        "clusterBoxMinPoints": (1, 20),
    }
    for key, (minimum, maximum) in integer_fields.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            overlay_settings[key] = int(max(minimum, min(maximum, round(value))))

    float_fields = {
        "clusterBoxAngleGateDeg": (0.5, 30.0),
        "clusterBoxDistanceGateM": (0.5, 30.0),
    }
    for key, (minimum, maximum) in float_fields.items():
        if key not in request.args:
            continue
        value = safe_float(request.args.get(key))
        if value is not None:
            overlay_settings[key] = float(max(minimum, min(maximum, value)))

    for key in {"showLidarPoints", "showSafeGround", "showLidarClusterBoxes"}:
        if key in request.args:
            overlay_settings[key] = str(request.args.get(key)).strip().lower() in {
                "1", "true", "yes", "on"
            }

    if "simLidarPointMode" in request.args:
        mode = str(request.args.get("simLidarPointMode", "valid_plus_high")).strip().lower()
        if mode in {
            "vehicle_clusters",
            "valid_only",
            "valid_plus_high",
            "all_obstacles",
        }:
            overlay_settings["simLidarPointMode"] = mode

    return jsonify({"status": "success", "overlay": dict(overlay_settings)})


@app.route("/sim_debug_setup", methods=["GET", "POST"])
def sim_debug_setup():
    """One-click setup for the user's current debugging workflow.

    It does not change the simulator map.  It only makes YOLO more permissive
    and makes object LiDAR hits visible on the simulator overlay returned by
    /detect.
    """
    mode = str(request.args.get("mode", "tank_object_overlay")).strip().lower()
    if mode not in {"tank_object_overlay", "cpu_light", "yolo_only"}:
        return jsonify({"status": "error", "message": "mode=tank_object_overlay, cpu_light, or yolo_only"}), 400

    if mode == "cpu_light":
        fusion_settings.update({
            "confidence": 0.16,
            "iou": 0.45,
            "imageSize": 416,
            "yoloIntervalSec": 0.80,
            "maxDisplayAgeSec": 4.0,
            "maxDisplayYawDeltaDeg": 32.0,
            "showFusedBoxes": True,
            "showUnmatchedYoloBoxes": True,
        })
        overlay_settings.update({
            "showLidarPoints": True,
            "showSafeGround": False,
            "simLidarPointMode": "valid_plus_high",
            "showLidarClusterBoxes": True,
            "obstacleBoxLimit": 450,
            "totalLidarBoxLimit": 620,
            "obstaclePixelCell": 3,
            "objectPointRadiusPx": 3,
        })
    elif mode == "yolo_only":
        fusion_settings.update({
            "confidence": 0.10,
            "iou": 0.45,
            "imageSize": 640,
            "yoloIntervalSec": 0.50,
            "maxDisplayAgeSec": 5.0,
            "showFusedBoxes": True,
            "showUnmatchedYoloBoxes": True,
        })
        overlay_settings.update({
            "showLidarPoints": False,
            "showSafeGround": False,
            "showLidarClusterBoxes": False,
        })
    else:
        fusion_settings.update({
            "confidence": 0.10,
            "iou": 0.45,
            "imageSize": 640,
            "yoloIntervalSec": 0.45,
            "roiFusionEnabled": True,
            "roiExpandRatio": 0.10,
            "roiMinObstaclePoints": 1,
            "roiSurfaceBandM": 4.0,
            "clusterFallbackEnabled": True,
            "maxFusionAngleGapDeg": 14.0,
            "maxDisplayAgeSec": 5.0,
            "maxDisplayYawDeltaDeg": 35.0,
            "maxDisplayPitchDeltaDeg": 14.0,
            "showFusedBoxes": True,
            "showUnmatchedYoloBoxes": True,
            "showYoloOnlyAngleLabel": True,
            "showYoloOnlyLidarHint": True,
        })
        overlay_settings.update({
            "showLidarPoints": True,
            "showSafeGround": False,
            "simLidarPointMode": "valid_plus_high",
            "showLidarClusterBoxes": True,
            "clusterBoxLimit": 16,
            "clusterBoxMinPoints": 1,
            "clusterBoxAngleGateDeg": 5.0,
            "clusterBoxDistanceGateM": 5.0,
            "obstacleBoxLimit": 900,
            "totalLidarBoxLimit": 1100,
            "obstaclePixelCell": 2,
            "objectPointRadiusPx": 3,
        })

    preload = safe_bool(request.args.get("preloadYolo"), False)
    preload_result = None
    if preload:
        try:
            get_yolo_model()
            preload_result = "loaded"
        except Exception as exc:
            preload_result = f"{type(exc).__name__}: {exc}"
    return jsonify({
        "status": "success",
        "mode": mode,
        "preloadYolo": preload_result,
        "fusion": dict(fusion_settings),
        "overlay": dict(overlay_settings),
        "note": "Simulator must still call /detect with camera images for YOLO boxes to appear.",
    })


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
    global latest_cache, _pending_vision_job, next_object_track_id

    with state_lock:
        latest_cache = EMPTY_CACHE
        object_tracks.clear()
        next_object_track_id = 1
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



@app.route("/map_gt_select", methods=["GET", "POST"])
def map_gt_select():
    """Load one .map from the web dashboard.

    Supports either:
      - filename=local_file.map  -> file next to this Python script
      - path=C:\\...\\file.map   -> pasted full path on the same PC running Python
    """
    filename = request.args.get("filename") or request.args.get("map")
    path_value = request.args.get("path") or request.args.get("mapPath")
    if not filename and not path_value:
        return jsonify({
            "status": "error",
            "message": "Use /map_gt_select?filename=YOUR.map or /map_gt_select?path=FULL_PATH.map",
            "baseDir": str(BASE_DIR),
            "mapFiles": [path.name for path in local_map_files()],
        }), 400
    clear_existing = safe_bool(request.args.get("clearExisting"), True)
    stop_cycle = safe_bool(request.args.get("stopCycle"), True)
    if path_value:
        result = load_map_ground_truth(path_value=path_value, clear_existing=clear_existing, persist_selection=True)
        loaded_name = Path(str(path_value)).name
    else:
        result = load_map_ground_truth(filename=filename, clear_existing=clear_existing, persist_selection=True)
        loaded_name = str(filename)
    with state_lock:
        if stop_cycle:
            map_cycle_settings["enabled"] = False
        if result.get("status") == "success":
            map_cycle_settings["currentMapFile"] = loaded_name
            map_cycle_settings["lastSwitchAt"] = now_text()
            map_cycle_settings["lastSwitchMonotonic"] = monotonic()
            map_cycle_settings["lastError"] = None
        else:
            map_cycle_settings["lastError"] = result.get("message", result.get("status"))
    return jsonify({
        "status": "success" if result.get("status") == "success" else "error",
        "loadedMap": loaded_name,
        "baseDir": str(BASE_DIR),
        "activeMapFile": ground_truth_state.get("activeMapFile"),
        "result": json_copy(result),
        "cycle": json_copy(map_cycle_settings),
    }), (200 if result.get("status") == "success" else 400)


def safe_map_upload_filename(name: str) -> str:
    base = Path(str(name or "uploaded.map")).name.strip().replace("\\", "_").replace("/", "_")
    if not base.lower().endswith(".map"):
        base = f"{base}.map"
    keep = []
    for ch in base:
        if ch.isalnum() or ch in {".", "_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    result = "".join(keep).strip("._")
    if not result.lower().endswith(".map"):
        result = f"{result}.map"
    return result or f"uploaded_{uuid.uuid4().hex[:8]}.map"


@app.route("/map_upload_load", methods=["POST"])
def map_upload_load():
    uploaded = request.files.get("mapfile") or request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"status": "error", "message": "Choose a .map file field named mapfile."}), 400
    filename = safe_map_upload_filename(uploaded.filename)
    path = BASE_DIR / filename
    try:
        uploaded.save(str(path))
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Save failed: {type(exc).__name__}: {exc}", "path": str(path)}), 500
    result = load_map_ground_truth(filename=filename, clear_existing=True, persist_selection=True)
    with state_lock:
        map_cycle_settings["enabled"] = False
        if result.get("status") == "success":
            map_cycle_settings["currentMapFile"] = filename
            map_cycle_settings["lastSwitchAt"] = now_text()
            map_cycle_settings["lastSwitchMonotonic"] = monotonic()
            map_cycle_settings["lastError"] = None
        else:
            map_cycle_settings["lastError"] = result.get("message", result.get("status"))
    return jsonify({
        "status": "success" if result.get("status") == "success" else "error",
        "loadedMap": filename,
        "savedPath": str(path),
        "result": json_copy(result),
        "mapFiles": [p.name for p in local_map_files()],
    }), (200 if result.get("status") == "success" else 400)



@app.route("/map_cycle_status", methods=["GET"])
def map_cycle_status():
    return jsonify({
        "status": "success",
        "settings": json_copy(map_cycle_settings),
        "baseDir": str(BASE_DIR),
        "mapFiles": [path.name for path in local_map_files()],
        "activeMapFile": ground_truth_state.get("activeMapFile"),
    })


@app.route("/map_cycle_update", methods=["GET", "POST"])
def map_cycle_update():
    if "enabled" in request.args:
        map_cycle_settings["enabled"] = safe_bool(request.args.get("enabled"), bool(map_cycle_settings.get("enabled", False)))
    if "mode" in request.args:
        mode = str(request.args.get("mode", "cycle")).strip().lower()
        if mode in {"cycle", "best"}:
            map_cycle_settings["mode"] = mode
    if "intervalSec" in request.args:
        value = safe_float(request.args.get("intervalSec"), None)
        if value is not None:
            map_cycle_settings["intervalSec"] = max(1.0, min(120.0, float(value)))
    if safe_bool(request.args.get("reset"), False):
        map_cycle_settings["currentIndex"] = -1
        map_cycle_settings["currentMapFile"] = None
        map_cycle_settings["lastSwitchAt"] = None
        map_cycle_settings["lastSwitchMonotonic"] = None
        map_cycle_settings["lastBestScan"] = None
        map_cycle_settings["lastError"] = None

    with state_lock:
        cache = latest_cache
    tick = maybe_rotate_active_map_gt(cache) if bool(map_cycle_settings.get("enabled", False)) else {"enabled": False}
    return jsonify({
        "status": "success",
        "settings": json_copy(map_cycle_settings),
        "tick": json_copy(tick),
        "baseDir": str(BASE_DIR),
        "mapFiles": [path.name for path in local_map_files()],
        "activeMapFile": ground_truth_state.get("activeMapFile"),
        "usage": {
            "cycle": "/map_cycle_update?enabled=true&mode=cycle&intervalSec=6&reset=true",
            "autoBest": "/map_cycle_update?enabled=true&mode=best&intervalSec=8&reset=true",
            "stop": "/map_cycle_update?enabled=false",
        },
        "warning": "This switches only the Python-side .map GT file. The simulator map itself is not changed by this endpoint.",
    })


@app.route("/map_auto_best", methods=["GET", "POST"])
def map_auto_best():
    with state_lock:
        cache = latest_cache
    apply_best = safe_bool(request.args.get("applyBest"), True)
    max_items = int(safe_float(request.args.get("limit"), 40) or 40)
    max_error = float(safe_float(request.args.get("maxWorldErrorM"), 18.0) or 18.0)
    result = score_all_local_maps_against_lidar(
        cache,
        apply_best=apply_best,
        max_items=max_items,
        max_match_world_error_m=max_error,
    )
    with state_lock:
        map_cycle_settings["lastBestScan"] = json_copy(result)
        best = result.get("best") or {}
        if result.get("status") == "success" and apply_best:
            map_cycle_settings["currentMapFile"] = best.get("filename")
            map_cycle_settings["lastSwitchAt"] = now_text()
            map_cycle_settings["lastSwitchMonotonic"] = monotonic()
    return jsonify(result)

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

# Optional terrain-height grid for more stable hill/object separation.
load_hill_map_height(force=False)

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
    print("Hill map     : http://127.0.0.1:5000/hill_map_height_status")
    print("Recommended  : Interval=0.5, Y=3, Channel=32, Minimap=16, Range=120, FPS=120, Graphics=Ultra")
    print("=" * 80)
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True, debug=False)
