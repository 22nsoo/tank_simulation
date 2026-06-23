from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from math import cos, floor, isfinite, radians, sin, tan
from pathlib import Path
from statistics import median
from threading import Lock
from time import monotonic
from typing import Any
import json
import re

from flask import Flask, jsonify, request

app = Flask(__name__)

# =============================================================================
# LiDAR sensing visualizer
# =============================================================================
# 목적:
# - /info 로 들어오는 LiDAR ray 중 실제로 센싱된 hit 위치를 그대로 시각화한다.
# - 기존 v8.1의 Top + Front angular view와 v8.3의 centerline vertical profile 아이디어를 합쳤다.
# - 객체 인식/주행 판단을 하지 않고, "LiDAR가 실제로 어디를 찍고 있는지" 확인하는 디버그 서버다.
#
# 접속:
#   python server_lidar_sensing_visualizer.py
#   http://127.0.0.1:5000/lidar_view
#   http://127.0.0.1:5000/lidar_patch_view
# =============================================================================

MAX_DISTANCE_M = 300.0

# Front angular point-cloud view. X = body-relative azimuth, Y = vertical angle.
FRONT_VIEW_HORIZONTAL_LIMIT_DEG = 70.0
FRONT_VIEW_VERTICAL_MIN_DEG = -22.5
FRONT_VIEW_VERTICAL_MAX_DEG = 22.5
FRONT_VIEW_MAX_DISTANCE_M = 120.0

# Side profile. Select the azimuth closest to this value.
PROFILE_TARGET_ANGLE_DEG = 0.0
PROFILE_SELECT_MAX_ERROR_DEG = 2.0
PROFILE_MAX_DISTANCE_M = 120.0
PROFILE_HEIGHT_MIN_M = -8.0
PROFILE_HEIGHT_MAX_M = 16.0

# Ground estimation. In this simulator, positive verticalAngle is downward.
GROUND_VERTICAL_MIN_DEG = 0.5
GROUND_VERTICAL_MAX_DEG = 22.5
GROUND_ESTIMATE_MAX_RANGE_M = 35.0
LOCAL_GROUND_STEEP_MIN_DEG = 8.0
LOCAL_GROUND_NEAR_MAX_RANGE_M = 12.0
LOCAL_GROUND_MIN_POINT_COUNT = 6

# Hit classification for visualization only.
GROUND_HEIGHT_MAX_M = 0.35
OBJECT_HEIGHT_MIN_M = 0.55
FAR_TERRAIN_EXTRA_RATIO = 1.35
FAR_TERRAIN_EXTRA_MIN_M = 3.0

# Nearest-hit DLL patch verification.
PATCH_VERIFY_FRONT_LIMIT_DEG = 60.0
PATCH_VERIFY_OBJECT_VERTICAL_MIN_DEG = -10.0
PATCH_VERIFY_OBJECT_VERTICAL_MAX_DEG = 1.0
PATCH_VERIFY_DISTANCE_GAP_M = 4.0
PATCH_VERIFY_GROUND_TOLERANCE_M = 0.35
PATCH_VERIFY_MIN_FOREGROUND_RAYS = 3
PATCH_VERIFY_DIR = Path.cwd() / "lidar_patch_verification"

state_lock = Lock()
latest_raw_info: dict[str, Any] = {}
latest_state: dict[str, Any] = {}


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def normalize_signed_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_distance_deg(a: float, b: float) -> float:
    return abs(normalize_signed_angle(a - b))


def quantile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    ratio = max(0.0, min(1.0, ratio))
    index = int(round((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def bin_center(value: float, width: float, offset: float = 0.0) -> float:
    index = floor((value - offset) / width)
    return offset + (index + 0.5) * width


# =============================================================================
# 1. Parse LiDAR rays
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


def valid_hit_rays(rays: list[dict[str, Any]]) -> list[dict[str, Any]]:
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



def lidar_patch_ray_key(ray: dict[str, Any]) -> str:
    return (
        f"{float(ray['angle']):.3f}|"
        f"{float(ray['verticalAngle']):.3f}|"
        f"{ray.get('channelIndex')}"
    )


def build_patch_verification(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
) -> dict[str, Any]:
    """
    Behavioral verification for the nearest-hit DLL patch.

    /info does not include collider names or a DLL hash, so this cannot directly
    prove which binary is loaded. It detects whether foreground returns form a
    continuous near surface and flags much farther ground-like hits inside that
    same angular silhouette.
    """
    front = [
        ray
        for ray in rays
        if (
            abs(float(ray["angle"])) <= PATCH_VERIFY_FRONT_LIMIT_DEG
            and FRONT_VIEW_VERTICAL_MIN_DEG
            <= float(ray["verticalAngle"])
            <= FRONT_VIEW_VERTICAL_MAX_DEG
        )
    ]

    by_angle: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for ray in front:
        if (
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
        ):
            by_angle[round(float(ray["angle"]), 3)].append(ray)

    candidate_columns: list[dict[str, Any]] = []
    total_foreground = 0
    total_suspected_rear_ground = 0

    for angle, column in sorted(by_angle.items()):
        object_band = [
            ray
            for ray in column
            if (
                PATCH_VERIFY_OBJECT_VERTICAL_MIN_DEG
                <= float(ray["verticalAngle"])
                <= PATCH_VERIFY_OBJECT_VERTICAL_MAX_DEG
            )
        ]

        foreground = []
        for ray in object_band:
            y = safe_float(ray["position"].get("y"))
            if y is None or local_ground_y is None:
                continue
            if y - float(local_ground_y) >= OBJECT_HEIGHT_MIN_M:
                foreground.append(ray)

        if len(foreground) < PATCH_VERIFY_MIN_FOREGROUND_RAYS:
            continue

        near_distance = float(median(float(ray["distance"]) for ray in foreground))
        vertical_min = min(float(ray["verticalAngle"]) for ray in foreground)
        vertical_max = max(float(ray["verticalAngle"]) for ray in foreground)

        silhouette = [
            ray
            for ray in object_band
            if vertical_min - 0.4 <= float(ray["verticalAngle"]) <= vertical_max + 0.4
        ]

        coherent_near = 0
        suspected_rear_ground = 0
        for ray in silhouette:
            distance = float(ray["distance"])
            y = safe_float(ray["position"].get("y"))
            near_ground = (
                y is not None
                and local_ground_y is not None
                and abs(y - float(local_ground_y))
                <= PATCH_VERIFY_GROUND_TOLERANCE_M
            )

            if (
                distance > near_distance + PATCH_VERIFY_DISTANCE_GAP_M
                and near_ground
            ):
                suspected_rear_ground += 1
            elif abs(distance - near_distance) <= PATCH_VERIFY_DISTANCE_GAP_M:
                coherent_near += 1

        denominator = max(1, coherent_near + suspected_rear_ground)
        continuity_ratio = coherent_near / denominator

        total_foreground += len(foreground)
        total_suspected_rear_ground += suspected_rear_ground

        candidate_columns.append({
            "angle": round(angle, 3),
            "nearDistance": round(near_distance, 3),
            "verticalMin": round(vertical_min, 3),
            "verticalMax": round(vertical_max, 3),
            "foregroundRayCount": len(foreground),
            "coherentNearRayCount": coherent_near,
            "suspectedRearGroundRayCount": suspected_rear_ground,
            "continuityRatio": round(continuity_ratio, 3),
            "status": (
                "GOOD_NEAREST_SURFACE"
                if suspected_rear_ground == 0 and continuity_ratio >= 0.75
                else "CHECK_POSSIBLE_PASSTHROUGH"
            ),
        })

    suspected_ratio = total_suspected_rear_ground / max(
        1,
        total_foreground + total_suspected_rear_ground,
    )

    if not candidate_columns:
        behavior_status = "NO_FOREGROUND_TEST_TARGET"
    elif suspected_ratio <= 0.05:
        behavior_status = "PATCH_BEHAVIOR_LOOKS_GOOD"
    else:
        behavior_status = "POSSIBLE_REAR_GROUND_SELECTION"

    missed = [ray for ray in rays if not ray.get("isDetected", False)]
    missed_distances = sorted(float(ray["distance"]) for ray in missed)
    inferred_max_range = (
        float(median(missed_distances))
        if missed_distances
        else None
    )

    return {
        "status": behavior_status,
        "message": (
            "Behavioral verification only. The definitive test is a baseline/"
            "patched snapshot comparison at the exact same pose."
        ),
        "totalRayCount": len(rays),
        "detectedRayCount": sum(1 for ray in rays if ray.get("isDetected", False)),
        "missedRayCount": len(missed),
        "inferredMaxRange": round_or_none(inferred_max_range),
        "candidateColumnCount": len(candidate_columns),
        "foregroundRayCount": total_foreground,
        "suspectedRearGroundRayCount": total_suspected_rear_ground,
        "suspectedRearGroundRatio": round(suspected_ratio, 5),
        "candidateColumns": candidate_columns[:200],
        "heatmapPoints": [
            {
                "angle": round(float(ray["angle"]), 3),
                "verticalAngle": round(float(ray["verticalAngle"]), 3),
                "distance": round(float(ray["distance"]), 3),
                "isDetected": bool(ray.get("isDetected", False)),
            }
            for ray in front
        ],
    }


def compact_patch_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    rays = parse_lidar_rays(data)
    return {
        "capturedAt": datetime.now().isoformat(timespec="milliseconds"),
        "simulationTime": data.get("time"),
        "lidarOrigin": deepcopy(data.get("lidarOrigin", {})),
        "lidarRotation": deepcopy(data.get("lidarRotation", {})),
        "rayCount": len(rays),
        "rays": {
            lidar_patch_ray_key(ray): {
                "angle": round(float(ray["angle"]), 3),
                "verticalAngle": round(float(ray["verticalAngle"]), 3),
                "channelIndex": ray.get("channelIndex"),
                "distance": round(float(ray["distance"]), 4),
                "isDetected": bool(ray.get("isDetected", False)),
                "y": round_or_none(safe_float(ray["position"].get("y")), 4),
            }
            for ray in rays
        },
    }


def load_patch_snapshot(label: str) -> dict[str, Any] | None:
    path = PATCH_VERIFY_DIR / f"{label}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_patch_snapshots(
    baseline: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    before_rays = baseline.get("rays", {})
    after_rays = patched.get("rays", {})
    common_keys = sorted(set(before_rays) & set(after_rays))

    nearer = []
    farther = []
    unchanged = 0

    for key in common_keys:
        before = before_rays[key]
        after = after_rays[key]
        before_distance = safe_float(before.get("distance"))
        after_distance = safe_float(after.get("distance"))
        if before_distance is None or after_distance is None:
            continue

        delta = float(after_distance) - float(before_distance)
        record = {
            "key": key,
            "angle": after.get("angle"),
            "verticalAngle": after.get("verticalAngle"),
            "channelIndex": after.get("channelIndex"),
            "baselineDistance": round(float(before_distance), 3),
            "patchedDistance": round(float(after_distance), 3),
            "delta": round(delta, 3),
        }

        if delta <= -PATCH_VERIFY_DISTANCE_GAP_M:
            nearer.append(record)
        elif delta >= PATCH_VERIFY_DISTANCE_GAP_M:
            farther.append(record)
        else:
            unchanged += 1

    nearer.sort(key=lambda item: item["delta"])
    farther.sort(key=lambda item: item["delta"], reverse=True)

    common_count = len(common_keys)
    nearer_ratio = len(nearer) / max(1, common_count)
    farther_ratio = len(farther) / max(1, common_count)

    return {
        "status": (
            "PATCH_EFFECT_DETECTED"
            if len(nearer) >= 10 and nearer_ratio > farther_ratio
            else "NO_CLEAR_PATCH_EFFECT"
        ),
        "warning": (
            "Both captures must use the same tank position, body yaw, turret pose, "
            "obstacle placement and LiDAR range."
        ),
        "baselineRayCount": baseline.get("rayCount"),
        "patchedRayCount": patched.get("rayCount"),
        "commonRayCount": common_count,
        "onlyBaselineCount": len(set(before_rays) - set(after_rays)),
        "onlyPatchedCount": len(set(after_rays) - set(before_rays)),
        "changedToNearerCount": len(nearer),
        "changedToFartherCount": len(farther),
        "unchangedCount": unchanged,
        "changedToNearerRatio": round(nearer_ratio, 5),
        "changedToFartherRatio": round(farther_ratio, 5),
        "largestNearerChanges": nearer[:100],
        "largestFartherChanges": farther[:30],
        "baselineMeta": {
            "capturedAt": baseline.get("capturedAt"),
            "simulationTime": baseline.get("simulationTime"),
            "lidarOrigin": baseline.get("lidarOrigin"),
            "lidarRotation": baseline.get("lidarRotation"),
        },
        "patchedMeta": {
            "capturedAt": patched.get("capturedAt"),
            "simulationTime": patched.get("simulationTime"),
            "lidarOrigin": patched.get("lidarOrigin"),
            "lidarRotation": patched.get("lidarRotation"),
        },
    }


# =============================================================================
# 2. Ground estimate and hit classification
# =============================================================================
def estimate_flat_ground_y(hits: list[dict[str, Any]]) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in hits
        if (
            GROUND_VERTICAL_MIN_DEG <= float(ray["verticalAngle"]) <= GROUND_VERTICAL_MAX_DEG
            and float(ray["horizontalRange"]) <= GROUND_ESTIMATE_MAX_RANGE_M
            and ray["position"].get("y") is not None
        )
    )
    return quantile(heights, 0.25)


def estimate_local_ground_y(rays: list[dict[str, Any]], fallback_ground_y: float | None) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in rays
        if (
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
            and float(ray["verticalAngle"]) >= LOCAL_GROUND_STEEP_MIN_DEG
            and float(ray["horizontalRange"]) <= LOCAL_GROUND_NEAR_MAX_RANGE_M
        )
    )
    if len(heights) >= LOCAL_GROUND_MIN_POINT_COUNT:
        return float(median(heights))
    return fallback_ground_y


def expected_flat_ground_range(sensor_height_m: float | None, vertical_angle_deg: float) -> float | None:
    """Expected horizontal range where a downward ray would hit flat ground."""
    if sensor_height_m is None or sensor_height_m <= 0.05 or vertical_angle_deg <= 0.0:
        return None
    tangent = tan(radians(vertical_angle_deg))
    return sensor_height_m / tangent if tangent > 0.0 else None


def classify_hit(ray: dict[str, Any], local_ground_y: float | None, lidar_origin_y: float | None) -> dict[str, Any]:
    pos_y = safe_float(ray["position"].get("y"))
    height_above_ground = (
        float(pos_y) - float(local_ground_y)
        if pos_y is not None and local_ground_y is not None
        else None
    )

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )
    expected_range = expected_flat_ground_range(sensor_height, float(ray["verticalAngle"]))

    # Visualization labels only. These are not semantic object labels.
    hit_type = "unknown_hit"
    if height_above_ground is not None:
        if abs(height_above_ground) <= GROUND_HEIGHT_MAX_M and float(ray["verticalAngle"]) >= 0.0:
            hit_type = "ground_like"
        elif height_above_ground >= OBJECT_HEIGHT_MIN_M and float(ray["verticalAngle"]) < 0.0:
            hit_type = "silhouette_or_object_top"
        elif height_above_ground >= OBJECT_HEIGHT_MIN_M:
            hit_type = "object_like"

    # Suspected terrain-through / delayed ground return.
    if (
        expected_range is not None
        and height_above_ground is not None
        and abs(height_above_ground) <= GROUND_HEIGHT_MAX_M
        and float(ray["horizontalRange"]) > expected_range * FAR_TERRAIN_EXTRA_RATIO + FAR_TERRAIN_EXTRA_MIN_M
    ):
        hit_type = "delayed_ground_suspect"

    return {
        "angle": round(float(ray["angle"]), 3),
        "verticalAngle": round(float(ray["verticalAngle"]), 3),
        "distance": round(float(ray["distance"]), 3),
        "horizontalRange": round(float(ray["horizontalRange"]), 3),
        "channelIndex": ray.get("channelIndex"),
        "position": {
            "x": round(float(ray["position"]["x"]), 3),
            "y": round(float(ray["position"]["y"]), 3),
            "z": round(float(ray["position"]["z"]), 3),
        },
        "heightAboveLocalGround": round_or_none(height_above_ground),
        "expectedFlatGroundRange": round_or_none(expected_range),
        "hitType": hit_type,
    }


# =============================================================================
# 3. Visualization payload builders
# =============================================================================
def make_front_view_points(classified_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for hit in classified_hits:
        angle = float(hit["angle"])
        vertical = float(hit["verticalAngle"])
        distance = float(hit["distance"])
        if abs(angle) > FRONT_VIEW_HORIZONTAL_LIMIT_DEG:
            continue
        if not (FRONT_VIEW_VERTICAL_MIN_DEG <= vertical <= FRONT_VIEW_VERTICAL_MAX_DEG):
            continue
        if distance > FRONT_VIEW_MAX_DISTANCE_M:
            continue
        points.append(hit)
    return points


def make_top_view_points(classified_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Body-relative polar map using LiDAR azimuth and horizontalRange.
    return [
        hit
        for hit in classified_hits
        if float(hit["horizontalRange"]) <= MAX_DISTANCE_M
    ]


def make_centerline_profile(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    if not rays:
        return {"selectedAngle": None, "rays": []}

    unique_angles = sorted({round(float(ray["angle"]), 3) for ray in rays})
    selected_angle = min(unique_angles, key=lambda angle: angular_distance_deg(angle, PROFILE_TARGET_ANGLE_DEG))

    if angular_distance_deg(selected_angle, PROFILE_TARGET_ANGLE_DEG) > PROFILE_SELECT_MAX_ERROR_DEG:
        return {
            "targetAngle": PROFILE_TARGET_ANGLE_DEG,
            "selectedAngle": selected_angle,
            "warning": "No near-forward azimuth ray found.",
            "rays": [],
        }

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )

    selected_rays = [ray for ray in rays if abs(float(ray["angle"]) - selected_angle) <= 0.15]
    selected_rays.sort(key=lambda ray: (int(ray["channelIndex"]) if isinstance(ray.get("channelIndex"), int) else 9999, float(ray["verticalAngle"])))

    out: list[dict[str, Any]] = []
    for ray in selected_rays:
        detected = bool(
            ray.get("isDetected", False)
            and ray["position"].get("x") is not None
            and ray["position"].get("y") is not None
            and ray["position"].get("z") is not None
        )
        height_above = None
        if detected and local_ground_y is not None:
            height_above = float(ray["position"]["y"]) - float(local_ground_y)

        expected_range = expected_flat_ground_range(sensor_height, float(ray["verticalAngle"]))
        out.append(
            {
                "channelIndex": ray.get("channelIndex"),
                "angle": round(float(ray["angle"]), 3),
                "verticalAngle": round(float(ray["verticalAngle"]), 3),
                "isDetected": detected,
                "distance": round(float(ray["distance"]), 3),
                "horizontalRange": round(float(ray["horizontalRange"]), 3),
                "positionY": round_or_none(safe_float(ray["position"].get("y"))),
                "heightAboveLocalGround": round_or_none(height_above),
                "expectedFlatGroundRange": round_or_none(expected_range),
            }
        )

    return {
        "targetAngle": PROFILE_TARGET_ANGLE_DEG,
        "selectedAngle": round(float(selected_angle), 3),
        "sensorHeightAboveLocalGround": round_or_none(sensor_height),
        "channelCount": len(out),
        "hitCount": sum(1 for ray in out if ray["isDetected"]),
        "missCount": sum(1 for ray in out if not ray["isDetected"]),
        "rays": out,
    }


def summarize_lidar(data: dict[str, Any]) -> dict[str, Any]:
    rays = parse_lidar_rays(data)
    hits = valid_hit_rays(rays)

    flat_ground_y = estimate_flat_ground_y(hits)
    local_ground_y = estimate_local_ground_y(rays, flat_ground_y)
    lidar_origin = data.get("lidarOrigin", {}) or {}
    lidar_origin_y = safe_float(lidar_origin.get("y"))
    lidar_rotation = data.get("lidarRotation", {}) or {}

    classified_hits = [classify_hit(ray, local_ground_y, lidar_origin_y) for ray in hits]
    patch_verification = build_patch_verification(rays, local_ground_y)

    hit_type_counts: dict[str, int] = defaultdict(int)
    for hit in classified_hits:
        hit_type_counts[str(hit["hitType"])] += 1

    return {
        "simulationTime": data.get("time"),
        "receivedAt": datetime.now().isoformat(timespec="seconds"),
        "lidarOrigin": lidar_origin,
        "lidarRotation": lidar_rotation,
        "rawRayCount": len(rays),
        "rawDetectedPointCount": len(hits),
        "estimatedGroundY": round_or_none(flat_ground_y),
        "localGroundY": round_or_none(local_ground_y),
        "lidarOriginY": round_or_none(lidar_origin_y),
        "hitTypeCounts": dict(sorted(hit_type_counts.items())),
        "topViewPoints": make_top_view_points(classified_hits),
        "frontViewPoints": make_front_view_points(classified_hits),
        "centerlineProfile": make_centerline_profile(rays, local_ground_y, lidar_origin_y),
        "patchVerification": patch_verification,
    }


# =============================================================================
# 4. Flask endpoints
# =============================================================================
@app.route("/info", methods=["POST"])
def info():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    scan = summarize_lidar(data)
    with state_lock:
        global latest_raw_info, latest_state
        latest_raw_info = deepcopy(data)
        latest_state = scan

    print(
        f"LiDAR frame | t={scan.get('simulationTime')} | rays={scan['rawRayCount']} | "
        f"hits={scan['rawDetectedPointCount']} | groundY={scan.get('localGroundY')} | "
        f"types={scan.get('hitTypeCounts')}"
    )
    return jsonify({"status": "success", "control": ""})


@app.route("/lidar_status", methods=["GET"])
def lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_state))


@app.route("/raw_lidar_status", methods=["GET"])
def raw_lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_raw_info))


@app.route("/export_snapshot", methods=["POST", "GET"])
def export_snapshot():
    label = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", request.args.get("label", "snapshot")).strip("_") or "snapshot"
    with state_lock:
        raw_copy = deepcopy(latest_raw_info)
        analyzed_copy = deepcopy(latest_state)
    if not raw_copy:
        return jsonify({"status": "error", "message": "No /info frame has been received yet."}), 400
    output_dir = Path.cwd() / "lidar_sensing_snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.json"
    path.write_text(json.dumps({"label": label, "rawInfo": raw_copy, "analysis": analyzed_copy}, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"status": "success", "savedPath": str(path), "label": label})


@app.route("/reset_state", methods=["POST", "GET"])
def reset_state():
    with state_lock:
        latest_raw_info.clear()
        latest_state.clear()
    return jsonify({"status": "success", "message": "LiDAR sensing visualizer state cleared."})


@app.route("/get_action", methods=["POST"])
def get_action():
    return jsonify({
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    })


@app.route("/init", methods=["GET"])
def init():
    with state_lock:
        latest_raw_info.clear()
        latest_state.clear()
    return jsonify({
        "startMode": "start",
        "blStartX": 150,
        "blStartY": 10,
        "blStartZ": 20,
        "rdStartX": 150,
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": False,
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True,
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": ""})



@app.route("/lidar_patch_status", methods=["GET"])
def lidar_patch_status():
    with state_lock:
        return jsonify(deepcopy(latest_state.get("patchVerification", {})))


@app.route("/capture_lidar_patch_snapshot", methods=["POST", "GET"])
def capture_lidar_patch_snapshot():
    label = re.sub(
        r"[^0-9A-Za-z가-힣_-]+",
        "_",
        request.args.get("label", "baseline"),
    ).strip("_") or "baseline"

    with state_lock:
        raw_copy = deepcopy(latest_raw_info)

    if not raw_copy or not isinstance(raw_copy.get("lidarPoints"), list):
        return jsonify({
            "status": "error",
            "message": "No /info frame containing lidarPoints has been received.",
        }), 400

    PATCH_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = compact_patch_snapshot(raw_copy)
    path = PATCH_VERIFY_DIR / f"{label}.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return jsonify({
        "status": "success",
        "label": label,
        "savedPath": str(path),
        "capturedAt": snapshot.get("capturedAt"),
        "rayCount": snapshot.get("rayCount"),
    })


@app.route("/compare_lidar_patch_snapshots", methods=["GET"])
def compare_lidar_patch_snapshots_endpoint():
    baseline_label = re.sub(
        r"[^0-9A-Za-z가-힣_-]+",
        "_",
        request.args.get("baseline", "baseline"),
    ).strip("_") or "baseline"
    patched_label = re.sub(
        r"[^0-9A-Za-z가-힣_-]+",
        "_",
        request.args.get("patched", "patched"),
    ).strip("_") or "patched"

    baseline = load_patch_snapshot(baseline_label)
    patched = load_patch_snapshot(patched_label)
    if baseline is None or patched is None:
        return jsonify({
            "status": "MISSING_SNAPSHOT",
            "baselineFound": baseline is not None,
            "patchedFound": patched is not None,
            "snapshotDirectory": str(PATCH_VERIFY_DIR),
        }), 404

    return jsonify(compare_patch_snapshots(baseline, patched))


@app.route("/lidar_patch_view", methods=["GET"])
def lidar_patch_view():
    return r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>LiDAR nearest-hit patch verifier</title>
<style>
body { margin:20px; background:#111; color:#eee; font-family:Arial,sans-serif; }
.panel { background:#191919; border:1px solid #555; padding:14px; margin-bottom:16px; }
button { padding:8px 12px; margin-right:8px; cursor:pointer; }
canvas { background:#181818; border:1px solid #555; display:block; }
.ok { color:#44d62c; font-weight:bold; }
.warn { color:#ffb703; font-weight:bold; }
.bad { color:#ff4d4d; font-weight:bold; }
.grid { display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; }
pre { white-space:pre-wrap; min-width:480px; max-width:760px; }
</style>
</head>
<body>
<h2>LiDAR nearest-hit DLL patch verifier</h2>

<div class="panel">
  <b>동일 조건 전후 비교</b>
  <p>
    원본 DLL에서 Baseline 저장 → 패치 DLL로 교체 → 같은 위치·자세·장애물 배치에서
    Patched 저장 → Compare 실행
  </p>
  <button id="baselineButton">Capture baseline</button>
  <button id="patchedButton">Capture patched</button>
  <button id="compareButton">Compare</button>
  <span id="compareResult">대기 중</span>
</div>

<div class="grid">
  <div>
    <div class="panel">
      <b>전방 LiDAR 각도 히트맵</b><br>
      X=차체 상대 수평각, Y=수직각<br>
      빨강 ≤20m, 노랑 ≤50m, 초록 &gt;50m, 회색=미탐지
    </div>
    <canvas id="heatmap" width="980" height="520"></canvas>
  </div>
  <pre id="status">/info LiDAR 프레임 대기 중...</pre>
</div>

<script>
const canvas=document.getElementById('heatmap');
const ctx=canvas.getContext('2d');
const statusEl=document.getElementById('status');
const resultEl=document.getElementById('compareResult');

function xy(angle,vertical) {
  const margin=44;
  return {
    x:margin+(angle+60)/120*(canvas.width-2*margin),
    y:margin+(vertical+22.5)/45*(canvas.height-2*margin)
  };
}
function pointColor(point) {
  if (!point.isDetected) return '#555';
  const distance=Number(point.distance);
  if (distance<=20) return '#ff3030';
  if (distance<=50) return '#ffca3a';
  return '#44d62c';
}
function drawGrid() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.font='12px Arial';
  ctx.strokeStyle='#444';
  ctx.fillStyle='#bbb';

  for (const angle of [-60,-30,0,30,60]) {
    const p=xy(angle,0);
    ctx.beginPath();
    ctx.moveTo(p.x,44);
    ctx.lineTo(p.x,canvas.height-44);
    ctx.stroke();
    ctx.fillText((angle>0?'+':'')+angle+'°',p.x-14,canvas.height-20);
  }

  for (const vertical of [-22.5,-10,0,10,22.5]) {
    const p=xy(0,vertical);
    ctx.beginPath();
    ctx.moveTo(44,p.y);
    ctx.lineTo(canvas.width-44,p.y);
    ctx.stroke();
    ctx.fillText((vertical>0?'+':'')+vertical+'°',4,p.y+4);
  }
}
function draw(status) {
  drawGrid();

  for (const point of (status.heatmapPoints||[])) {
    const p=xy(Number(point.angle),Number(point.verticalAngle));
    ctx.fillStyle=pointColor(point);
    ctx.fillRect(p.x-1,p.y-1,2,2);
  }

  for (const column of (status.candidateColumns||[])) {
    const top=xy(Number(column.angle)-0.45,Number(column.verticalMin));
    const bottom=xy(Number(column.angle)+0.45,Number(column.verticalMax));
    ctx.strokeStyle=
      column.status==='GOOD_NEAREST_SURFACE' ? '#00ffff' : '#ff00ff';
    ctx.lineWidth=2;
    ctx.strokeRect(
      top.x,
      top.y,
      Math.max(3,bottom.x-top.x),
      Math.max(3,bottom.y-top.y)
    );
  }

  let className='warn';
  if (status.status==='PATCH_BEHAVIOR_LOOKS_GOOD') className='ok';
  if (status.status==='POSSIBLE_REAR_GROUND_SELECTION') className='bad';
  statusEl.className=className;

  statusEl.textContent=[
    'behaviorStatus: '+status.status,
    'totalRayCount: '+status.totalRayCount,
    'detectedRayCount: '+status.detectedRayCount,
    'missedRayCount: '+status.missedRayCount,
    'inferredMaxRange: '+status.inferredMaxRange,
    'candidateColumnCount: '+status.candidateColumnCount,
    'foregroundRayCount: '+status.foregroundRayCount,
    'suspectedRearGroundRayCount: '+status.suspectedRearGroundRayCount,
    'suspectedRearGroundRatio: '+status.suspectedRearGroundRatio,
    '',
    '청록 박스: 가까운 표면 연속성이 양호',
    '자홍 박스: 뒤쪽 지면 선택 가능성 확인 필요',
    '',
    status.message||''
  ].join('\n');
}

async function refresh() {
  try {
    const response=await fetch('/lidar_patch_status',{cache:'no-store'});
    draw(await response.json());
  } catch (error) {
    statusEl.textContent=String(error);
  }
}
async function capture(label) {
  resultEl.textContent=label+' 저장 중...';
  const response=await fetch(
    '/capture_lidar_patch_snapshot?label='+label,
    {method:'POST'}
  );
  const payload=await response.json();
  resultEl.className=response.ok?'ok':'bad';
  resultEl.textContent=response.ok
    ? label+' 저장 완료: '+payload.rayCount+' rays'
    : JSON.stringify(payload);
}
async function compareSnapshots() {
  resultEl.textContent='비교 중...';
  const response=await fetch(
    '/compare_lidar_patch_snapshots?baseline=baseline&patched=patched',
    {cache:'no-store'}
  );
  const payload=await response.json();
  resultEl.className=
    payload.status==='PATCH_EFFECT_DETECTED' ? 'ok' : 'warn';
  resultEl.textContent=[
    payload.status,
    'common='+Number(payload.commonRayCount||0),
    'nearer='+Number(payload.changedToNearerCount||0),
    'farther='+Number(payload.changedToFartherCount||0),
    'nearerRatio='+Number(payload.changedToNearerRatio||0).toFixed(5)
  ].join(' | ');
}

document.getElementById('baselineButton').onclick=()=>capture('baseline');
document.getElementById('patchedButton').onclick=()=>capture('patched');
document.getElementById('compareButton').onclick=compareSnapshots;

setInterval(refresh,500);
refresh();
</script>
</body>
</html>
"""


@app.route("/lidar_view", methods=["GET"])
def lidar_view():
    return r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>LiDAR Sensing Point Visualizer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 18px; background: #111; color: #eee; }
    canvas { background: #181818; border: 1px solid #555; }
    .dashboard { display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap; }
    .views { display: flex; flex-direction: column; gap: 18px; }
    .panel-title { font-size: 16px; font-weight: bold; margin: 0 0 6px 0; }
    pre { min-width: 650px; max-width: 960px; white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
    .legend { margin: 4px 0 0 0; font-size: 13px; color: #ddd; }
    .note { max-width: 1480px; color: #ddd; }
  </style>
</head>
<body>
  <h2>LiDAR Sensing Point Visualizer</h2>
  <p class="note">목적: 후처리 객체 판단이 아니라, <b>LiDAR가 실제로 반환한 hit 위치</b>를 확인한다. 지면 관통/뒤 지면 hit 의심 상황을 보기 위해 ground-like, object-like, delayed-ground-suspect를 색으로 분리한다.</p>
  <div class="dashboard">
    <div class="views">
      <div>
        <div class="panel-title">Top view: body-relative hit positions</div>
        <canvas id="topView" width="860" height="860"></canvas>
        <div class="legend">Top: body-forward = up. Points are actual detected LiDAR returns in polar angle/range.</div>
      </div>
      <div>
        <div class="panel-title">Front angular view: azimuth × vertical angle</div>
        <canvas id="frontView" width="860" height="430"></canvas>
        <div class="legend">Front: X=body-relative azimuth, Y=vertical angle. 아래쪽은 positive verticalAngle.</div>
      </div>
      <div>
        <div class="panel-title">Centerline side profile: selected 0° azimuth channel rays</div>
        <canvas id="profileView" width="860" height="520"></canvas>
        <div class="legend">Profile: X=forward horizontal range, Y=height above local ground. Dashed gray means miss if raw miss rays are available.</div>
      </div>
    </div>
    <pre id="status">Waiting for /info data...</pre>
  </div>
<script>
const topCanvas = document.getElementById('topView');
const topCtx = topCanvas.getContext('2d');
const frontCanvas = document.getElementById('frontView');
const frontCtx = frontCanvas.getContext('2d');
const profileCanvas = document.getElementById('profileView');
const profileCtx = profileCanvas.getContext('2d');
const statusText = document.getElementById('status');

const MAX_DISTANCE = 120.0;
const FRONT_HORIZONTAL_LIMIT = 70.0;
const FRONT_VERTICAL_MIN = -22.5;
const FRONT_VERTICAL_MAX = 22.5;
const PROFILE_MAX_DISTANCE = 120.0;
const PROFILE_HEIGHT_MIN = -8.0;
const PROFILE_HEIGHT_MAX = 16.0;

function hitColor(type) {
  if (type === 'ground_like') return '#44d62c';
  if (type === 'object_like') return '#ffb703';
  if (type === 'silhouette_or_object_top') return '#4cc9f0';
  if (type === 'delayed_ground_suspect') return '#ff3030';
  return '#dddddd';
}
function distanceAlpha(distance) {
  if (distance <= 30) return 1.0;
  if (distance <= 70) return 0.85;
  return 0.70;
}
function polar(angleDeg, distance, cx, cy, radius) {
  const a = angleDeg * Math.PI / 180.0;
  const r = Math.min(distance, MAX_DISTANCE) / MAX_DISTANCE * radius;
  return { x: cx + Math.sin(a) * r, y: cy - Math.cos(a) * r };
}
function drawTop(scan) {
  const ctx = topCtx, canvas = topCanvas;
  const w = canvas.width, h = canvas.height, cx = w/2, cy = h/2, radius = Math.min(w,h)*0.46;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#555'; ctx.fillStyle = '#bbb'; ctx.font = '13px Arial';

  for (const d of [30,60,90,120]) {
    const r = radius * d / MAX_DISTANCE;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();
    ctx.fillText(d + ' m', cx + 5, cy - r + 16);
  }
  for (const a of [-90,-60,-30,0,30,60,90]) {
    const p = polar(a, MAX_DISTANCE, cx, cy, radius);
    ctx.strokeStyle = a === 0 ? '#999' : '#333';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p.x, p.y); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.fillText((a>0?'+':'') + a + '°', p.x - 18, p.y + 14);
  }

  const points = scan.topViewPoints || [];
  for (const p of points) {
    const q = polar(p.angle, p.horizontalRange, cx, cy, radius);
    ctx.globalAlpha = distanceAlpha(p.distance);
    ctx.fillStyle = hitColor(p.hitType);
    let size = 2.6;
    if (p.hitType === 'delayed_ground_suspect') size = 4.2;
    if (p.hitType === 'object_like' || p.hitType === 'silhouette_or_object_top') size = 3.3;
    ctx.beginPath(); ctx.arc(q.x, q.y, size, 0, Math.PI*2); ctx.fill();
  }
  ctx.globalAlpha = 1.0;

  ctx.fillStyle = '#4cc9f0'; ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#eee'; ctx.fillText('LiDAR', cx+10, cy-8);
  ctx.fillText('green=ground-like | yellow=object-like | cyan=silhouette/top | red=delayed ground suspect', 18, h-20);
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
  const w = canvas.width, h = canvas.height, margin = 44;
  ctx.clearRect(0,0,w,h);
  ctx.font = '12px Arial';
  ctx.strokeStyle = '#444'; ctx.fillStyle = '#bbb';
  for (const a of [-60,-30,0,30,60]) {
    const p = frontXY(a,0,w,h,margin);
    ctx.beginPath(); ctx.moveTo(p.x, margin); ctx.lineTo(p.x, h-margin); ctx.stroke();
    ctx.fillText((a>0?'+':'')+a+'°', p.x-14, h-margin+18);
  }
  for (const v of [-22.5,-10,0,10,22.5]) {
    const p = frontXY(0,v,w,h,margin);
    ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke();
    ctx.fillText((v>0?'+':'')+v+'°', 5, p.y+4);
  }
  ctx.fillStyle = '#eee';
  ctx.fillText('body-forward 0°', w/2-42, 18);
  ctx.fillText('up / horizon side (-)', 8, margin-12);
  ctx.fillText('down / ground side (+)', 8, h-10);

  for (const p of (scan.frontViewPoints || [])) {
    const q = frontXY(p.angle, p.verticalAngle, w, h, margin);
    ctx.globalAlpha = distanceAlpha(p.distance);
    ctx.fillStyle = hitColor(p.hitType);
    const size = p.hitType === 'delayed_ground_suspect' ? 4.0 : 2.6;
    ctx.beginPath(); ctx.arc(q.x, q.y, size, 0, Math.PI*2); ctx.fill();
  }
  ctx.globalAlpha = 1.0;
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
function drawProfile(scan) {
  const ctx = profileCtx, canvas = profileCanvas;
  const w = canvas.width, h = canvas.height, margin = 54;
  const profile = scan.centerlineProfile || {};
  const sensorHeight = profile.sensorHeightAboveLocalGround == null ? 1.0 : profile.sensorHeightAboveLocalGround;
  const origin = profileXY(0, sensorHeight, w, h, margin);
  ctx.clearRect(0,0,w,h);
  ctx.font='12px Arial';
  ctx.strokeStyle='#444'; ctx.fillStyle='#bbb';

  for (const d of [0,20,40,60,80,100,120]) {
    const p = profileXY(d,0,w,h,margin);
    ctx.beginPath(); ctx.moveTo(p.x, margin); ctx.lineTo(p.x, h-margin); ctx.stroke();
    ctx.fillText(d+'m', p.x-10, h-margin+18);
  }
  for (const y of [-5,0,5,10,15]) {
    const p = profileXY(0,y,w,h,margin);
    ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke();
    ctx.fillText((y>0?'+':'')+y+'m', 8, p.y+4);
  }

  const groundLeft = profileXY(0,0,w,h,margin);
  const groundRight = profileXY(PROFILE_MAX_DISTANCE,0,w,h,margin);
  ctx.strokeStyle='#00cfd5'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(groundLeft.x, groundLeft.y); ctx.lineTo(groundRight.x, groundRight.y); ctx.stroke();
  ctx.lineWidth=1;

  for (const ray of (profile.rays || [])) {
    const range = Math.min(PROFILE_MAX_DISTANCE, ray.horizontalRange || ray.distance || PROFILE_MAX_DISTANCE);
    let endpointHeight;
    if (ray.isDetected && ray.heightAboveLocalGround !== null) {
      endpointHeight = ray.heightAboveLocalGround;
    } else {
      endpointHeight = sensorHeight - range * Math.tan(ray.verticalAngle * Math.PI / 180.0);
    }
    const end = profileXY(range, endpointHeight, w, h, margin);
    ctx.save();
    ctx.globalAlpha = ray.isDetected ? 0.32 : 0.20;
    ctx.strokeStyle = ray.isDetected ? '#44d62c' : '#888';
    if (!ray.isDetected) ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y); ctx.stroke();
    ctx.restore();
    if (ray.isDetected) {
      ctx.fillStyle = '#ffb703';
      ctx.beginPath(); ctx.arc(end.x, end.y, 3.0, 0, Math.PI*2); ctx.fill();
    }
    if (ray.expectedFlatGroundRange !== null && ray.expectedFlatGroundRange <= PROFILE_MAX_DISTANCE) {
      const eg = profileXY(ray.expectedFlatGroundRange, 0, w, h, margin);
      ctx.fillStyle = '#00cfd5';
      ctx.fillRect(eg.x-1, eg.y-4, 2, 8);
    }
  }
  ctx.fillStyle = '#4cc9f0'; ctx.beginPath(); ctx.arc(origin.x, origin.y, 7, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#eee';
  ctx.fillText('LiDAR origin', origin.x+10, origin.y-8);
  ctx.fillText('selected azimuth: ' + (profile.selectedAngle == null ? 'n/a' : profile.selectedAngle + '°') +
    ' | channels: ' + (profile.channelCount || 0) +
    ' | hits: ' + (profile.hitCount || 0) +
    ' | misses: ' + (profile.missCount || 0), margin, 34);
  ctx.fillText('cyan ticks on ground line = expected flat-ground range for each downward ray', margin, 50);
}

function draw(scan) {
  drawTop(scan);
  drawFront(scan);
  drawProfile(scan);

  const counts = scan.hitTypeCounts || {};
  const countLines = Object.keys(counts).sort().map(k => '  ' + k + ': ' + counts[k]);
  statusText.textContent = [
    'simulationTime: ' + scan.simulationTime,
    'receivedAt: ' + scan.receivedAt,
    'rawRayCount: ' + scan.rawRayCount,
    'rawDetectedPointCount: ' + scan.rawDetectedPointCount,
    'topViewPointCount: ' + ((scan.topViewPoints || []).length),
    'frontViewPointCount: ' + ((scan.frontViewPoints || []).length),
    'estimatedGroundY: ' + scan.estimatedGroundY,
    'localGroundY: ' + scan.localGroundY,
    'lidarOriginY: ' + scan.lidarOriginY,
    '',
    'Hit type counts:', ...(countLines.length ? countLines : ['  none']),
    '',
    'Interpretation:',
    '  ground_like = local ground 근처 hit',
    '  object_like = local ground보다 높은 hit',
    '  silhouette_or_object_top = 지평선/상단 쪽 object-like hit',
    '  delayed_ground_suspect = 예상 평지 hit 지점보다 훨씬 뒤에서 잡힌 ground-like hit',
    '',
    'APIs:',
    '  /lidar_status',
    '  /raw_lidar_status',
    '  /export_snapshot?label=test'
  ].join('\n');
}

async function refresh() {
  try {
    const r = await fetch('/lidar_status', {cache: 'no-store'});
    const scan = await r.json();
    if (!scan || !scan.rawRayCount) {
      statusText.textContent = 'Waiting for /info data...';
      return;
    }
    draw(scan);
  } catch (e) {
    statusText.textContent = String(e);
  }
}
setInterval(refresh, 200);
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Patch verifier: http://127.0.0.1:5000/lidar_patch_view")
    print("Open LiDAR sensing visualizer: http://127.0.0.1:5000/lidar_view")
    print("JSON status: http://127.0.0.1:5000/lidar_status")
    print("Raw JSON: http://127.0.0.1:5000/raw_lidar_status")
    print("Export snapshot: http://127.0.0.1:5000/export_snapshot?label=test")
    app.run(host="0.0.0.0", port=5000, threaded=True)
