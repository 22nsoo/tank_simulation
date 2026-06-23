import csv
import json
from time import time

from config import (
    LIDAR_DIR,
    LIDAR_NEAR_DISTANCE,
    LIDAR_MAX_DISTANCE,
    LIDAR_POINT_MARGIN,
    LIDAR_MAX_OBSTACLES,
    LIDAR_API_STALE_SECONDS,
    MAP_X_MIN,
    MAP_X_MAX,
    MAP_Z_MIN,
    MAP_Z_MAX,
)

from geometry import distance_2d


def latest_lidar_file():
    """CSV/JSON 저장 방식으로 들어온 LiDAR 파일 중 가장 최신 파일을 찾습니다."""
    if not LIDAR_DIR.exists():
        return None

    files = list(LIDAR_DIR.glob("*.csv")) + list(LIDAR_DIR.glob("*.json"))

    if not files:
        return None

    return max(files, key=lambda path: path.stat().st_mtime)


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def empty_lidar_summary(source="none"):
    return {
        "source": source,
        "file": None,
        "mtime": 0.0,
        "point_count": 0,
        "detected_count": 0,
        "near_count": 0,
        "min_distance": None,
        "obstacle_count": 0,
        "quality": {
            "grade": "no_data",
            "detection_rate": 0.0,
            "mean_distance": None,
            "p50_distance": None,
            "p90_distance": None,
            "front_min_distance": None,
            "front_near_count": 0,
            "angle_coverage": 0,
            "channel_count": 0,
            "vertical_channel_count": 0,
            "stale_seconds": None,
        },
    }


def lidar_raw_items(data):
    raw_points = (
        data.get("points")
        or data.get("lidarData")
        or data.get("data")
        or data.get("ranges")
        or []
    )

    if isinstance(raw_points, dict):
        raw_points = raw_points.get("points") or raw_points.get("data") or []

    return raw_points if isinstance(raw_points, list) else []


def normalize_lidar_points(data):
    """실시간 API LiDAR 입력을 내부 공통 포맷(x, z, distance)으로 정리합니다."""
    raw_points = lidar_raw_items(data)

    points = []

    for item in raw_points:
        if isinstance(item, dict):
            detected = item.get("isDetected", item.get("detected", True))

            if not parse_bool(detected):
                continue

            try:
                distance = float(item.get("distance", item.get("range", 0)))
                position = item.get("position") if isinstance(item.get("position"), dict) else {}
                x = float(item.get("x", position.get("x")))
                z = float(item.get("z", position.get("z")))

            except (KeyError, TypeError, ValueError):
                continue

        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                distance = float(item[2])
                x = float(item[3])
                z = float(item[5] if len(item) > 5 else item[4])

            except (TypeError, ValueError):
                continue

        else:
            continue

        point = {
            "x": x,
            "z": z,
            "distance": distance,
        }

        for key in ("angle", "verticalAngle", "channelIndex"):
            if isinstance(item, dict) and item.get(key) is not None:
                try:
                    point[key] = float(item[key])
                except (TypeError, ValueError):
                    point[key] = item[key]

        points.append(point)

    return points


def percentile(values, ratio):
    if not values:
        return None

    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * ratio))),
    )

    return ordered[index]


def analyze_lidar_quality(points, raw_count=None, source="none", mtime=0.0):
    raw_count = raw_count if raw_count is not None else len(points)
    detected_count = len(points)

    distances = [
        point["distance"]
        for point in points
        if point.get("distance") is not None
    ]

    near_count = sum(
        1
        for value in distances
        if value <= LIDAR_NEAR_DISTANCE
    )

    detection_rate = detected_count / raw_count if raw_count else 0.0

    angles = [
        point.get("angle")
        for point in points
        if point.get("angle") is not None
    ]

    verticals = [
        point.get("verticalAngle")
        for point in points
        if point.get("verticalAngle") is not None
    ]

    channels = [
        point.get("channelIndex")
        for point in points
        if point.get("channelIndex") is not None
    ]

    front_distances = [
        point["distance"]
        for point in points
        if point.get("angle") is not None and abs(float(point["angle"])) <= 30.0
    ]

    stale_seconds = max(0.0, time() - mtime) if mtime else None
    mean_distance = sum(distances) / len(distances) if distances else None

    if not points:
        grade = "no_data"
    elif stale_seconds is not None and stale_seconds > LIDAR_API_STALE_SECONDS * 2 and source == "api":
        grade = "stale"
    elif detected_count < 30 or detection_rate < 0.15:
        grade = "weak"
    elif near_count > 0 and len(set(channels)) >= 2:
        grade = "good"
    else:
        grade = "ok"

    return {
        "grade": grade,
        "raw_count": raw_count,
        "detection_rate": round(detection_rate, 3),
        "mean_distance": round(mean_distance, 3) if mean_distance is not None else None,
        "p50_distance": percentile(distances, 0.5),
        "p90_distance": percentile(distances, 0.9),
        "front_min_distance": min(front_distances) if front_distances else None,
        "front_near_count": sum(
            1
            for value in front_distances
            if value <= LIDAR_NEAR_DISTANCE
        ),
        "angle_coverage": len(
            set(round(float(angle), 1) for angle in angles)
        ),
        "channel_count": len(set(channels)),
        "vertical_channel_count": len(
            set(round(float(value), 1) for value in verticals)
        ),
        "stale_seconds": round(stale_seconds, 2) if stale_seconds is not None else None,
    }


def points_to_obstacles(points, pos_x=None, pos_z=None):
    """LiDAR 점군 중 가까운 점만 작은 장애물 박스로 변환합니다."""
    filtered = []
    min_distance = None

    for point in points:
        distance = point["distance"]
        x = point["x"]
        z = point["z"]

        min_distance = distance if min_distance is None else min(min_distance, distance)

        if distance > LIDAR_MAX_DISTANCE:
            continue

        if pos_x is not None and pos_z is not None:
            if distance_2d(pos_x, pos_z, x, z) > LIDAR_NEAR_DISTANCE:
                continue

        filtered.append((x, z, distance))

    filtered.sort(key=lambda item: item[2])

    obstacles_from_lidar = []

    for x, z, _ in filtered[:LIDAR_MAX_OBSTACLES]:
        obstacles_from_lidar.append({
            "x_min": max(MAP_X_MIN, x - LIDAR_POINT_MARGIN),
            "x_max": min(MAP_X_MAX, x + LIDAR_POINT_MARGIN),
            "z_min": max(MAP_Z_MIN, z - LIDAR_POINT_MARGIN),
            "z_max": min(MAP_Z_MAX, z + LIDAR_POINT_MARGIN),
            "source": "lidar",
        })

    return obstacles_from_lidar, len(filtered), min_distance


def read_lidar_points_from_file(path):
    """
    최신 LiDAR CSV/JSON 파일을 읽어서
    points, point_count, detected_count를 반환합니다.
    """
    points = []
    point_count = 0
    detected_count = 0

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_items = lidar_raw_items(payload)
        point_count = len(raw_items)
        points = normalize_lidar_points(payload)
        detected_count = len(points)

        return points, point_count, detected_count

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            point_count += 1

            if not parse_bool(row.get("isDetected", "")):
                continue

            try:
                distance = float(row["distance"])
                x = float(row["x"])
                z = float(row["z"])

            except (KeyError, TypeError, ValueError):
                continue

            detected_count += 1

            point = {
                "x": x,
                "z": z,
                "distance": distance,
            }

            for key in ("angle", "verticalAngle", "channelIndex"):
                if row.get(key) is not None:
                    try:
                        point[key] = float(row[key])
                    except (TypeError, ValueError):
                        point[key] = row[key]

            points.append(point)

    return points, point_count, detected_count