# -*- coding: utf-8 -*-
"""Live LiDAR/YOLO tactical context storage for the RAG dashboard.

This module keeps a compact ChromaDB collection with the latest battlefield
snapshot from the Tank Challenge LiDAR/YOLO server. It is intentionally
defensive: if the live server, ChromaDB, or embedding model is unavailable, the
caller receives a structured error and the main RAG flow can continue.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_CHROMA_DIR = os.path.join(ROOT, "rag_decision_support", "chroma_store")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TACTICAL_COLLECTION = "tank_tactical_context"
DEFAULT_SOURCE_BASE = "http://127.0.0.1:5000"
_MODEL_CACHE: dict[str, Any] = {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _fetch_json(url: str, timeout: float = 1.2) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_live_payloads(source_base: str = DEFAULT_SOURCE_BASE, timeout: float = 0.35) -> dict:
    base = source_base.rstrip("/")
    payloads: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in ("fire_status", "lidar_status", "vision_status", "fusion_status", "aim_status", "action_debug"):
        try:
            payloads[name] = _fetch_json(f"{base}/{name}", timeout=timeout)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {"source_base": base, "payloads": payloads, "errors": errors}


def _find_lists(value: Any, key_names: set[str]) -> list[list]:
    found: list[list] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in key_names and isinstance(child, list):
                found.append(child)
            else:
                found.extend(_find_lists(child, key_names))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_lists(child, key_names))
    return found


def _find_first_key(value: Any, key_names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in key_names:
                return child
        for child in value.values():
            found = _find_first_key(child, key_names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first_key(child, key_names)
            if found is not None:
                return found
    return None


def _extract_xyz(raw: Any) -> dict | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        x = _safe_float(raw[0])
        y = _safe_float(raw[1], 0.0)
        z = _safe_float(raw[2])
        if x is None or z is None:
            return None
        return {"x": x, "y": y or 0.0, "z": z}
    if not isinstance(raw, dict):
        return None
    nested = raw.get("position") or raw.get("worldPosition") or raw.get("world_position")
    if isinstance(nested, dict):
        nested_result = _extract_xyz(nested)
        if nested_result is not None:
            return nested_result
    x = _safe_float(raw.get("x", raw.get("X")))
    z = _safe_float(raw.get("z", raw.get("Z")))
    y = _safe_float(raw.get("y", raw.get("Y")), 0.0)
    if x is None or z is None:
        return None
    return {"x": x, "y": y or 0.0, "z": z}


def _extract_player_pose(payloads: dict) -> dict:
    position_raw = None
    yaw_raw = None
    for payload in payloads.values():
        if position_raw is None:
            position_raw = _find_first_key(payload, {"playerPos", "player_position", "latestPlayer", "latestPlayerPosition", "latest_player"})
        if yaw_raw is None:
            yaw_raw = _find_first_key(payload, {"playerBodyX", "bodyYawDeg", "body_yaw", "playerHeading", "headingDeg", "turretBodyYawDeg"})
    position = _extract_xyz(position_raw)
    if position is None and isinstance(position_raw, dict):
        nested_position = position_raw.get("position")
        position = _extract_xyz(nested_position)
    return {
        "position": position,
        "body_yaw_deg": _safe_float(yaw_raw),
    }


def _item_distance(item: dict) -> float | None:
    cluster = item.get("lidarCluster") if isinstance(item.get("lidarCluster"), dict) else {}
    for key in ("distanceM", "distance", "target_distance", "surfaceDistanceM", "medianDistanceM"):
        value = _safe_float(item.get(key))
        if value is not None:
            return value
    for key in ("distanceM", "surfaceDistanceM", "medianDistanceM"):
        value = _safe_float(cluster.get(key))
        if value is not None:
            return value
    return None


def _item_angle(item: dict) -> float | None:
    cluster = item.get("lidarCluster") if isinstance(item.get("lidarCluster"), dict) else {}
    for key in ("angleDeg", "lidarBodyAngleDeg", "bodyRelativeAngleDeg", "target_angle"):
        value = _safe_float(item.get(key))
        if value is not None:
            return value
    for key in ("angleDeg", "lidarBodyAngleDeg"):
        value = _safe_float(cluster.get(key))
        if value is not None:
            return value
    return None


def _item_label(item: dict) -> str:
    for key in ("semanticClass", "className", "label", "name", "candidateLabel", "target_type"):
        value = item.get(key)
        if value:
            return str(value)
    return "object"


def _is_enemy_like(item: dict) -> bool:
    text = json.dumps(item, ensure_ascii=False).lower()
    return any(token in text for token in ("enemy", "tank_enemy", "enemy_tank", "tank"))


def _compact_object(item: dict) -> dict:
    return {
        "label": _item_label(item),
        "distanceM": _item_distance(item),
        "angleDeg": _item_angle(item),
        "confidence": _safe_float(item.get("confidence")),
        "fusionMatched": bool(item.get("fusionMatched", False)),
    }


def summarize_tactical_payloads(live: dict) -> dict:
    payloads = live.get("payloads") or {}
    all_fused: list[dict] = []
    for payload in payloads.values():
        for rows in _find_lists(payload, {"latestFusedObjects", "lastYoloFusedObjects", "targets"}):
            all_fused.extend(item for item in rows if isinstance(item, dict))

    all_clusters: list[dict] = []
    for payload in payloads.values():
        for rows in _find_lists(payload, {"lidarClusters", "validClusters", "clusters"}):
            all_clusters.extend(item for item in rows if isinstance(item, dict))

    enemies = [_compact_object(item) for item in all_fused if _is_enemy_like(item)]
    if not enemies:
        enemies = [_compact_object(item) for item in all_fused[:4]]
    enemies = [item for item in enemies if item.get("distanceM") is not None or item.get("angleDeg") is not None]
    enemies.sort(key=lambda item: (_safe_float(item.get("distanceM"), 9999.0) or 9999.0, abs(_safe_float(item.get("angleDeg"), 0.0) or 0.0)))

    obstacles = [_compact_object(item) for item in all_clusters]
    obstacles = [item for item in obstacles if item.get("distanceM") is not None or item.get("angleDeg") is not None]
    obstacles.sort(key=lambda item: (_safe_float(item.get("distanceM"), 9999.0) or 9999.0, abs(_safe_float(item.get("angleDeg"), 0.0) or 0.0)))

    fire = payloads.get("fire_status") if isinstance(payloads.get("fire_status"), dict) else {}
    lidar = payloads.get("lidar_status") if isinstance(payloads.get("lidar_status"), dict) else {}
    vision = payloads.get("vision_status") if isinstance(payloads.get("vision_status"), dict) else {}

    front_obstacles = [
        item for item in obstacles
        if (_safe_float(item.get("distanceM"), 9999.0) or 9999.0) <= 35.0
        and abs(_safe_float(item.get("angleDeg"), 999.0) or 999.0) <= 15.0
    ]

    return {
        "source_base": live.get("source_base"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "available_payloads": sorted(payloads.keys()),
        "errors": live.get("errors") or {},
        "player_pose": _extract_player_pose(payloads),
        "enemy_count": len(enemies),
        "nearest_enemy": enemies[0] if enemies else None,
        "enemies": enemies[:6],
        "obstacle_count": len(obstacles),
        "front_blocked": bool(front_obstacles),
        "front_obstacles": front_obstacles[:8],
        "nearest_obstacles": obstacles[:10],
        "fire_status": {
            "fire": fire.get("fire"),
            "can_fire": fire.get("can_fire"),
            "decision": fire.get("decision"),
            "reason": fire.get("reason"),
            "distance": fire.get("distance") or fire.get("distance_fire") or fire.get("target_distance"),
            "body_error": fire.get("body_error") or fire.get("body_error_fire"),
            "turret_error": fire.get("turret_error") or fire.get("turret_error_fire"),
            "pitch_error": fire.get("pitch_error") or fire.get("pitch_error_fire"),
        },
        "counts": {
            "obstaclePointCount": _safe_int(lidar.get("obstaclePointCount")),
            "validObjectCount": _safe_int(lidar.get("validObjectCount")),
            "latestFusedObjectCount": _safe_int(vision.get("latestFusedObjectCount")),
        },
    }


def tactical_document(summary: dict) -> str:
    nearest_enemy = summary.get("nearest_enemy") or {}
    enemy_text = "none"
    if nearest_enemy:
        enemy_text = (
            f"{nearest_enemy.get('label')} at {nearest_enemy.get('distanceM')}m, "
            f"{nearest_enemy.get('angleDeg')}deg"
        )
    obstacle_lines = [
        f"{item.get('label')} at {item.get('distanceM')}m, {item.get('angleDeg')}deg"
        for item in summary.get("front_obstacles", [])[:5]
    ]
    fire = summary.get("fire_status") or {}
    return "\n".join(
        [
            "[Live Tactical Context]",
            f"timestamp: {summary.get('timestamp')}",
            f"available_payloads: {', '.join(summary.get('available_payloads', []))}",
            f"enemy_count: {summary.get('enemy_count')}",
            f"nearest_enemy: {enemy_text}",
            f"obstacle_count: {summary.get('obstacle_count')}",
            f"front_blocked: {summary.get('front_blocked')}",
            "front_obstacles: " + ("; ".join(obstacle_lines) if obstacle_lines else "none"),
            (
                "fire_status: "
                f"fire={fire.get('fire')}, can_fire={fire.get('can_fire')}, "
                f"decision={fire.get('decision')}, reason={fire.get('reason')}"
            ),
        ]
    )


def _embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _collection(persist_dir: str = DEFAULT_CHROMA_DIR):
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(TACTICAL_COLLECTION)


def upsert_tactical_summary(summary: dict, persist_dir: str = DEFAULT_CHROMA_DIR, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    document = tactical_document(summary)
    metadata = {
        "source": "lidar_yolo",
        "timestamp": str(summary.get("timestamp") or ""),
        "enemy_count": int(summary.get("enemy_count") or 0),
        "obstacle_count": int(summary.get("obstacle_count") or 0),
        "front_blocked": bool(summary.get("front_blocked")),
        "nearest_enemy_distance": _safe_float((summary.get("nearest_enemy") or {}).get("distanceM"), 0.0) or 0.0,
        "nearest_enemy_angle": _safe_float((summary.get("nearest_enemy") or {}).get("angleDeg"), 0.0) or 0.0,
    }
    model = _embedding_model(model_name)
    embedding = model.encode([document], show_progress_bar=False)[0].tolist()
    collection = _collection(persist_dir)
    collection.upsert(
        ids=["live_latest"],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    return {"status": "success", "id": "live_latest", "metadata": metadata, "document": document}


def ingest_live_tactical_context(source_base: str = DEFAULT_SOURCE_BASE) -> dict:
    live = fetch_live_payloads(source_base=source_base)
    summary = summarize_tactical_payloads(live)
    if not summary.get("available_payloads"):
        return {"status": "unavailable", "summary": summary, "error": "no live LiDAR/YOLO payloads were reachable"}
    try:
        stored = upsert_tactical_summary(summary)
    except Exception as exc:
        return {"status": "store_failed", "summary": summary, "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "success", "summary": summary, "stored": stored}


def search_tactical_context(text: str, top_k: int = 3, persist_dir: str = DEFAULT_CHROMA_DIR, model_name: str = DEFAULT_MODEL_NAME) -> list[dict]:
    try:
        model = _embedding_model(model_name)
        query_embedding = model.encode([text], show_progress_bar=False)[0].tolist()
        result = _collection(persist_dir).query(query_embeddings=[query_embedding], n_results=max(1, top_k))
    except Exception:
        return []

    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    rows = []
    for doc_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
        rows.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": metadata or {},
                "score": 1.0 / (1.0 + float(distance or 0.0)),
            }
        )
    return rows


def _split_side_obstacles(summary: dict) -> tuple[list[dict], list[dict]]:
    left: list[dict] = []
    right: list[dict] = []
    for item in summary.get("front_obstacles", []) or []:
        angle = _safe_float(item.get("angleDeg"), 0.0) or 0.0
        if angle < 0:
            left.append(item)
        else:
            right.append(item)
    return left, right


def _side_clearance_score(items: list[dict]) -> float:
    if not items:
        return 1000.0
    nearest = min((_safe_float(item.get("distanceM"), 999.0) or 999.0) for item in items)
    return nearest - 4.0 * len(items)


def _relative_destination(summary: dict, relative_angle_deg: float, distance_m: float) -> dict:
    player_pose = summary.get("player_pose") or {}
    position = player_pose.get("position") if isinstance(player_pose.get("position"), dict) else None
    body_yaw = _safe_float(player_pose.get("body_yaw_deg"))
    result = {
        "relative_angle_deg": round(float(relative_angle_deg), 3),
        "relative_distance_m": round(float(distance_m), 3),
        "world": None,
        "basis": "body_relative",
    }
    if position is None or body_yaw is None:
        result["note"] = "World coordinate unavailable because live player position or body yaw was not found."
        return result

    world_yaw = math.radians(float(body_yaw) + float(relative_angle_deg))
    x = float(position["x"]) + float(distance_m) * math.sin(world_yaw)
    z = float(position["z"]) + float(distance_m) * math.cos(world_yaw)
    result["world"] = {
        "x": round(x, 3),
        "y": round(float(position.get("y", 0.0) or 0.0), 3),
        "z": round(z, 3),
    }
    result["player_pose"] = {
        "x": round(float(position["x"]), 3),
        "y": round(float(position.get("y", 0.0) or 0.0), 3),
        "z": round(float(position["z"]), 3),
        "body_yaw_deg": round(float(body_yaw), 3),
    }
    return result


def _safe_destination_for_action(action: str, summary: dict) -> dict | None:
    if action == "MOVE_LEFT":
        return _relative_destination(summary, -45.0, 12.0)
    if action == "MOVE_RIGHT":
        return _relative_destination(summary, 45.0, 12.0)
    if action == "REVERSE":
        return _relative_destination(summary, 180.0, 8.0)
    if action == "SCAN":
        return _relative_destination(summary, 0.0, 0.0)
    return None


def recommend_mobility_action(summary: dict, shot_recommendation: dict | None = None, query: dict | None = None) -> dict:
    """Return a high-level driving/fire action from tactical context and RAG fire advice.

    This is intentionally conservative and does not directly actuate the tank.
    A downstream controller can map `control_hint` to its own `/get_action`
    command format.
    """
    shot_recommendation = shot_recommendation or {}
    query = query or {}
    available = bool(summary.get("available_payloads"))
    enemy_count = int(summary.get("enemy_count") or 0)
    nearest_enemy = summary.get("nearest_enemy") or {}
    front_blocked = bool(summary.get("front_blocked"))
    fire_status = summary.get("fire_status") or {}

    turret_error = _safe_float(query.get("turret_error"), _safe_float(fire_status.get("turret_error"), 0.0)) or 0.0
    pitch_error = _safe_float(query.get("pitch_error"), _safe_float(fire_status.get("pitch_error"), 0.0)) or 0.0
    body_error = _safe_float(query.get("body_error"), _safe_float(fire_status.get("body_error"), 0.0)) or 0.0
    shot_fire = bool(shot_recommendation.get("fire", False))

    reasons: list[str] = []
    control_hint = {"move": "STOP", "turn": "STRAIGHT", "fire": False}

    if not available:
        return {
            "action": "HOLD_AIM",
            "confidence": 0.2,
            "reason": "Live LiDAR/YOLO context is unavailable, so the system should not drive aggressively.",
            "reasons": ["NO_LIVE_CONTEXT"],
            "control_hint": control_hint,
            "safe_destination": None,
            "enemy_count": 0,
            "nearest_enemy": None,
            "front_blocked": None,
        }

    if front_blocked:
        reasons.append("FRONT_BLOCKED")
    if enemy_count <= 0:
        reasons.append("NO_CONFIRMED_ENEMY")
    if abs(turret_error) > 1.2:
        reasons.append("TURRET_NOT_ALIGNED")
    if abs(pitch_error) > 0.35:
        reasons.append("PITCH_NOT_STABLE")
    if abs(body_error) > 20.0:
        reasons.append("BODY_ALIGNMENT_RISK")

    if shot_fire and enemy_count > 0 and not front_blocked and abs(turret_error) <= 1.2 and abs(pitch_error) <= 0.35:
        return {
            "action": "FIRE",
            "confidence": max(0.55, float(shot_recommendation.get("confidence", 0.55) or 0.55)),
            "reason": "Historical RAG recommends fire, a live enemy is present, and the front firing lane is not blocked.",
            "reasons": ["RAG_FIRE", "ENEMY_PRESENT", "LANE_CLEAR"],
            "control_hint": {"move": "STOP", "turn": "STRAIGHT", "fire": True},
            "safe_destination": None,
            "enemy_count": enemy_count,
            "nearest_enemy": nearest_enemy,
            "front_blocked": False,
        }

    if front_blocked:
        left, right = _split_side_obstacles(summary)
        left_score = _side_clearance_score(left)
        right_score = _side_clearance_score(right)
        if left_score >= right_score:
            action = "MOVE_LEFT"
            control_hint = {"move": "FORWARD", "turn": "LEFT", "fire": False}
            side_reason = "left side appears less blocked than right side"
        else:
            action = "MOVE_RIGHT"
            control_hint = {"move": "FORWARD", "turn": "RIGHT", "fire": False}
            side_reason = "right side appears less blocked than left side"
        nearest_front = (summary.get("front_obstacles") or [{}])[0]
        nearest_dist = _safe_float(nearest_front.get("distanceM"), 999.0) or 999.0
        if nearest_dist <= 8.0:
            action = "REVERSE"
            control_hint = {"move": "BACKWARD", "turn": "STRAIGHT", "fire": False}
            side_reason = "nearest front obstacle is too close, reverse before steering"
        return {
            "action": action,
            "confidence": 0.72,
            "reason": f"Front lane is blocked; {side_reason}.",
            "reasons": reasons,
            "control_hint": control_hint,
            "safe_destination": _safe_destination_for_action(action, summary),
            "enemy_count": enemy_count,
            "nearest_enemy": nearest_enemy or None,
            "front_blocked": True,
            "front_obstacles": summary.get("front_obstacles", [])[:5],
        }

    if enemy_count <= 0:
        return {
            "action": "SCAN",
            "confidence": 0.5,
            "reason": "No live enemy-like object is confirmed; scan or continue cautious navigation.",
            "reasons": reasons,
            "control_hint": {"move": "STOP", "turn": "LEFT", "fire": False},
            "safe_destination": _safe_destination_for_action("SCAN", summary),
            "enemy_count": 0,
            "nearest_enemy": None,
            "front_blocked": False,
        }

    if abs(turret_error) > 1.2 or abs(pitch_error) > 0.35:
        turn = "LEFT" if turret_error < 0 else "RIGHT"
        return {
            "action": "HOLD_AIM",
            "confidence": 0.68,
            "reason": "Enemy is present but aim is not stable enough for firing.",
            "reasons": reasons,
            "control_hint": {"move": "STOP", "turn": turn, "fire": False},
            "safe_destination": None,
            "enemy_count": enemy_count,
            "nearest_enemy": nearest_enemy,
            "front_blocked": False,
        }

    return {
        "action": "HOLD_AIM",
        "confidence": float(shot_recommendation.get("confidence", 0.5) or 0.5),
        "reason": "Enemy is present and lane is clear, but RAG did not recommend immediate fire.",
        "reasons": reasons or ["RAG_HOLD"],
        "control_hint": control_hint,
        "safe_destination": None,
        "enemy_count": enemy_count,
        "nearest_enemy": nearest_enemy,
        "front_blocked": False,
    }
