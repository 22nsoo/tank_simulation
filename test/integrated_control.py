# -*- coding: utf-8 -*-
"""Integrated drive/fire controller for the LiDAR fusion server.

The LiDAR/YOLO server owns perception.  This bridge converts that perception
into simulator /get_action commands:
- local-planner style waypoint driving via rag_decision_support.drive_planner
- ballistic aiming/firing via shot_analysis.fire_logic
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SHOT_DIR = PROJECT_ROOT / "shot_analysis"
for path in (PROJECT_ROOT, SHOT_DIR):
    token = str(path)
    if token not in sys.path:
        sys.path.insert(0, token)

from rag_decision_support.drive_planner import plan_drive


def _load_fire_logic():
    module = importlib.import_module("fire_logic")
    # We use this module as a library, not as its standalone obstacle-test app.
    if hasattr(module, "OBSTACLE_TARGET_MODE"):
        module.OBSTACLE_TARGET_MODE = False
    return module


try:
    fire_logic = _load_fire_logic()
    FIRE_LOGIC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - reported through /control_status.
    fire_logic = None
    FIRE_LOGIC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def _empty_action() -> dict[str, Any]:
    return {
        "moveWS": {"command": "", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


def _copy_axis(action: dict[str, Any], key: str) -> dict[str, Any]:
    value = (action or {}).get(key)
    if not isinstance(value, dict):
        return {"command": "", "weight": 0.0}
    return {
        "command": str(value.get("command") or ""),
        "weight": max(0.0, min(1.0, float(value.get("weight") or 0.0))),
    }


def _extract_position(raw: Any) -> dict[str, float] | None:
    if isinstance(raw, dict):
        x = _safe_float(raw.get("x"))
        y = _safe_float(raw.get("y"), 0.0)
        z = _safe_float(raw.get("z"))
        if x is not None and z is not None:
            return {"x": x, "y": y or 0.0, "z": z}
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        x = _safe_float(raw[0])
        y = _safe_float(raw[1], 0.0)
        z = _safe_float(raw[2])
        if x is not None and z is not None:
            return {"x": x, "y": y or 0.0, "z": z}
    return None


def _extract_destination(data: dict[str, Any]) -> dict[str, float]:
    if "destination" in data and isinstance(data["destination"], str):
        x, y, z = [float(part.strip()) for part in data["destination"].split(",")[:3]]
        return {"x": x, "y": y, "z": z}
    if isinstance(data.get("destination"), dict):
        raw = data["destination"]
    else:
        raw = data
    x = _safe_float(raw.get("x"))
    y = _safe_float(raw.get("y"), 0.0)
    z = _safe_float(raw.get("z"))
    if x is None or z is None:
        raise ValueError("destination requires x and z")
    return {"x": x, "y": y or 0.0, "z": z}


def _body_yaw_from(body: dict[str, Any], cache: Any, fallback: float = 0.0) -> float:
    for key in ("playerBodyX", "bodyYawDeg", "body_yaw_deg", "headingDeg"):
        value = _safe_float(body.get(key), None)
        if value is not None:
            return value
    pose = getattr(cache, "pose", {}) or {}
    for key in ("playerBodyX", "bodyYawDeg", "headingDeg"):
        value = _safe_float(pose.get(key), None)
        if value is not None:
            return value
    return fallback


def _compact_obstacle(cluster: dict[str, Any]) -> dict[str, Any] | None:
    angle = _safe_float(cluster.get("angleDeg"), None)
    distance = _safe_float(cluster.get("distanceM"), None)
    if angle is None or distance is None:
        return None
    radius = max(
        1.0,
        0.5 * max(
            _safe_float(cluster.get("visibleWidthM"), 3.0) or 3.0,
            _safe_float(cluster.get("depthSpanM"), 3.0) or 3.0,
        ),
    )
    return {
        "label": cluster.get("candidateLabel") or cluster.get("clusterId") or "lidar_object",
        "distanceM": round(distance, 3),
        "angleDeg": round(angle, 3),
        "radiusM": round(radius, 3),
        "pointCount": int(cluster.get("pointCount", 0) or 0),
    }


def _target_world_position(target: dict[str, Any]) -> dict[str, float] | None:
    world = target.get("world") if isinstance(target, dict) else {}
    if not isinstance(world, dict):
        return None
    for key in ("aimPoint", "surfaceCenter", "center"):
        pos = _extract_position(world.get(key))
        if pos is not None:
            return pos
    return None


class IntegratedDriveFireController:
    def __init__(self) -> None:
        self.drive_enabled = True
        self.fire_enabled = True
        self.engage_pauses_drive = True
        self.destination: dict[str, float] | None = None
        self.drive_state: dict[str, Any] = {}
        self.last_plan: dict[str, Any] = {}
        self.last_fire_action: dict[str, Any] = _empty_action()
        self.last_debug: dict[str, Any] = {"status": "boot"}
        self._last_position: dict[str, float] | None = None
        self._last_position_t: float | None = None

    def set_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.destination = _extract_destination(payload)
        self.drive_state = {}
        return {"status": "OK", "destination": self.destination, "driveEnabled": self.drive_enabled}

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "driveEnabled" in payload:
            self.drive_enabled = bool(payload.get("driveEnabled"))
        if "fireEnabled" in payload:
            self.fire_enabled = bool(payload.get("fireEnabled"))
        if "engagePausesDrive" in payload:
            self.engage_pauses_drive = bool(payload.get("engagePausesDrive"))
        if payload.get("clearDestination"):
            self.destination = None
            self.drive_state = {}
        return self.status()

    def _estimate_speed(self, position: dict[str, float]) -> float | None:
        now = time.monotonic()
        if self._last_position is None or self._last_position_t is None:
            self._last_position = dict(position)
            self._last_position_t = now
            return None
        dt = max(1e-3, now - self._last_position_t)
        speed = math.hypot(
            position["x"] - self._last_position["x"],
            position["z"] - self._last_position["z"],
        ) / dt
        self._last_position = dict(position)
        self._last_position_t = now
        return max(0.0, min(25.0, speed))

    def _drive_action(
        self,
        position: dict[str, float] | None,
        body_yaw_deg: float,
        clusters: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.drive_enabled or self.destination is None or position is None:
            self.last_plan = {
                "available": False,
                "control_mode": "idle",
                "reason": "drive disabled or destination not set",
                "command": {"move_ws": "", "move_ad": "", "move_weight": 0.0, "turn_weight": 0.0},
                "waypoints": [],
            }
            return _empty_action()

        obstacles = []
        for cluster in clusters or ():
            item = _compact_obstacle(cluster)
            if item is not None:
                obstacles.append(item)
        front_obstacles = [
            item for item in obstacles
            if item["distanceM"] <= 35.0 and abs(item["angleDeg"]) <= 25.0
        ]
        summary = {
            "player_pose": {"position": position, "body_yaw_deg": body_yaw_deg},
            "front_obstacles": front_obstacles,
            "nearest_obstacles": obstacles[:10],
        }
        tactical_action = {
            "action": "MOVE_TO_GOAL",
            "safe_destination": {
                "world": self.destination,
                "relative_angle_deg": 0.0,
                "relative_distance_m": 0.0,
            },
        }
        speed_ms = self._estimate_speed(position)
        plan = plan_drive(summary, tactical_action, previous_state=self.drive_state, speed_ms=speed_ms)
        self.last_plan = plan
        self.drive_state = plan.get("state") or self.drive_state
        cmd = plan.get("command") or {}
        action = _empty_action()
        action["moveWS"] = {
            "command": str(cmd.get("move_ws") or ""),
            "weight": float(cmd.get("move_weight") or 0.0),
        }
        action["moveAD"] = {
            "command": str(cmd.get("move_ad") or ""),
            "weight": float(cmd.get("turn_weight") or 0.0),
        }
        return action

    def _fire_action(
        self,
        body: dict[str, Any],
        cache: Any,
        turret_state: dict[str, Any],
        position: dict[str, float] | None,
        body_yaw_deg: float,
        targets: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.fire_enabled:
            return _empty_action(), {"available": False, "reason": "fire disabled"}
        if fire_logic is None:
            return _empty_action(), {"available": False, "reason": FIRE_LOGIC_IMPORT_ERROR}
        if position is None:
            return _empty_action(), {"available": False, "reason": "missing player position"}
        target = targets[0] if targets else None
        enemy_pos = _target_world_position(target or {})
        if target is None or enemy_pos is None:
            return _empty_action(), {"available": False, "reason": "no YOLO/LiDAR fused fire target"}

        info = {
            "playerPos": position,
            "enemyPos": enemy_pos,
            "enemyHealth": 100.0,
            "playerBodyX": body_yaw_deg,
            "playerTurretX": _safe_float(turret_state.get("x"), 0.0) or 0.0,
            "playerTurretY": _safe_float(turret_state.get("y"), 0.0) or 0.0,
            "targetIndex": target.get("targetId") or target.get("frameSeq") or 0,
            "targetType": target.get("semanticClass") or target.get("className") or "enemy_tank",
        }
        action = fire_logic.make_aim_action(info)
        self.last_fire_action = action
        return action, {
            "available": True,
            "target": target,
            "fireDebug": action.get("debug", {}),
        }

    def build_action(
        self,
        body: dict[str, Any],
        cache: Any,
        turret_state: dict[str, Any],
        targets: list[dict[str, Any]],
        legacy_aim_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        position = _extract_position(body.get("position"))
        if position is None:
            latest_position = getattr(cache, "pose", {}).get("playerPos") if getattr(cache, "pose", None) else None
            position = _extract_position(latest_position)
        body_yaw = _body_yaw_from(body, cache)
        clusters = getattr(cache, "clusters", ()) or ()

        drive = self._drive_action(position, body_yaw, clusters)
        fire, fire_debug = self._fire_action(body, cache, turret_state, position, body_yaw, targets)

        action = _empty_action()
        action["moveWS"] = _copy_axis(drive, "moveWS")
        action["moveAD"] = _copy_axis(drive, "moveAD")

        # Use fire_logic for the actual ballistic turret/fire decision.  If it is
        # unavailable, keep lidar_cluster's original seek/aim turret command.
        aim_source = "shot_analysis.fire_logic" if fire_debug.get("available") else "lidar_cluster_legacy"
        aim_action = fire if fire_debug.get("available") else (legacy_aim_action or {})
        action["turretQE"] = _copy_axis(aim_action, "turretQE")
        action["turretRF"] = _copy_axis(aim_action, "turretRF")
        action["fire"] = bool(fire.get("fire", False)) if fire_debug.get("available") else bool((legacy_aim_action or {}).get("fire", False))

        target_present = bool(fire_debug.get("available"))
        fire_info = fire.get("debug", {}) if isinstance(fire, dict) else {}
        should_pause = (
            target_present
            and self.engage_pauses_drive
            and (
                bool(action["fire"])
                or bool(fire_info.get("aim_aligned"))
                or _safe_float(fire_info.get("distance"), 9999.0) <= 200.0
            )
        )
        if should_pause:
            action["moveWS"] = {"command": "STOP", "weight": 0.0}
            # Body yaw alignment belongs to fire_logic while engaging.
            action["moveAD"] = _copy_axis(fire, "moveAD")

        action["debug"] = {
            "controller": "integrated_drive_fire",
            "aimSource": aim_source,
            "drive": self.last_plan,
            "fire": fire_debug,
            "destination": self.destination,
            "engagePausesDrive": self.engage_pauses_drive,
        }
        self.last_debug = action["debug"]
        return action

    def handle_bullet(self, payload: dict[str, Any]) -> dict[str, Any]:
        if fire_logic is None:
            return {"forwarded": False, "reason": FIRE_LOGIC_IMPORT_ERROR}
        try:
            fire_logic.log_bullet_impact(payload)
            return {"forwarded": True}
        except Exception as exc:
            return {"forwarded": False, "reason": f"{type(exc).__name__}: {exc}"}

    def status(self) -> dict[str, Any]:
        return {
            "driveEnabled": self.drive_enabled,
            "fireEnabled": self.fire_enabled,
            "engagePausesDrive": self.engage_pauses_drive,
            "destination": self.destination,
            "fireLogicAvailable": fire_logic is not None,
            "fireLogicError": FIRE_LOGIC_IMPORT_ERROR,
            "lastPlan": self.last_plan,
            "lastFireAction": self.last_fire_action,
            "lastDebug": self.last_debug,
        }
