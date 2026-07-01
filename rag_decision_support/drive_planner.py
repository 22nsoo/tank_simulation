# -*- coding: utf-8 -*-
"""Local drive planner used by the RAG tactical dashboard.

This is a lightweight extraction of the Tank Challenge local-planner behavior:
waypoint following, lookahead target selection, PD steering, clearance stop,
TTC stop, and replan signaling. It does not actuate the simulator directly.
"""

from __future__ import annotations

import math
from typing import Any


ARRIVAL_RADIUS = 6.0
HEADING_THRESHOLD = 5.0
MOVE_WEIGHT = 0.45
LOOKAHEAD_MIN = 5.0
LOOKAHEAD_MAX = 12.0
KP = 0.015
KD = 0.020
LOCAL_TANK_WIDTH_M = 3.6
LOCAL_TANK_LENGTH_M = 10.8
LOCAL_SAFETY_CLEARANCE_M = 0.8
CLEARANCE_EMERGENCY_M = 2.8
REPLAN_CORRIDOR_CLEARANCE_M = 4.8
REPLAN_CORRIDOR_STOP_AHEAD_M = 12.0
TTC_SLOWDOWN_S = 4.0
TTC_EMERGENCY_STOP_S = 2.0
DEFAULT_SPEED_MS = 4.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _dist(ax: float, az: float, bx: float, bz: float) -> float:
    return math.hypot(bx - ax, bz - az)


def _angle_to(pos_x: float, pos_z: float, tx: float, tz: float) -> float:
    return math.degrees(math.atan2(tx - pos_x, tz - pos_z)) % 360.0


def _heading_error(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _point_from_relative(pos: dict, yaw_deg: float, angle_deg: float, distance_m: float) -> dict:
    world_yaw = math.radians(yaw_deg + angle_deg)
    return {
        "x": round(_safe_float(pos.get("x")) + distance_m * math.sin(world_yaw), 3),
        "y": round(_safe_float(pos.get("y")), 3),
        "z": round(_safe_float(pos.get("z")) + distance_m * math.cos(world_yaw), 3),
    }


def _make_waypoints(start: dict, goal: dict, segments: int = 4) -> list[dict]:
    sx, sz = _safe_float(start.get("x")), _safe_float(start.get("z"))
    gx, gz = _safe_float(goal.get("x")), _safe_float(goal.get("z"))
    sy = _safe_float(start.get("y"), _safe_float(goal.get("y")))
    waypoints = []
    count = max(1, int(segments))
    for index in range(1, count + 1):
        ratio = index / count
        waypoints.append(
            {
                "x": round(sx + (gx - sx) * ratio, 3),
                "y": round(sy, 3),
                "z": round(sz + (gz - sz) * ratio, 3),
            }
        )
    return waypoints


def _find_lookahead(pos_x: float, pos_z: float, waypoints: list[dict], lookahead: float) -> tuple[int, dict]:
    for index, waypoint in enumerate(waypoints):
        if _dist(pos_x, pos_z, _safe_float(waypoint.get("x")), _safe_float(waypoint.get("z"))) >= lookahead:
            return index, waypoint
    return len(waypoints) - 1, waypoints[-1]


def _obstacle_from_relative(item: dict, player_pos: dict, heading_deg: float) -> dict | None:
    distance = _safe_float(item.get("distanceM"), -1.0)
    angle = _safe_float(item.get("angleDeg"), 0.0)
    if distance < 0.0:
        return None
    point = _point_from_relative(player_pos, heading_deg, angle, distance)
    radius = max(1.0, _safe_float(item.get("radiusM"), 1.8))
    return {
        "cx": point["x"],
        "cz": point["z"],
        "radius": radius,
        "label": item.get("label") or "obstacle",
        "distanceM": distance,
        "angleDeg": angle,
    }


def _obstacle_clearance(pos_x: float, pos_z: float, obstacle: dict) -> float:
    radius = _safe_float(obstacle.get("radius"), 1.8)
    return _dist(pos_x, pos_z, _safe_float(obstacle.get("cx")), _safe_float(obstacle.get("cz"))) - radius


def _minimum_clearance(pos_x: float, pos_z: float, obstacles: list[dict]) -> tuple[float, dict | None]:
    if not obstacles:
        return float("inf"), None
    nearest = min(obstacles, key=lambda item: _obstacle_clearance(pos_x, pos_z, item))
    return _obstacle_clearance(pos_x, pos_z, nearest), nearest


def _minimum_ttc(pos_x: float, pos_z: float, heading_deg: float, speed_ms: float, obstacles: list[dict]) -> float:
    if speed_ms < 0.2 or not obstacles:
        return float("inf")
    yaw = math.radians(heading_deg)
    forward_x, forward_z = math.sin(yaw), math.cos(yaw)
    right_x, right_z = math.cos(yaw), -math.sin(yaw)
    best = float("inf")
    for obstacle in obstacles:
        dx = _safe_float(obstacle.get("cx")) - pos_x
        dz = _safe_float(obstacle.get("cz")) - pos_z
        forward = dx * forward_x + dz * forward_z
        if forward <= 0.0:
            continue
        lateral = abs(dx * right_x + dz * right_z)
        radius = _safe_float(obstacle.get("radius"), 1.8)
        corridor = radius + 0.5 * LOCAL_TANK_WIDTH_M + LOCAL_SAFETY_CLEARANCE_M
        if lateral > corridor:
            continue
        longitudinal_clearance = forward - radius - 0.5 * LOCAL_TANK_LENGTH_M - LOCAL_SAFETY_CLEARANCE_M
        best = min(best, max(0.0, longitudinal_clearance) / speed_ms)
    return best


def _corridor_risk(pos_x: float, pos_z: float, goal: dict, obstacles: list[dict]) -> tuple[float, float]:
    if not obstacles:
        return float("inf"), float("inf")
    gx, gz = _safe_float(goal.get("x")), _safe_float(goal.get("z"))
    vx, vz = gx - pos_x, gz - pos_z
    length_sq = vx * vx + vz * vz
    if length_sq <= 1e-9:
        return float("inf"), 0.0
    best_clearance = float("inf")
    best_ahead = float("inf")
    for obstacle in obstacles:
        ox, oz = _safe_float(obstacle.get("cx")), _safe_float(obstacle.get("cz"))
        t = max(0.0, min(1.0, ((ox - pos_x) * vx + (oz - pos_z) * vz) / length_sq))
        closest_x, closest_z = pos_x + vx * t, pos_z + vz * t
        clearance = _dist(ox, oz, closest_x, closest_z) - _safe_float(obstacle.get("radius"), 1.8)
        ahead = math.sqrt(length_sq) * t
        if clearance < best_clearance:
            best_clearance = clearance
            best_ahead = ahead
    return best_clearance, best_ahead


def _command(move_ws: str, move_ad: str, move_weight: float, turn_weight: float, **extra: Any) -> dict:
    command = {
        "move_ws": move_ws,
        "move_ad": move_ad,
        "move_weight": round(max(0.0, min(1.0, move_weight)), 3),
        "turn_weight": round(max(0.0, min(1.0, turn_weight)), 3),
    }
    command.update(extra)
    return command


def _fallback_goal(action: dict, summary: dict) -> dict | None:
    destination = action.get("safe_destination") if isinstance(action, dict) else None
    if isinstance(destination, dict) and isinstance(destination.get("world"), dict):
        return destination["world"]
    pose = summary.get("player_pose") or {}
    position = pose.get("position") if isinstance(pose.get("position"), dict) else None
    yaw = _safe_float(pose.get("body_yaw_deg"), None)
    if position is None or yaw is None:
        return None
    if not isinstance(destination, dict):
        return None
    return _point_from_relative(
        position,
        yaw,
        _safe_float(destination.get("relative_angle_deg")),
        _safe_float(destination.get("relative_distance_m")),
    )


def plan_drive(summary: dict, action: dict, previous_state: dict | None = None, speed_ms: float | None = None) -> dict:
    """Return local planner output for the action selected by RAG/tactical logic."""
    previous_state = previous_state or {}
    pose = summary.get("player_pose") or {}
    position = pose.get("position") if isinstance(pose.get("position"), dict) else None
    heading_deg = _safe_float(pose.get("body_yaw_deg"), None)
    if position is None or heading_deg is None:
        return {
            "available": False,
            "control_mode": "no_pose",
            "reason": "player position/body yaw not available from live context",
            "command": _command("", "", 0.0, 0.0),
            "waypoints": [],
        }

    selected_action = str(action.get("action") or "HOLD_AIM")
    if selected_action in {"FIRE", "HOLD_AIM", "SCAN"}:
        return {
            "available": True,
            "control_mode": "stationary_action",
            "reason": f"{selected_action} does not require local path following",
            "command": _command("", "A" if selected_action == "SCAN" else "", 0.0, 0.35 if selected_action == "SCAN" else 0.0),
            "waypoints": [],
        }

    goal = _fallback_goal(action, summary)
    if goal is None:
        return {
            "available": False,
            "control_mode": "no_goal",
            "reason": "safe destination is unavailable",
            "command": _command("", "", 0.0, 0.0),
            "waypoints": [],
        }

    pos_x, pos_z = _safe_float(position.get("x")), _safe_float(position.get("z"))
    obstacles = []
    for item in (summary.get("front_obstacles") or []) + (summary.get("nearest_obstacles") or []):
        obstacle = _obstacle_from_relative(item, position, heading_deg)
        if obstacle is not None:
            obstacles.append(obstacle)

    waypoints = _make_waypoints(position, goal, segments=4)
    wp_index = int(previous_state.get("wp_index") or 0)
    while wp_index < len(waypoints):
        wp = waypoints[wp_index]
        if _dist(pos_x, pos_z, _safe_float(wp.get("x")), _safe_float(wp.get("z"))) < ARRIVAL_RADIUS:
            wp_index += 1
        else:
            break
    if wp_index >= len(waypoints):
        return {
            "available": True,
            "control_mode": "arrived",
            "reason": "safe destination reached",
            "command": _command("", "", 0.0, 0.0, done=True, wp_index=wp_index),
            "waypoints": waypoints,
            "goal": goal,
        }

    current_wp = waypoints[wp_index]
    dist_to_wp = _dist(pos_x, pos_z, _safe_float(current_wp.get("x")), _safe_float(current_wp.get("z")))
    lookahead = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, dist_to_wp * 0.5))
    _, target_wp = _find_lookahead(pos_x, pos_z, waypoints[wp_index:], lookahead)
    target_yaw = _angle_to(pos_x, pos_z, _safe_float(target_wp.get("x")), _safe_float(target_wp.get("z")))
    error = _heading_error(heading_deg, target_yaw)
    prev_error = _safe_float(previous_state.get("pid_prev_error"), 0.0)
    d_error = error - prev_error
    turn_weight = min(1.0, max(0.0, KP * abs(error) + KD * abs(d_error)))

    min_clearance, nearest = _minimum_clearance(pos_x, pos_z, obstacles)
    corridor_clear, corridor_ahead = _corridor_risk(pos_x, pos_z, goal, obstacles)
    tank_speed = DEFAULT_SPEED_MS if speed_ms is None else max(0.0, float(speed_ms))
    min_ttc = _minimum_ttc(pos_x, pos_z, heading_deg, tank_speed, obstacles)

    if min_clearance <= CLEARANCE_EMERGENCY_M:
        return {
            "available": True,
            "control_mode": "clearance_stop",
            "reason": "nearest obstacle violates emergency clearance",
            "command": _command("", "", 0.0, 0.0, wp_index=wp_index, stop_reason="pos_clearance"),
            "waypoints": waypoints,
            "goal": goal,
            "min_clearance_m": round(min_clearance, 3),
            "min_ttc_s": 0.0,
            "nearest_obstacle": nearest,
            "replan_requested": True,
        }

    if corridor_clear < REPLAN_CORRIDOR_CLEARANCE_M and corridor_ahead <= REPLAN_CORRIDOR_STOP_AHEAD_M:
        return {
            "available": True,
            "control_mode": "corridor_replan_stop",
            "reason": "planned corridor is blocked nearby",
            "command": _command("", "", 0.0, 0.0, wp_index=wp_index, stop_reason="corridor_near"),
            "waypoints": waypoints,
            "goal": goal,
            "corridor_clearance_m": round(corridor_clear, 3),
            "corridor_ahead_m": round(corridor_ahead, 3),
            "nearest_obstacle": nearest,
            "replan_requested": True,
        }

    if min_ttc <= TTC_EMERGENCY_STOP_S:
        return {
            "available": True,
            "control_mode": "ttc_stop",
            "reason": "time-to-collision is below emergency threshold",
            "command": _command("", "", 0.0, 0.0, wp_index=wp_index, stop_reason="ttc"),
            "waypoints": waypoints,
            "goal": goal,
            "min_clearance_m": round(min_clearance, 3),
            "min_ttc_s": round(min_ttc, 3),
            "nearest_obstacle": nearest,
            "replan_requested": True,
        }

    speed_factor = 0.5 if dist_to_wp < 6.0 else (0.7 if dist_to_wp < 10.0 else 1.0)
    abs_error = abs(error)
    if abs_error <= HEADING_THRESHOLD:
        move_ad = ""
        move_weight = MOVE_WEIGHT * speed_factor
        turn_weight = 0.0
    elif abs_error <= 30.0:
        move_ad = "D" if error > 0.0 else "A"
        move_weight = MOVE_WEIGHT * speed_factor * 0.8
    else:
        move_ad = "D" if error > 0.0 else "A"
        move_weight = 0.0
        turn_weight = max(turn_weight, 0.6)

    if min_ttc < TTC_SLOWDOWN_S:
        move_weight *= max(0.15, min_ttc / TTC_SLOWDOWN_S)

    return {
        "available": True,
        "control_mode": "path_follow" if corridor_clear >= REPLAN_CORRIDOR_CLEARANCE_M else "path_follow_replan_ahead",
        "reason": "following local waypoints toward safe destination",
        "command": _command(
            "W" if move_weight > 0.0 else "",
            move_ad,
            move_weight,
            turn_weight,
            wp_index=wp_index,
            heading_error=round(error, 3),
            target_yaw=round(target_yaw, 3),
            distance_to_wp_m=round(dist_to_wp, 3),
        ),
        "waypoints": waypoints,
        "goal": goal,
        "lookahead_m": round(lookahead, 3),
        "min_clearance_m": None if math.isinf(min_clearance) else round(min_clearance, 3),
        "min_ttc_s": None if math.isinf(min_ttc) else round(min_ttc, 3),
        "corridor_clearance_m": None if math.isinf(corridor_clear) else round(corridor_clear, 3),
        "corridor_ahead_m": None if math.isinf(corridor_ahead) else round(corridor_ahead, 3),
        "nearest_obstacle": nearest,
        "replan_requested": corridor_clear < REPLAN_CORRIDOR_CLEARANCE_M,
        "state": {"wp_index": wp_index, "pid_prev_error": round(error, 3)},
    }
