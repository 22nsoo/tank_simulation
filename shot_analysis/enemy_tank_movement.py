from flask import Flask, request, jsonify
import math
import time


app = Flask(__name__)

# Rotate approximately 90 degrees once, then move forward/backward on that axis.
BODY_YAW_SPEED_PER_WEIGHT = 37.254
INITIAL_ROTATION_DEG = 90.0
INITIAL_TURN_COMMAND = "D"
INITIAL_TURN_WEIGHT = 0.30
INITIAL_TURN_SECONDS = (
    INITIAL_ROTATION_DEG
    / (BODY_YAW_SPEED_PER_WEIGHT * INITIAL_TURN_WEIGHT)
)
ROTATION_SETTLE_SECONDS = 0.5

DRIVE_WEIGHT = 0.20
ALIGN_DRIVE_WEIGHT = 0.08
X_HALF_SPAN_METERS = 20.0
X_DIRECTION_DETECT_METERS = 0.25
X_AXIS_HEADING_TOLERANCE_DEG = 0.8
X_AXIS_DRIVE_HEADING_TOLERANCE_DEG = 6.0
X_AXIS_STEER_KP = 0.015
X_AXIS_STEER_MAX_WEIGHT = 0.20
REVERSAL_STOP_SECONDS = 0.7

movement_started_at = time.monotonic()
movement_center_x = None
movement_center_z = None
drive_command = "W"
forward_x_sign = None
last_position = None
last_position_time = None
estimated_body_yaw = None
pending_drive_command = None
reversal_stop_until = 0.0


def normalize_angle(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def update_heading_estimate(position, now, ws_command):
    global last_position, last_position_time, estimated_body_yaw

    try:
        current_x = float(position["x"])
        current_z = float(position["z"])
    except (KeyError, TypeError, ValueError):
        return

    if last_position is not None and last_position_time is not None:
        dt = now - last_position_time
        dx = current_x - last_position["x"]
        dz = current_z - last_position["z"]
        distance = math.hypot(dx, dz)

        if 0.02 <= dt <= 1.0 and distance >= 0.03:
            velocity_yaw = math.degrees(math.atan2(dx, dz))
            estimated_body_yaw = (
                velocity_yaw
                if ws_command == "W"
                else normalize_angle(velocity_yaw + 180.0)
            )

    last_position = {"x": current_x, "z": current_z}
    last_position_time = now


def stop_action():
    return {
        "moveWS": {"command": "STOP", "weight": 0.0},
        "moveAD": {"command": "STOP", "weight": 0.0},
        "turretQE": {"command": "STOP", "weight": 0.0},
        "turretRF": {"command": "STOP", "weight": 0.0},
        "fire": False,
    }


def enemy_movement_action(position=None, now=None):
    """
    Rotate the chassis about 90 degrees once, settle, and then repeat
    forward/backward movement inside a fixed world-X interval.
    """
    if now is None:
        now = time.monotonic()

    elapsed = max(0.0, now - movement_started_at)

    global movement_center_x, movement_center_z
    global drive_command, forward_x_sign
    global pending_drive_command, reversal_stop_until

    position = position if isinstance(position, dict) else {}
    try:
        position_x = float(position["x"])
        position_z = float(position["z"])
    except (KeyError, TypeError, ValueError):
        position_x = None
        position_z = None

    if movement_center_x is None and position_x is not None:
        movement_center_x = position_x
        movement_center_z = position_z

    if elapsed < INITIAL_TURN_SECONDS:
        move_ws_command = "STOP"
        move_ws_weight = 0.0
        move_ad_command = INITIAL_TURN_COMMAND
        move_ad_weight = INITIAL_TURN_WEIGHT
        movement_phase = "initial_rotate_90_deg"
        phase_elapsed = elapsed
        x_axis_heading_error = None
    elif elapsed < INITIAL_TURN_SECONDS + ROTATION_SETTLE_SECONDS:
        move_ws_command = "STOP"
        move_ws_weight = 0.0
        move_ad_command = "STOP"
        move_ad_weight = 0.0
        movement_phase = "rotation_settle"
        phase_elapsed = elapsed - INITIAL_TURN_SECONDS
        x_axis_heading_error = None
    else:
        if (
            forward_x_sign is None
            and position_x is not None
            and movement_center_x is not None
            and abs(position_x - movement_center_x) >= X_DIRECTION_DETECT_METERS
        ):
            forward_x_sign = 1.0 if position_x > movement_center_x else -1.0

        if (
            forward_x_sign is not None
            and position_x is not None
            and movement_center_x is not None
        ):
            x_min = movement_center_x - X_HALF_SPAN_METERS
            x_max = movement_center_x + X_HALF_SPAN_METERS

            next_command = None
            if forward_x_sign > 0:
                if position_x >= x_max:
                    next_command = "S"
                elif position_x <= x_min:
                    next_command = "W"
            else:
                if position_x <= x_min:
                    next_command = "S"
                elif position_x >= x_max:
                    next_command = "W"

            if (
                pending_drive_command is None
                and next_command is not None
                and next_command != drive_command
            ):
                pending_drive_command = next_command
                reversal_stop_until = now + REVERSAL_STOP_SECONDS

        if pending_drive_command is not None and now >= reversal_stop_until:
            drive_command = pending_drive_command
            pending_drive_command = None

        reversing = pending_drive_command is not None
        move_ws_command = "STOP" if reversing else drive_command
        move_ws_weight = 0.0 if reversing else DRIVE_WEIGHT
        movement_phase = (
            "reversal_stop"
            if reversing
            else ("x_positive" if drive_command == "W" else "x_negative")
        )
        phase_elapsed = (
            elapsed - INITIAL_TURN_SECONDS - ROTATION_SETTLE_SECONDS
        )

        if not reversing:
            update_heading_estimate(position, now, drive_command)

        if estimated_body_yaw is not None:
            desired_body_yaw = (
                90.0
                if forward_x_sign is None or forward_x_sign > 0
                else -90.0
            )
            x_axis_heading_error = normalize_angle(
                desired_body_yaw - estimated_body_yaw
            )
        else:
            x_axis_heading_error = None

        if (
            not reversing
            and
            x_axis_heading_error is not None
            and abs(x_axis_heading_error) > X_AXIS_HEADING_TOLERANCE_DEG
        ):
            move_ad_command = "D" if x_axis_heading_error > 0 else "A"
            move_ad_weight = min(
                X_AXIS_STEER_MAX_WEIGHT,
                max(0.03, abs(x_axis_heading_error) * X_AXIS_STEER_KP),
            )

            if abs(x_axis_heading_error) > X_AXIS_DRIVE_HEADING_TOLERANCE_DEG:
                move_ws_command = drive_command
                move_ws_weight = ALIGN_DRIVE_WEIGHT
                movement_phase = "x_axis_align_while_slow_drive"
        else:
            move_ad_command = "STOP"
            move_ad_weight = 0.0

    return {
        "moveWS": {
            "command": move_ws_command,
            "weight": move_ws_weight,
        },
        "moveAD": {
            "command": move_ad_command,
            "weight": move_ad_weight,
        },
        "turretQE": {"command": "STOP", "weight": 0.0},
        "turretRF": {"command": "STOP", "weight": 0.0},
        "fire": False,
        "debug": {
            "movementPhase": movement_phase,
            "phaseElapsed": round(phase_elapsed, 3),
            "driveWeight": DRIVE_WEIGHT,
            "alignDriveWeight": ALIGN_DRIVE_WEIGHT,
            "initialRotationDeg": INITIAL_ROTATION_DEG,
            "initialTurnWeight": INITIAL_TURN_WEIGHT,
            "initialTurnSeconds": round(INITIAL_TURN_SECONDS, 3),
            "movementCenterX": movement_center_x,
            "movementCenterZ": movement_center_z,
            "xHalfSpanMeters": X_HALF_SPAN_METERS,
            "forwardXSign": forward_x_sign,
            "estimatedBodyYaw": (
                round(estimated_body_yaw, 3)
                if estimated_body_yaw is not None
                else None
            ),
            "xAxisHeadingError": (
                round(x_axis_heading_error, 3)
                if x_axis_heading_error is not None
                else None
            ),
            "xAxisDriveHeadingTolerance": X_AXIS_DRIVE_HEADING_TOLERANCE_DEG,
            "zDeviation": (
                round(position_z - movement_center_z, 3)
                if position_z is not None and movement_center_z is not None
                else None
            ),
            "reversalStopSeconds": REVERSAL_STOP_SECONDS,
            "reversalStopRemaining": round(
                max(0.0, reversal_stop_until - now),
                3,
            ),
        },
    }


@app.route("/init", methods=["GET"])
def init():
    global movement_started_at, movement_center_x, movement_center_z
    global drive_command, forward_x_sign
    global last_position, last_position_time, estimated_body_yaw
    global pending_drive_command, reversal_stop_until

    movement_started_at = time.monotonic()
    movement_center_x = None
    movement_center_z = None
    drive_command = "W"
    forward_x_sign = None
    last_position = None
    last_position_time = None
    estimated_body_yaw = None
    pending_drive_command = None
    reversal_stop_until = 0.0

    return jsonify({
        "startMode": "start",
        "blStartX": 150.0,
        "blStartY": 10.0,
        "blStartZ": 250.0,
        "trackingMode": True,
        "detectMode": False,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": "start"})


@app.route("/get_action", methods=["POST"])
def get_action():
    data = request.get_json(force=True, silent=True) or {}
    position = data.get("position", {})
    turret = data.get("turret", {})
    action = enemy_movement_action(position=position)

    print(
        "[ENEMY MOVE] "
        f"position=({position.get('x')}, {position.get('y')}, "
        f"{position.get('z')}) "
        f"turret=({turret.get('x')}, {turret.get('y')}) "
        f"phase={action['debug']['movementPhase']} "
        f"WS={action['moveWS']} AD={action['moveAD']}"
    )

    return jsonify(action)


@app.route("/movement_status", methods=["GET"])
def movement_status():
    action = enemy_movement_action()
    return jsonify({
        "mode": "world_x_axis_feedback_motion",
        "action": action,
        "initialRotation": {
            "command": INITIAL_TURN_COMMAND,
            "degrees": INITIAL_ROTATION_DEG,
            "weight": INITIAL_TURN_WEIGHT,
            "estimatedSeconds": round(INITIAL_TURN_SECONDS, 3),
        },
        "xRangeMeters": 2.0 * X_HALF_SPAN_METERS,
        "forwardBackwardSpeedEqual": True,
        "positionFeedback": True,
    })


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({
            "status": "ERROR",
            "message": "Invalid request data",
        }), 400

    print(
        "[BULLET IMPACT] "
        f"x={data.get('x')} y={data.get('y')} z={data.get('z')} "
        f"hit={data.get('hit')}"
    )

    return jsonify({
        "status": "OK",
        "message": "Bullet impact data received",
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5100,
        debug=True,
        use_reloader=False,
    )
