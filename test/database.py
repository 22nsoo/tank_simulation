import json
from datetime import datetime
from time import time

from config import DB_CONFIG

try:
    import psycopg
except ImportError:
    psycopg = None


DB_TABLES = {
    "drive_logs": [
        "id", "created_at", "x", "y", "z", "dest_x", "dest_z", "body_yaw",
        "move_ws", "move_ws_weight", "move_ad", "move_ad_weight", "fire",
        "path_index", "path_length", "obstacle_count", "lidar_source",
        "lidar_near_count", "lidar_min_distance", "target_state",
        "target_label", "target_confidence", "collision", "event",
    ],
    "detections": [
        "id", "created_at", "frame_name", "class_name", "confidence",
        "bbox", "image_width", "image_height",
    ],
    "lidar_summaries": [
        "id", "created_at", "source", "file_name", "point_count",
        "detected_count", "near_count", "min_distance", "obstacle_count",
    ],
    "events": [
        "id", "created_at", "event_type", "message", "payload",
    ],
}

db_available = False
db_last_error = "psycopg not installed" if psycopg is None else None
db_last_retry = 0.0


def db_connect():
    if psycopg is None:
        raise RuntimeError("psycopg not installed")
    return psycopg.connect(**DB_CONFIG)


def init_database(force=False):
    global db_available, db_last_error, db_last_retry

    if not force and time() - db_last_retry < 10:
        return db_available

    db_last_retry = time()

    if psycopg is None:
        db_available = False
        db_last_error = "psycopg not installed"
        return False

    schema = """
    CREATE TABLE IF NOT EXISTS drive_logs (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        timestamp_text TEXT,
        x DOUBLE PRECISION,
        y DOUBLE PRECISION,
        z DOUBLE PRECISION,
        dest_x DOUBLE PRECISION,
        dest_z DOUBLE PRECISION,
        body_yaw DOUBLE PRECISION,
        move_ws TEXT,
        move_ws_weight DOUBLE PRECISION,
        move_ad TEXT,
        move_ad_weight DOUBLE PRECISION,
        turret_qe TEXT,
        turret_rf TEXT,
        fire BOOLEAN,
        path_index INTEGER,
        path_length INTEGER,
        obstacle_count INTEGER,
        lidar_source TEXT,
        lidar_points INTEGER,
        lidar_near_count INTEGER,
        lidar_min_distance DOUBLE PRECISION,
        target_state TEXT,
        target_label TEXT,
        target_class TEXT,
        target_confidence DOUBLE PRECISION,
        collision BOOLEAN,
        event TEXT,
        action_json JSONB,
        raw_json JSONB
    );

    CREATE TABLE IF NOT EXISTS detections (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        frame_name TEXT,
        class_name TEXT,
        confidence DOUBLE PRECISION,
        bbox JSONB,
        image_width INTEGER,
        image_height INTEGER
    );

    CREATE TABLE IF NOT EXISTS lidar_summaries (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        source TEXT,
        file_name TEXT,
        point_count INTEGER,
        detected_count INTEGER,
        near_count INTEGER,
        min_distance DOUBLE PRECISION,
        obstacle_count INTEGER
    );

    CREATE TABLE IF NOT EXISTS events (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        event_type TEXT,
        message TEXT,
        payload JSONB
    );
    """

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(schema)

        db_available = True
        db_last_error = None
        return True

    except Exception as exc:
        db_available = False
        db_last_error = str(exc)
        return False


def db_execute(query, params=()):
    global db_available, db_last_error

    if not init_database():
        return False

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
        return True

    except Exception as exc:
        db_available = False
        db_last_error = str(exc)
        return False


def db_fetch_all(query, params=()):
    global db_available, db_last_error

    if not init_database():
        return []

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    except Exception as exc:
        db_available = False
        db_last_error = str(exc)
        return []


def record_event(event_type, message, payload=None):
    db_execute(
        "INSERT INTO events (event_type, message, payload) VALUES (%s, %s, %s::jsonb)",
        (event_type, message, json.dumps(payload or {}, ensure_ascii=False)),
    )


def json_ready(value):
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")

    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}

    if isinstance(value, list):
        return [json_ready(item) for item in value]

    return value


def get_db_status():
    return {
        "available": db_available,
        "error": db_last_error,
        "host": DB_CONFIG["host"],
        "port": DB_CONFIG["port"],
        "name": DB_CONFIG["dbname"],
    }