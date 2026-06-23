from time import time
from config import MODEL_PATH

class RuntimeState:
    def __init__(self):
        self.destination = {"x": 100.0, "y": 0.0, "z": 250.0}
        self.obstacles = []
        self.active_navigation_obstacles = []

        self.current_path = []
        self.current_path_index = 0

        self.latest_player_body_yaw = 0.0
        self.latest_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.latest_turret = {"x": 0.0, "y": 0.0}

        self.latest_action = None
        self.latest_detections = []
        self.latest_detection_time = 0.0

        self.target_candidate = None
        self.target_candidate_hits = 0
        self.target_candidate_time = 0.0
        self.retained_target = None
        self.retained_target_time = 0.0

        self.latest_frame_name = None
        self.latest_frame_bytes = b""

        self.latest_event = "server ready"
        self.latest_bullet = None
        self.latest_collision = None

        self.latest_lidar = {
            "source": "none",
            "file": None,
            "mtime": 0.0,
            "point_count": 0,
            "detected_count": 0,
            "near_count": 0,
            "min_distance": None,
            "obstacle_count": 0,
        }
        self.latest_lidar_points = []
        self.latest_lidar_api_time = 0.0

        self.latest_target_status = {
            "state": "no_target",
            "label": "탐지 없음",
        }

        self.latest_drive_logs = []
        self.use_lidar_navigation = False
        self.fire_approval_until = 0.0
        self.last_update_time = time()

state = RuntimeState()