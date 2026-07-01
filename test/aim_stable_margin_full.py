from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime
from math import atan, atan2, cos, degrees, floor, hypot, isfinite, radians, tan
from pathlib import Path
from statistics import median
from threading import Lock
from time import monotonic, sleep
from typing import Any
from io import BytesIO
import base64
import json
import re

from flask import Flask, jsonify, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# =============================================================================
# 0. ?쒕떇 ?뚮씪誘명꽣
# =============================================================================
MAX_DISTANCE_M = 120.0

# ----------------------------------------------------------------------------
# 媛앹껜 ?꾨낫援? ?ш린?쒕뒗 理쒖쥌 ?섎? 遺꾨쪟媛 ?꾨땲??LiDAR ?뺤긽 ?꾨낫留?留뚮뱺??
# TH = ?뉗? ?꾨낫(?щ엺/?섎Т/湲곕뫁 怨꾩뿴), BK = 遺?쇨? ???꾨낫
# (?꾩감/諛붿쐞/踰?怨꾩뿴), UK = 誘명솗???꾨낫. YOLO 寃고빀? ???④퀎?먯꽌 ?섑뻾?쒕떎.
# ----------------------------------------------------------------------------
OBJECT_VERTICAL_MIN_DEG = -10.0
OBJECT_VERTICAL_MAX_DEG = -0.1
OBJECT_DETECTION_MAX_DISTANCE_M = 120.0
# Match server_lidar_sensing_visualizer_patch_verify.py's hit filter.
GROUND_HEIGHT_MAX_M = 0.35
OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M = 0.55
FAR_TERRAIN_EXTRA_RATIO = 1.35
FAR_TERRAIN_EXTRA_MIN_M = 3.0
OBJECT_AZIMUTH_BIN_WIDTH_DEG = 1.0
OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG = 2.1
OBJECT_CLUSTER_MAX_DISTANCE_GAP_M = 3.0
THIN_MAX_WIDTH_M = 2.20
BULKY_MIN_WIDTH_M = 2.50
THIN_MAX_DISTANCE_M = 70.0
BULKY_CONFIDENT_DISTANCE_M = 100.0

# v16.6 terrain-residual + compact vertical-plane object filter.
TERRAIN_PROFILE_ANGLE_BIN_DEG = 2.0
TERRAIN_PROFILE_RANGE_BIN_M = 1.0
TERRAIN_GROUND_RESIDUAL_TOL_M = 0.70
VALID_OBJECT_STACK_ANGLE_BIN_DEG = 1.5
VALID_OBJECT_STACK_RANGE_BIN_M = 1.5
VALID_OBJECT_STACK_MIN_SPAN_M = 0.60
VALID_OBJECT_STACK_MIN_POINTS = 2
VALID_OBJECT_MIN_DISTANCE_M = 3.0
VALID_OBJECT_MAX_DISTANCE_M = 120.0
VALID_OBJECT_MAX_RANGE_SPAN_M = 1.35
VALID_OBJECT_MIN_VERTICALITY_RATIO = 0.85
OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M = 0.95
OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M = 1.05
OBJECT_ON_HILL_MIN_HIGH_POINTS = 2
OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO = 0.25
OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO = 0.80
FLAT_OBJECT_MIN_HEIGHT_SPAN_M = 0.90
FLAT_OBJECT_MIN_POINTS = 3
FLAT_OBJECT_MIN_VERTICALITY_RATIO = 0.95
FLAT_OBJECT_MAX_RANGE_SPAN_M = 1.20
VALID_OBJECT_MIN_ABOVE_STACK_BASE_M = 0.45

# 吏?뺤뿉 遺숈뼱 ?덈뒗 ?대윭?ㅽ꽣???덉젙 媛앹껜 ?몃옓?먯꽌 ?④릿??
# ?붾쾭源낆슜?쇰줈 rawObjects?먮뒗 洹몃?濡??④릿??
TERRAIN_OBJECT_OVERLAP_ANGLE_DEG = 7.0
TERRAIN_OBJECT_OVERLAP_RANGE_M = 5.0
# 吏?뺤뿉 遺숈? ??? ?대윭?ㅽ꽣留??듭젣?쒕떎.
# 寃쎌궗 洹쇱쿂???????щ엺?대굹 ?꾩감???꾨낫濡?怨꾩냽 蹂댁뿬???쒕떎.
TERRAIN_CONNECTED_LOW_HEIGHT_M = 0.35
TERRAIN_CONNECTED_HAZARD_LOW_HEIGHT_M = 0.95
LOCAL_GROUND_LOOKUP_MAX_RANGE_GAP_M = 4.0

# ----------------------------------------------------------------------------
# 濡쒖뺄 吏??紐⑤뜽怨??꾪뿕 媛먯?
# ----------------------------------------------------------------------------
TERRAIN_VERTICAL_MIN_DEG = 0.5
TERRAIN_VERTICAL_MAX_DEG = 22.5
TERRAIN_FRONT_LIMIT_DEG = 60.0
TERRAIN_ANALYSIS_MAX_DISTANCE_M = 35.0
TERRAIN_SECTOR_WIDTH_DEG = 10.0
TERRAIN_RANGE_BIN_M = 1.5
LOCAL_GROUND_GRID_ANGLE_WIDTH_DEG = 5.0
LOCAL_GROUND_GRID_RANGE_BIN_M = 2.0

PASSABLE_UP_SLOPE_MAX_DEG = 12.0
CAUTION_UP_SLOPE_MAX_DEG = 20.0
PASSABLE_DOWN_SLOPE_MAX_DEG = 10.0
CAUTION_DOWN_SLOPE_MAX_DEG = 18.0
CAUTION_UP_STEP_M = 0.30
BLOCKED_UP_STEP_M = 0.60
CAUTION_DROP_M = 0.30
BLOCKED_DROP_M = 0.65

WALL_RANGE_BIN_M = 1.2
WALL_MIN_HEIGHT_SPAN_M = 1.0
WALL_MIN_UNIQUE_CHANNELS = 3

# Vertical-profile shape filter used to separate hills from upright obstacles.
PROFILE_SHAPE_RANGE_BIN_M = 1.5
PROFILE_SHAPE_MIN_POINTS = 5
PROFILE_SHAPE_MIN_RANGE_SPAN_M = 8.0
PROFILE_SHAPE_MIN_HEIGHT_GAIN_M = 0.8
PROFILE_SHAPE_MAX_POINT_GAP_M = 8.0
PROFILE_SHAPE_MAX_SLOPE_DEG = 35.0
PROFILE_SHAPE_MAX_SLOPE_CHANGE_DEG = 18.0
PROFILE_SHAPE_MONOTONIC_RATIO_MIN = 0.70
PROFILE_SHAPE_MIN_HILL_COLUMNS = 2
PROFILE_SHAPE_HILL_VOTE_RATIO_MIN = 0.35
OBSTACLE_RESIDUAL_MIN_HEIGHT_M = 0.60
OBSTACLE_RESIDUAL_MIN_CHANNELS = 3
OBSTACLE_RESIDUAL_MAX_RANGE_SPAN_M = 2.5
OBSTACLE_ADJACENT_ANGLE_DEG = 2.5
OBSTACLE_ADJACENT_RANGE_M = 3.0
OBSTACLE_MIN_ADJACENT_COLUMNS = 2
ISOLATED_RETURN_MAX_AZIMUTH_BINS = 2
ISOLATED_RETURN_MAX_POINTS = 4
# Sparse far returns from a rising terrain surface often split into one-bin
# UK/BK fragments.  They have almost no vertical thickness and no wall stack.
TERRAIN_SURFACE_FRAGMENT_MIN_DISTANCE_M = 35.0
TERRAIN_SURFACE_FRAGMENT_MAX_HEIGHT_SPAN_M = 0.55
TERRAIN_SURFACE_FRAGMENT_MIN_ABOVE_GROUND_M = 0.55

# ?덉긽 吏硫?紐⑤뜽: 遺꾩꽍 踰붿쐞 諛뽰쓽 ?됱????먯뿰?ㅻ읇寃??우쓣 ?뺤? 愿묒꽑? 臾댁떆?쒕떎.
# ?덈꼍 ?ㅽ깘??以꾩씠湲??꾪븳 泥섎━??
LOCAL_GROUND_STEEP_MIN_DEG = 8.0
LOCAL_GROUND_NEAR_MAX_DISTANCE_M = 12.0
LOCAL_GROUND_MIN_POINT_COUNT = 6
EXPECTED_GROUND_MIN_DISTANCE_M = 1.5
EXPECTED_GROUND_MAX_DISTANCE_M = 30.0
EXPECTED_GROUND_MIN_RAY_COUNT = 8
EXPECTED_GROUND_RANGE_TOLERANCE_M = 2.5
EXPECTED_GROUND_DELAY_RATIO = 1.35
CLIFF_CAUTION_EXPECTED_MISS_RATIO = 0.35
CLIFF_BLOCKED_EXPECTED_MISS_RATIO = 0.60
CLIFF_CAUTION_DELAYED_RETURN_RATIO = 0.35
CLIFF_BLOCKED_DELAYED_RETURN_RATIO = 0.60
CLIFF_MIN_PROFILE_GAP_M = 4.5
CLIFF_NEAR_GROUND_EVIDENCE_M = 12.0

# ----------------------------------------------------------------------------
# ?쒓컙 ?덉젙?붿? 異붿쟻
# ----------------------------------------------------------------------------
TERRAIN_HISTORY_SIZE = 5
TERRAIN_BLOCKED_CONFIRM_FRAMES = 2
TERRAIN_CAUTION_CONFIRM_FRAMES = 2
TERRAIN_PASSABLE_CONFIRM_FRAMES = 3
DEAD_END_FRONT_LIMIT_DEG = 55.0
DEAD_END_BLOCKED_RATIO = 0.70
DEAD_END_MAX_PASSABLE_SECTORS = 1

TRACK_HISTORY_SIZE = 5
# ?꾩떆 ?꾨낫??利됱떆 蹂댁뿬以??
# 諛섏쓳?깆쓣 ?꾪빐 留ㅼ묶 ?꾨젅?꾩씠 ?볦씠硫??덉젙 ?몃옓?쇰줈 ?뺤젙?쒕떎.
TRACK_CONFIRM_HITS = 1
TRACK_MAX_MISSES = 8
TRACK_ASSOCIATION_DISTANCE_M = 6.5
PROVISIONAL_OBJECT_LIMIT = 30
TRACK_PROCESS_NOISE = 0.8
TRACK_MEASUREMENT_NOISE = 1.5
IMPACT_HISTORY_SIZE = 20

# ----------------------------------------------------------------------------
# LiDAR -> YOLO 寃고빀 ?ㅼ?以꾨쭅 ?곗꽑?쒖쐞
# ----------------------------------------------------------------------------
# 360???꾨낫瑜?紐⑤몢 ?좎??섎릺, ?볦? BK ?꾨낫瑜?癒쇱? 泥섎━?쒕떎.
# 50m 湲곗?? ?붿껌??洹쇨굅由?媛앹껜 踰붿쐞瑜??삵븳??
# ??먯꽌 ???꾧꺽??洹쇨굅由?湲곗????곌린濡??섎㈃ 40.0?쇰줈 ??텧 ???덈떎.
PRIORITY_NEAR_MAX_DISTANCE_M = 45.0
PRIORITY_MAX_QUEUE_SIZE = 20
PRIORITY_DUPLICATE_ANGLE_TOLERANCE_DEG = 3.0
PRIORITY_DUPLICATE_DISTANCE_TOLERANCE_M = 4.0

# ----------------------------------------------------------------------------
# 媛???꾪뿕??洹쇨굅由??쒖쟻???ν븳 ?먮룞 李⑥껜 ?뺣젹
# ----------------------------------------------------------------------------
# BK????꾩감/諛붿쐞泥섎읆 遺?쇨? ???꾨낫?쇰뒗 ?살씠硫? 理쒖쥌 ?꾩감 ?쇰꺼? ?꾨땲??
# 李⑥껜 ?뺣젹? LiDAR ?몃옓 ID瑜??ㅻ쫫李⑥닚?쇰줈 ?묒쑝硫??꾨낫瑜?議곗??쒕떎.
# 議곗???ID媛 ???꾩감?몄? YOLO媛 ?뺤씤?????덈룄濡?2珥덉쓽 ?먮떒 ?쒓컙??以??
AUTO_BODY_ALIGN_ENABLED = True
BODY_ALIGN_USE_LIDAR_BULKY_FALLBACK = True
BODY_ALIGN_CONFIRMED_ONLY = True
BODY_ALIGN_TARGET_MAX_DISTANCE_M = 90.0
BODY_ALIGN_LOCK_RELEASE_DISTANCE_M = 95.0
BODY_ALIGN_DEADBAND_DEG = 0.8
BODY_ALIGN_SLOW_ZONE_DEG = 18.0
BODY_ALIGN_MEDIUM_ZONE_DEG = 45.0
BODY_ALIGN_FAST_ZONE_DEG = 90.0
BODY_ALIGN_WEIGHT_SLOW = 0.08
BODY_ALIGN_WEIGHT_MEDIUM = 0.16
BODY_ALIGN_WEIGHT_FAST = 0.28
BODY_ALIGN_WEIGHT_MAX = 0.40
BODY_ALIGN_USE_PRIMARY_FOR_RECOGNITION = False
BODY_ALIGN_STICKY_LOCK_ENABLED = True
BODY_ALIGN_LOCK_CONFIRM_HITS = 3
BODY_ALIGN_SPATIAL_LOCK_ANGLE_DEG = 5.0
BODY_ALIGN_SPATIAL_LOCK_DISTANCE_M = 8.0
BODY_ALIGN_POST_AIM_DECISION_SECONDS = 2.0
BODY_ALIGN_YOLO_FRAME_WAIT_SECONDS = 5.0
BODY_ALIGN_YOLO_PREALIGN_DEG = 6.0
BODY_ALIGN_SEQUENTIAL_ID_SCAN = True

# ----------------------------------------------------------------------------
# UI? 而⑦듃濡ㅻ윭
# ----------------------------------------------------------------------------
CONTOUR_ANGLE_BIN_DEG = 3.0
CONTOUR_MAX_DISTANCE_M = 120.0
FRONT_CLEARANCE_HALF_WIDTH_DEG = 15.0

# ?꾨갑 媛곷룄 湲곕컲 ?ъ씤???대씪?곕뱶 酉?
# ?꾨갑 罹붾쾭?ㅻ뒗 ?댁쟾?먭? ?뺣㈃??蹂대뒗 寃껋쿂??LiDAR瑜??쒖떆?쒕떎.
# X = 李⑥껜 湲곗? 諛⑹쐞媛? Y = ?섏쭅媛?
FRONT_VIEW_HORIZONTAL_LIMIT_DEG = 60.0
FRONT_VIEW_VERTICAL_MIN_DEG = -22.5
FRONT_VIEW_VERTICAL_MAX_DEG = 22.5
FRONT_VIEW_MAX_DISTANCE_M = 120.0

# 以묒떖???섏쭅 ?꾨줈?뚯씪.
# 李⑥껜 ?뺣㈃ 0?꾩뿉 媛??媛源뚯슫 ?섑룊 鍮붿쓣 怨⑤씪 紐⑤뱺 ?섏쭅 梨꾨꼸??痢〓㈃ ?⑤㈃?쇰줈 ?쒖떆?쒕떎.
FRONT_PROFILE_TARGET_ANGLE_DEG = 0.0

# ----------------------------------------------------------------------------
# YOLO ?쒓컖 ?쒖쟻 ?몄떇怨?議곗? 蹂댁“
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "best_8s.pt"
VISION_CONFIDENCE_MIN = 0.25
YOLO_CLASS_TO_LIDAR_GEOMETRY = {
    "ally": "thin",
    "enemy": "thin",
    "rock": "bulky",
    "rock_l": "bulky",
    "tank": "bulky",
    "tank_ally": "bulky",
    "tank_enemy": "bulky",
    "tank_ally_back": "bulky",
    "tank_ally_front": "bulky",
    "tank_ally_side": "bulky",
    "tank_enemy_back": "bulky",
    "tank_enemy_front": "bulky",
    "tank_enemy_side": "bulky",
    "tent": "bulky",
    "car": "bulky",
    "car1": "bulky",
    "car2": "bulky",
    "human": "thin",
}
VISION_TARGET_CLASSES = {
    "enemy",
    "tank",
    "tank_enemy",
    "tank_enemy_back",
    "tank_enemy_front",
    "tank_enemy_side",
}
VISION_NEVER_ATTACK_CLASSES = {
    "ally",
    "car",
    "car1",
    "car2",
    "human",
    "rock",
    "rock_l",
    "tent",
}
VISION_TARGET_HOLD_SECONDS = 1.2
VISION_DETECT_MAX_FPS = 20.0
VISION_DETECT_MIN_INTERVAL_SECONDS = 1.0 / VISION_DETECT_MAX_FPS
VISION_AIM_DEADBAND_X = 0.015
VISION_AIM_DEADBAND_Y = 0.025
VISION_DECISION_CENTER_X = 0.12
VISION_DECISION_CENTER_Y = 0.25
VISION_TANK_GEOMETRY_OVERRIDE_X = 0.08
VISION_TANK_GEOMETRY_OVERRIDE_CONFIDENCE = 0.65
VISION_TANK_GEOMETRY_OVERRIDE_MIN_POINTS = 4
LIDAR_TANK_RESCUE_MIN_WIDTH_M = 5.5
LIDAR_TANK_RESCUE_MIN_POINTS = 5
VISION_AIM_ZERO_STEP_Y = 0.01
VISION_TURRET_WEIGHT_MIN = 0.06
VISION_TURRET_WEIGHT_MAX = 0.18
VISION_BODY_WEIGHT_MIN = 0.06
VISION_BODY_WEIGHT_MAX = 0.18
LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG = 10.0
FIRE_APPROVAL_SECONDS = 1.5

# LiDAR ?꾩슜 議곗? 紐⑤뱶.
# 議곗?怨?諛쒖궗?먯꽌??YOLO 媛먯?/留ㅽ븨??臾댁떆?쒕떎.
# YOLO ?대옒???몄떇??遺덉븞?뺥븷 ???ъ슜?쒕떎.
USE_YOLO_FOR_AIM = False
USE_YOLO_FOR_RECOGNITION = True
USE_YOLO_FIRE_GUARD = False
# LiDAR ???쒖뼱: ???곕뱶諛대뱶瑜??섏쑝硫??ы깙 蹂댁젙???쒖옉?쒕떎.
LIDAR_TURRET_YAW_DEADBAND_DEG = 0.25
LIDAR_TURRET_YAW_WEIGHT_MIN = 0.06
LIDAR_TURRET_YAW_WEIGHT_MAX = 0.18
WORLD_TARGET_LOCK_ENABLED = True
WORLD_TARGET_LOCK_HOLD_SECONDS = 2.0
WORLD_TARGET_LOCK_REFRESH_DISTANCE_M = 8.0
WORLD_TARGET_LOCK_SWITCH_DISTANCE_M = 14.0
WORLD_TURRET_PD_KP = 0.018
WORLD_TURRET_PD_KD = 0.003
WORLD_TURRET_DEADBAND_DEG = 0.35
WORLD_TURRET_WEIGHT_MIN = 0.014
WORLD_TURRET_WEIGHT_MAX = 0.12

# ----------------------------------------------------------------------------
# 조준 안정화 제어
# ----------------------------------------------------------------------------
# 목표가 조준점 근처에 들어왔을 때 계속 회전 명령을 보내면,
# 포탑이 목표를 지나쳤다가 다시 되돌아오는 좌우 진동이 생기기 쉽다.
# 이 구간에서는 큰 연속 제어 대신 짧은 단발 펄스를 보내고,
# 다음 LiDAR/API 상태가 갱신될 때까지 잠깐 기다리며 오차 변화를 확인한다.
AIM_STABILIZER_ENABLED = True
AIM_CLOSE_ZONE_DEG = 2.0
AIM_FINE_ZONE_DEG = 0.75
AIM_SETTLE_SECONDS = 0.22
AIM_NEAR_COMMAND_INTERVAL_SECONDS = 0.16
AIM_TRACK_SWITCH_RESET_DEG = 4.0
AIM_FINE_TURRET_WEIGHT_MIN = 0.025
AIM_FINE_TURRET_WEIGHT_MAX = 0.075
AIM_NEAR_BODY_LOCK_DEG = 6.0
AIM_BODY_LOCK_DISTANCE_M = 95.0

# 포탑 펄스 제어:
# Q/E를 계속 누르는 대신 "한 번 작게 움직이고 -> 쿨다운 동안 관찰"한다.
# 오차가 클수록 조금 더 큰 weight를 쓰고, 정조준에 가까울수록 micro pulse만 쓴다.
# 값이 너무 크면 목표를 지나쳐 흔들리고, 너무 작으면 수렴이 느려진다.
TURRET_PULSE_AIM_ENABLED = True
TURRET_PULSE_COOLDOWN_SECONDS = 0.14
TURRET_PULSE_WEIGHT_FAR = 0.075
TURRET_PULSE_WEIGHT_CLOSE = 0.050
TURRET_PULSE_WEIGHT_FINE = 0.028
TURRET_FINE_ZONE_DEG = 1.2
TURRET_CLOSE_ZONE_DEG = 3.0
TURRET_STOP_DEADBAND_DEG = 0.55

# 조준 안정화 응답 지연:
# 목표 근처에서는 /get_action 응답을 아주 짧게 늦춰서
# LiDAR 스캔과 상태 업데이트가 따라올 시간을 준다.
# 값이 너무 길면 전체 컨트롤러가 둔해지므로 작게 유지한다.
AIM_RESPONSE_SLEEP_ENABLED = True
AIM_FINE_RESPONSE_SLEEP_SECONDS = 0.10
AIM_CLOSE_RESPONSE_SLEEP_SECONDS = 0.06

# 諛쒖궗 ?꾩뿉???대떦 LiDAR ?몃옓???좎떆 臾댁떆?쒕떎.
# ?쒕??덉씠?곌? ?쒖쟻???쒓굅?섎뒗 ?숈븞 媛숈? ?쒖쟻???ㅼ떆 ?좉린吏 ?딄퀬,
# ?ㅼ쓬?쇰줈 媛源뚯슫 遺?????꾪뿕 ?꾨낫濡??섏뼱媛寃??섍린 ?꾪븿?대떎.
ELIMINATED_TARGET_IGNORE_SECONDS = 1.75

# ?먮룞 諛쒖궗??湲곕낯?곸쑝濡?爰쇱졇 ?덈떎.
# FIRE 踰꾪듉? 諛쒖궗瑜??뱀씤/臾댁옣留??섎ŉ, /get_action? ?뱀씤 ?쒓컙怨??꾧꺽??議곗? 以鍮꾧?
# 寃뱀튌 ?뚮쭔 ?ㅼ젣 諛쒖궗瑜??몃떎.
AUTO_FIRE_WHEN_STABLE = False
AUTO_FIRE_COOLDOWN_SECONDS = 0.85

# FIRE 踰꾪듉 ?쒖꽦 議곌굔? ?ㅼ젣 諛쒖궗 議곌굔蹂대떎 ?먯뒯?댁빞 ?쒕떎.
# ?ъ슜?먭? ?꾨꼍???뺣젹?섍린 吏곸쟾??FIRE瑜??꾨? ???덇퀬,
# ?뱀씤 ?쒓컙 ?숈븞 ?뺣? 議곗? ?덉젙?붿? ?꾧꺽?????쇱튂 以鍮꾨? 湲곕떎由ш쾶 ?쒕떎.
FIRE_BUTTON_ENABLE_YAW_DEG = 2.2
FIRE_BUTTON_ENABLE_PITCH_DEG = 1.2
FIRE_BUTTON_APPROVAL_SECONDS = 3.0
# ?섎룞 諛쒖궗 ?숈옉:
# ?쒖쟻???대? 異⑸텇??媛源뚯씠 留욎떠議뚮떎硫?FIRE 踰꾪듉?쇰줈 ?ㅼ젣 諛쒖궗媛 ?섍????쒕떎.
# ?붾쾭洹몄슜 strict ready???좎??섏?留? ?섎룞 諛쒖궗?????먯뒯??寃뚯씠?몃? ?ъ슜?쒕떎.
# ??API 吏?? YOLO rawY ?꾨씫, 怨쇰룄?섍쾶 鍮〓묀??嫄곕━ 湲곕컲 ???꾧퀎媛??뚮Ц??# "Fire armed: waiting for fine aim" ?곹깭??媛뉙엳??寃껋쓣 留됰뒗??
MANUAL_FIRE_USE_LOOSE_GATE = True
MANUAL_FIRE_YAW_DEG = 1.2
MANUAL_FIRE_REQUIRE_PITCH_CLOSE = False

# LiDAR ?꾩슜 諛쒖궗 ??媛?? 癒??쒖쟻?쇱닔濡????묒? 媛곷룄 ?ㅼ감瑜??붽뎄?쒕떎.
# ?덉슜?섎뒗 ?〓갑??鍮쀫굹媛?嫄곕━瑜?????쇱젙 踰붿쐞濡?臾띠뼱 ?붾떎.
LIDAR_FIRE_LATERAL_TOLERANCE_M = 0.45
LIDAR_FIRE_YAW_DEADBAND_MIN_DEG = 0.25
LIDAR_FIRE_YAW_DEADBAND_MAX_DEG = 0.70
# ?명솚?깃낵 ?붾쾭洹??쒖떆 fallback?⑹쑝濡??좎??쒕떎. ?ㅼ젣 諛쒖궗???숈쟻 ?곕뱶諛대뱶瑜??ъ슜?쒕떎.
LIDAR_FIRE_YAW_DEADBAND_DEG = 0.35

AUTO_BALLISTIC_PITCH_ENABLED = True
# 嫄곕━ 湲곕컲 ?쇱튂 ?ㅽ봽?? 癒??쒖쟻? 媛源뚯슫 ?쒖쟻蹂대떎 ???믪? ?꾨룄 蹂댁젙??諛쏅뒗??
BALLISTIC_PITCH_OFFSET_NEAR_DEG = 0.20
BALLISTIC_PITCH_OFFSET_FAR_DEG = 0.65
BALLISTIC_PITCH_OFFSET_FAR_START_M = 50.0
BALLISTIC_PITCH_OFFSET_FAR_END_M = 85.0
BALLISTIC_PITCH_DEADBAND_DEG = 0.05
BALLISTIC_PITCH_WEIGHT_MIN = 0.10
BALLISTIC_PITCH_WEIGHT_MAX = 0.25
BALLISTIC_DISTANCE_PITCH_TABLE = [
    (20.8, -5.00),
    (22.0, -4.72),
    (24.2, -3.86),
    (31.0, -2.00),
    (33.2, -1.54),
    (36.6, -0.89),
    (41.1, -0.20),
    (45.5, 0.55),
    (50.0, 1.13),
    (55.6, 1.82),
    (60.1, 2.51),
    (64.5, 2.96),
    (70.1, 3.60),
    (75.6, 4.29),
    (83.3, 5.17),
    (89.9, 5.82),
    (97.6, 6.60),
    (100.8, 7.00),
    (104.1, 7.36),
    (107.3, 7.65),
    (108.4, 7.83),
    (111.7, 8.08),
    (115.9, 8.55),
    (120.2, 8.98),
    (129.8, 9.76),
]
FRONT_PROFILE_MAX_SELECT_ANGLE_ERROR_DEG = 2.0
FRONT_PROFILE_MAX_DISTANCE_M = 120.0
FRONT_PROFILE_FIT_MAX_DISTANCE_M = 35.0
FRONT_PROFILE_MIN_FIT_POINTS = 3

PRINT_INTERVAL_SECONDS = 0.5
AUTO_DRIVE_ENABLED = False

state_lock = Lock()
vision_detect_lock = Lock()
last_print_time = 0.0
last_vision_detect_run_time = 0.0
yolo_model = None
terrain_history: deque[dict[float, dict[str, Any]]] = deque(maxlen=TERRAIN_HISTORY_SIZE)
latest_raw_info: dict[str, Any] = {}
latest_state: dict[str, Any] = {
    "simulationTime": None,
    "terrainSectors": [],
    "terrainDecision": {},
    "contourPoints": [],
    "frontVerticalProfile": {},
    "rawObjects": [],
    "trackedObjects": [],
    "fusionPriorityQueue": [],
    "primaryFusionTarget": None,
    "scanTarget": None,
    "bodyAlignment": {},
    "visionTarget": None,
    "visionDetections": [],
    "lidarVisionFusion": None,
    "impactMarkers": [],
}

body_alignment_state: dict[str, Any] = {
    "enabled": AUTO_BODY_ALIGN_ENABLED,
    "lockedTrackId": None,
    "target": None,
    "moveAD": {"command": "", "weight": 0.0},
    "aligned": False,
    "reason": "waiting_for_target",
    "stickyLock": False,
    "pendingTrackId": None,
    "pendingHits": 0,
    "pendingCenterAngle": None,
    "pendingDistance": None,
    "lockedCenterAngle": None,
    "lockedDistance": None,
    "alignedSince": None,
    "decisionReadyAt": None,
    "postAimDecisionHoldRemaining": None,
    "nextTrackIdMin": 1,
    "rejectedTrackIds": [],
    "acceptedTankTrackIds": [],
    "lastJudgedTrackId": None,
    "lastJudgement": None,
    "lastExhaustedCandidateSignature": None,
    "scanRound": 1,
    "lastScanResetReason": None,
}

action_debug_state: dict[str, Any] = {
    "getActionRequestCount": 0,
    "lastRequestBody": {},
    "lastResponse": {},
    "lastRequestedAt": None,
}

fire_control_state: dict[str, Any] = {
    "approvedUntil": 0.0,
    "approvedAt": None,
    "lastFiredAt": None,
    "lastAutoFireAt": 0.0,
    "fireCount": 0,
}
aim_zero_state: dict[str, Any] = {
    "offsetY": 0.0,
    "updatedAt": None,
}

aim_stabilizer_state: dict[str, Any] = {
    "trackId": None,
    "insideFineSince": None,
    "lastNearCommandAt": 0.0,
    "lastYawCommandAt": 0.0,
    "settled": False,
    "lastYawErrorDeg": None,
    "lastDistance": None,
    "lastPulseWeight": 0.0,
    "reason": "waiting_for_target",
}

world_target_lock_state: dict[str, Any] = {
    "active": False,
    "targetKey": None,
    "trackId": None,
    "worldX": None,
    "worldZ": None,
    "distance": None,
    "lockedAt": 0.0,
    "lastSeenAt": 0.0,
    "lastErrorDeg": None,
    "lastErrorAt": 0.0,
    "lastCommand": {"command": "", "weight": 0.0},
    "yawReference": None,
    "reason": "waiting_for_target",
}

eliminated_target_state: dict[str, Any] = {
    "ignoredTracks": {},
    "lastFiredTrackId": None,
    "lastMarkedAt": None,
}

vision_state: dict[str, Any] = {
    "target": None,
    "detections": [],
    "lastDetectedAt": 0.0,
    "lastProcessedAt": 0.0,
    "lastProcessedTrackId": None,
    "lastDetectRequestAt": 0.0,
    "lastDetectRequestMode": None,
    "lastDetectRequestHadImage": False,
    "modelPath": str(MODEL_PATH),
    "modelLoaded": False,
    "modelLoadError": None,
    "lastInferenceError": None,
    "lidarFusion": None,
}
recognized_lidar_objects: dict[int, dict[str, Any]] = {}
impact_history: deque[dict[str, Any]] = deque(maxlen=IMPACT_HISTORY_SIZE)
next_impact_id = 1


# =============================================================================
# 1. 湲곕낯 ?ы띁
# =============================================================================
def normalize_signed_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def angular_distance_deg(a: float, b: float) -> float:
    return abs(normalize_signed_angle(a - b))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        try:
            yolo_model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            with state_lock:
                vision_state["modelLoaded"] = False
                vision_state["lastDetectRequestMode"] = "model_load_error"
                vision_state["modelLoadError"] = str(exc)
                vision_state["modelPath"] = str(MODEL_PATH)
            raise
        with state_lock:
            vision_state["modelLoaded"] = True
            vision_state["modelLoadError"] = None
            vision_state["modelPath"] = str(MODEL_PATH)
    return yolo_model


def aim_weight(error_ratio: float, deadband: float, low: float, high: float) -> float:
    magnitude = abs(error_ratio)
    if magnitude <= deadband:
        return 0.0
    scaled = clamp((magnitude - deadband) / max(0.001, 0.5 - deadband), 0.0, 1.0)
    return round(low + (high - low) * scaled, 3)


def interpolate_ballistic_pitch(distance_m: float | None) -> float | None:
    if distance_m is None:
        return None
    table = BALLISTIC_DISTANCE_PITCH_TABLE
    distance = float(distance_m)
    if distance <= table[0][0]:
        return table[0][1]
    if distance >= table[-1][0]:
        return table[-1][1]
    for (left_distance, left_pitch), (right_distance, right_pitch) in zip(table, table[1:]):
        if left_distance <= distance <= right_distance:
            ratio = (distance - left_distance) / max(0.001, right_distance - left_distance)
            return left_pitch + (right_pitch - left_pitch) * ratio
    return None


def distance_based_pitch_offset(distance_m: float | None) -> float:
    """?꾩옱 ?쒖쟻 嫄곕━ 湲곗???異붽? ?ъ떊 怨좉컖??諛섑솚?쒕떎.

    媛源뚯슫 ?쒖쟻? ?묒? ?ㅽ봽?뗭쓣 ?좎??섍퀬, 癒??쒖쟻? ?????ㅽ봽?뗭쓣 諛쏆븘
    嫄곕━媛 ?섏뼱???꾩씠 吏㏐쾶 ?⑥뼱吏吏 ?딄쾶 ?쒕떎.
    """
    if distance_m is None:
        return BALLISTIC_PITCH_OFFSET_NEAR_DEG

    distance = float(distance_m)
    if distance <= BALLISTIC_PITCH_OFFSET_FAR_START_M:
        return BALLISTIC_PITCH_OFFSET_NEAR_DEG
    if distance >= BALLISTIC_PITCH_OFFSET_FAR_END_M:
        return BALLISTIC_PITCH_OFFSET_FAR_DEG

    ratio = (
        (distance - BALLISTIC_PITCH_OFFSET_FAR_START_M)
        / max(0.001, BALLISTIC_PITCH_OFFSET_FAR_END_M - BALLISTIC_PITCH_OFFSET_FAR_START_M)
    )
    return BALLISTIC_PITCH_OFFSET_NEAR_DEG + (
        BALLISTIC_PITCH_OFFSET_FAR_DEG - BALLISTIC_PITCH_OFFSET_NEAR_DEG
    ) * ratio


def distance_based_fire_yaw_deadband(distance_m: float | None) -> float:
    """諛쒖궗??嫄곕━ 湲곕컲 ???곕뱶諛대뱶瑜?諛섑솚?쒕떎.

    怨좎젙 媛곷룄 ?덉슜移섎뒗 ?κ굅由ъ뿉???덈Т ?먯뒯?섎떎.
    誘명꽣 ?⑥쐞 ?〓갑???덉슜 ?ㅼ감瑜?媛곷룄 ?꾧퀎媛믪쑝濡?蹂?섑븳 ??踰붿쐞 ?덉뿉 怨좎젙?쒕떎.
    """
    if distance_m is None or distance_m <= 0:
        return LIDAR_FIRE_YAW_DEADBAND_MAX_DEG

    yaw_deg = degrees(atan(LIDAR_FIRE_LATERAL_TOLERANCE_M / float(distance_m)))
    return clamp(
        yaw_deg,
        LIDAR_FIRE_YAW_DEADBAND_MIN_DEG,
        LIDAR_FIRE_YAW_DEADBAND_MAX_DEG,
    )


def ballistic_pitch_control(lidar_target: dict[str, Any] | None = None) -> dict[str, Any]:
    lidar_target = lidar_target or current_lidar_fusion_target(latest_state)
    turret = latest_turret_from_info(latest_raw_info)
    current_pitch = safe_float(turret.get("pitch"))
    distance = safe_float((lidar_target or {}).get("nearestDistance"))
    target_pitch = interpolate_ballistic_pitch(distance)
    pitch_offset = distance_based_pitch_offset(distance)
    if target_pitch is not None:
        target_pitch += pitch_offset

    status = {
        "enabled": AUTO_BALLISTIC_PITCH_ENABLED,
        "ready": False,
        "reason": "waiting_for_lidar_target",
        "distance": round_or_none(distance),
        "currentPitch": round_or_none(current_pitch),
        "targetPitch": round_or_none(target_pitch),
        "pitchOffset": round_or_none(pitch_offset),
        "pitchError": None,
        "turretRF": {"command": "", "weight": 0.0},
    }
    if not AUTO_BALLISTIC_PITCH_ENABLED:
        status["reason"] = "ballistic_pitch_disabled"
        return status
    if not lidar_target or distance is None:
        return status
    if current_pitch is None or target_pitch is None:
        status["reason"] = "missing_current_or_target_pitch"
        return status

    pitch_error = float(target_pitch) - float(current_pitch)
    status["pitchError"] = round(pitch_error, 3)
    if abs(pitch_error) <= BALLISTIC_PITCH_DEADBAND_DEG:
        status["ready"] = True
        status["reason"] = "pitch_inside_ballistic_deadband"
        return status

    weight = aim_weight(
        pitch_error / 10.0,
        BALLISTIC_PITCH_DEADBAND_DEG / 10.0,
        BALLISTIC_PITCH_WEIGHT_MIN,
        BALLISTIC_PITCH_WEIGHT_MAX,
    )
    # ?꾩옱 ?쒕??덉씠??留ㅽ븨?먯꽌??R???쇱튂瑜??щ━怨? F媛 ?쇱튂瑜??대┛??
    status["turretRF"] = {"command": "R" if pitch_error > 0 else "F", "weight": weight}
    status["reason"] = "raise_pitch_to_ballistic_solution" if pitch_error > 0 else "lower_pitch_to_ballistic_solution"
    return status


def vision_aim_commands(target: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not target:
        empty = {"command": "", "weight": 0.0}
        return deepcopy(empty), deepcopy(empty), deepcopy(empty)

    error_x = float(target.get("errorX", 0.0))
    raw_error_y = float(target.get("errorY", 0.0))
    error_y = raw_error_y - float(aim_zero_state.get("offsetY", 0.0) or 0.0)
    body_weight = aim_weight(error_x, VISION_AIM_DEADBAND_X, VISION_BODY_WEIGHT_MIN, VISION_BODY_WEIGHT_MAX)
    turret_yaw_weight = aim_weight(error_x, VISION_AIM_DEADBAND_X, VISION_TURRET_WEIGHT_MIN, VISION_TURRET_WEIGHT_MAX)
    turret_pitch_weight = aim_weight(error_y, VISION_AIM_DEADBAND_Y, VISION_TURRET_WEIGHT_MIN, VISION_TURRET_WEIGHT_MAX)

    move_ad = {"command": "", "weight": 0.0}
    turret_qe = {"command": "", "weight": 0.0}
    turret_rf = {"command": "", "weight": 0.0}

    if body_weight:
        move_ad = {"command": "D" if error_x > 0 else "A", "weight": body_weight}
    if turret_yaw_weight:
        turret_qe = {"command": "E" if error_x > 0 else "Q", "weight": turret_yaw_weight}
    if turret_pitch_weight:
        turret_rf = {"command": "F" if error_y > 0 else "R", "weight": turret_pitch_weight}

    return move_ad, turret_qe, turret_rf


def reset_aim_stabilizer(reason: str = "reset") -> None:
    aim_stabilizer_state.update({
        "trackId": None,
        "insideFineSince": None,
        "lastNearCommandAt": 0.0,
        "lastYawCommandAt": 0.0,
        "settled": False,
        "lastYawErrorDeg": None,
        "lastDistance": None,
        "lastPulseWeight": 0.0,
        "reason": reason,
    })


def lidar_turret_yaw_control(lidar_target: dict[str, Any] | None) -> dict[str, Any]:
    """LiDAR 목표를 기준으로 포탑 yaw(Q/E)를 안정적으로 보정한다.

    Q = 포탑 좌회전, E = 포탑 우회전.

    목표 근처에서는 연속 입력보다 단발 펄스가 안정적이다. 한 번 움직인 뒤
    TURRET_PULSE_COOLDOWN_SECONDS 동안 새 LiDAR/API 상태를 기다려 오차가
    줄었는지 확인한다. 이 방식은 조준점 근처의 좌우 진동을 줄이기 위한
    "한 번 움직임 -> 관찰 -> 다시 판단" 제어다.
    """
    if not lidar_target:
        reset_aim_stabilizer("waiting_for_lidar_target")
        return {"command": "", "weight": 0.0}

    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
        999.0,
    )
    if angle_error is None:
        reset_aim_stabilizer("missing_lidar_yaw_error")
        return {"command": "", "weight": 0.0}

    now = monotonic()
    angle_error = normalize_signed_angle(float(angle_error))
    abs_error = abs(angle_error)
    distance = safe_float(lidar_target.get("nearestDistance"))
    track_id = lidar_target.get("trackId")

    fire_deadband = distance_based_fire_yaw_deadband(distance)
    stop_deadband = max(float(TURRET_STOP_DEADBAND_DEG), fire_deadband)
    fine_zone = max(float(TURRET_FINE_ZONE_DEG), stop_deadband * 1.25)
    close_zone = max(float(TURRET_CLOSE_ZONE_DEG), fine_zone * 2.0)

    previous_track = aim_stabilizer_state.get("trackId")
    previous_error = safe_float(aim_stabilizer_state.get("lastYawErrorDeg"))
    switched_track = previous_track is not None and track_id is not None and previous_track != track_id
    jumped_error = previous_error is not None and abs(angle_error - previous_error) >= AIM_TRACK_SWITCH_RESET_DEG

    if switched_track or jumped_error:
        aim_stabilizer_state.update({
            "trackId": track_id,
            "insideFineSince": None,
            "lastYawCommandAt": 0.0,
            "settled": False,
            "lastPulseWeight": 0.0,
            "reason": "target_or_error_changed_reset_pulse_timer",
        })
    else:
        aim_stabilizer_state["trackId"] = track_id

    aim_stabilizer_state["lastYawErrorDeg"] = round(angle_error, 3)
    aim_stabilizer_state["lastDistance"] = round_or_none(distance)
    # 정지 구간: 이미 발사 허용 오차 안에 들어왔으므로 더 움직이지 않는다.
    # 바로 settled=True로 두지 않고 AIM_SETTLE_SECONDS만큼 유지되는지 본다.
    # 순간적으로 오차가 작아진 프레임과 실제로 안정된 상태를 구분하기 위해서다.
    if abs_error <= stop_deadband:
        if aim_stabilizer_state.get("insideFineSince") is None:
            aim_stabilizer_state["insideFineSince"] = now
            aim_stabilizer_state["settled"] = False
            aim_stabilizer_state["reason"] = "inside_stop_deadband_waiting_for_settle"
        settled_for = now - float(aim_stabilizer_state.get("insideFineSince") or now)
        if settled_for >= AIM_SETTLE_SECONDS:
            aim_stabilizer_state["settled"] = True
            aim_stabilizer_state["reason"] = "pulse_aim_settled"
        return {"command": "", "weight": 0.0}
    # 정지 구간을 벗어났으므로 안정 판정은 취소하고 다시 조준을 진행한다.
    aim_stabilizer_state["insideFineSince"] = None
    aim_stabilizer_state["settled"] = False
    # 아주 작은 LiDAR yaw 오차는 센서/반올림 노이즈일 수 있으므로 무시한다.
    # stop_deadband보다 좁은 보호막 역할을 하며, 미세 진동 명령을 막는다.
    if abs_error <= LIDAR_TURRET_YAW_DEADBAND_DEG:
        aim_stabilizer_state["reason"] = "inside_lidar_yaw_deadband"
        return {"command": "", "weight": 0.0}
    # 펄스 쿨다운: 이전 Q/E 명령 직후에는 새 명령을 보내지 않는다.
    # 시뮬레이터와 LiDAR 상태가 반영되기 전에 추가 입력을 보내면
    # 포탑이 과하게 움직여 목표를 지나칠 수 있다.
    if TURRET_PULSE_AIM_ENABLED:
        last_cmd = float(aim_stabilizer_state.get("lastYawCommandAt", 0.0) or 0.0)
        elapsed = now - last_cmd
        if elapsed < TURRET_PULSE_COOLDOWN_SECONDS:
            aim_stabilizer_state["reason"] = "pulse_cooldown_waiting_for_fresh_state"
            return {"command": "", "weight": 0.0}
    # 오차 구간에 따라 보수적인 펄스 가중치를 선택한다.
    # fine_zone: 정조준 근처라 micro pulse만 사용한다.
    # close_zone: 아직 약간 벗어나 있으므로 작은 단발 펄스를 쓴다.
    # far_zone: 오차가 큰 편이지만 그래도 연속 회전 대신 제한된 펄스만 보낸다.

    if abs_error <= fine_zone:
        weight = float(TURRET_PULSE_WEIGHT_FINE)
        reason = "fine_zone_single_micro_pulse"
    elif abs_error <= close_zone:
        weight = float(TURRET_PULSE_WEIGHT_CLOSE)
        reason = "close_zone_single_pulse"
    else:
        weight = float(TURRET_PULSE_WEIGHT_FAR)
        reason = "far_zone_single_pulse"

    aim_stabilizer_state["lastYawCommandAt"] = now
    aim_stabilizer_state["lastNearCommandAt"] = now
    aim_stabilizer_state["lastPulseWeight"] = round(weight, 3)
    aim_stabilizer_state["reason"] = reason

    return {
        "command": "E" if angle_error > 0.0 else "Q",
        "weight": round(weight, 3),
    }


def lidar_target_world_position(lidar_target: dict[str, Any] | None) -> dict[str, float] | None:
    if not lidar_target:
        return None
    pos = lidar_target.get("filteredWorldPosition") or lidar_target.get("worldPosition") or {}
    if not isinstance(pos, dict):
        return None
    x = safe_float(pos.get("x"))
    z = safe_float(pos.get("z"))
    if x is None or z is None:
        return None
    return {"x": float(x), "z": float(z)}


def clear_world_target_lock(reason: str = "cleared") -> None:
    world_target_lock_state.update({
        "active": False,
        "targetKey": None,
        "trackId": None,
        "worldX": None,
        "worldZ": None,
        "distance": None,
        "lockedAt": 0.0,
        "lastSeenAt": 0.0,
        "lastErrorDeg": None,
        "lastErrorAt": 0.0,
        "lastCommand": {"command": "", "weight": 0.0},
        "yawReference": None,
        "reason": reason,
    })


def refresh_world_target_lock(lidar_target: dict[str, Any] | None, now: float | None = None) -> bool:
    if not WORLD_TARGET_LOCK_ENABLED or not lidar_target:
        clear_world_target_lock("waiting_for_lidar_target")
        return False
    pos = lidar_target_world_position(lidar_target)
    if not pos:
        clear_world_target_lock("target_missing_world_position")
        return False

    now = monotonic() if now is None else now
    target_key = target_ignore_key(lidar_target)
    track_id = lidar_target.get("trackId")
    current_x = safe_float(world_target_lock_state.get("worldX"))
    current_z = safe_float(world_target_lock_state.get("worldZ"))
    active = bool(world_target_lock_state.get("active"))
    expired = active and now - float(world_target_lock_state.get("lastSeenAt", 0.0) or 0.0) > WORLD_TARGET_LOCK_HOLD_SECONDS
    same_target = bool(
        active
        and (
            (target_key is not None and target_key == world_target_lock_state.get("targetKey"))
            or (track_id is not None and track_id == world_target_lock_state.get("trackId"))
        )
    )
    close_to_lock = bool(
        active
        and current_x is not None
        and current_z is not None
        and hypot(float(pos["x"]) - float(current_x), float(pos["z"]) - float(current_z))
        <= WORLD_TARGET_LOCK_REFRESH_DISTANCE_M
    )
    far_new_target = bool(
        active
        and current_x is not None
        and current_z is not None
        and hypot(float(pos["x"]) - float(current_x), float(pos["z"]) - float(current_z))
        >= WORLD_TARGET_LOCK_SWITCH_DISTANCE_M
    )

    if not active or expired or same_target or close_to_lock or far_new_target:
        locked_at = float(world_target_lock_state.get("lockedAt", 0.0) or 0.0)
        if not active or expired or far_new_target:
            locked_at = now
        world_target_lock_state.update({
            "active": True,
            "targetKey": target_key,
            "trackId": track_id,
            "worldX": round(float(pos["x"]), 3),
            "worldZ": round(float(pos["z"]), 3),
            "distance": round_or_none(lidar_target.get("nearestDistance")),
            "lockedAt": locked_at,
            "lastSeenAt": now,
            "reason": "world_position_locked" if not active or expired or far_new_target else "world_position_refreshed",
        })
        return True

    world_target_lock_state["reason"] = "holding_previous_world_lock"
    return True


def world_target_yaw_error_deg() -> float | None:
    if not bool(world_target_lock_state.get("active")):
        return None
    pos = latest_player_position_from_info(latest_raw_info)
    player_x = safe_float(pos.get("x"))
    player_z = safe_float(pos.get("z"))
    target_x = safe_float(world_target_lock_state.get("worldX"))
    target_z = safe_float(world_target_lock_state.get("worldZ"))
    if player_x is None or player_z is None or target_x is None or target_z is None:
        world_target_lock_state["reason"] = "missing_player_or_target_position"
        return None
    body_yaw = body_yaw_from_info(latest_raw_info)
    turret_yaw = safe_float(latest_turret_from_info(latest_raw_info).get("yaw"))
    desired_absolute = normalize_signed_angle(degrees(atan2(target_x - player_x, target_z - player_z)))
    if turret_yaw is None:
        desired_body_relative = normalize_signed_angle(desired_absolute - body_yaw)
        return desired_body_relative
    absolute_error = normalize_signed_angle(desired_absolute - float(turret_yaw))
    desired_body_relative = normalize_signed_angle(desired_absolute - body_yaw)
    relative_error = normalize_signed_angle(desired_body_relative - float(turret_yaw))
    if abs(absolute_error) <= abs(relative_error):
        world_target_lock_state["yawReference"] = "absolute_turret_yaw"
        return absolute_error
    world_target_lock_state["yawReference"] = "relative_turret_yaw"
    return relative_error


def world_target_turret_yaw_control(lidar_target: dict[str, Any] | None) -> dict[str, Any]:
    # 월드 좌표로 잠근 목표가 있으면, 현재 LiDAR 프레임의 흔들림보다
    # 고정된 world position을 우선해 포탑 yaw 오차를 계산한다.
    # 잠금이 없거나 갱신에 실패하면 일반 LiDAR 기반 펄스 제어로 되돌아간다.
    now = monotonic()
    if not refresh_world_target_lock(lidar_target, now):
        reset_aim_stabilizer(world_target_lock_state.get("reason", "world_lock_unavailable"))
        return lidar_turret_yaw_control(lidar_target)

    angle_error = world_target_yaw_error_deg()
    if angle_error is None:
        return lidar_turret_yaw_control(lidar_target)

    abs_error = abs(angle_error)
    distance = safe_float((lidar_target or {}).get("nearestDistance"), world_target_lock_state.get("distance"))
    fire_deadband = max(distance_based_fire_yaw_deadband(distance), WORLD_TURRET_DEADBAND_DEG)
    previous_error = safe_float(world_target_lock_state.get("lastErrorDeg"))
    previous_time = float(world_target_lock_state.get("lastErrorAt", 0.0) or 0.0)

    world_target_lock_state["lastErrorDeg"] = round(angle_error, 3)
    world_target_lock_state["lastErrorAt"] = now
    aim_stabilizer_state["trackId"] = world_target_lock_state.get("trackId")
    aim_stabilizer_state["lastYawErrorDeg"] = round(angle_error, 3)
    aim_stabilizer_state["lastDistance"] = round_or_none(distance)

    # 발사 허용 오차 안에서는 더 움직이지 않고 안정 지속 시간을 확인한다.
    # 일반 펄스 제어와 같은 aim_stabilizer_state를 써서 발사 준비 판단을 공유한다.
    if abs_error <= fire_deadband:
        if aim_stabilizer_state.get("insideFineSince") is None:
            aim_stabilizer_state["insideFineSince"] = now
            aim_stabilizer_state["settled"] = False
        settled_for = now - float(aim_stabilizer_state.get("insideFineSince") or now)
        if settled_for >= AIM_SETTLE_SECONDS:
            aim_stabilizer_state["settled"] = True
        command = {"command": "", "weight": 0.0}
        aim_stabilizer_state["reason"] = "world_lock_inside_deadband"
        world_target_lock_state["lastCommand"] = command
        world_target_lock_state["reason"] = "world_lock_aim_settling" if not aim_stabilizer_state["settled"] else "world_lock_aim_settled"
        return command

    aim_stabilizer_state["insideFineSince"] = None
    aim_stabilizer_state["settled"] = False
    # 월드락 제어는 단발 펄스 대신 PD 제어를 사용한다.
    # P는 현재 오차를 줄이고, D는 오차 변화 속도를 보며 과한 추월을 줄인다.
    derivative = 0.0
    if previous_error is not None and previous_time > 0.0:
        dt = max(0.03, now - previous_time)
        derivative = (angle_error - float(previous_error)) / dt
    control = WORLD_TURRET_PD_KP * angle_error + WORLD_TURRET_PD_KD * derivative
    weight = min(WORLD_TURRET_WEIGHT_MAX, max(WORLD_TURRET_WEIGHT_MIN, abs(control)))
    command = {"command": "E" if angle_error > 0.0 else "Q", "weight": round(weight, 3)}

    aim_stabilizer_state["lastYawCommandAt"] = now
    aim_stabilizer_state["lastNearCommandAt"] = now
    aim_stabilizer_state["lastPulseWeight"] = round(weight, 3)
    aim_stabilizer_state["reason"] = "world_lock_pd_turret_control"
    world_target_lock_state["lastCommand"] = command
    world_target_lock_state["reason"] = "world_lock_pd_turret_control"
    return command


def should_lock_body_near_lidar_target(lidar_target: dict[str, Any] | None) -> bool:
    if not AIM_STABILIZER_ENABLED or not lidar_target:
        return False
    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
        999.0,
    )
    distance = safe_float(lidar_target.get("nearestDistance"))
    if angle_error is None:
        return False
    if distance is not None and distance > AIM_BODY_LOCK_DISTANCE_M:
        return False
    return abs(normalize_signed_angle(float(angle_error))) <= AIM_NEAR_BODY_LOCK_DEG


def aim_stabilizer_ready() -> bool:
    return bool(AIM_STABILIZER_ENABLED and aim_stabilizer_state.get("settled"))


def cleanup_eliminated_targets(now: float | None = None) -> None:
    now = monotonic() if now is None else now
    ignored = eliminated_target_state.get("ignoredTracks", {}) or {}
    expired = [
        key for key, until in ignored.items()
        if safe_float(until, 0.0) is not None and now >= float(until)
    ]
    for key in expired:
        ignored.pop(key, None)
    eliminated_target_state["ignoredTracks"] = ignored


def target_ignore_key(obj: dict[str, Any] | None) -> str | None:
    if not obj:
        return None
    track_id = obj.get("trackId")
    if track_id is not None:
        return f"track:{track_id}"
    object_id = obj.get("id")
    if object_id is not None:
        return f"object:{object_id}"
    return None


def is_temporarily_eliminated_target(obj: dict[str, Any] | None, now: float | None = None) -> bool:
    key = target_ignore_key(obj)
    if key is None:
        return False
    cleanup_eliminated_targets(now)
    ignored = eliminated_target_state.get("ignoredTracks", {}) or {}
    return key in ignored


def mark_target_as_eliminated_after_fire(lidar_target: dict[str, Any] | None) -> None:
    key = target_ignore_key(lidar_target)
    if key is None:
        return
    now = monotonic()
    cleanup_eliminated_targets(now)
    ignored = eliminated_target_state.get("ignoredTracks", {}) or {}
    ignored[key] = now + ELIMINATED_TARGET_IGNORE_SECONDS
    eliminated_target_state.update({
        "ignoredTracks": ignored,
        "lastFiredTrackId": (lidar_target or {}).get("trackId"),
        "lastMarkedAt": datetime.now().isoformat(timespec="milliseconds"),
    })
    # ?ㅼ쓬 LiDAR ?붿빟?먯꽌 ?ㅼ쓬 ?곗꽑?쒖쐞 ?쒖쟻??怨좊Ⅴ寃?媛뺤젣?쒕떎.
    body_alignment_state["lockedTrackId"] = None
    body_alignment_state["target"] = None
    body_alignment_state["moveAD"] = {"command": "", "weight": 0.0}
    body_alignment_state["stickyLock"] = False
    body_alignment_state["pendingTrackId"] = None
    body_alignment_state["pendingHits"] = 0
    body_alignment_state["pendingCenterAngle"] = None
    body_alignment_state["pendingDistance"] = None
    body_alignment_state["lockedCenterAngle"] = None
    body_alignment_state["lockedDistance"] = None
    body_alignment_state["alignedSince"] = None
    body_alignment_state["decisionReadyAt"] = None
    body_alignment_state["postAimDecisionHoldRemaining"] = None
    track_id = track_id_int(lidar_target)
    if track_id is not None:
        body_alignment_state["nextTrackIdMin"] = max(
            int(body_alignment_state.get("nextTrackIdMin", 1) or 1),
            int(track_id) + 1,
        )
        body_alignment_state["lastJudgedTrackId"] = int(track_id)
        body_alignment_state["lastJudgement"] = "fired_then_advance_to_next_id"
    clear_world_target_lock("shot_emitted_switch_to_next_target")
    reset_aim_stabilizer("shot_emitted_switch_to_next_target")


def precision_aim_response_sleep_seconds(lidar_target: dict[str, Any] | None) -> float:
    if not AIM_RESPONSE_SLEEP_ENABLED or not lidar_target:
        return 0.0
    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
    )
    if angle_error is None:
        return 0.0
    distance = safe_float(lidar_target.get("nearestDistance"))
    fine_zone = max(float(AIM_FINE_ZONE_DEG), distance_based_fire_yaw_deadband(distance))
    close_zone = max(float(AIM_CLOSE_ZONE_DEG), fine_zone * 2.5)
    abs_error = abs(normalize_signed_angle(float(angle_error)))
    if abs_error <= fine_zone:
        return float(AIM_FINE_RESPONSE_SLEEP_SECONDS)
    if abs_error <= close_zone:
        return float(AIM_CLOSE_RESPONSE_SLEEP_SECONDS)
    return 0.0


def active_vision_target() -> dict[str, Any] | None:
    if not USE_YOLO_FOR_AIM:
        return None
    target = vision_state.get("target")
    detected_at = float(vision_state.get("lastDetectedAt", 0.0) or 0.0)
    if target and monotonic() - detected_at <= VISION_TARGET_HOLD_SECONDS:
        return deepcopy(target)
    return None


def fire_readiness_status(now: float | None = None) -> dict[str, Any]:
    now = monotonic() if now is None else now
    target = active_vision_target()
    lidar_target = current_lidar_fusion_target(latest_state)
    fusion = vision_state.get("lidarFusion") or latest_state.get("lidarVisionFusion") or {}
    alignment = latest_state.get("bodyAlignment", {}) or {}

    error_x = abs(float(target.get("errorX", 999.0))) if target else 999.0
    raw_error_y = float(target.get("errorY", 999.0)) if target else 999.0
    adjusted_error_y = raw_error_y - float(aim_zero_state.get("offsetY", 0.0) or 0.0)
    error_y = abs(adjusted_error_y)
    vision_ready = bool(target) and error_x <= VISION_AIM_DEADBAND_X
    fusion_ready = bool(fusion.get("isAttackTarget")) and bool(fusion.get("alignedForFusion"))
    body_angle_error = abs(float((alignment.get("target") or {}).get("bodyRelativeAngleErrorDeg", 999.0)))
    body_ready = bool(
        bool(alignment.get("aligned"))
        or (
            alignment.get("target")
            and body_angle_error <= max(BODY_ALIGN_DEADBAND_DEG, AIM_NEAR_BODY_LOCK_DEG)
        )
    )
    pitch_status = ballistic_pitch_control(lidar_target)
    pitch_ready = bool(pitch_status.get("ready"))

    # LiDAR ?꾩슜 諛쒖궗瑜??꾪븳 異붽? ??媛??
    # 二쇳뻾??李⑥껜 ?뺣젹? ??1???ㅼ감?먯꽌??"?뺣젹???쇰줈 蹂????덉?留?
    # 諛쒖궗?먮뒗 ???꾧꺽????寃뚯씠?멸? ?꾩슂?섎떎.
    aim_angle_error = 999.0
    lock_yaw_error = world_target_yaw_error_deg()
    if lidar_target:
        raw_aim_angle_error = safe_float(
            lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
            999.0,
        )
        aim_angle_error = abs(float(raw_aim_angle_error if raw_aim_angle_error is not None else 999.0))
    if lock_yaw_error is not None:
        aim_angle_error = abs(float(lock_yaw_error))
    lidar_distance = safe_float((lidar_target or {}).get("nearestDistance"))
    fire_yaw_deadband = distance_based_fire_yaw_deadband(lidar_distance)
    lidar_yaw_ready = aim_angle_error <= fire_yaw_deadband

    # LiDAR ?꾩슜 紐⑤뱶: YOLO ?대옒??留ㅽ븨? 議곗??대굹 諛쒖궗瑜?留됱? ?딅뒗??
    fusion_status = str(fusion.get("status", ""))
    fusion_has_vision = bool(fusion.get("vision"))
    known_non_attack = bool(USE_YOLO_FIRE_GUARD and fusion_has_vision and not bool(fusion.get("isAttackTarget")))
    geometry_mismatch = bool(USE_YOLO_FIRE_GUARD and fusion_status == "recognized_geometry_mismatch")
    lidar_attack_confirmed = (
        body_align_track_is_accepted_tank(lidar_target)
        if BODY_ALIGN_SEQUENTIAL_ID_SCAN
        else True
    )

    lidar_ready = (
        bool(lidar_target)
        and lidar_attack_confirmed
        and body_ready
        and pitch_ready
        and lidar_yaw_ready
        and not known_non_attack
        and not geometry_mismatch
    )
    ready = bool(lidar_ready or (USE_YOLO_FOR_AIM and vision_ready and fusion_ready and body_ready and pitch_ready))

    # ?먯뒯??UI 寃뚯씠?? ?꾧꺽??諛쒖궗 以鍮꾧? ?꾩쭅 ?꾨땲?대룄 ?쒖쟻 洹쇱쿂?먯꽌??    # 踰꾪듉?쇰줈 諛쒖궗瑜?臾댁옣?????덇쾶 ?쒕떎.
    # ?ㅼ젣 諛쒖궗???뱀씤 ?쒓컙???쒖꽦?붾맂 ?숈븞 ready == True???뚮쭔 ?섍컙??
    pitch_error_for_button = safe_float(pitch_status.get("pitchError"))
    pitch_close_for_button = (
        pitch_error_for_button is None
        or abs(float(pitch_error_for_button)) <= FIRE_BUTTON_ENABLE_PITCH_DEG
        or pitch_ready
    )
    yaw_close_for_button = (
        bool(lidar_target)
        and aim_angle_error != 999.0
        and aim_angle_error <= FIRE_BUTTON_ENABLE_YAW_DEG
    )
    can_approve_fire = bool(
        ready
        or (
            bool(lidar_target)
            and lidar_attack_confirmed
            and yaw_close_for_button
            and pitch_close_for_button
            and not known_non_attack
            and not geometry_mismatch
        )
        or (USE_YOLO_FOR_AIM and vision_ready and fusion_ready and pitch_close_for_button)
    )
    approved = now <= float(fire_control_state.get("approvedUntil", 0.0) or 0.0)

    if vision_ready and fusion_ready and body_ready:
        reason = "ready_to_fire_yolo_confirmed"
    elif lidar_ready:
        reason = "ready_to_fire_lidar_aligned"
    elif known_non_attack:
        reason = "blocked_by_yolo_non_attack_class"
    elif geometry_mismatch:
        reason = "blocked_by_lidar_yolo_geometry_mismatch"
    elif not lidar_attack_confirmed:
        reason = "waiting_for_enemy_tank_id_confirmation"
    elif not pitch_ready:
        reason = pitch_status.get("reason", "pitch_not_ready")
    elif not lidar_yaw_ready:
        reason = "lidar_yaw_not_ready"
    elif not lidar_target:
        reason = "no_lidar_target"
    elif not fusion_ready:
        reason = "lidar_yolo_fusion_not_attack_ready"
    elif not body_ready:
        reason = "body_not_aligned"
    else:
        reason = "aim_not_centered"

    return {
        "ready": ready,
        "buttonEnabled": can_approve_fire,
        "canApproveFire": can_approve_fire,
        "approved": approved,
        "manualFireGate": bool(
            can_approve_fire
            and (not bool(MANUAL_FIRE_REQUIRE_PITCH_CLOSE) or pitch_close_for_button)
            and (aim_angle_error == 999.0 or aim_angle_error <= MANUAL_FIRE_YAW_DEG)
        ),
        "fireOnNextAction": bool(
            approved
            and (
                ready
                or (
                    MANUAL_FIRE_USE_LOOSE_GATE
                    and can_approve_fire
                    and (not bool(MANUAL_FIRE_REQUIRE_PITCH_CLOSE) or pitch_close_for_button)
                    and (aim_angle_error == 999.0 or aim_angle_error <= MANUAL_FIRE_YAW_DEG)
                )
            )
        ),
        "reason": reason,
        "approvalSeconds": FIRE_BUTTON_APPROVAL_SECONDS,
        "approvedUntil": fire_control_state.get("approvedUntil"),
        "approvedAt": fire_control_state.get("approvedAt"),
        "lastFiredAt": fire_control_state.get("lastFiredAt"),
        "fireCount": fire_control_state.get("fireCount", 0),
        "target": deepcopy(target),
        "worldTargetLock": deepcopy(world_target_lock_state),
        "aimError": {
            "x": round(error_x, 4) if target else None,
            "rawY": round(raw_error_y, 4) if target else None,
            "adjustedY": round(adjusted_error_y, 4) if target else None,
            "y": round(error_y, 4) if target else None,
            "worldLockYawDeg": round_or_none(lock_yaw_error),
            "deadbandX": VISION_AIM_DEADBAND_X,
            "deadbandY": VISION_AIM_DEADBAND_Y,
            "zeroOffsetY": round(float(aim_zero_state.get("offsetY", 0.0) or 0.0), 4),
        },
        "aimZero": deepcopy(aim_zero_state),
        "ballisticPitch": pitch_status,
        "fusionReady": fusion_ready,
        "bodyReady": body_ready,
        "lidarReady": lidar_ready,
        "lidarAttackConfirmed": lidar_attack_confirmed,
        "lidarYawReady": lidar_yaw_ready,
        "lidarYawCloseForButton": yaw_close_for_button,
        "lidarYawErrorDeg": round(aim_angle_error, 3) if aim_angle_error != 999.0 else None,
        "worldLockYawErrorDeg": round_or_none(lock_yaw_error),
        "lidarFireYawDeadbandDeg": round(fire_yaw_deadband, 3),
        "fireButtonYawDeg": FIRE_BUTTON_ENABLE_YAW_DEG,
        "manualFireYawDeg": MANUAL_FIRE_YAW_DEG,
        "manualFireUseLooseGate": MANUAL_FIRE_USE_LOOSE_GATE,
        "manualFireRequirePitchClose": MANUAL_FIRE_REQUIRE_PITCH_CLOSE,
        "fireButtonPitchDeg": FIRE_BUTTON_ENABLE_PITCH_DEG,
        "pitchCloseForButton": pitch_close_for_button,
        "lidarDistance": round_or_none(lidar_distance),
        "pitchReady": pitch_ready,
        "visionReady": vision_ready,
        "knownNonAttack": known_non_attack,
        "geometryMismatch": geometry_mismatch,
        "fusionStatus": fusion_status,
    }


def current_lidar_fusion_target(scan: dict[str, Any] | None = None) -> dict[str, Any] | None:
    scan = scan or latest_state
    alignment = scan.get("bodyAlignment", {}) or {}
    if alignment.get("target"):
        return deepcopy(alignment["target"])
    primary = scan.get("primaryFusionTarget")
    if primary and semantic_is_tank_candidate(primary):
        return deepcopy(primary)
    return None


def current_lidar_recognition_target(scan: dict[str, Any] | None = None) -> dict[str, Any] | None:
    scan = scan or latest_state
    if scan.get("scanTarget"):
        return deepcopy(scan["scanTarget"])
    alignment = scan.get("bodyAlignment", {}) or {}
    if alignment.get("target"):
        return deepcopy(alignment["target"])
    if scan.get("primaryFusionTarget"):
        return deepcopy(scan["primaryFusionTarget"])
    return None


def build_lidar_vision_fusion(
    lidar_target: dict[str, Any] | None,
    vision_target: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not lidar_target:
        return None

    angle_error = safe_float(
        lidar_target.get("bodyRelativeAngleErrorDeg", lidar_target.get("centerAngle")),
        999.0,
    )
    aligned_for_fusion = abs(float(angle_error if angle_error is not None else 999.0)) <= LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG
    fusion = {
        "status": "waiting_for_alignment" if not aligned_for_fusion else "waiting_for_detection",
        "alignedForFusion": aligned_for_fusion,
        "maxBodyErrorDeg": LIDAR_VISION_FUSION_MAX_BODY_ERROR_DEG,
        "lidar": {
            "trackId": body_align_target_id(lidar_target),
            "rawTrackId": lidar_target.get("trackId"),
            "id": lidar_target.get("id"),
            "candidateLabel": lidar_target.get("candidateLabel"),
            "geometryClass": lidar_target.get("geometryClass"),
            "nearestDistance": lidar_target.get("nearestDistance"),
            "centerAngle": lidar_target.get("centerAngle"),
            "bodyRelativeAngleErrorDeg": angle_error,
            "fusionPriorityRank": lidar_target.get("fusionPriorityRank"),
            "fusionPriorityTier": lidar_target.get("fusionPriorityTier"),
        },
        "vision": None,
        "semanticClass": None,
        "isAttackTarget": False,
    }

    if not aligned_for_fusion or not vision_target:
        return fusion

    class_name = str(vision_target.get("className", "unknown"))
    semantic = normalize_yolo_class_name(class_name)
    mapped_geometry = lidar_geometry_for_yolo_class(class_name)
    lidar_geometry = str(lidar_target.get("geometryClass", "unknown"))
    geometry_matches = mapped_geometry is None or mapped_geometry == lidar_geometry
    rescued_by_lidar = lidar_wide_bk_tank_rescue(lidar_target, class_name)
    point_count = int(lidar_target.get("pointCount", 0) or 0)
    centered_tank_override = bool(
        is_attack_yolo_class(class_name)
        and safe_float(vision_target.get("confidence"), 0.0) >= VISION_TANK_GEOMETRY_OVERRIDE_CONFIDENCE
        and abs(safe_float(vision_target.get("errorX"), 999.0) or 999.0) <= VISION_TANK_GEOMETRY_OVERRIDE_X
        and lidar_geometry in {"thin", "bulky"}
        and point_count >= VISION_TANK_GEOMETRY_OVERRIDE_MIN_POINTS
    )
    geometry_matches_for_attack = bool(geometry_matches or centered_tank_override)
    is_attack_target = is_attack_yolo_class(class_name) or rescued_by_lidar
    fusion.update({
        "status": (
            "recognized_attack_target"
            if is_attack_target and geometry_matches_for_attack
            else (
                "recognized_geometry_mismatch"
                if not geometry_matches
                else "recognized_non_attack_target"
            )
        ),
        "vision": deepcopy(vision_target),
        "semanticClass": class_name,
        "mappedLidarGeometry": mapped_geometry,
        "lidarGeometryMatchesYolo": geometry_matches,
        "geometryOverrideByCenteredTank": centered_tank_override,
        "isAttackTarget": is_attack_target and geometry_matches_for_attack,
        "rescuedByLidarWideBk": rescued_by_lidar,
    })
    return fusion


def normalize_yolo_class_name(class_name: Any) -> str:
    return str(class_name or "").strip().lower()


def lidar_geometry_for_yolo_class(class_name: Any) -> str | None:
    return YOLO_CLASS_TO_LIDAR_GEOMETRY.get(normalize_yolo_class_name(class_name))


def is_attack_yolo_class(class_name: Any) -> bool:
    semantic = normalize_yolo_class_name(class_name)
    if semantic in VISION_NEVER_ATTACK_CLASSES:
        return False
    return semantic in VISION_TARGET_CLASSES or semantic.startswith("tank_enemy_")


def lidar_wide_bk_tank_rescue(
    lidar_target: dict[str, Any] | None,
    class_name: Any,
) -> bool:
    if not lidar_target:
        return False
    semantic = normalize_yolo_class_name(class_name)
    if semantic in VISION_NEVER_ATTACK_CLASSES:
        return False
    return False


def semantic_is_known_non_attack(obj: dict[str, Any]) -> bool:
    if obj.get("isAttackTarget"):
        return False
    semantic = normalize_yolo_class_name(
        obj.get("recognizedClass") or obj.get("semanticClass")
    )
    return bool(semantic) and not is_attack_yolo_class(semantic)


def enrich_with_recognition(obj: dict[str, Any]) -> dict[str, Any]:
    track_id = body_align_target_id(obj)
    if track_id is None:
        return obj
    recognition = recognized_lidar_objects.get(int(track_id))
    if not recognition:
        return obj

    enriched = deepcopy(obj)
    class_name = recognition.get("className", "unknown")
    enriched.update({
        "semanticClass": class_name,
        "recognizedClass": class_name,
        "recognizedConfidence": recognition.get("confidence"),
        "recognizedAt": recognition.get("recognizedAt"),
        "recognizedBy": "yolo_lidar_geometry_mapping",
        "isAttackTarget": bool(recognition.get("isAttackTarget", False)),
        "rescuedByLidarWideBk": bool(recognition.get("rescuedByLidarWideBk", False)),
    })
    return enriched


def remember_recognized_lidar_object(
    lidar_target: dict[str, Any] | None,
    vision_target: dict[str, Any] | None,
    fusion: dict[str, Any] | None,
) -> None:
    if not lidar_target or not vision_target or not fusion:
        return
    if not fusion.get("lidarGeometryMatchesYolo", False) and not fusion.get("isAttackTarget", False):
        return
    track_id = body_align_target_id(lidar_target)
    if track_id is None:
        return

    class_name = str(vision_target.get("className", "unknown"))
    recognized_lidar_objects[int(track_id)] = {
        "trackId": int(track_id),
        "rawTrackId": lidar_target.get("trackId"),
        "id": lidar_target.get("id"),
        "className": class_name,
        "confidence": vision_target.get("confidence"),
        "mappedLidarGeometry": lidar_geometry_for_yolo_class(class_name),
        "isAttackTarget": fusion.get("isAttackTarget", False),
        "rescuedByLidarWideBk": fusion.get("rescuedByLidarWideBk", False),
        "recognizedMonoAt": monotonic(),
        "recognizedAt": datetime.now().isoformat(timespec="milliseconds"),
    }


def refresh_latest_state_recognitions() -> None:
    for key in ("trackedObjects", "confirmedObjects", "fusionPriorityQueue"):
        latest_state[key] = [
            enrich_with_recognition(obj)
            for obj in latest_state.get(key, [])
        ]
    if latest_state.get("primaryFusionTarget"):
        latest_state["primaryFusionTarget"] = enrich_with_recognition(latest_state["primaryFusionTarget"])
    alignment = latest_state.get("bodyAlignment", {})
    if alignment.get("target"):
        alignment["target"] = enrich_with_recognition(alignment["target"])
        latest_state["bodyAlignment"] = alignment


def record_impact(data: dict[str, Any]) -> dict[str, Any]:
    marker = make_impact_marker(data)
    with state_lock:
        impact_history.append(marker)
        latest_state["impactMarkers"] = list(impact_history)
    return marker


def bin_center(value: float, width: float, offset: float = 0.0) -> float:
    index = floor((value - offset) / width)
    return offset + (index + 0.5) * width


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def first_present_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = safe_float(data.get(key))
        if value is not None:
            return value
    return None


def nested_position(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("position", "impactPosition", "hitPosition", "bulletPosition", "location", "point"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data if any(key in data for key in ("x", "X", "z", "Z")) else None


def latest_player_position_from_info(data: dict[str, Any]) -> dict[str, float | None]:
    pos = data.get("position") or data.get("playerPosition") or data.get("player_position") or {}
    if not isinstance(pos, dict):
        pos = {}
    x = first_present_float(data, ("x", "X", "playerPosX", "Player_Pos_X", "player_pos_x"))
    y = first_present_float(data, ("y", "Y", "playerPosY", "Player_Pos_Y", "player_pos_y"))
    z = first_present_float(data, ("z", "Z", "playerPosZ", "Player_Pos_Z", "player_pos_z"))
    if x is None:
        x = first_present_float(pos, ("x", "X"))
    if y is None:
        y = first_present_float(pos, ("y", "Y"))
    if z is None:
        z = first_present_float(pos, ("z", "Z"))
    return {"x": round_or_none(x), "y": round_or_none(y), "z": round_or_none(z)}


def body_yaw_from_info(data: dict[str, Any]) -> float:
    yaw = first_present_float(
        data,
        (
            "playerBodyX",
            "Player_Body_X",
            "player_body_x",
            "bodyX",
            "body_x",
            "bodyYaw",
            "body_yaw",
        ),
    )
    if yaw is not None:
        return yaw
    rotation = data.get("lidarRotation", {}) or {}
    if isinstance(rotation, dict):
        return safe_float(rotation.get("y"), 0.0) or 0.0
    return 0.0


def latest_turret_from_info(data: dict[str, Any]) -> dict[str, float | None]:
    turret = data.get("turret", {}) or data.get("playerTurret", {}) or data.get("Player_Turret", {}) or {}
    if not isinstance(turret, dict):
        turret = {}
    yaw = first_present_float(
        data,
        (
            "playerTurretX",
            "Player_Turret_X",
            "player_turret_x",
            "turretX",
            "turret_x",
            "turretYaw",
            "turret_yaw",
        ),
    )
    pitch = first_present_float(
        data,
        (
            "playerTurretY",
            "Player_Turret_Y",
            "player_turret_y",
            "turretY",
            "turret_y",
            "turretPitch",
            "turret_pitch",
        ),
    )
    if yaw is None:
        yaw = first_present_float(turret, ("x", "X", "yaw", "Yaw"))
    if pitch is None:
        pitch = first_present_float(turret, ("y", "Y", "pitch", "Pitch"))
    return {"yaw": round_or_none(yaw), "pitch": round_or_none(pitch)}


def make_impact_marker(data: dict[str, Any]) -> dict[str, Any]:
    global next_impact_id

    position = nested_position(data) or {}
    angle = first_present_float(data, ("angle", "centerAngle", "bodyRelativeAngleDeg"))
    distance = first_present_float(data, ("distance", "nearestDistance", "range"))
    world_x = first_present_float(position, ("x", "X"))
    world_y = first_present_float(position, ("y", "Y"))
    world_z = first_present_float(position, ("z", "Z"))

    with state_lock:
        origin = deepcopy(latest_state.get("lidarOrigin", {}) or {})
        raw_info = deepcopy(latest_raw_info)
        last_action = deepcopy(action_debug_state.get("lastResponse", {}))
        last_fire_status = deepcopy(action_debug_state.get("lastFireStatus", {}))
        aim_zero = deepcopy(aim_zero_state)

    origin_x = first_present_float(origin, ("x", "X"))
    origin_z = first_present_float(origin, ("z", "Z"))
    if angle is None and distance is None and None not in (world_x, world_z, origin_x, origin_z):
        dx = float(world_x) - float(origin_x)
        dz = float(world_z) - float(origin_z)
        distance = hypot(dx, dz)
        world_angle = degrees(atan2(dx, dz))
        angle = normalize_signed_angle(world_angle - body_yaw_from_info(raw_info))

    marker_id = next_impact_id
    next_impact_id += 1

    return {
        "id": marker_id,
        "receivedAt": datetime.now().isoformat(timespec="milliseconds"),
        "objectName": data.get("objectName") or data.get("name") or data.get("target"),
        "angle": round_or_none(angle),
        "distance": round_or_none(distance),
        "position": {
            "x": round_or_none(world_x),
            "y": round_or_none(world_y),
            "z": round_or_none(world_z),
        },
        "turret": latest_turret_from_info(raw_info),
        "bodyYaw": round_or_none(body_yaw_from_info(raw_info)),
        "lastAction": {
            "turretQE": last_action.get("turretQE", {"command": "", "weight": 0.0}),
            "turretRF": last_action.get("turretRF", {"command": "", "weight": 0.0}),
            "moveAD": last_action.get("moveAD", {"command": "", "weight": 0.0}),
            "fire": bool(last_action.get("fire", False)),
        },
        "aimZero": aim_zero,
        "fireStatus": last_fire_status,
        "raw": deepcopy(data),
    }


def quantile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    ratio = max(0.0, min(1.0, ratio))
    index = int(round((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


def state_severity(state: str) -> int:
    return {"unknown": 0, "passable": 1, "caution": 2, "blocked": 3}.get(state, 0)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


# =============================================================================
# 2. 愿묒꽑 ?뚯떛
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


def detected_rays(rays: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


# =============================================================================
# 3. 濡쒖뺄 吏硫?紐⑤뜽
# =============================================================================
def estimate_flat_ground_y(hits: list[dict[str, Any]]) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in hits
        if (
            TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG
            and ray["horizontalRange"] <= TERRAIN_ANALYSIS_MAX_DISTANCE_M
        )
    )
    return quantile(heights, 0.25)


def estimate_local_ground_y(
    rays: list[dict[str, Any]],
    fallback_ground_y: float | None,
) -> float | None:
    heights = sorted(
        float(ray["position"]["y"])
        for ray in rays
        if (
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
            and ray["verticalAngle"] >= LOCAL_GROUND_STEEP_MIN_DEG
            and ray["horizontalRange"] <= LOCAL_GROUND_NEAR_MAX_DISTANCE_M
        )
    )
    if len(heights) >= LOCAL_GROUND_MIN_POINT_COUNT:
        return float(median(heights))
    return fallback_ground_y


def expected_flat_ground_range(
    sensor_height_m: float | None,
    vertical_angle_deg: float,
) -> float | None:
    """Return the expected horizontal range of a downward ray on flat ground."""
    if sensor_height_m is None or sensor_height_m <= 0.05 or vertical_angle_deg <= 0.0:
        return None
    tangent = tan(radians(vertical_angle_deg))
    return sensor_height_m / tangent if tangent > 0.0 else None


def classify_lidar_hit(
    ray: dict[str, Any],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    """Classify a valid hit using the visualizer's ground/object filter."""
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
    expected_range = expected_flat_ground_range(
        sensor_height,
        float(ray["verticalAngle"]),
    )

    hit_type = "unknown_hit"
    if height_above_ground is not None:
        if (
            abs(height_above_ground) <= GROUND_HEIGHT_MAX_M
            and float(ray["verticalAngle"]) >= 0.0
        ):
            hit_type = "ground_like"
        elif (
            height_above_ground >= OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M
            and float(ray["verticalAngle"]) < 0.0
        ):
            hit_type = "silhouette_or_object_top"
        elif height_above_ground >= OBJECT_MIN_HEIGHT_ABOVE_LOCAL_GROUND_M:
            hit_type = "object_like"

    if (
        expected_range is not None
        and height_above_ground is not None
        and abs(height_above_ground) <= GROUND_HEIGHT_MAX_M
        and float(ray["horizontalRange"])
        > expected_range * FAR_TERRAIN_EXTRA_RATIO + FAR_TERRAIN_EXTRA_MIN_M
    ):
        hit_type = "delayed_ground_suspect"

    enriched = deepcopy(ray)
    enriched["localGroundY"] = local_ground_y
    enriched["heightAboveLocalGround"] = height_above_ground
    enriched["expectedFlatGroundRange"] = expected_range
    enriched["hitType"] = hit_type
    return enriched


def local_grid_angle_center(angle_deg: float) -> float:
    return round(bin_center(angle_deg, LOCAL_GROUND_GRID_ANGLE_WIDTH_DEG, offset=-180.0), 3)


def local_grid_range_center(horizontal_range_m: float) -> float:
    return round(bin_center(horizontal_range_m, LOCAL_GROUND_GRID_RANGE_BIN_M), 3)


def build_local_ground_grid(hits: list[dict[str, Any]]) -> dict[float, list[dict[str, float]]]:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)

    for ray in hits:
        if not (TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        if ray["horizontalRange"] > TERRAIN_ANALYSIS_MAX_DISTANCE_M:
            continue
        y = ray["position"].get("y")
        if y is None:
            continue
        grouped[(local_grid_angle_center(ray["angle"]), local_grid_range_center(ray["horizontalRange"]))].append(float(y))

    grid: dict[float, list[dict[str, float]]] = defaultdict(list)
    for (angle_center, range_center), ys in grouped.items():
        grid[angle_center].append({"range": range_center, "groundY": min(ys)})

    for values in grid.values():
        values.sort(key=lambda item: item["range"])
    return dict(grid)


def lookup_local_ground_y(
    grid: dict[float, list[dict[str, float]]],
    angle_deg: float,
    horizontal_range_m: float,
    fallback_ground_y: float | None,
) -> float | None:
    angle_candidates = sorted(
        grid.keys(),
        key=lambda key: angular_distance_deg(float(key), angle_deg),
    )[:3]

    best: tuple[float, float] | None = None
    for angle_key in angle_candidates:
        for entry in grid.get(angle_key, []):
            range_gap = abs(float(entry["range"]) - horizontal_range_m)
            angle_gap = angular_distance_deg(float(angle_key), angle_deg)
            score = range_gap + 0.25 * angle_gap
            if range_gap <= LOCAL_GROUND_LOOKUP_MAX_RANGE_GAP_M:
                if best is None or score < best[0]:
                    best = (score, float(entry["groundY"]))

    return best[1] if best is not None else fallback_ground_y


# =============================================================================
# 4. 吏???꾪뿕 遺꾩꽍湲?# =============================================================================
def terrain_sector_center(angle_deg: float) -> float:
    return round(bin_center(angle_deg, TERRAIN_SECTOR_WIDTH_DEG, offset=-TERRAIN_FRONT_LIMIT_DEG), 3)


def build_ground_profile(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        y = point["position"].get("y")
        if y is None:
            continue
        grouped[round(bin_center(point["horizontalRange"], TERRAIN_RANGE_BIN_M), 3)].append(point)

    profile: list[dict[str, float]] = []
    for range_center, bin_points in grouped.items():
        lowest = min(bin_points, key=lambda item: float(item["position"]["y"]))
        profile.append({"horizontalRange": float(range_center), "height": float(lowest["position"]["y"])})
    profile.sort(key=lambda item: item["horizontalRange"])
    return profile


def analyze_profile_metrics(profile: list[dict[str, float]]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "maxUpSlopeDeg": None,
        "maxDownSlopeDeg": None,
        "maxUpStep": None,
        "maxDrop": None,
        "maxProfileGap": None,
        "maxUpStepRange": None,
        "maxDropRange": None,
    }
    if len(profile) < 2:
        return metrics

    for left, right in zip(profile, profile[1:]):
        dx = right["horizontalRange"] - left["horizontalRange"]
        dy = right["height"] - left["height"]
        if dx <= 0.3:
            continue
        slope = degrees(atan2(dy, dx))
        if slope >= 0:
            if metrics["maxUpSlopeDeg"] is None or slope > float(metrics["maxUpSlopeDeg"]):
                metrics["maxUpSlopeDeg"] = slope
        else:
            down = abs(slope)
            if metrics["maxDownSlopeDeg"] is None or down > float(metrics["maxDownSlopeDeg"]):
                metrics["maxDownSlopeDeg"] = down
        if dy >= 0:
            if metrics["maxUpStep"] is None or dy > float(metrics["maxUpStep"]):
                metrics["maxUpStep"] = dy
                metrics["maxUpStepRange"] = right["horizontalRange"]
        else:
            drop = abs(dy)
            if metrics["maxDrop"] is None or drop > float(metrics["maxDrop"]):
                metrics["maxDrop"] = drop
                metrics["maxDropRange"] = right["horizontalRange"]
        if metrics["maxProfileGap"] is None or dx > float(metrics["maxProfileGap"]):
            metrics["maxProfileGap"] = dx
    return metrics


def detect_wall_stack(points: list[dict[str, Any]]) -> tuple[bool, float, int, float | None]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        y = point["position"].get("y")
        if y is None:
            continue
        grouped[round(bin_center(point["horizontalRange"], WALL_RANGE_BIN_M), 3)].append(point)

    best_span = 0.0
    best_channels = 0
    best_range: float | None = None
    for range_center, bin_points in grouped.items():
        ys = [float(point["position"]["y"]) for point in bin_points]
        channels = {point.get("channelIndex") for point in bin_points}
        span = max(ys) - min(ys)
        if span > best_span:
            best_span = span
            best_channels = len(channels)
            best_range = float(range_center)
    return (
        best_span >= WALL_MIN_HEIGHT_SPAN_M and best_channels >= WALL_MIN_UNIQUE_CHANNELS,
        best_span,
        best_channels,
        best_range,
    )


def build_vertical_shape_columns(
    hits: list[dict[str, Any]],
    local_ground_y: float | None,
) -> dict[float, dict[str, Any]]:
    """Describe each azimuth column as an upright stack or smooth uphill."""
    if local_ground_y is None:
        return {}

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in hits:
        y = point["position"].get("y")
        if y is None or point["horizontalRange"] > OBJECT_DETECTION_MAX_DISTANCE_M:
            continue
        if not (
            OBJECT_VERTICAL_MIN_DEG
            <= float(point["verticalAngle"])
            <= TERRAIN_VERTICAL_MAX_DEG
        ):
            continue
        angle_bin = round(
            bin_center(float(point["angle"]), OBJECT_AZIMUTH_BIN_WIDTH_DEG),
            3,
        )
        grouped[angle_bin].append(point)

    columns: dict[float, dict[str, Any]] = {}
    for angle, points in grouped.items():
        is_wall, wall_span, wall_channels, wall_range = detect_wall_stack(points)

        residual_best_span = 0.0
        residual_best_channels = 0
        residual_best_range: float | None = None
        residual_bins: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            range_center = round(
                bin_center(
                    float(point["horizontalRange"]),
                    OBSTACLE_RESIDUAL_MAX_RANGE_SPAN_M,
                ),
                3,
            )
            residual_bins[range_center].append(point)
        for range_center, bin_points in residual_bins.items():
            heights = sorted(
                float(point["position"]["y"]) - float(local_ground_y)
                for point in bin_points
            )
            if not heights:
                continue
            lower_surface = heights[0]
            elevated = [
                point
                for point in bin_points
                if (
                    float(point["position"]["y"])
                    - float(local_ground_y)
                    - lower_surface
                )
                >= OBSTACLE_RESIDUAL_MIN_HEIGHT_M
            ]
            channels = {
                point.get("channelIndex")
                for point in elevated
                if point.get("channelIndex") is not None
            }
            span = heights[-1] - heights[0]
            if len(channels) > residual_best_channels or (
                len(channels) == residual_best_channels
                and span > residual_best_span
            ):
                residual_best_span = span
                residual_best_channels = len(channels)
                residual_best_range = float(range_center)

        residual_stack = bool(
            residual_best_channels >= OBSTACLE_RESIDUAL_MIN_CHANNELS
            and residual_best_span >= OBSTACLE_RESIDUAL_MIN_HEIGHT_M
        )

        range_bins: dict[float, list[float]] = defaultdict(list)
        for point in points:
            height = float(point["position"]["y"]) - float(local_ground_y)
            range_center = round(
                bin_center(
                    float(point["horizontalRange"]),
                    PROFILE_SHAPE_RANGE_BIN_M,
                ),
                3,
            )
            range_bins[range_center].append(height)

        profile = [
            {
                "range": float(range_center),
                "height": float(median(heights)),
            }
            for range_center, heights in sorted(range_bins.items())
        ]
        relevant = [
            item
            for item in profile
            if item["height"] >= -GROUND_HEIGHT_MAX_M
        ]

        slopes: list[float] = []
        monotonic_steps = 0
        usable_steps = 0
        max_gap = 0.0
        for left, right in zip(relevant, relevant[1:]):
            dx = float(right["range"]) - float(left["range"])
            if dx <= 0.3:
                continue
            dy = float(right["height"]) - float(left["height"])
            max_gap = max(max_gap, dx)
            slopes.append(degrees(atan2(dy, dx)))
            usable_steps += 1
            if dy >= -0.20:
                monotonic_steps += 1

        range_span = (
            float(relevant[-1]["range"]) - float(relevant[0]["range"])
            if len(relevant) >= 2
            else 0.0
        )
        height_gain = (
            float(relevant[-1]["height"]) - float(relevant[0]["height"])
            if len(relevant) >= 2
            else 0.0
        )
        monotonic_ratio = monotonic_steps / usable_steps if usable_steps else 0.0
        max_abs_slope = max((abs(value) for value in slopes), default=0.0)
        max_slope_change = max(
            (
                abs(right - left)
                for left, right in zip(slopes, slopes[1:])
            ),
            default=0.0,
        )

        is_hill = bool(
            not is_wall
            and len(relevant) >= PROFILE_SHAPE_MIN_POINTS
            and range_span >= PROFILE_SHAPE_MIN_RANGE_SPAN_M
            and height_gain >= PROFILE_SHAPE_MIN_HEIGHT_GAIN_M
            and max_gap <= PROFILE_SHAPE_MAX_POINT_GAP_M
            and monotonic_ratio >= PROFILE_SHAPE_MONOTONIC_RATIO_MIN
            and max_abs_slope <= PROFILE_SHAPE_MAX_SLOPE_DEG
            and max_slope_change <= PROFILE_SHAPE_MAX_SLOPE_CHANGE_DEG
        )

        columns[angle] = {
            "isWallLike": bool(is_wall),
            "isResidualStack": residual_stack,
            "isHillLike": is_hill,
            "pointCount": len(relevant),
            "rangeSpan": round(range_span, 3),
            "heightGain": round(height_gain, 3),
            "maxPointGap": round(max_gap, 3),
            "monotonicRatio": round(monotonic_ratio, 3),
            "maxAbsSlopeDeg": round(max_abs_slope, 3),
            "maxSlopeChangeDeg": round(max_slope_change, 3),
            "wallHeightSpan": round(wall_span, 3),
            "wallUniqueChannelCount": wall_channels,
            "wallRange": round_or_none(wall_range),
            "residualHeightSpan": round(residual_best_span, 3),
            "residualChannelCount": residual_best_channels,
            "residualRange": round_or_none(residual_best_range),
        }

    # A real obstacle should repeat in neighboring azimuth columns at nearly
    # the same range. Isolated high returns are treated as noise/terrain.
    for angle, metrics in columns.items():
        reference_range = safe_float(
            metrics.get("residualRange"),
            safe_float(metrics.get("wallRange")),
        )
        adjacent_matches = 0
        if reference_range is not None and (
            metrics.get("isWallLike") or metrics.get("isResidualStack")
        ):
            for other_angle, other in columns.items():
                if angular_distance_deg(float(other_angle), float(angle)) > OBSTACLE_ADJACENT_ANGLE_DEG:
                    continue
                other_range = safe_float(
                    other.get("residualRange"),
                    safe_float(other.get("wallRange")),
                )
                if other_range is None:
                    continue
                if not (other.get("isWallLike") or other.get("isResidualStack")):
                    continue
                if abs(float(other_range) - float(reference_range)) <= OBSTACLE_ADJACENT_RANGE_M:
                    adjacent_matches += 1
        metrics["adjacentObstacleColumnCount"] = adjacent_matches
        metrics["isPersistentObstacle"] = (
            adjacent_matches >= OBSTACLE_MIN_ADJACENT_COLUMNS
        )
    return columns


def expected_flat_ground_range(sensor_height_m: float | None, vertical_angle_deg: float) -> float | None:
    if sensor_height_m is None or sensor_height_m <= 0.05 or vertical_angle_deg <= 0.0:
        return None
    tangent = tan(radians(vertical_angle_deg))
    return sensor_height_m / tangent if tangent > 0.0 else None


def classify_terrain_sector_raw(
    sector_angle: float,
    sector_rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    downward = [
        ray for ray in sector_rays
        if TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG
    ]
    hit_points = [
        ray for ray in downward
        if ray.get("isDetected", False)
        and ray["horizontalRange"] <= TERRAIN_ANALYSIS_MAX_DISTANCE_M
        and ray["position"].get("y") is not None
    ]
    profile = build_ground_profile(hit_points)
    metrics = analyze_profile_metrics(profile)
    is_wall, wall_span, wall_channels, wall_range = detect_wall_stack(hit_points)

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )

    expected_ranges: list[float] = []
    missing_expected_ranges: list[float] = []
    delayed_ranges: list[float] = []

    for ray in downward:
        expected_range = expected_flat_ground_range(sensor_height, ray["verticalAngle"])
        if expected_range is None or not (EXPECTED_GROUND_MIN_DISTANCE_M <= expected_range <= EXPECTED_GROUND_MAX_DISTANCE_M):
            continue
        expected_ranges.append(expected_range)
        if not ray.get("isDetected", False):
            missing_expected_ranges.append(expected_range)
            continue
        hit_y = ray["position"].get("y")
        if hit_y is None:
            missing_expected_ranges.append(expected_range)
            continue
        tolerance = max(EXPECTED_GROUND_RANGE_TOLERANCE_M, expected_range * (EXPECTED_GROUND_DELAY_RATIO - 1.0))
        if (
            ray["horizontalRange"] > expected_range + tolerance
            and local_ground_y is not None
            and float(hit_y) < float(local_ground_y) - CAUTION_DROP_M
        ):
            delayed_ranges.append(ray["horizontalRange"])

    expected_count = len(expected_ranges)
    miss_ratio = len(missing_expected_ranges) / expected_count if expected_count else None
    delayed_ratio = len(delayed_ranges) / expected_count if expected_count else None
    nearest_range = min((point["horizontalRange"] for point in hit_points), default=None)
    farthest_range = max((point["horizontalRange"] for point in hit_points), default=None)
    has_near_ground = any(point["horizontalRange"] <= CLIFF_NEAR_GROUND_EVIDENCE_M for point in hit_points)

    max_up_slope = metrics["maxUpSlopeDeg"]
    max_down_slope = metrics["maxDownSlopeDeg"]
    max_up_step = metrics["maxUpStep"]
    max_drop = metrics["maxDrop"]
    profile_gap = metrics["maxProfileGap"]

    hazard_range = nearest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M
    state = "unknown"
    reason = "insufficient_terrain_points"
    hazard_type = "unknown"

    if is_wall:
        state, reason, hazard_type = "blocked", "wall_like_vertical_stack", "wall_or_obstacle"
        hazard_range = wall_range or hazard_range
    elif max_drop is not None and max_drop >= BLOCKED_DROP_M:
        state, reason, hazard_type = "blocked", "cliff_or_pit_drop", "cliff_or_pit"
        hazard_range = metrics["maxDropRange"] or hazard_range
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio >= CLIFF_BLOCKED_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "blocked", "missing_expected_ground_returns_possible_cliff", "possible_cliff"
        hazard_range = min(missing_expected_ranges, default=hazard_range)
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and delayed_ratio is not None and delayed_ratio >= CLIFF_BLOCKED_DELAYED_RETURN_RATIO:
        state, reason, hazard_type = "blocked", "delayed_lower_ground_returns_possible_pit", "possible_pit"
        hazard_range = min(delayed_ranges, default=hazard_range)
    elif profile_gap is not None and profile_gap >= CLIFF_MIN_PROFILE_GAP_M and has_near_ground and (
        (miss_ratio is not None and miss_ratio >= CLIFF_CAUTION_EXPECTED_MISS_RATIO)
        or (delayed_ratio is not None and delayed_ratio >= CLIFF_CAUTION_DELAYED_RETURN_RATIO)
    ):
        state, reason, hazard_type = "blocked", "terrain_profile_gap_with_drop_evidence", "possible_cliff"
    elif max_up_step is not None and max_up_step >= BLOCKED_UP_STEP_M:
        state, reason, hazard_type = "blocked", "large_upward_step", "step_or_wall"
        hazard_range = metrics["maxUpStepRange"] or hazard_range
    elif max_up_slope is not None and max_up_slope > CAUTION_UP_SLOPE_MAX_DEG:
        state, reason, hazard_type = "blocked", "steep_uphill", "steep_slope"
    elif max_down_slope is not None and max_down_slope > CAUTION_DOWN_SLOPE_MAX_DEG:
        state, reason, hazard_type = "blocked", "steep_downhill", "steep_slope"
    elif max_drop is not None and max_drop >= CAUTION_DROP_M:
        state, reason, hazard_type = "caution", "moderate_drop", "drop"
        hazard_range = metrics["maxDropRange"] or hazard_range
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio >= CLIFF_CAUTION_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "caution", "reduced_expected_ground_returns", "possible_cliff"
        hazard_range = min(missing_expected_ranges, default=hazard_range)
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and delayed_ratio is not None and delayed_ratio >= CLIFF_CAUTION_DELAYED_RETURN_RATIO:
        state, reason, hazard_type = "caution", "delayed_lower_ground_returns", "possible_pit"
        hazard_range = min(delayed_ranges, default=hazard_range)
    elif max_up_step is not None and max_up_step >= CAUTION_UP_STEP_M:
        state, reason, hazard_type = "caution", "moderate_upward_step", "step"
        hazard_range = metrics["maxUpStepRange"] or hazard_range
    elif max_up_slope is not None and max_up_slope > PASSABLE_UP_SLOPE_MAX_DEG:
        state, reason, hazard_type = "caution", "moderate_uphill", "slope"
    elif max_down_slope is not None and max_down_slope > PASSABLE_DOWN_SLOPE_MAX_DEG:
        state, reason, hazard_type = "caution", "moderate_downhill", "slope"
    elif expected_count >= EXPECTED_GROUND_MIN_RAY_COUNT and miss_ratio is not None and miss_ratio < CLIFF_CAUTION_EXPECTED_MISS_RATIO:
        state, reason, hazard_type = "passable", "expected_ground_returns_consistent", "ground"
        hazard_range = farthest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M
    elif len(profile) >= 2:
        state, reason, hazard_type = "passable", "continuous_gentle_profile", "ground"
        hazard_range = farthest_range or TERRAIN_ANALYSIS_MAX_DISTANCE_M

    return {
        "centerAngle": round(sector_angle, 3),
        "rawState": state,
        "state": state,
        "rawReason": reason,
        "reason": reason,
        "hazardType": hazard_type,
        "hazardBoundaryRange": round(float(hazard_range), 3),
        "nearestHorizontalRange": round_or_none(nearest_range),
        "farthestHorizontalRange": round_or_none(farthest_range),
        "localGroundY": round_or_none(local_ground_y),
        "sensorHeightAboveLocalGround": round_or_none(sensor_height),
        "expectedGroundRayCount": expected_count,
        "missingExpectedGroundCount": len(missing_expected_ranges),
        "expectedGroundMissRatio": round_or_none(miss_ratio),
        "delayedGroundReturnCount": len(delayed_ranges),
        "delayedGroundReturnRatio": round_or_none(delayed_ratio),
        "hitPointCount": len(hit_points),
        "profilePointCount": len(profile),
        "maxUpSlopeDeg": round_or_none(max_up_slope),
        "maxDownSlopeDeg": round_or_none(max_down_slope),
        "maxUpStep": round_or_none(max_up_step),
        "maxDrop": round_or_none(max_drop),
        "maxProfileGap": round_or_none(profile_gap),
        "wallHeightSpan": round(wall_span, 3),
        "wallUniqueChannelCount": wall_channels,
    }


def stabilize_terrain_sectors(raw_sectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terrain_history.append({float(sector["centerAngle"]): deepcopy(sector) for sector in raw_sectors})
    stabilized: list[dict[str, Any]] = []

    for raw in raw_sectors:
        angle = float(raw["centerAngle"])
        history = [frame[angle] for frame in terrain_history if angle in frame]
        states = [item["rawState"] for item in history]
        counts = Counter(states)
        blocked_count = counts["blocked"]
        caution_count = counts["caution"]
        passable_count = counts["passable"]

        stable_state = raw["rawState"]
        reason = raw["rawReason"]
        if blocked_count >= TERRAIN_BLOCKED_CONFIRM_FRAMES:
            stable_state = "blocked"
            blocked_reasons = [item["rawReason"] for item in history if item["rawState"] == "blocked"]
            reason = Counter(blocked_reasons).most_common(1)[0][0]
        elif raw["rawState"] == "blocked":
            stable_state = "caution"
            reason = "pending_confirmation_" + raw["rawReason"]
        elif blocked_count + caution_count >= TERRAIN_CAUTION_CONFIRM_FRAMES:
            stable_state = "caution"
            caution_reasons = [item["rawReason"] for item in history if item["rawState"] in ("blocked", "caution")]
            reason = Counter(caution_reasons).most_common(1)[0][0]
        elif passable_count >= TERRAIN_PASSABLE_CONFIRM_FRAMES:
            stable_state = "passable"
            reason = "temporal_passable_confirmation"
        else:
            stable_state = raw["rawState"] if raw["rawState"] != "passable" else "unknown"
            reason = raw["rawReason"] if stable_state != "unknown" else "collecting_temporal_evidence"

        hazard_ranges = [
            float(item["hazardBoundaryRange"])
            for item in history
            if item.get("hazardBoundaryRange") is not None
            and item["rawState"] in (stable_state, "blocked" if stable_state == "caution" else stable_state)
        ]
        stable = deepcopy(raw)
        stable["state"] = stable_state
        stable["reason"] = reason
        stable["historyCount"] = len(history)
        stable["blockedVoteCount"] = blocked_count
        stable["cautionVoteCount"] = caution_count
        stable["passableVoteCount"] = passable_count
        if hazard_ranges:
            stable["hazardBoundaryRange"] = round(float(median(hazard_ranges)), 3)
        stabilized.append(stable)
    return stabilized


def analyze_terrain(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for ray in rays:
        if abs(ray["angle"]) > TERRAIN_FRONT_LIMIT_DEG:
            continue
        if not (TERRAIN_VERTICAL_MIN_DEG <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        grouped[terrain_sector_center(ray["angle"])].append(ray)
        count += 1

    raw_sectors: list[dict[str, Any]] = []
    angle = -TERRAIN_FRONT_LIMIT_DEG + TERRAIN_SECTOR_WIDTH_DEG / 2.0
    while angle < TERRAIN_FRONT_LIMIT_DEG:
        raw_sectors.append(classify_terrain_sector_raw(angle, grouped.get(round(angle, 3), []), local_ground_y, lidar_origin_y))
        angle += TERRAIN_SECTOR_WIDTH_DEG
    return stabilize_terrain_sectors(raw_sectors), count


def summarize_front_terrain_decision(sectors: list[dict[str, Any]]) -> dict[str, Any]:
    front = [sector for sector in sectors if abs(float(sector["centerAngle"])) <= FRONT_CLEARANCE_HALF_WIDTH_DEG]
    blocked = [sector for sector in front if sector["state"] == "blocked"]
    caution = [sector for sector in front if sector["state"] == "caution"]
    unknown = [sector for sector in front if sector["state"] == "unknown"]

    dead_zone = [sector for sector in sectors if abs(float(sector["centerAngle"])) <= DEAD_END_FRONT_LIMIT_DEG]
    dead_blocked = [sector for sector in dead_zone if sector["state"] == "blocked"]
    dead_passable = [sector for sector in dead_zone if sector["state"] == "passable"]
    blocked_ratio = len(dead_blocked) / len(dead_zone) if dead_zone else 0.0
    dead_end = blocked_ratio >= DEAD_END_BLOCKED_RATIO and len(dead_passable) <= DEAD_END_MAX_PASSABLE_SECTORS

    if dead_end:
        state, action, reason = "blocked", "stop_or_turn", "possible_dead_end"
    elif blocked:
        state, action, reason = "blocked", "stop_or_turn", blocked[0]["reason"]
    elif caution:
        state, action, reason = "caution", "slow_forward", caution[0]["reason"]
    elif unknown:
        state, action, reason = "unknown", "slow_or_recheck", unknown[0]["reason"]
    elif front and all(sector["state"] == "passable" for sector in front):
        state, action, reason = "passable", "forward", "front_path_is_gentle"
    else:
        state, action, reason = "unknown", "slow_or_recheck", "no_front_sector_evidence"

    return {
        "state": state,
        "recommendedAction": action,
        "reason": reason,
        "deadEndDetected": dead_end,
        "deadEndBlockedRatio": round(blocked_ratio, 3),
        "frontSectorCount": len(front),
    }


# =============================================================================
# 5. 媛앹껜 ?꾨낫? ?쒓컙 湲곕컲 ?몃옓
# =============================================================================
def interpolate_profile_height(
    samples: list[tuple[float, float]],
    query_range: float,
) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1 or query_range <= samples[0][0]:
        return float(samples[0][1])
    if query_range >= samples[-1][0]:
        return float(samples[-1][1])
    for left, right in zip(samples, samples[1:]):
        if left[0] <= query_range <= right[0]:
            span = right[0] - left[0]
            if span <= 1e-6:
                return float(left[1])
            ratio = (query_range - left[0]) / span
            return float(left[1] + ratio * (right[1] - left[1]))
    return float(samples[-1][1])


def terrain_profile_for_hits(
    hits: list[dict[str, Any]],
) -> dict[float, list[tuple[float, float]]]:
    """Build the v16.6 per-azimuth lower terrain envelope."""
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for point in hits:
        y = point["position"].get("y")
        if y is None:
            continue
        angle_bin = round(
            bin_center(
                float(point["angle"]),
                TERRAIN_PROFILE_ANGLE_BIN_DEG,
                offset=-180.0,
            ),
            3,
        )
        range_bin = round(
            bin_center(
                float(point["horizontalRange"]),
                TERRAIN_PROFILE_RANGE_BIN_M,
            ),
            3,
        )
        grouped[(angle_bin, range_bin)].append(float(y))

    profiles: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for (angle_bin, range_bin), heights in grouped.items():
        profiles[angle_bin].append((float(range_bin), min(heights)))
    for samples in profiles.values():
        samples.sort(key=lambda item: item[0])
    return dict(profiles)


def filter_valid_object_hits(
    hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    v16.6 object filter:
    - estimate a lower hill profile per azimuth;
    - require a compact, near-vertical stack;
    - retain only points sufficiently above terrain or stack base.
    """
    if not hits:
        return [], {"method": "terrain_residual_plus_flat_vertical_stack", "validPointCount": 0}

    profiles = terrain_profile_for_hits(hits)
    enriched_hits: list[dict[str, Any]] = []
    stack_groups: dict[tuple[float, float], list[int]] = defaultdict(list)

    for point in hits:
        angle_bin = round(
            bin_center(
                float(point["angle"]),
                TERRAIN_PROFILE_ANGLE_BIN_DEG,
                offset=-180.0,
            ),
            3,
        )
        terrain_y = interpolate_profile_height(
            profiles.get(angle_bin, []),
            float(point["horizontalRange"]),
        )
        enriched = deepcopy(point)
        enriched["terrainProfileY"] = terrain_y
        enriched["heightAboveTerrain"] = float(point["position"]["y"]) - terrain_y
        enriched_hits.append(enriched)

        stack_key = (
            round(
                bin_center(
                    float(point["angle"]),
                    VALID_OBJECT_STACK_ANGLE_BIN_DEG,
                    offset=-180.0,
                ),
                3,
            ),
            round(
                bin_center(
                    float(point["horizontalRange"]),
                    VALID_OBJECT_STACK_RANGE_BIN_M,
                ),
                3,
            ),
        )
        stack_groups[stack_key].append(len(enriched_hits) - 1)

    valid_indices: set[int] = set()
    vertical_bin_count = 0
    hill_object_bin_count = 0
    flat_object_bin_count = 0

    for indices in stack_groups.values():
        points = [enriched_hits[index] for index in indices]
        ys = [float(point["position"]["y"]) for point in points]
        ranges = [float(point["horizontalRange"]) for point in points]
        residuals = [float(point["heightAboveTerrain"]) for point in points]
        local_min_y = min(ys)
        local_span = max(ys) - local_min_y
        local_range_span = max(ranges) - min(ranges)
        verticality_ratio = local_span / max(0.15, local_range_span)
        high_count = sum(
            1 for value in residuals
            if value >= OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M
        )
        ground_like_count = sum(
            1 for value in residuals
            if value <= TERRAIN_GROUND_RESIDUAL_TOL_M
        )
        high_ratio = high_count / max(1, len(points))
        ground_like_ratio = ground_like_count / max(1, len(points))

        vertical_plane = bool(
            local_span >= VALID_OBJECT_STACK_MIN_SPAN_M
            and len(points) >= VALID_OBJECT_STACK_MIN_POINTS
            and local_range_span <= VALID_OBJECT_MAX_RANGE_SPAN_M
            and verticality_ratio >= VALID_OBJECT_MIN_VERTICALITY_RATIO
        )
        object_on_hill = bool(
            vertical_plane
            and max(residuals) >= OBJECT_ON_HILL_MIN_CLUSTER_HEIGHT_M
            and high_count >= OBJECT_ON_HILL_MIN_HIGH_POINTS
            and high_ratio >= OBJECT_ON_HILL_MIN_HIGH_POINT_RATIO
            and ground_like_ratio <= OBJECT_ON_HILL_MAX_GROUNDLIKE_RATIO
        )
        flat_object = bool(
            local_span >= FLAT_OBJECT_MIN_HEIGHT_SPAN_M
            and len(points) >= FLAT_OBJECT_MIN_POINTS
            and local_range_span <= FLAT_OBJECT_MAX_RANGE_SPAN_M
            and verticality_ratio >= FLAT_OBJECT_MIN_VERTICALITY_RATIO
        )

        if vertical_plane:
            vertical_bin_count += 1
        if object_on_hill:
            hill_object_bin_count += 1
        if flat_object:
            flat_object_bin_count += 1

        for index in indices:
            point = enriched_hits[index]
            above_stack_base = float(point["position"]["y"]) - local_min_y
            accepted = (
                (
                    object_on_hill
                    and float(point["heightAboveTerrain"])
                    >= OBJECT_ON_HILL_MIN_TOP_CLEARANCE_M
                )
                or (
                    flat_object
                    and above_stack_base >= VALID_OBJECT_MIN_ABOVE_STACK_BASE_M
                )
            )
            if accepted and (
                VALID_OBJECT_MIN_DISTANCE_M
                <= float(point["distance"])
                <= VALID_OBJECT_MAX_DISTANCE_M
            ):
                point["objectFilter"] = (
                    "object_on_hill"
                    if object_on_hill
                    else "flat_vertical_stack"
                )
                point["heightAboveLocalGround"] = float(point["heightAboveTerrain"])
                point["stackHeightSpan"] = local_span
                point["stackRangeSpan"] = local_range_span
                point["stackVerticalityRatio"] = verticality_ratio
                valid_indices.add(index)

    valid = [enriched_hits[index] for index in sorted(valid_indices)]
    return valid, {
        "method": "terrain_residual_plus_flat_vertical_stack",
        "validPointCount": len(valid),
        "verticalPlaneBinCount": vertical_bin_count,
        "objectOnHillBinCount": hill_object_bin_count,
        "flatObjectBinCount": flat_object_bin_count,
    }


def make_object_azimuth_summaries(
    hits: list[dict[str, Any]],
    ground_grid: dict[float, list[dict[str, float]]],
    fallback_ground_y: float | None,
    lidar_origin_y: float | None,
) -> tuple[list[dict[str, Any]], int]:
    valid_hits, _ = filter_valid_object_hits(hits)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for point in valid_hits:
        if point["distance"] > OBJECT_DETECTION_MAX_DISTANCE_M:
            continue
        grouped[round(bin_center(point["angle"], OBJECT_AZIMUTH_BIN_WIDTH_DEG), 3)].append(point)
        count += 1

    summaries: list[dict[str, Any]] = []
    for azimuth, points in grouped.items():
        nearest = min(point["distance"] for point in points)
        surface = [point for point in points if point["distance"] <= nearest + OBJECT_CLUSTER_MAX_DISTANCE_GAP_M]
        summaries.append({
            "azimuth": float(azimuth),
            "nearestDistance": min(point["distance"] for point in surface),
            "medianDistance": float(median(point["distance"] for point in surface)),
            "points": surface,
        })
    summaries.sort(key=lambda item: item["azimuth"])
    return summaries, count


def can_merge_object_bins(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        angular_distance_deg(right["azimuth"], left["azimuth"]) <= OBJECT_CLUSTER_MAX_ANGLE_GAP_DEG
        and abs(float(right["medianDistance"]) - float(left["medianDistance"])) <= OBJECT_CLUSTER_MAX_DISTANCE_GAP_M
    )


def classify_candidate(width: float, distance: float, point_count: int) -> tuple[str, str, str]:
    if point_count < 2:
        return "unknown", "low", "too_few_points"
    if width >= BULKY_MIN_WIDTH_M:
        return "bulky", "high" if distance <= BULKY_CONFIDENT_DISTANCE_M and point_count >= 4 else "medium", "wide_visible_footprint"
    if width <= THIN_MAX_WIDTH_M:
        if distance > THIN_MAX_DISTANCE_M:
            return "unknown", "low", "thin_candidate_beyond_reliable_range"
        return "thin", "high" if point_count >= 3 else "medium", "narrow_visible_footprint"
    return "unknown", "medium", "ambiguous_visible_width"


def overlap_with_hazard(center_angle: float, distance: float, sectors: list[dict[str, Any]]) -> bool:
    for sector in sectors:
        if sector["state"] not in ("blocked", "caution"):
            continue
        if angular_distance_deg(float(sector["centerAngle"]), center_angle) > TERRAIN_OBJECT_OVERLAP_ANGLE_DEG:
            continue
        boundary = safe_float(sector.get("hazardBoundaryRange"))
        if boundary is not None and abs(boundary - distance) <= TERRAIN_OBJECT_OVERLAP_RANGE_M:
            return True
    return False


def summarize_object_cluster(
    cluster: list[dict[str, Any]],
    object_id: int,
    sectors: list[dict[str, Any]],
    shape_columns: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    points = [point for summary in cluster for point in summary["points"]]
    distances = [float(point["distance"]) for point in points]
    angles = [float(summary["azimuth"]) for summary in cluster]
    ys = [float(point["position"]["y"]) for point in points]
    xs = [float(point["position"]["x"]) for point in points]
    zs = [float(point["position"]["z"]) for point in points]
    above_values = [float(point["heightAboveLocalGround"]) for point in points if point.get("heightAboveLocalGround") is not None]

    median_distance = float(median(distances))
    angular_width = max(OBJECT_AZIMUTH_BIN_WIDTH_DEG, max(angles) - min(angles) + OBJECT_AZIMUTH_BIN_WIDTH_DEG)
    width = 2.0 * median_distance * tan(radians(angular_width / 2.0))
    height = max(ys) - min(ys)
    center_angle = float(median(angles))
    geometry, confidence, reason = classify_candidate(width, median_distance, len(points))
    median_above = float(median(above_values)) if above_values else None
    hazard_overlap = overlap_with_hazard(center_angle, median_distance, sectors)
    cluster_shape_columns = [
        metrics
        for angle, metrics in shape_columns.items()
        if min(angles) - OBJECT_AZIMUTH_BIN_WIDTH_DEG
        <= float(angle)
        <= max(angles) + OBJECT_AZIMUTH_BIN_WIDTH_DEG
    ]
    hill_column_count = sum(
        1 for metrics in cluster_shape_columns if metrics.get("isHillLike")
    )
    wall_column_count = sum(
        1 for metrics in cluster_shape_columns if metrics.get("isWallLike")
    )
    persistent_obstacle_column_count = sum(
        1
        for metrics in cluster_shape_columns
        if metrics.get("isPersistentObstacle")
    )
    persistent_obstacle = persistent_obstacle_column_count > 0
    hill_vote_ratio = hill_column_count / max(1, len(cluster_shape_columns))
    smooth_hill_profile = bool(
        not persistent_obstacle
        and hill_column_count >= PROFILE_SHAPE_MIN_HILL_COLUMNS
        and hill_vote_ratio >= PROFILE_SHAPE_HILL_VOTE_RATIO_MIN
        and hill_column_count > wall_column_count
    )
    sparse_terrain_surface = bool(
        min(distances) >= TERRAIN_SURFACE_FRAGMENT_MIN_DISTANCE_M
        and height <= TERRAIN_SURFACE_FRAGMENT_MAX_HEIGHT_SPAN_M
        and median_above is not None
        and median_above >= TERRAIN_SURFACE_FRAGMENT_MIN_ABOVE_GROUND_M
        and wall_column_count == 0
        and not persistent_obstacle
        and geometry in ("unknown", "bulky")
    )
    isolated_nonpersistent_return = bool(
        not persistent_obstacle
        and len(cluster) <= ISOLATED_RETURN_MAX_AZIMUTH_BINS
        and len(points) <= ISOLATED_RETURN_MAX_POINTS
    )

    # v8.3?먯꽌???꾨낫瑜??덈Т 怨듦꺽?곸쑝濡??듭젣?덈떎:
    #   hazard_overlap OR median_above < 0.80
    # ??議곌굔? 寃쎌궗??踰?洹쇱쿂???볦씤 ?ㅼ젣 媛앹껜瑜??④만 ???덈떎.
    close_to_ground = (
        median_above is not None
        and median_above < TERRAIN_CONNECTED_LOW_HEIGHT_M
    )
    low_profile_hazard_overlap = (
        hazard_overlap
        and median_above is not None
        and median_above < TERRAIN_CONNECTED_HAZARD_LOW_HEIGHT_M
    )
    # Points reaching this stage already passed the v16.6 terrain-residual
    # and compact vertical-plane filter. Do not re-suppress them with the
    # older hill/isolated-return heuristics.
    terrain_connected = False

    return {
        "id": object_id,
        "geometryClass": geometry,
        "shapeConfidence": confidence,
        "shapeReason": reason,
        "candidateLabel": {"thin": "TH", "bulky": "BK", "unknown": "UK"}.get(geometry, "UK"),
        "candidateMeaning": {"thin": "person_or_tree_like", "bulky": "tank_or_rock_like", "unknown": "unknown"}.get(geometry, "unknown"),
        "semanticClass": "unassigned_until_yolo_fusion",
        "centerAngle": round(center_angle, 3),
        "angleMin": round(min(angles), 3),
        "angleMax": round(max(angles), 3),
        "nearestDistance": round(min(distances), 3),
        "medianDistance": round(median_distance, 3),
        "estimatedWidth": round(width, 3),
        "estimatedHeight": round(height, 3),
        "medianHeightAboveLocalGround": round_or_none(median_above),
        "pointCount": len(points),
        "azimuthBinCount": len(cluster),
        "worldPosition": {"x": round(float(median(xs)), 3), "y": round(float(median(ys)), 3), "z": round(float(median(zs)), 3)},
        "terrainConnected": terrain_connected,
        "hazardSurfaceOverlap": hazard_overlap,
        "profileShapeColumnCount": len(cluster_shape_columns),
        "profileHillColumnCount": hill_column_count,
        "profileWallColumnCount": wall_column_count,
        "persistentObstacleColumnCount": persistent_obstacle_column_count,
        "persistentObstacle": persistent_obstacle,
        "profileHillVoteRatio": round(hill_vote_ratio, 3),
        "smoothHillProfile": smooth_hill_profile,
        "sparseTerrainSurface": sparse_terrain_surface,
        "isolatedNonpersistentReturn": isolated_nonpersistent_return,
        "terrainSuppressionReason": None,
        "objectFilter": (
            Counter(
                str(point.get("objectFilter", "terrain_residual_plus_vertical_plane"))
                for point in points
            ).most_common(1)[0][0]
        ),
    }


def classify_object_geometry(
    hits: list[dict[str, Any]],
    ground_grid: dict[float, list[dict[str, float]]],
    fallback_ground_y: float | None,
    lidar_origin_y: float | None,
    sectors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    shape_columns = build_vertical_shape_columns(hits, fallback_ground_y)
    summaries, count = make_object_azimuth_summaries(
        hits,
        ground_grid,
        fallback_ground_y,
        lidar_origin_y,
    )
    if not summaries:
        return [], count
    clusters: list[list[dict[str, Any]]] = []
    current = [summaries[0]]
    for summary in summaries[1:]:
        if can_merge_object_bins(current[-1], summary):
            current.append(summary)
        else:
            clusters.append(current)
            current = [summary]
    clusters.append(current)
    objects = [
        summarize_object_cluster(
            cluster,
            idx + 1,
            sectors,
            shape_columns,
        )
        for idx, cluster in enumerate(clusters)
    ]
    objects.sort(key=lambda obj: float(obj["nearestDistance"]))
    for idx, obj in enumerate(objects, start=1):
        obj["id"] = idx
    return objects, count


class AxisKalman:
    def __init__(self, position: float) -> None:
        self.position = float(position)
        self.velocity = 0.0
        self.p00, self.p01, self.p10, self.p11 = 4.0, 0.0, 0.0, 4.0

    def predict(self, dt: float) -> None:
        dt = max(0.05, min(dt, 2.0))
        self.position += self.velocity * dt
        q = TRACK_PROCESS_NOISE
        p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + q * dt * dt
        p01 = self.p01 + dt * self.p11
        p10 = self.p10 + dt * self.p11
        p11 = self.p11 + q
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11

    def update(self, measurement: float) -> None:
        residual = float(measurement) - self.position
        s = self.p00 + TRACK_MEASUREMENT_NOISE
        k0, k1 = self.p00 / s, self.p10 / s
        self.position += k0 * residual
        self.velocity += k1 * residual
        p00, p01 = self.p00, self.p01
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p10 = self.p10 - k1 * p00
        self.p11 = self.p11 - k1 * p01
        if abs(self.velocity) < 0.05:
            self.velocity = 0.0


class ObjectTrack:
    def __init__(self, track_id: int, observation: dict[str, Any], sim_time: float) -> None:
        pos = observation["worldPosition"]
        self.id = track_id
        self.kx = AxisKalman(float(pos["x"]))
        self.kz = AxisKalman(float(pos["z"]))
        self.last_time = sim_time
        self.misses = 0
        self.hit_history: deque[int] = deque([1], maxlen=TRACK_HISTORY_SIZE)
        self.labels: deque[str] = deque([observation["geometryClass"]], maxlen=TRACK_HISTORY_SIZE)
        self.last_observation = deepcopy(observation)

    def predict(self, sim_time: float) -> None:
        dt = max(0.05, sim_time - self.last_time)
        self.kx.predict(dt)
        self.kz.predict(dt)
        self.last_time = sim_time

    def distance_to(self, observation: dict[str, Any]) -> float:
        pos = observation["worldPosition"]
        return hypot(float(pos["x"]) - self.kx.position, float(pos["z"]) - self.kz.position)

    def update(self, observation: dict[str, Any]) -> None:
        pos = observation["worldPosition"]
        self.kx.update(float(pos["x"]))
        self.kz.update(float(pos["z"]))
        self.misses = 0
        self.hit_history.append(1)
        self.labels.append(observation["geometryClass"])
        self.last_observation = deepcopy(observation)

    def miss(self) -> None:
        self.misses += 1
        self.hit_history.append(0)

    def to_dict(self) -> dict[str, Any]:
        observation = deepcopy(self.last_observation)
        geometry = Counter(self.labels).most_common(1)[0][0]
        confirmed = sum(self.hit_history) >= TRACK_CONFIRM_HITS and self.misses < TRACK_MAX_MISSES
        observation.update({
            "trackId": self.id,
            "geometryClass": geometry,
            "candidateLabel": {"thin": "TH", "bulky": "BK", "unknown": "UK"}.get(geometry, "UK"),
            "confirmed": confirmed,
            "persistenceHits": sum(self.hit_history),
            "historySize": len(self.hit_history),
            "misses": self.misses,
            "filteredWorldPosition": {"x": round(self.kx.position, 3), "z": round(self.kz.position, 3)},
            "estimatedVelocity": {"vx": round(self.kx.velocity, 3), "vz": round(self.kz.velocity, 3)},
        })
        return observation


class ObjectTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: list[ObjectTrack] = []

    def update(self, observations: list[dict[str, Any]], sim_time: float) -> list[dict[str, Any]]:
        usable = [obj for obj in observations if not obj.get("terrainConnected", False)]
        for track in self.tracks:
            track.predict(sim_time)

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for oi, obs in enumerate(usable):
                distance = track.distance_to(obs)
                if distance <= TRACK_ASSOCIATION_DISTANCE_M:
                    candidates.append((distance, ti, oi))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        for _, ti, oi in candidates:
            if ti in matched_tracks or oi in matched_obs:
                continue
            self.tracks[ti].update(usable[oi])
            matched_tracks.add(ti)
            matched_obs.add(oi)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.miss()

        for oi, obs in enumerate(usable):
            if oi not in matched_obs:
                self.tracks.append(ObjectTrack(self.next_id, obs, sim_time))
                self.next_id += 1

        self.tracks = [track for track in self.tracks if track.misses < TRACK_MAX_MISSES]
        return [track.to_dict() for track in self.tracks]


object_tracker = ObjectTracker()


def reset_runtime_state() -> None:
    """?쒓컙 ?ы몴, 媛앹껜 ?몃옓, 李⑥껜 ?뺣젹 ?좉툑??珥덇린?뷀븳??"""
    global object_tracker, latest_state, latest_raw_info, body_alignment_state, action_debug_state, vision_state, recognized_lidar_objects, fire_control_state, aim_zero_state, aim_stabilizer_state, world_target_lock_state, eliminated_target_state, next_impact_id

    terrain_history.clear()
    impact_history.clear()
    next_impact_id = 1
    object_tracker = ObjectTracker()
    recognized_lidar_objects = {}
    latest_raw_info = {}
    latest_state = {
        "simulationTime": None,
        "terrainSectors": [],
        "terrainDecision": {},
        "contourPoints": [],
        "frontVerticalProfile": {},
        "rawObjects": [],
        "trackedObjects": [],
        "fusionPriorityQueue": [],
        "primaryFusionTarget": None,
        "scanTarget": None,
        "bodyAlignment": {},
        "visionTarget": None,
        "visionDetections": [],
        "lidarVisionFusion": None,
        "impactMarkers": [],
    }
    fire_control_state = {
        "approvedUntil": 0.0,
        "approvedAt": None,
        "lastFiredAt": None,
        "lastAutoFireAt": 0.0,
        "fireCount": 0,
    }
    aim_zero_state = {
        "offsetY": 0.0,
        "updatedAt": None,
    }
    aim_stabilizer_state = {
        "trackId": None,
        "insideFineSince": None,
        "lastNearCommandAt": 0.0,
        "lastYawCommandAt": 0.0,
        "settled": False,
        "lastYawErrorDeg": None,
        "lastDistance": None,
        "lastPulseWeight": 0.0,
        "reason": "waiting_for_target",
    }
    world_target_lock_state = {
        "active": False,
        "targetKey": None,
        "trackId": None,
        "worldX": None,
        "worldZ": None,
        "distance": None,
        "lockedAt": 0.0,
        "lastSeenAt": 0.0,
        "lastErrorDeg": None,
        "lastErrorAt": 0.0,
        "lastCommand": {"command": "", "weight": 0.0},
        "yawReference": None,
        "reason": "waiting_for_target",
    }
    body_alignment_state = {
        "enabled": AUTO_BODY_ALIGN_ENABLED,
        "lockedTrackId": None,
        "target": None,
        "moveAD": {"command": "", "weight": 0.0},
        "aligned": False,
        "reason": "waiting_for_target",
        "stickyLock": False,
        "pendingTrackId": None,
        "pendingHits": 0,
        "pendingCenterAngle": None,
        "pendingDistance": None,
        "lockedCenterAngle": None,
        "lockedDistance": None,
        "alignedSince": None,
        "decisionReadyAt": None,
        "postAimDecisionHoldRemaining": None,
        "nextTrackIdMin": 1,
        "rejectedTrackIds": [],
        "acceptedTankTrackIds": [],
        "lastJudgedTrackId": None,
        "lastJudgement": None,
        "lastExhaustedCandidateSignature": None,
        "scanRound": 1,
        "lastScanResetReason": None,
    }
    action_debug_state = {
        "getActionRequestCount": 0,
        "lastRequestBody": {},
        "lastResponse": {},
        "lastRequestedAt": None,
    }
    vision_state = {
        "target": None,
        "detections": [],
        "lastDetectedAt": 0.0,
        "lastProcessedAt": 0.0,
        "lastProcessedTrackId": None,
        "lastDetectRequestAt": 0.0,
        "lastDetectRequestMode": None,
        "lastDetectRequestHadImage": False,
        "modelPath": str(MODEL_PATH),
        "modelLoaded": yolo_model is not None,
        "modelLoadError": None,
        "lastInferenceError": None,
        "lidarFusion": None,
    }


# =============================================================================
# 6. ?ㅺ낸 吏?꾩? 理쒖긽???붿빟
# =============================================================================
def make_contour_points(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for ray in hits:
        if ray["horizontalRange"] > CONTOUR_MAX_DISTANCE_M:
            continue
        if not (-2.0 <= ray["verticalAngle"] <= TERRAIN_VERTICAL_MAX_DEG):
            continue
        grouped[round(bin_center(ray["angle"], CONTOUR_ANGLE_BIN_DEG, offset=-180.0), 3)].append(ray)

    points: list[dict[str, Any]] = []
    for angle, rays in grouped.items():
        nearest = min(rays, key=lambda ray: ray["horizontalRange"])
        points.append({"angle": float(angle), "distance": round(float(nearest["horizontalRange"]), 3)})
    points.sort(key=lambda point: point["angle"])
    return points


def make_front_view_points(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """?댁쟾???쒖젏??媛源뚯슫 ?꾨갑 媛곷룄 ?ъ쁺??媛먯? ?ъ씤?몃? 諛섑솚?쒕떎."""
    points: list[dict[str, Any]] = []

    for ray in hits:
        angle = float(ray["angle"])
        vertical = float(ray["verticalAngle"])
        distance = float(ray["distance"])

        if abs(angle) > FRONT_VIEW_HORIZONTAL_LIMIT_DEG:
            continue
        if not (FRONT_VIEW_VERTICAL_MIN_DEG <= vertical <= FRONT_VIEW_VERTICAL_MAX_DEG):
            continue
        if distance > FRONT_VIEW_MAX_DISTANCE_M:
            continue

        points.append(
            {
                "angle": round(angle, 3),
                "verticalAngle": round(vertical, 3),
                "distance": round(distance, 3),
                "horizontalRange": round(float(ray["horizontalRange"]), 3),
                "channelIndex": ray.get("channelIndex"),
            }
        )

    return points



def linear_regression_slope_deg(profile: list[dict[str, float]]) -> float | None:
    """?섎떒 ?멸낸???꾨줈?뚯씪?먯꽌 ??듭쟻??濡쒖뺄 吏??寃쎌궗瑜?異붿젙?쒕떎."""
    if len(profile) < FRONT_PROFILE_MIN_FIT_POINTS:
        return None

    xs = [float(point["horizontalRange"]) for point in profile]
    ys = [float(point["height"]) for point in profile]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if denominator <= 1e-9:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return degrees(atan2(slope, 1.0))


def make_front_vertical_profile(
    rays: list[dict[str, Any]],
    local_ground_y: float | None,
    lidar_origin_y: float | None,
) -> dict[str, Any]:
    """
    李⑥껜 ?뺣㈃ 0?꾩뿉 媛??媛源뚯슫 諛⑹쐞媛곸쓣 ?좏깮?섍퀬,
    ?대떦 ?섏쭅 梨꾨꼸?ㅼ쓣 痢〓㈃ ?⑤㈃?쇰줈 ?몄텧?쒕떎.

    ???쒕??덉씠?곗뿉?쒕뒗 ?묒닔 verticalAngle???꾨옒履쎌쓣 媛由ы궓??
    """
    if not rays:
        return {
            "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
            "selectedAngle": None,
            "channelCount": 0,
            "hitCount": 0,
            "missCount": 0,
            "sensorHeightAboveLocalGround": None,
            "approxGroundSlopeDeg": None,
            "maxUpSlopeDeg": None,
            "maxDownSlopeDeg": None,
            "maxDrop": None,
            "maxUpStep": None,
            "rays": [],
            "groundProfilePoints": [],
        }

    unique_angles = sorted({round(float(ray["angle"]), 3) for ray in rays})
    selected_angle = min(
        unique_angles,
        key=lambda angle: angular_distance_deg(angle, FRONT_PROFILE_TARGET_ANGLE_DEG),
    )

    if angular_distance_deg(selected_angle, FRONT_PROFILE_TARGET_ANGLE_DEG) > FRONT_PROFILE_MAX_SELECT_ANGLE_ERROR_DEG:
        return {
            "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
            "selectedAngle": selected_angle,
            "channelCount": 0,
            "hitCount": 0,
            "missCount": 0,
            "sensorHeightAboveLocalGround": None,
            "approxGroundSlopeDeg": None,
            "maxUpSlopeDeg": None,
            "maxDownSlopeDeg": None,
            "maxDrop": None,
            "maxUpStep": None,
            "rays": [],
            "groundProfilePoints": [],
            "warning": "No near-forward azimuth ray found.",
        }

    selected_rays = [
        ray
        for ray in rays
        if abs(float(ray["angle"]) - selected_angle) <= 0.15
    ]

    selected_rays.sort(
        key=lambda ray: (
            int(ray["channelIndex"]) if isinstance(ray.get("channelIndex"), int) else 9999,
            float(ray["verticalAngle"]),
        )
    )

    sensor_height = (
        float(lidar_origin_y) - float(local_ground_y)
        if lidar_origin_y is not None and local_ground_y is not None
        else None
    )

    output_rays: list[dict[str, Any]] = []
    downward_hits: list[dict[str, Any]] = []

    for ray in selected_rays:
        detected = bool(
            ray.get("isDetected", False)
            and ray["position"].get("y") is not None
        )
        position_y = safe_float(ray["position"].get("y"))
        relative_height = (
            float(position_y) - float(local_ground_y)
            if position_y is not None and local_ground_y is not None
            else None
        )
        expected_range = expected_flat_ground_range(
            sensor_height,
            float(ray["verticalAngle"]),
        )

        item = {
            "channelIndex": ray.get("channelIndex"),
            "angle": round(float(ray["angle"]), 3),
            "verticalAngle": round(float(ray["verticalAngle"]), 3),
            "isDetected": detected,
            "distance": round(float(ray["distance"]), 3),
            "horizontalRange": round(float(ray["horizontalRange"]), 3),
            "positionY": round_or_none(position_y),
            "heightAboveLocalGround": round_or_none(relative_height),
            "expectedFlatGroundRange": round_or_none(expected_range),
        }
        output_rays.append(item)

        if (
            detected
            and TERRAIN_VERTICAL_MIN_DEG <= float(ray["verticalAngle"]) <= TERRAIN_VERTICAL_MAX_DEG
            and float(ray["horizontalRange"]) <= FRONT_PROFILE_FIT_MAX_DISTANCE_M
        ):
            downward_hits.append(ray)

    profile = build_ground_profile(downward_hits)
    metrics = analyze_profile_metrics(profile)
    approx_slope = linear_regression_slope_deg(profile)

    profile_output = [
        {
            "horizontalRange": round(float(point["horizontalRange"]), 3),
            "heightAboveLocalGround": (
                round(float(point["height"]) - float(local_ground_y), 3)
                if local_ground_y is not None
                else None
            ),
            "worldY": round(float(point["height"]), 3),
        }
        for point in profile
    ]

    hit_count = sum(1 for ray in output_rays if ray["isDetected"])

    return {
        "targetAngle": FRONT_PROFILE_TARGET_ANGLE_DEG,
        "selectedAngle": round(float(selected_angle), 3),
        "channelCount": len(output_rays),
        "hitCount": hit_count,
        "missCount": len(output_rays) - hit_count,
        "sensorHeightAboveLocalGround": round_or_none(sensor_height),
        "approxGroundSlopeDeg": round_or_none(approx_slope),
        "maxUpSlopeDeg": round_or_none(metrics["maxUpSlopeDeg"]),
        "maxDownSlopeDeg": round_or_none(metrics["maxDownSlopeDeg"]),
        "maxDrop": round_or_none(metrics["maxDrop"]),
        "maxUpStep": round_or_none(metrics["maxUpStep"]),
        "rays": output_rays,
        "groundProfilePoints": profile_output,
    }


def make_front_clearance(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    front = [obj for obj in objects if obj.get("confirmed") and abs(float(obj["centerAngle"])) <= FRONT_CLEARANCE_HALF_WIDTH_DEG]
    return min(front, key=lambda obj: float(obj["nearestDistance"]), default=None)



def priority_geometry_rank(geometry_class: str) -> int:
    """
    ?レ옄媛 ?묒쓣?섎줉 ?곗꽑?쒖쐞媛 ?믩떎.

    BK / bulky ?꾨낫???꾩감??諛붿쐞 怨꾩뿴??媛?μ꽦???덉뼱 癒쇱? 泥섎━?쒕떎.
    TH / thin ?꾨낫???댄썑 ?щ엺 媛먯?瑜??꾪빐 怨꾩냽 ?④꺼 ?붾떎.
    """
    return {
        "bulky": 0,
        "thin": 1,
        "unknown": 2,
    }.get(geometry_class, 3)


def is_same_priority_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        angular_distance_deg(
            float(left.get("centerAngle", 0.0)),
            float(right.get("centerAngle", 0.0)),
        )
        <= PRIORITY_DUPLICATE_ANGLE_TOLERANCE_DEG
        and abs(
            float(left.get("medianDistance", left.get("nearestDistance", 0.0)))
            - float(right.get("medianDistance", right.get("nearestDistance", 0.0)))
        )
        <= PRIORITY_DUPLICATE_DISTANCE_TOLERANCE_M
    )


def enrich_priority_candidate(
    obj: dict[str, Any],
    tracking_stage: str,
) -> dict[str, Any]:
    enriched = deepcopy(obj)

    distance = float(
        enriched.get(
            "nearestDistance",
            enriched.get("medianDistance", MAX_DISTANCE_M),
        )
    )
    geometry = str(enriched.get("geometryClass", "unknown"))
    is_close = distance <= PRIORITY_NEAR_MAX_DISTANCE_M
    is_confirmed = tracking_stage == "confirmed"

    # ?쒖쟻 ?좏깮 ?꾪뿕??怨꾩링:
    # 1. ?꾩감/諛붿쐞 怨꾩뿴?????덉쑝誘濡?bulky / ?먭볼??媛앹껜 ?곗꽑
    # 2. 洹몃떎??媛源뚯슫 嫄곕━ ?곗꽑
    # 3. ?꾩떆 媛앹껜蹂대떎 ?뺤젙 ?몃옓 ?곗꽑
    # 4. 근거리 객체에는 마지막 동률 해소 보너스를 부여한다.
    priority_key = (
        priority_geometry_rank(geometry),
        distance,
        0 if is_confirmed else 1,
        0 if is_close else 1,
    )

    enriched.update(
        {
            "trackingStage": tracking_stage,
            "isNearPriorityRange": is_close,
            "fusionPriorityKey": list(priority_key),
            "fusionPriorityTier": (
                "P1_bulky_near"
                if geometry == "bulky" and is_close
                else (
                    "P2_bulky_far"
                    if geometry == "bulky"
                    else (
                        "P3_non_bulky_near"
                        if is_close
                        else "P4_non_bulky_far"
                    )
                )
            ),
            "fusionPriorityReason": (
                "bulky candidate and within close-range threshold"
                if geometry == "bulky" and is_close
                else (
                    "bulky candidate outside close-range threshold"
                    if geometry == "bulky"
                    else (
                        "non-bulky fallback within close-range threshold"
                        if is_close
                        else "non-bulky fallback outside close-range threshold"
                    )
                )
            ),
            "recommendedTurretBodyRelativeAngleDeg": round(
                float(enriched.get("centerAngle", 0.0)),
                3,
            ),
        }
    )
    return enriched


def make_fusion_priority_queue(
    confirmed_objects: list[dict[str, Any]],
    provisional_objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    ?댄썑 YOLO 寃고빀???꾪븳 360??LiDAR ?꾨낫 ?먮? 留뚮뱺??

    ?뺤젙 媛앹껜? ?꾩떆 媛앹껜瑜?紐⑤몢 ?좎??쒕떎.
    ?꾩떆 ?꾨낫媛 ?뺤젙 ?몃옓怨?寃뱀튂硫??뺤젙 ?몃옓留??④릿??
    """
    candidates: list[dict[str, Any]] = []

    cleanup_eliminated_targets()

    for obj in confirmed_objects:
        candidate = enrich_priority_candidate(obj, "confirmed")
        if is_temporarily_eliminated_target(candidate):
            continue
        candidates.append(candidate)

    for obj in provisional_objects:
        provisional = enrich_priority_candidate(obj, "provisional")
        if is_temporarily_eliminated_target(provisional):
            continue
        if any(is_same_priority_candidate(provisional, existing) for existing in candidates):
            continue
        candidates.append(provisional)

    candidates.sort(
        key=lambda obj: tuple(obj.get("fusionPriorityKey", [99, 99, 99, 999.0]))
    )

    candidates = candidates[:PRIORITY_MAX_QUEUE_SIZE]

    for index, obj in enumerate(candidates, start=1):
        obj["fusionPriorityRank"] = index
        obj["isPrimaryFusionTarget"] = index == 1

    primary = deepcopy(candidates[0]) if candidates else None
    return candidates, primary



def semantic_is_tank_candidate(obj: dict[str, Any]) -> bool:
    """
    YOLO 寃고빀 ?댄썑 ?ъ슜?????덈뒗 ?섎? 遺꾨쪟 寃??

    ?ν썑 ?섎? ?쇰꺼 ??
    - Tank_enemy_front
    - Tank_enemy_side
    - Tank_ally_back
    """
    semantic = normalize_yolo_class_name(
        obj.get("recognizedClass") or obj.get("semanticClass")
    )
    return is_attack_yolo_class(semantic)


def track_id_int(obj: dict[str, Any] | None) -> int | None:
    if not obj:
        return None
    for key in ("trackId", "id", "objectId"):
        try:
            value = obj.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def body_align_target_id(obj: dict[str, Any] | None) -> int | None:
    return track_id_int(obj)


def body_align_target_id_label(obj: dict[str, Any] | None) -> str:
    target_id = body_align_target_id(obj)
    return "n/a" if target_id is None else str(target_id)


def rejected_body_align_track_ids() -> set[int]:
    rejected = body_alignment_state.get("rejectedTrackIds", []) or []
    ids: set[int] = set()
    for item in rejected:
        try:
            ids.add(int(item))
        except (TypeError, ValueError):
            pass
    return ids


def body_align_live_candidate_signature(
    priority_queue: list[dict[str, Any]],
) -> tuple[tuple[int, int, int], ...]:
    signature: list[tuple[int, int, int]] = []
    for obj in priority_queue:
        track_id = track_id_int(obj)
        if track_id is None:
            continue
        distance = safe_float(obj.get("nearestDistance"))
        if distance is None or distance > BODY_ALIGN_LOCK_RELEASE_DISTANCE_M:
            continue
        if BODY_ALIGN_CONFIRMED_ONLY and obj.get("trackingStage") != "confirmed":
            continue
        angle = safe_float(obj.get("centerAngle"), 0.0) or 0.0
        signature.append((int(track_id), int(round(float(angle) * 10.0)), int(round(float(distance) * 10.0))))
    return tuple(sorted(signature))


def reset_body_align_sequential_scan(reason: str) -> None:
    body_alignment_state["nextTrackIdMin"] = 1
    body_alignment_state["rejectedTrackIds"] = []
    body_alignment_state["acceptedTankTrackIds"] = []
    body_alignment_state["lastJudgedTrackId"] = None
    body_alignment_state["lastJudgement"] = None
    body_alignment_state["lockedTrackId"] = None
    body_alignment_state["target"] = None
    body_alignment_state["lockedCenterAngle"] = None
    body_alignment_state["lockedDistance"] = None
    body_alignment_state["alignedSince"] = None
    body_alignment_state["decisionReadyAt"] = None
    body_alignment_state["postAimDecisionHoldRemaining"] = None
    body_alignment_state["scanRound"] = int(body_alignment_state.get("scanRound", 1) or 1) + 1
    body_alignment_state["lastScanResetReason"] = reason
    clear_pending_body_align_target()
    clear_world_target_lock(reason)
    reset_aim_stabilizer(reason)


def maybe_reset_exhausted_sequential_scan(priority_queue: list[dict[str, Any]]) -> None:
    if not BODY_ALIGN_SEQUENTIAL_ID_SCAN:
        return
    if body_alignment_state.get("target") is not None:
        return
    signature = body_align_live_candidate_signature(priority_queue)
    if not signature:
        body_alignment_state["lastExhaustedCandidateSignature"] = None
        return
    live_ids = [item[0] for item in signature]
    min_track_id = int(body_alignment_state.get("nextTrackIdMin", 1) or 1)
    rejected = rejected_body_align_track_ids()
    has_allowed_candidate = any(track_id >= min_track_id and track_id not in rejected for track_id in live_ids)
    if has_allowed_candidate:
        body_alignment_state["lastExhaustedCandidateSignature"] = None
        return
    previous_signature = body_alignment_state.get("lastExhaustedCandidateSignature")
    if previous_signature is None:
        body_alignment_state["lastExhaustedCandidateSignature"] = signature
        return
    if previous_signature != signature:
        reset_body_align_sequential_scan("new_lidar_candidate_set_reset_sequential_scan")
        return
    body_alignment_state["lastExhaustedCandidateSignature"] = signature


def sequential_track_allowed(obj: dict[str, Any]) -> bool:
    track_id = track_id_int(obj)
    if track_id is None:
        return False
    min_track_id = int(body_alignment_state.get("nextTrackIdMin", 1) or 1)
    if track_id < min_track_id:
        return False
    if track_id in rejected_body_align_track_ids():
        return False
    return True


def is_body_align_candidate(obj: dict[str, Any]) -> bool:
    distance = safe_float(obj.get("nearestDistance"))
    if distance is None or distance > BODY_ALIGN_LOCK_RELEASE_DISTANCE_M:
        return False

    if BODY_ALIGN_CONFIRMED_ONLY and obj.get("trackingStage") != "confirmed":
        return False

    if BODY_ALIGN_SEQUENTIAL_ID_SCAN:
        return sequential_track_allowed(obj)

    if semantic_is_known_non_attack(obj):
        return False

    if semantic_is_tank_candidate(obj):
        return True

    return (
        BODY_ALIGN_USE_LIDAR_BULKY_FALLBACK
        and obj.get("geometryClass") == "bulky"
        and distance <= BODY_ALIGN_TARGET_MAX_DISTANCE_M
    )


def body_align_target_sort_key(obj: dict[str, Any]) -> tuple[int, int, float]:
    """
    ?꾧꺽???꾩감 ?섎? 遺꾨쪟媛 LiDAR ?꾩슜 BK fallback蹂대떎 ?곗꽑?쒕떎.
    媛숈? 遺꾨쪟 ?덉뿉?쒕뒗 ??媛源앷퀬 ?덉젙?곸씤 媛앹껜瑜?癒쇱? ?좏깮?쒕떎.
    """
    if BODY_ALIGN_SEQUENTIAL_ID_SCAN:
        track_id = track_id_int(obj)
        confirmed_rank = 0 if obj.get("trackingStage") == "confirmed" else 1
        distance = float(obj.get("nearestDistance", MAX_DISTANCE_M))
        return track_id if track_id is not None else 999999, confirmed_rank, distance

    semantic_rank = 0 if semantic_is_tank_candidate(obj) else 1
    confirmed_rank = 0 if obj.get("trackingStage") == "confirmed" else 1
    distance = float(obj.get("nearestDistance", MAX_DISTANCE_M))
    return semantic_rank, confirmed_rank, distance


def body_align_spatial_match(
    obj: dict[str, Any],
    center_angle: Any,
    distance: Any,
) -> bool:
    obj_angle = safe_float(obj.get("centerAngle"))
    obj_distance = safe_float(obj.get("nearestDistance"))
    ref_angle = safe_float(center_angle)
    ref_distance = safe_float(distance)
    if obj_angle is None or obj_distance is None or ref_angle is None or ref_distance is None:
        return False
    angle_delta = abs(normalize_signed_angle(float(obj_angle) - float(ref_angle)))
    distance_delta = abs(float(obj_distance) - float(ref_distance))
    return (
        angle_delta <= BODY_ALIGN_SPATIAL_LOCK_ANGLE_DEG
        and distance_delta <= BODY_ALIGN_SPATIAL_LOCK_DISTANCE_M
    )


def remember_locked_body_align_target(obj: dict[str, Any]) -> None:
    body_alignment_state["lockedTrackId"] = body_align_target_id(obj)
    body_alignment_state["lockedCenterAngle"] = obj.get("centerAngle")
    body_alignment_state["lockedDistance"] = obj.get("nearestDistance")


def active_post_aim_decision_hold(now: float | None = None) -> bool:
    aligned_since = safe_float(body_alignment_state.get("alignedSince"))
    if aligned_since is None:
        return False
    now = monotonic() if now is None else now
    return now - float(aligned_since) < BODY_ALIGN_POST_AIM_DECISION_SECONDS


def active_yolo_recognition_window(now: float | None = None) -> bool:
    if not USE_YOLO_FOR_RECOGNITION:
        return False
    target = body_alignment_state.get("target")
    if not target or track_id_int(target) is None:
        return False
    angle_error = safe_float(target.get("bodyRelativeAngleErrorDeg"))
    if angle_error is None:
        angle_error = safe_float(target.get("centerAngle"))
    if angle_error is not None and abs(float(angle_error)) <= BODY_ALIGN_YOLO_PREALIGN_DEG:
        return True
    aligned_since = safe_float(body_alignment_state.get("alignedSince"))
    if aligned_since is None:
        return False
    now = monotonic() if now is None else now
    elapsed = now - float(aligned_since)
    if 0.0 <= elapsed <= BODY_ALIGN_POST_AIM_DECISION_SECONDS:
        return True
    track_id = track_id_int(target)
    if yolo_frame_processed_for_track(track_id, float(aligned_since)):
        return False
    return 0.0 <= elapsed <= BODY_ALIGN_YOLO_FRAME_WAIT_SECONDS


def yolo_recognition_window_payload(now: float | None = None) -> dict[str, Any]:
    now = monotonic() if now is None else now
    aligned_since = safe_float(body_alignment_state.get("alignedSince"))
    target = body_alignment_state.get("target")
    elapsed = None if aligned_since is None else now - float(aligned_since)
    remaining = (
        None
        if elapsed is None
        else max(0.0, BODY_ALIGN_POST_AIM_DECISION_SECONDS - float(elapsed))
    )
    return {
        "active": active_yolo_recognition_window(now),
        "trackId": track_id_int(target),
        "elapsed": round(elapsed, 3) if elapsed is not None else None,
        "remaining": round(remaining, 3) if remaining is not None else None,
        "duration": BODY_ALIGN_POST_AIM_DECISION_SECONDS,
        "maxFrameWait": BODY_ALIGN_YOLO_FRAME_WAIT_SECONDS,
        "frameProcessed": yolo_frame_processed_for_track(
            track_id_int(target),
            float(aligned_since) if aligned_since is not None else None,
        ),
    }


def find_sticky_locked_target(
    priority_queue: list[dict[str, Any]],
    locked_track_id: Any,
) -> dict[str, Any] | None:
    if not BODY_ALIGN_STICKY_LOCK_ENABLED:
        return None
    for obj in priority_queue:
        obj_track_id = track_id_int(obj)
        try:
            locked_track_id_int = int(locked_track_id) if locked_track_id is not None else None
        except (TypeError, ValueError):
            locked_track_id_int = None
        if (
            BODY_ALIGN_SEQUENTIAL_ID_SCAN
            and locked_track_id_int is not None
            and obj_track_id is not None
            and obj_track_id < locked_track_id_int
        ):
            continue
        same_track = locked_track_id_int is not None and obj_track_id == locked_track_id_int
        same_spatial_target = body_align_spatial_match(
            obj,
            body_alignment_state.get("lockedCenterAngle"),
            body_alignment_state.get("lockedDistance"),
        )
        if not same_track and not same_spatial_target:
            continue
        distance = safe_float(obj.get("nearestDistance"))
        if distance is None or distance > BODY_ALIGN_LOCK_RELEASE_DISTANCE_M:
            return None
        if (
            semantic_is_known_non_attack(obj)
            and not active_post_aim_decision_hold()
        ) or is_temporarily_eliminated_target(obj):
            return None
        selected = deepcopy(obj)
        selected["stickyLock"] = True
        selected["stickyLockBy"] = "track_id" if same_track else "spatial_match"
        remember_locked_body_align_target(selected)
        return selected
    return None


def find_previous_body_align_target(
    priority_queue: list[dict[str, Any]],
) -> dict[str, Any] | None:
    previous = body_alignment_state.get("target")
    if not previous:
        return None
    previous_track_id = previous.get("trackId")
    for obj in priority_queue:
        obj_track_id = track_id_int(obj)
        try:
            previous_track_id_int = int(previous_track_id) if previous_track_id is not None else None
        except (TypeError, ValueError):
            previous_track_id_int = None
        if (
            BODY_ALIGN_SEQUENTIAL_ID_SCAN
            and previous_track_id_int is not None
            and obj_track_id is not None
            and obj_track_id < previous_track_id_int
        ):
            continue
        same_track = previous_track_id_int is not None and obj_track_id == previous_track_id_int
        same_spatial_target = body_align_spatial_match(
            obj,
            previous.get("centerAngle"),
            previous.get("nearestDistance"),
        )
        if not same_track and not same_spatial_target:
            continue
        if (
            semantic_is_known_non_attack(obj)
            and not active_post_aim_decision_hold()
        ) or is_temporarily_eliminated_target(obj):
            return None
        selected = deepcopy(obj)
        selected["postAimDecisionHold"] = True
        selected["stickyLockBy"] = "post_aim_hold_track" if same_track else "post_aim_hold_spatial"
        return selected
    return None


def confirmed_new_body_align_target(obj: dict[str, Any]) -> dict[str, Any] | None:
    track_id = body_align_target_id(obj)
    if track_id is None:
        return None

    pending_track_id = body_alignment_state.get("pendingTrackId")
    allow_spatial_pending_match = True
    try:
        allow_spatial_pending_match = pending_track_id is None or track_id >= int(pending_track_id)
    except (TypeError, ValueError):
        allow_spatial_pending_match = True
    same_pending_target = (
        pending_track_id == track_id
        or (
            allow_spatial_pending_match
            and body_align_spatial_match(
                obj,
                body_alignment_state.get("pendingCenterAngle"),
                body_alignment_state.get("pendingDistance"),
            )
        )
    )
    if same_pending_target:
        pending_hits = int(body_alignment_state.get("pendingHits", 0) or 0) + 1
    else:
        pending_hits = 1

    body_alignment_state["pendingTrackId"] = track_id
    body_alignment_state["pendingHits"] = pending_hits
    body_alignment_state["pendingCenterAngle"] = obj.get("centerAngle")
    body_alignment_state["pendingDistance"] = obj.get("nearestDistance")

    if pending_hits < BODY_ALIGN_LOCK_CONFIRM_HITS:
        return None

    selected = deepcopy(obj)
    selected["lockConfirmHits"] = pending_hits
    return selected


def clear_pending_body_align_target() -> None:
    body_alignment_state["pendingTrackId"] = None
    body_alignment_state["pendingHits"] = 0
    body_alignment_state["pendingCenterAngle"] = None
    body_alignment_state["pendingDistance"] = None


def recognition_for_track(track_id: int | None, since: float | None = None) -> dict[str, Any] | None:
    if track_id is None:
        return None
    recognition = recognized_lidar_objects.get(int(track_id))
    if not recognition:
        return None
    if since is not None:
        recognized_at = safe_float(recognition.get("recognizedMonoAt"), 0.0) or 0.0
        if float(recognized_at) < float(since):
            return None
    return recognition


def yolo_frame_processed_for_track(track_id: int | None, since: float | None = None) -> bool:
    if track_id is None:
        return False
    try:
        processed_track_id = int(vision_state.get("lastProcessedTrackId"))
    except (TypeError, ValueError):
        return False
    if processed_track_id != int(track_id):
        return False
    processed_at = safe_float(vision_state.get("lastProcessedAt"), 0.0) or 0.0
    if since is not None and float(processed_at) < float(since):
        return False
    return processed_at > 0.0


def mark_body_align_track_rejected(track_id: int, reason: str) -> None:
    rejected = rejected_body_align_track_ids()
    rejected.add(int(track_id))
    body_alignment_state["rejectedTrackIds"] = sorted(rejected)
    body_alignment_state["nextTrackIdMin"] = max(
        int(body_alignment_state.get("nextTrackIdMin", 1) or 1),
        int(track_id) + 1,
    )
    body_alignment_state["lastJudgedTrackId"] = int(track_id)
    body_alignment_state["lastJudgement"] = reason
    body_alignment_state["lockedTrackId"] = None
    body_alignment_state["target"] = None
    body_alignment_state["lockedCenterAngle"] = None
    body_alignment_state["lockedDistance"] = None
    body_alignment_state["alignedSince"] = None
    body_alignment_state["decisionReadyAt"] = None
    body_alignment_state["postAimDecisionHoldRemaining"] = None
    clear_pending_body_align_target()
    clear_world_target_lock(reason)
    reset_aim_stabilizer(reason)


def mark_body_align_track_accepted(track_id: int) -> None:
    accepted = set()
    for item in body_alignment_state.get("acceptedTankTrackIds", []) or []:
        try:
            accepted.add(int(item))
        except (TypeError, ValueError):
            pass
    accepted.add(int(track_id))
    body_alignment_state["acceptedTankTrackIds"] = sorted(accepted)
    body_alignment_state["lastJudgedTrackId"] = int(track_id)
    body_alignment_state["lastJudgement"] = "enemy_tank_confirmed"


def body_align_track_is_accepted_tank(obj: dict[str, Any] | None) -> bool:
    track_id = track_id_int(obj)
    if track_id is None:
        return False
    if semantic_is_tank_candidate(obj or {}):
        return True
    accepted = set()
    for item in body_alignment_state.get("acceptedTankTrackIds", []) or []:
        try:
            accepted.add(int(item))
        except (TypeError, ValueError):
            pass
    return int(track_id) in accepted


def judge_completed_body_align_target(now: float | None = None) -> None:
    target = body_alignment_state.get("target")
    track_id = track_id_int(target)
    if target is None or track_id is None:
        return
    aligned_since = safe_float(body_alignment_state.get("alignedSince"))

    # YOLO can process a frame as soon as the target enters the wider
    # BODY_ALIGN_YOLO_PREALIGN_DEG window.  Do not require the body to also
    # reach the much narrower alignment deadband before consuming that result;
    # otherwise an empty/non-tank result can leave the sequential scan locked
    # on the same LiDAR ID forever.
    if aligned_since is None:
        if not yolo_frame_processed_for_track(track_id):
            return
        recognition = recognition_for_track(track_id)
        if recognition and (
            bool(recognition.get("isAttackTarget", False))
            or is_attack_yolo_class(recognition.get("className"))
        ):
            mark_body_align_track_accepted(track_id)
        elif recognition:
            mark_body_align_track_rejected(
                track_id,
                "rejected_non_tank_yolo_class_before_full_alignment",
            )
        else:
            mark_body_align_track_rejected(
                track_id,
                "rejected_no_enemy_tank_in_prealign_yolo_frame",
            )
        return

    now = monotonic() if now is None else now
    if now - float(aligned_since) < BODY_ALIGN_POST_AIM_DECISION_SECONDS:
        return

    if not yolo_frame_processed_for_track(track_id, since=float(aligned_since)):
        body_alignment_state["lastJudgement"] = "waiting_for_yolo_frame"
        body_alignment_state["postAimDecisionHoldRemaining"] = round(
            max(0.0, BODY_ALIGN_YOLO_FRAME_WAIT_SECONDS - (now - float(aligned_since))),
            3,
        )
        return

    recognition = recognition_for_track(track_id, since=float(aligned_since))
    if recognition and (
        bool(recognition.get("isAttackTarget", False))
        or is_attack_yolo_class(recognition.get("className"))
    ):
        mark_body_align_track_accepted(track_id)
        return

    if recognition:
        mark_body_align_track_rejected(track_id, "rejected_non_tank_yolo_class")
    else:
        mark_body_align_track_rejected(track_id, "rejected_no_enemy_tank_after_2s_yolo_window")


def choose_body_align_target(
    priority_queue: list[dict[str, Any]],
) -> dict[str, Any] | None:
    global body_alignment_state

    if BODY_ALIGN_SEQUENTIAL_ID_SCAN:
        judge_completed_body_align_target()
        maybe_reset_exhausted_sequential_scan(priority_queue)

    locked_track_id = body_alignment_state.get("lockedTrackId")
    locked_target = find_sticky_locked_target(priority_queue, locked_track_id)
    if locked_target:
        clear_pending_body_align_target()
        return locked_target

    if active_post_aim_decision_hold():
        previous_target = find_previous_body_align_target(priority_queue)
        if previous_target:
            clear_pending_body_align_target()
            return previous_target
        return None

    candidates = [
        obj for obj in priority_queue
        if is_body_align_candidate(obj) and not is_temporarily_eliminated_target(obj)
    ]
    if not candidates:
        body_alignment_state["lockedTrackId"] = None
        body_alignment_state["lockedCenterAngle"] = None
        body_alignment_state["lockedDistance"] = None
        clear_pending_body_align_target()
        return None

    if locked_track_id is not None:
        for obj in candidates:
            if obj.get("trackId") == locked_track_id:
                clear_pending_body_align_target()
                remember_locked_body_align_target(obj)
                return deepcopy(obj)

    candidates.sort(key=body_align_target_sort_key)
    selected = confirmed_new_body_align_target(candidates[0])
    if selected is None:
        return None

    clear_pending_body_align_target()
    remember_locked_body_align_target(selected)
    return selected


def body_turn_weight(angle_error_deg: float) -> float:
    error = abs(angle_error_deg)
    if error <= BODY_ALIGN_SLOW_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_SLOW
    if error <= BODY_ALIGN_MEDIUM_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_MEDIUM
    if error <= BODY_ALIGN_FAST_ZONE_DEG:
        return BODY_ALIGN_WEIGHT_FAST
    return BODY_ALIGN_WEIGHT_MAX


def update_body_alignment(
    priority_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    李⑥껜 ?뚯쟾 ?붿껌留??앹꽦?쒕떎.

    ?덉쟾 ?숈옉:
    - ?뺣젹 以묒뿉???꾩쭊?섏? ?딅뒗??
    - 諛쒖궗?섏? ?딅뒗??
    - ?곕뱶諛대뱶 ?덉뿉?쒕뒗 ?뚯쟾??硫덉텣??
    """
    global body_alignment_state

    enabled = bool(body_alignment_state.get("enabled", AUTO_BODY_ALIGN_ENABLED))
    if not enabled:
        body_alignment_state.update(
            {
                "target": None,
                "moveAD": {"command": "", "weight": 0.0},
                "aligned": False,
                "reason": "automatic_body_alignment_disabled",
                "stickyLock": False,
                "alignedSince": None,
                "decisionReadyAt": None,
                "postAimDecisionHoldRemaining": None,
            }
        )
        return deepcopy(body_alignment_state)

    target = choose_body_align_target(priority_queue)
    if target is None:
        pending_hits = int(body_alignment_state.get("pendingHits", 0) or 0)
        pending_track_id = body_alignment_state.get("pendingTrackId")
        reason = (
            "confirming_new_target"
            if pending_track_id is not None and pending_hits > 0
            else "no_remaining_lidar_id_target"
        )
        body_alignment_state.update(
            {
                "target": None,
                "moveAD": {"command": "", "weight": 0.0},
                "aligned": False,
                "reason": reason,
                "stickyLock": False,
                "alignedSince": None,
                "decisionReadyAt": None,
                "postAimDecisionHoldRemaining": None,
            }
        )
        return deepcopy(body_alignment_state)

    angle_error = normalize_signed_angle(float(target.get("centerAngle", 0.0)))
    target["bodyRelativeAngleErrorDeg"] = round(angle_error, 3)
    now = monotonic()
    previous_target = body_alignment_state.get("target")
    same_target_as_before = bool(
        previous_target
        and (
            (
                previous_target.get("trackId") is not None
                and target.get("trackId") == previous_target.get("trackId")
            )
            or body_align_spatial_match(
                target,
                previous_target.get("centerAngle"),
                previous_target.get("nearestDistance"),
            )
        )
    )

    if abs(angle_error) <= BODY_ALIGN_DEADBAND_DEG:
        move_ad = {"command": "", "weight": 0.0}
        aligned = True
        aligned_since = safe_float(body_alignment_state.get("alignedSince")) if same_target_as_before else None
        if aligned_since is None:
            aligned_since = now
        decision_ready_at = float(aligned_since) + BODY_ALIGN_POST_AIM_DECISION_SECONDS
        hold_remaining = max(0.0, decision_ready_at - now)
        reason = (
            "post_aim_decision_hold"
            if hold_remaining > 0.0
            else "target_inside_alignment_deadband"
        )
    else:
        # Tank Challenge API:
        # A = 李⑥껜 醫뚰쉶?? D = 李⑥껜 ?고쉶??
        move_ad = {
            "command": "A" if angle_error < 0.0 else "D",
            "weight": round(body_turn_weight(angle_error), 3),
        }
        aligned = False
        reason = "turn_left_toward_target" if angle_error < 0.0 else "turn_right_toward_target"
        aligned_since = None
        decision_ready_at = None
        hold_remaining = None

    body_alignment_state.update(
        {
            "target": target,
            "moveAD": move_ad,
            "aligned": aligned,
            "reason": reason,
            "stickyLock": bool(target.get("stickyLock")),
            "alignedSince": aligned_since,
            "decisionReadyAt": decision_ready_at,
            "postAimDecisionHoldRemaining": round(hold_remaining, 3) if hold_remaining is not None else None,
        }
    )
    return deepcopy(body_alignment_state)


def summarize_lidar(data: dict[str, Any]) -> dict[str, Any]:
    rays = parse_lidar_rays(data)
    hits = detected_rays(rays)
    flat_ground_y = estimate_flat_ground_y(hits)
    lidar_origin = data.get("lidarOrigin", {}) or {}
    lidar_origin_y = safe_float(lidar_origin.get("y"))
    local_ground_y = estimate_local_ground_y(rays, flat_ground_y)
    ground_grid = build_local_ground_grid(hits)

    terrain_sectors, terrain_ray_count = analyze_terrain(rays, local_ground_y, lidar_origin_y)
    terrain_decision = summarize_front_terrain_decision(terrain_sectors)
    raw_objects, object_point_count = classify_object_geometry(
        hits,
        ground_grid,
        local_ground_y,
        lidar_origin_y,
        terrain_sectors,
    )

    sim_time = safe_float(data.get("time"), monotonic()) or monotonic()
    provisional_objects = [
        obj for obj in raw_objects
        if not obj.get("terrainConnected", False)
    ][:PROVISIONAL_OBJECT_LIMIT]

    tracked_objects = [
        enrich_with_recognition(obj)
        for obj in object_tracker.update(raw_objects, float(sim_time))
    ]
    confirmed_objects = [obj for obj in tracked_objects if obj.get("confirmed")]

    fusion_priority_queue, primary_fusion_target = make_fusion_priority_queue(
        confirmed_objects=confirmed_objects,
        provisional_objects=provisional_objects,
    )
    body_alignment = update_body_alignment(fusion_priority_queue)
    scan_target = deepcopy(body_alignment.get("target")) if body_alignment.get("target") else None

    return {
        "simulationTime": data.get("time"),
        "lidarOrigin": lidar_origin,
        "lidarRotation": data.get("lidarRotation", {}),
        "rawRayCount": len(rays),
        "rawDetectedPointCount": len(hits),
        "estimatedGroundY": round_or_none(flat_ground_y),
        "localGroundY": round_or_none(local_ground_y),
        "lidarOriginY": round_or_none(lidar_origin_y),
        "localGroundGridCellCount": sum(len(values) for values in ground_grid.values()),
        "terrainRayCount": terrain_ray_count,
        "terrainSectors": terrain_sectors,
        "terrainDecision": terrain_decision,
        "contourPoints": make_contour_points(hits),
        "frontVerticalProfile": make_front_vertical_profile(rays, local_ground_y, lidar_origin_y),
        "objectCandidatePointCount": object_point_count,
        "rawObjectCount": len(raw_objects),
        "rawObjects": raw_objects,
        "suppressedTerrainObjectCount": sum(1 for obj in raw_objects if obj.get("terrainConnected")),
        "provisionalObjectCount": len(provisional_objects),
        "provisionalObjects": provisional_objects,
        "trackedObjectCount": len(tracked_objects),
        "confirmedObjectCount": len(confirmed_objects),
        "trackedObjects": tracked_objects,
        "confirmedObjects": confirmed_objects,
        "fusionPriorityQueue": fusion_priority_queue,
        "fusionPriorityQueueCount": len(fusion_priority_queue),
        "primaryFusionTarget": primary_fusion_target,
        "scanTarget": scan_target,
        "bodyAlignment": body_alignment,
        "visionTarget": deepcopy(vision_state.get("target")),
        "visionDetections": deepcopy(vision_state.get("detections", [])),
        "lidarVisionFusion": deepcopy(vision_state.get("lidarFusion")),
        "impactMarkers": list(impact_history),
        "frontClearance": make_front_clearance(confirmed_objects),
    }


def print_status(scan: dict[str, Any]) -> None:
    global last_print_time
    now = monotonic()
    if now - last_print_time < PRINT_INTERVAL_SECONDS:
        return
    last_print_time = now
    print("\n" + "=" * 112)
    print(
        f"time={scan.get('simulationTime')} | rays={scan.get('rawRayCount')} | hits={scan.get('rawDetectedPointCount')} | "
        f"groundGrid={scan.get('localGroundGridCellCount')} | rawObjects={scan.get('rawObjectCount')} | "
        f"suppressedTerrainObjects={scan.get('suppressedTerrainObjectCount')} | confirmedTracks={scan.get('confirmedObjectCount')}"
    )
    decision = scan.get("terrainDecision", {})
    print(
        f"terrainDecision={decision.get('state')} | action={decision.get('recommendedAction')} | "
        f"reason={decision.get('reason')} | deadEnd={decision.get('deadEndDetected')}"
    )
    primary = scan.get("primaryFusionTarget")
    if primary:
        print(
            "priorityFusionTarget="
            f"#{primary.get('fusionPriorityRank')} | "
            f"{primary.get('candidateLabel')}? | "
            f"geometry={primary.get('geometryClass')} | "
            f"dist={float(primary.get('nearestDistance', 0.0)):.1f}m | "
            f"angle={float(primary.get('centerAngle', 0.0)):+.1f}deg | "
            f"tier={primary.get('fusionPriorityTier')} | "
            f"stage={primary.get('trackingStage')}"
        )
    else:
        print("priorityFusionTarget=none")

    alignment = scan.get("bodyAlignment", {})
    alignment_target = alignment.get("target")
    if alignment_target:
        print(
            "bodyAlignment="
            f"enabled={alignment.get('enabled')} | "
            f"reason={alignment.get('reason')} | "
            f"moveAD={alignment.get('moveAD')} | "
            f"target={alignment_target.get('candidateLabel')}? "
            f"{float(alignment_target.get('nearestDistance', 0.0)):.1f}m "
            f"{float(alignment_target.get('bodyRelativeAngleErrorDeg', 0.0)):+.1f}deg"
        )
    else:
        print(
            "bodyAlignment="
            f"enabled={alignment.get('enabled')} | "
            f"reason={alignment.get('reason')} | target=none"
        )

    for obj in scan.get("confirmedObjects", [])[:20]:
        print(
            f"  {obj['candidateLabel']}{obj['trackId']}? | geometry={obj['geometryClass']} | "
            f"dist={obj['nearestDistance']:.1f}m | angle={obj['centerAngle']:+.1f}deg | "
            f"width={obj['estimatedWidth']:.2f}m | persistence={obj['persistenceHits']}/{obj['historySize']}"
        )


def image_from_request() -> Image.Image | None:
    image_file = request.files.get("image") or request.files.get("file")
    if image_file:
        return Image.open(image_file.stream).convert("RGB")

    data = request.get_json(silent=True) or {}
    encoded = data.get("image") or data.get("frame") or data.get("capture")
    if isinstance(encoded, str):
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        except (OSError, ValueError, TypeError):
            return None
    return None


def detect_visual_targets(image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    model = load_yolo_model()
    width, height = image.size
    results = model.predict(image, conf=VISION_CONFIDENCE_MIN, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        names = result.names or {}
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id))
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            error_x = (center_x - width / 2.0) / max(1.0, width)
            error_y = (center_y - height / 2.0) / max(1.0, height)
            in_decision_center = (
                abs(error_x) <= VISION_DECISION_CENTER_X
                and abs(error_y) <= VISION_DECISION_CENTER_Y
            )
            detections.append({
                "classId": class_id,
                "className": class_name,
                "confidence": round(confidence, 4),
                "mappedLidarGeometry": lidar_geometry_for_yolo_class(class_name),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
                "imageSize": {"width": width, "height": height},
                "errorX": round(error_x, 4),
                "errorY": round(error_y, 4),
                "area": round(area, 1),
                "isAimTarget": is_attack_yolo_class(class_name),
                "inDecisionCenter": in_decision_center,
            })

    if not detections:
        return [], None

    target_candidates = [
        item for item in detections
        if item["inDecisionCenter"] and item["isAimTarget"]
    ]
    if not target_candidates:
        return detections, None
    target_candidates.sort(key=lambda item: (item["confidence"], item["area"]), reverse=True)
    return detections, deepcopy(target_candidates[0])


# =============================================================================
# 7. Flask ?붾뱶?ъ씤??# =============================================================================
@app.route("/info", methods=["POST"])
def info():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    scan = summarize_lidar(data)
    with state_lock:
        global latest_state, latest_raw_info
        latest_state = scan
        latest_raw_info = deepcopy(data)
    print_status(scan)
    return jsonify({"status": "success", "control": ""})


@app.route("/detect", methods=["POST"])
def detect():
    global last_vision_detect_run_time
    now = monotonic()
    with state_lock:
        vision_state["lastDetectRequestAt"] = now
        vision_state["lastDetectRequestMode"] = "received"
        vision_state["lastDetectRequestHadImage"] = False

    image = image_from_request()
    if image is None:
        with state_lock:
            vision_state["lastDetectRequestMode"] = "no_image"
        return jsonify({"status": "error", "message": "No image received"}), 400
    with state_lock:
        vision_state["lastDetectRequestHadImage"] = True

    if not USE_YOLO_FOR_RECOGNITION:
        with state_lock:
            vision_state["lastDetectRequestMode"] = "yolo_recognition_disabled"
            vision_state["detections"] = []
            vision_state["target"] = None
            vision_state["lidarFusion"] = None
            latest_state["visionDetections"] = []
            latest_state["visionTarget"] = None
            latest_state["lidarVisionFusion"] = None
        if request.args.get("format") == "debug":
            return jsonify({
                "status": "success",
                "mode": "yolo_recognition_disabled",
                "detections": [],
                "target": None,
                "lidarVisionFusion": None,
            })
        return jsonify([])

    with state_lock:
        cached_detections = deepcopy(vision_state.get("detections", []))
        cached_target = deepcopy(vision_state.get("target"))
        cached_fusion = deepcopy(vision_state.get("lidarFusion"))
        recognition_window = yolo_recognition_window_payload(now)
    if not recognition_window.get("active"):
        with state_lock:
            vision_state["lastDetectRequestMode"] = "outside_yolo_recognition_window"
        if request.args.get("format") == "debug":
            return jsonify({
                "status": "success",
                "mode": "outside_yolo_recognition_window",
                "recognitionWindow": recognition_window,
                "detections": [],
                "target": None,
                "lidarVisionFusion": cached_fusion,
            })
        return jsonify([])

    if now - last_vision_detect_run_time < VISION_DETECT_MIN_INTERVAL_SECONDS:
        with state_lock:
            vision_state["lastDetectRequestMode"] = "vision_throttled"
        if request.args.get("format") == "debug":
            return jsonify({
                "status": "success",
                "mode": "vision_throttled",
                "maxFps": VISION_DETECT_MAX_FPS,
                "recognitionWindow": recognition_window,
                "detections": cached_detections,
                "target": cached_target,
                "lidarVisionFusion": cached_fusion,
            })
        return jsonify([
            {
                "className": item["className"],
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "color": "#00FF00" if item.get("isAimTarget") else "#FFD166",
                "filled": False,
                "updateBoxWhileMoving": True,
            }
            for item in cached_detections
        ])

    if not vision_detect_lock.acquire(blocking=False):
        with state_lock:
            vision_state["lastDetectRequestMode"] = "vision_busy"
        if request.args.get("format") == "debug":
            return jsonify({
                "status": "success",
                "mode": "vision_busy",
                "maxFps": VISION_DETECT_MAX_FPS,
                "recognitionWindow": recognition_window,
                "detections": cached_detections,
                "target": cached_target,
                "lidarVisionFusion": cached_fusion,
            })
        return jsonify([
            {
                "className": item["className"],
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "color": "#00FF00" if item.get("isAimTarget") else "#FFD166",
                "filled": False,
                "updateBoxWhileMoving": True,
            }
            for item in cached_detections
        ])

    last_vision_detect_run_time = now

    try:
        try:
            detections, target = detect_visual_targets(image)
        except Exception as exc:
            with state_lock:
                vision_state["lastDetectRequestMode"] = "vision_inference_error"
                vision_state["lastInferenceError"] = str(exc)
            raise
        with state_lock:
            lidar_target = current_lidar_recognition_target(latest_state)
            fusion = build_lidar_vision_fusion(lidar_target, target)
            fused_target = target if fusion and fusion.get("isAttackTarget") else None
            remember_recognized_lidar_object(lidar_target, target, fusion)
            vision_state["detections"] = detections
            vision_state["target"] = fused_target
            vision_state["lidarFusion"] = fusion
            vision_state["lastDetectedAt"] = now if fused_target else vision_state.get("lastDetectedAt", 0.0)
            vision_state["lastProcessedAt"] = now
            vision_state["lastProcessedTrackId"] = track_id_int(lidar_target)
            vision_state["modelPath"] = str(MODEL_PATH)
            vision_state["modelLoaded"] = True
            vision_state["lastDetectRequestMode"] = "processed"
            vision_state["lastInferenceError"] = None
            refresh_latest_state_recognitions()
            latest_state["visionDetections"] = deepcopy(detections)
            latest_state["visionTarget"] = deepcopy(fused_target)
            latest_state["lidarVisionFusion"] = deepcopy(fusion)
    finally:
        vision_detect_lock.release()

    return jsonify({
        "status": "success",
        "model": MODEL_PATH.name,
        "recognitionWindow": recognition_window,
        "detections": detections,
        "target": fused_target,
        "lidarVisionFusion": fusion,
    }) if request.args.get("format") == "debug" else jsonify([
        {
            "className": item["className"],
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "color": "#00FF00" if item.get("isAimTarget") else "#FFD166",
            "filled": False,
            "updateBoxWhileMoving": True,
        }
        for item in detections
    ])


@app.route("/lidar_status", methods=["GET"])
def lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_state))


@app.route("/lidar_monitor_status", methods=["GET"])
def lidar_monitor_status():
    """
    釉뚮씪?곗???寃쎈웾 payload.

    ?곷떒 ?ㅺ낸 酉곗? 以묒떖??64梨꾨꼸 ?꾨줈?뚯씪??鍮좊Ⅴ寃?媛깆떊?섎룄濡?
    raw 媛앹껜 諛곗뿴怨??꾩껜 ?꾨갑 ?ъ씤???대씪?곕뱶???쒖쇅?쒕떎.
    """
    with state_lock:
        scan = deepcopy(latest_state)

    return jsonify(
        {
            "simulationTime": scan.get("simulationTime"),
            "rawRayCount": scan.get("rawRayCount", 0),
            "rawDetectedPointCount": scan.get("rawDetectedPointCount", 0),
            "localGroundGridCellCount": scan.get("localGroundGridCellCount", 0),
            "terrainDecision": scan.get("terrainDecision", {}),
            "terrainSectors": scan.get("terrainSectors", []),
            "contourPoints": scan.get("contourPoints", []),
            "frontVerticalProfile": scan.get("frontVerticalProfile", {}),
            "provisionalObjects": scan.get("provisionalObjects", []),
            "confirmedObjects": scan.get("confirmedObjects", []),
            "rawObjectCount": scan.get("rawObjectCount", 0),
            "suppressedTerrainObjectCount": scan.get("suppressedTerrainObjectCount", 0),
            "provisionalObjectCount": scan.get("provisionalObjectCount", 0),
            "trackedObjectCount": scan.get("trackedObjectCount", 0),
            "confirmedObjectCount": scan.get("confirmedObjectCount", 0),
            "fusionPriorityQueue": scan.get("fusionPriorityQueue", []),
            "fusionPriorityQueueCount": scan.get("fusionPriorityQueueCount", 0),
            "primaryFusionTarget": scan.get("primaryFusionTarget"),
            "bodyAlignment": scan.get("bodyAlignment", {}),
            "visionTarget": scan.get("visionTarget"),
            "visionDetectionCount": len(scan.get("visionDetections", [])),
            "lidarVisionFusion": scan.get("lidarVisionFusion"),
            "impactMarkers": scan.get("impactMarkers", []),
            "fireControl": fire_readiness_status(),
            "aimZero": deepcopy(aim_zero_state),
            "aimStabilizer": deepcopy(aim_stabilizer_state),
        }
    )


@app.route("/body_align_status", methods=["GET"])
def body_align_status():
    with state_lock:
        return jsonify(deepcopy(latest_state.get("bodyAlignment", {})))


@app.route("/vision_status", methods=["GET"])
def vision_status():
    with state_lock:
        payload = deepcopy(vision_state)
        payload["activeTarget"] = active_vision_target()
        payload["currentLidarTarget"] = current_lidar_fusion_target(latest_state)
    return jsonify(payload)


@app.route("/fire_status", methods=["GET"])
def fire_status():
    with state_lock:
        return jsonify(fire_readiness_status())


@app.route("/fire_confirm", methods=["POST", "GET"])
def fire_confirm():
    with state_lock:
        status = fire_readiness_status()
        if not (status.get("buttonEnabled") or status.get("canApproveFire") or status.get("ready")):
            return jsonify({"status": "not_ready", "fireControl": status}), 409
        now = monotonic()
        fire_control_state["approvedUntil"] = now + FIRE_BUTTON_APPROVAL_SECONDS
        fire_control_state["approvedAt"] = datetime.now().isoformat(timespec="milliseconds")
        status = fire_readiness_status(now)
    return jsonify({"status": "approved", "fireControl": status})


@app.route("/aim_zero", methods=["POST", "GET"])
def aim_zero():
    raw_action = str(request.args.get("action", "")).strip().lower()
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", raw_action or "set")).strip().lower()
    current = float(aim_zero_state.get("offsetY", 0.0) or 0.0)

    if action == "reset":
        offset = 0.0
    elif action == "capture":
        target = active_vision_target()
        if not target:
            return jsonify({"status": "not_ready", "message": "No active vision target to capture."}), 409
        offset = float(target.get("errorY", 0.0))
    elif action in {"up", "increase"}:
        offset = current + VISION_AIM_ZERO_STEP_Y
    elif action in {"down", "decrease"}:
        offset = current - VISION_AIM_ZERO_STEP_Y
    else:
        offset = safe_float(data.get("offsetY", request.args.get("offsetY")), current) or current

    offset = clamp(float(offset), -0.35, 0.35)
    with state_lock:
        aim_zero_state["offsetY"] = round(offset, 4)
        aim_zero_state["updatedAt"] = datetime.now().isoformat(timespec="milliseconds")
        status = fire_readiness_status()
    return jsonify({"status": "success", "aimZero": deepcopy(aim_zero_state), "fireControl": status})


@app.route("/body_align_enable", methods=["POST", "GET"])
def body_align_enable():
    global body_alignment_state

    raw = str(request.args.get("enabled", "true")).strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}

    with state_lock:
        body_alignment_state["enabled"] = enabled
        if not enabled:
            body_alignment_state["lockedTrackId"] = None
            body_alignment_state["target"] = None
            body_alignment_state["moveAD"] = {"command": "", "weight": 0.0}
            body_alignment_state["aligned"] = False
            body_alignment_state["reason"] = "automatic_body_alignment_disabled"

    return jsonify(
        {
            "status": "success",
            "enabled": enabled,
            "message": "Automatic body alignment setting updated.",
        }
    )


@app.route("/priority_status", methods=["GET"])
def priority_status():
    """?댄썑 YOLO 寃고빀???ъ슜??360??LiDAR ?꾨낫 ?먮? 諛섑솚?쒕떎."""
    with state_lock:
        return jsonify(
            {
                "nearPriorityMaxDistanceM": PRIORITY_NEAR_MAX_DISTANCE_M,
                "primaryFusionTarget": deepcopy(latest_state.get("primaryFusionTarget")),
                "fusionPriorityQueue": deepcopy(latest_state.get("fusionPriorityQueue", [])),
            }
        )


@app.route("/raw_lidar_status", methods=["GET"])
def raw_lidar_status():
    with state_lock:
        return jsonify(deepcopy(latest_raw_info))


@app.route("/front_vertical_profile", methods=["GET"])
def front_vertical_profile():
    with state_lock:
        return jsonify(deepcopy(latest_state.get("frontVerticalProfile", {})))


@app.route("/export_snapshot", methods=["POST", "GET"])
def export_snapshot():
    label = re.sub(r"[^0-9A-Za-z媛-??-]+", "_", request.args.get("label", "snapshot")).strip("_") or "snapshot"
    with state_lock:
        raw_copy = deepcopy(latest_raw_info)
        analyzed_copy = deepcopy(latest_state)
    if not raw_copy:
        return jsonify({"status": "error", "message": "No /info frame has been received yet."}), 400
    output_dir = Path.cwd() / "lidar_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.json"
    path.write_text(json.dumps({"label": label, "rawInfo": raw_copy, "analysis": analyzed_copy}, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"status": "success", "savedPath": str(path), "label": label})


@app.route("/lidar_view", methods=["GET"])
def lidar_view():
    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tank LiDAR Auto Body Alignment v8.7 Tracking Fix</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }
    canvas { background: #181818; border: 1px solid #555; }
    .dashboard { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
    .views { display: flex; flex-direction: column; gap: 18px; }
    .panel-title { font-size: 16px; font-weight: bold; margin: 0 0 6px 0; }
    pre { min-width: 720px; max-width: 980px; white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
    .legend { margin: 4px 0 0 0; font-size: 13px; color: #ddd; }
    .side-panel { display: flex; flex-direction: column; gap: 10px; }
    .firebar { display: flex; align-items: center; gap: 10px; }
    .firebar button { border: 1px solid #7a2424; background: #3a1515; color: #eee; padding: 9px 16px; font-weight: bold; border-radius: 6px; cursor: pointer; }
    .firebar button[data-ready="true"] { background: #cf2f2f; border-color: #ff6b6b; color: #fff; }
    .firebar button:disabled { opacity: 0.45; cursor: not-allowed; }
    .firebar span { color: #ddd; font-size: 13px; }
    .zerobar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .zerobar button { border: 1px solid #555; background: #222; color: #eee; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
    .zerobar span { color: #ddd; font-size: 13px; }
  </style>
</head>
<body>
  <h2>LiDAR Auto Body Alignment v8.7: Tracking Mode Fix + Action Debug</h2>
  <p>Top view: green contour = observed terrain outline. Terrain arcs: green=passable, yellow=caution, red=blocked, gray=unknown.</p>
  <p>Heavy front point-cloud view is disabled to reduce browser rendering and JSON transfer load.</p>
  <p>64-channel profile: select the azimuth closest to body-forward 0째. X = forward horizontal range, Y = height relative to local ground. Cyan polyline = estimated terrain profile.</p>
  <p>Object labels remain LiDAR-only candidates: TH?=thin(person/tree-like), BK?=bulky(tank/rock-like), UK?=unknown. Final person/tank labels require later YOLO fusion.</p>
  <p>Fusion queue priority: BK? bulky candidates first ??candidates within 50 m ??confirmed tracks ??nearer distance. The white ring marks the current fusion priority #1.</p>
  <p>Automatic body alignment: LiDAR track IDs are aimed in ascending order. After alignment, YOLO gets 2 seconds to confirm enemy tank; non-tanks are skipped and lower IDs are not revisited.</p>
  <div class="dashboard">
    <div class="views">
      <div>
        <div class="panel-title">Top view: local terrain outline</div>
        <canvas id="topRadar" width="860" height="860"></canvas>
      </div>
      <div>
        <div class="panel-title">Centerline side profile: nearest body-forward azimuth 횞 vertical channels</div>
        <canvas id="verticalProfile" width="860" height="520"></canvas>
        <div class="legend">Side profile: green ray = detected | gray dashed ray = missed | cyan line = local ground profile | blue point = LiDAR origin</div>
      </div>
    </div>
    <div class="side-panel">
      <div class="firebar">
        <button id="fireButton" disabled data-ready="false">FIRE</button>
        <span id="fireText">Waiting for aim lock</span>
      </div>
      <div class="zerobar">
        <button id="zeroDownButton">Zero -</button>
        <button id="zeroUpButton">Zero +</button>
        <button id="zeroCaptureButton">Use current Y</button>
        <button id="zeroResetButton">Reset zero</button>
        <span id="zeroText">Zero Y 0.000</span>
      </div>
      <pre id="status">Waiting for /info data...</pre>
    </div>
  </div>
<script>
const topCanvas = document.getElementById('topRadar');
const topCtx = topCanvas.getContext('2d');
const profileCanvas = document.getElementById('verticalProfile');
const profileCtx = profileCanvas.getContext('2d');
const statusText = document.getElementById('status');
const fireButton = document.getElementById('fireButton');
const fireText = document.getElementById('fireText');
const zeroText = document.getElementById('zeroText');
const MAX_DISTANCE = 120.0;
const FRONT_HORIZONTAL_LIMIT = 60.0;
const FRONT_VERTICAL_MIN = -22.5;
const FRONT_VERTICAL_MAX = 22.5;
const PROFILE_MAX_DISTANCE = 120.0;
const PROFILE_HEIGHT_MIN = -12.0;
const PROFILE_HEIGHT_MAX = 12.0;

function polar(angleDeg, distance, cx, cy, radius) {
  const a = angleDeg * Math.PI / 180.0;
  const r = Math.min(distance, MAX_DISTANCE) / MAX_DISTANCE * radius;
  return { x: cx + Math.sin(a) * r, y: cy - Math.cos(a) * r };
}
function terrainColor(state) {
  if (state === 'passable') return '#44d62c';
  if (state === 'caution') return '#ffb703';
  if (state === 'blocked') return '#ff3030';
  return '#777777';
}
function objectColor(kind) {
  if (kind === 'thin') return '#4cc9f0';
  if (kind === 'bulky') return '#f72585';
  return '#f1fa8c';
}
function objectLabel(obj) {
  const base = (obj.candidateLabel || 'UK') + (obj.trackId == null ? '?' : obj.trackId + '?');
  return obj.recognizedClass ? obj.recognizedClass + ' <- ' + base : base;
}
function drawImpactMarker(ctx, marker, cx, cy, radius) {
  if (marker.angle == null || marker.distance == null) return;
  const q=polar(Number(marker.angle),Number(marker.distance),cx,cy,radius);
  ctx.save();
  ctx.strokeStyle='#ff3030';
  ctx.fillStyle='#ff3030';
  ctx.lineWidth=4;
  ctx.beginPath();
  ctx.moveTo(q.x-10,q.y-10);
  ctx.lineTo(q.x+10,q.y+10);
  ctx.moveTo(q.x+10,q.y-10);
  ctx.lineTo(q.x-10,q.y+10);
  ctx.stroke();
  ctx.font='bold 13px Arial';
  ctx.fillText('IMPACT '+Number(marker.distance).toFixed(1)+'m',q.x+13,q.y-12);
  ctx.restore();
}
function updateFireControl(control) {
  control = control || {};
  const ready = !!control.ready;
  const buttonEnabled = !!(control.buttonEnabled || control.canApproveFire || ready);
  const approved = !!control.approved;
  fireButton.disabled = !buttonEnabled;
  fireButton.dataset.ready = buttonEnabled ? 'true' : 'false';
  if (control.fireOnNextAction) {
    fireText.textContent = 'Approved: next action will fire';
  } else if (approved && ready) {
    fireText.textContent = 'Fire approved';
  } else if (approved) {
    fireText.textContent = 'Fire armed: waiting for fine aim';
  } else if (ready) {
    fireText.textContent = 'Aim locked: press FIRE';
  } else if (buttonEnabled) {
    fireText.textContent = 'Near target: press FIRE to arm';
  } else {
    fireText.textContent = 'Not ready: '+(control.reason || 'waiting');
  }
}
function updateAimZero(scan) {
  const zero = (scan && scan.aimZero) || {};
  const control = (scan && scan.fireControl) || {};
  const err = control.aimError || {};
  zeroText.textContent =
    'Zero Y '+Number(zero.offsetY || 0).toFixed(3)
    +' | rawY='+(err.rawY == null ? 'n/a' : Number(err.rawY).toFixed(3))
    +' | adjY='+(err.adjustedY == null ? 'n/a' : Number(err.adjustedY).toFixed(3));
}
function distanceColor(distance) {
  if (distance <= 20) return '#ff3030';
  if (distance <= 50) return '#ffca3a';
  return '#44d62c';
}
function fusionColor(status) {
  if (status === 'recognized_attack_target') return '#00ff88';
  if (status === 'recognized_non_attack_target') return '#4cc9f0';
  if (status === 'recognized_geometry_mismatch') return '#ff3030';
  if (status === 'waiting_for_detection') return '#ffd166';
  if (status === 'waiting_for_alignment') return '#ff9f1c';
  return '#aaaaaa';
}
function drawArc(ctx, sector, cx, cy, radius) {
  const boundary = Math.max(3, Math.min(sector.hazardBoundaryRange || 20, 120));
  const r = boundary / MAX_DISTANCE * radius;
  const center = sector.centerAngle * Math.PI / 180.0;
  const half = 4.5 * Math.PI / 180.0;
  ctx.strokeStyle = terrainColor(sector.state);
  ctx.lineWidth = sector.state === 'blocked' ? 11 : 7;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(18, r), -Math.PI / 2 + center - half, -Math.PI / 2 + center + half);
  ctx.stroke();
  ctx.lineWidth = 1;
}
function drawTop(scan) {
  const ctx = topCtx, canvas = topCanvas;
  const w = canvas.width, h = canvas.height, cx = w/2, cy = h/2, radius = Math.min(w,h)*0.46;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#555'; ctx.fillStyle = '#bbb'; ctx.font = '13px Arial';
  for (const d of [30,60,90,120]) { const r=radius*d/MAX_DISTANCE; ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke(); ctx.fillText(d+' m',cx+5,cy-r+16); }
  ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx,cy-radius); ctx.stroke();
  ctx.fillStyle='#eee'; ctx.fillText('0 deg body-forward',cx+8,cy-radius+16); ctx.fillText('-90 deg',cx-radius+4,cy-8); ctx.fillText('+90 deg',cx+radius-60,cy-8);

  for (const p of (scan.contourPoints || [])) { const q=polar(p.angle,p.distance,cx,cy,radius); ctx.fillStyle='#35e835'; ctx.beginPath(); ctx.arc(q.x,q.y,2.5,0,Math.PI*2); ctx.fill(); }
  for (const sector of (scan.terrainSectors || [])) drawArc(ctx,sector,cx,cy,radius);

  if (scan.terrainDecision && scan.terrainDecision.deadEndDetected) {
    ctx.fillStyle='#ff3030'; ctx.font='bold 22px Arial'; ctx.fillText('POSSIBLE DEAD END', 20, 34);
  }

  ctx.font='bold 14px Arial';
  // ?꾩떆 ?꾨낫??鍮?留덉빱濡?利됱떆 ?쒖떆?쒕떎.
  ctx.save();
  ctx.globalAlpha=0.60;
  for (const obj of (scan.provisionalObjects || [])) {
    const q=polar(obj.centerAngle,obj.medianDistance,cx,cy,radius);
    ctx.strokeStyle=objectColor(obj.geometryClass); ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(q.x,q.y,5,0,Math.PI*2); ctx.stroke();
    ctx.fillStyle='#ddd'; ctx.fillText('~'+obj.candidateLabel+'? '+obj.nearestDistance.toFixed(1)+'m',q.x+7,q.y+13);
  }
  ctx.restore();

  // ?뺤젙 ?몃옓? 梨꾩썙吏?留덉빱濡??쒖떆?쒕떎.
  for (const obj of (scan.confirmedObjects || [])) {
    const q=polar(obj.centerAngle,obj.medianDistance,cx,cy,radius);
    ctx.fillStyle=objectColor(obj.geometryClass); ctx.beginPath(); ctx.arc(q.x,q.y,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=obj.recognizedClass ? '#00ff88' : '#fff';
    ctx.fillText(objectLabel(obj)+' '+obj.nearestDistance.toFixed(1)+'m '+obj.centerAngle.toFixed(1)+'deg',q.x+8,q.y-8);
  }

  // ?꾩옱 LiDAR -> YOLO 寃고빀 ?쒖쟻? ?곗깋 留곸쑝濡?媛뺤“?쒕떎.
  if (scan.primaryFusionTarget) {
    const target=scan.primaryFusionTarget;
    const q=polar(target.centerAngle,target.medianDistance,cx,cy,radius);
    ctx.strokeStyle='#ffffff'; ctx.lineWidth=4;
    ctx.beginPath(); ctx.arc(q.x,q.y,14,0,Math.PI*2); ctx.stroke();
    ctx.lineWidth=1;
    ctx.fillStyle='#ffffff';
    ctx.fillText(
      'PRIORITY #1 '+objectLabel(target)+' '+target.nearestDistance.toFixed(1)+'m',
      q.x+18,q.y+24
    );
  }

  if (scan.scanTarget) {
    const target=scan.scanTarget;
    const q=polar(target.centerAngle,target.medianDistance,cx,cy,radius);
    ctx.strokeStyle='#00e5ff'; ctx.lineWidth=3;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.arc(q.x,q.y,25,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth=1;
    ctx.fillStyle='#00e5ff';
    ctx.fillText(
      'SCAN ID '+(target.trackId == null ? (target.id == null ? 'n/a' : target.id) : target.trackId)+' '+objectLabel(target),
      q.x+18,q.y+52
    );
  }

  if (scan.bodyAlignment && scan.bodyAlignment.target) {
    const target=scan.bodyAlignment.target;
    const q=polar(target.centerAngle,target.medianDistance,cx,cy,radius);
    ctx.strokeStyle='#ff9f1c'; ctx.lineWidth=4;
    ctx.beginPath(); ctx.arc(q.x,q.y,21,0,Math.PI*2); ctx.stroke();
    ctx.lineWidth=1;
    ctx.fillStyle='#ff9f1c';
    ctx.fillText(
      'BODY TURN '+(scan.bodyAlignment.moveAD.command || 'ALIGNED')
        +' '+objectLabel(target)+' '+target.nearestDistance.toFixed(1)+'m',
      q.x+18,q.y+42
    );
  }

  if (scan.lidarVisionFusion && scan.lidarVisionFusion.lidar) {
    const fusion=scan.lidarVisionFusion;
    const target=fusion.lidar;
    const distance=target.nearestDistance || 0;
    const q=polar(target.centerAngle || 0,distance,cx,cy,radius);
    ctx.strokeStyle=fusionColor(fusion.status); ctx.lineWidth=5;
    ctx.setLineDash(fusion.alignedForFusion ? [] : [9,6]);
    ctx.beginPath(); ctx.arc(q.x,q.y,30,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth=1;
    ctx.fillStyle=fusionColor(fusion.status);
    ctx.font='bold 14px Arial';
    const semantic=fusion.semanticClass || 'YOLO pending';
    ctx.fillText(
      'LiDAR?봜OLO '+fusion.status+' | '+semantic,
      q.x+18,q.y+62
    );
  }

  for (const marker of (scan.impactMarkers || [])) {
    drawImpactMarker(ctx, marker, cx, cy, radius);
  }

  ctx.fillStyle='#4cc9f0'; ctx.beginPath(); ctx.arc(cx,cy,7,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fff'; ctx.font='13px Arial';
  ctx.fillText('Contour: green | Hollow ~=provisional | Solid=confirmed | green label=YOLO mapped | red X=impact',18,h-20);
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
  const w = canvas.width, h = canvas.height, margin = 42;
  ctx.clearRect(0,0,w,h);
  ctx.font='12px Arial';

  // 寃⑹옄
  ctx.strokeStyle='#444'; ctx.fillStyle='#bbb';
  for (const a of [-60,-30,0,30,60]) {
    const p=frontXY(a,0,w,h,margin); ctx.beginPath(); ctx.moveTo(p.x,margin); ctx.lineTo(p.x,h-margin); ctx.stroke(); ctx.fillText((a>0?'+':'')+a+'째',p.x-12,h-margin+18);
  }
  for (const v of [-22.5,-10,0,10,22.5]) {
    const p=frontXY(0,v,w,h,margin); ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke(); ctx.fillText((v>0?'+':'')+v+'째',5,p.y+4);
  }
  ctx.fillStyle='#eee'; ctx.fillText('left', margin, 18); ctx.fillText('body-forward 0째', w/2-42, 18); ctx.fillText('right', w-margin-28, 18);
  ctx.fillText('up (- vertical angle)', 8, margin-10); ctx.fillText('down (+ vertical angle)', 8, h-10);

  // Front-view point cloud.
  for (const p of (scan.frontViewPoints || [])) {
    const q=frontXY(p.angle,p.verticalAngle,w,h,margin);
    ctx.fillStyle=distanceColor(p.distance);
    const size=p.distance <= 20 ? 3.0 : (p.distance <= 50 ? 2.4 : 1.8);
    ctx.beginPath(); ctx.arc(q.x,q.y,size,0,Math.PI*2); ctx.fill();
  }

  // ?덉젙?붾맂 ?꾪뿕 ?뱁꽣瑜??꾨갑 酉??섎떒 ?좊줈 洹몃┛??
  for (const s of (scan.terrainSectors || [])) {
    const left=frontXY(s.centerAngle-4.5,22.5,w,h,margin).x;
    const right=frontXY(s.centerAngle+4.5,22.5,w,h,margin).x;
    ctx.fillStyle=terrainColor(s.state);
    ctx.fillRect(left,h-margin-10,Math.max(2,right-left),8);
  }

  // ?뺤젙 媛앹껜 ?꾨낫瑜??섏쭅 以묒븰??洹몃┛??
  // ?댄썑 YOLO 寃고빀?쇰줈 ?щ엺/?꾩감 諛붿슫??諛뺤뒪濡??泥댄븷 ???덈떎.
  ctx.font='bold 13px Arial';
  for (const obj of (scan.confirmedObjects || [])) {
    if (Math.abs(obj.centerAngle) > FRONT_HORIZONTAL_LIMIT) continue;
    const q=frontXY(obj.centerAngle,0,w,h,margin);
    ctx.strokeStyle=objectColor(obj.geometryClass); ctx.lineWidth=2;
    ctx.strokeRect(q.x-18,q.y-22,36,44); ctx.lineWidth=1;
    ctx.fillStyle=obj.recognizedClass ? '#00ff88' : '#fff';
    ctx.fillText(objectLabel(obj),q.x+22,q.y-8);
    ctx.fillText(obj.nearestDistance.toFixed(1)+'m',q.x+22,q.y+10);
  }
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
function drawVerticalProfile(scan) {
  const ctx = profileCtx, canvas = profileCanvas;
  const w = canvas.width, h = canvas.height, margin = 54;
  const profile = scan.frontVerticalProfile || {};
  const sensorHeight = profile.sensorHeightAboveLocalGround == null ? 1.0 : profile.sensorHeightAboveLocalGround;
  const origin = profileXY(0, sensorHeight, w, h, margin);

  ctx.clearRect(0,0,w,h);
  ctx.font='12px Arial';
  ctx.strokeStyle='#444'; ctx.fillStyle='#bbb';

  for (const d of [0,20,40,60,80,100,120]) {
    const p=profileXY(d,0,w,h,margin);
    ctx.beginPath(); ctx.moveTo(p.x,margin); ctx.lineTo(p.x,h-margin); ctx.stroke();
    ctx.fillText(d+'m',p.x-10,h-margin+18);
  }
  for (const y of [-10,-5,0,5,10]) {
    const p=profileXY(0,y,w,h,margin);
    ctx.beginPath(); ctx.moveTo(margin,p.y); ctx.lineTo(w-margin,p.y); ctx.stroke();
    ctx.fillText((y>0?'+':'')+y+'m',8,p.y+4);
  }

  // 濡쒖뺄 吏硫?湲곗???
  const groundLeft=profileXY(0,0,w,h,margin);
  const groundRight=profileXY(PROFILE_MAX_DISTANCE,0,w,h,margin);
  ctx.strokeStyle='#00cfd5'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(groundLeft.x,groundLeft.y); ctx.lineTo(groundRight.x,groundRight.y); ctx.stroke();
  ctx.lineWidth=1;

  // LiDAR ?먯젏?먯꽌 ?좏깮??媛??섏쭅 梨꾨꼸 愿묒꽑??洹몃┛??
  for (const ray of (profile.rays || [])) {
    const range = Math.min(PROFILE_MAX_DISTANCE, ray.horizontalRange || ray.distance || PROFILE_MAX_DISTANCE);
    let endpointHeight;

    if (ray.isDetected && ray.heightAboveLocalGround !== null) {
      endpointHeight = ray.heightAboveLocalGround;
    } else {
      // ?묒닔 ?섏쭅媛곸? ?꾨옒履쎌쓣 媛由ы궓??
      endpointHeight = sensorHeight - range * Math.tan(ray.verticalAngle * Math.PI / 180.0);
    }

    const end=profileXY(range,endpointHeight,w,h,margin);

    ctx.save();
    ctx.globalAlpha=ray.isDetected ? 0.30 : 0.22;
    ctx.strokeStyle=ray.isDetected ? '#44d62c' : '#888';
    if (!ray.isDetected) ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(origin.x,origin.y); ctx.lineTo(end.x,end.y); ctx.stroke();
    ctx.restore();

    if (ray.isDetected) {
      ctx.fillStyle=distanceColor(ray.distance);
      ctx.beginPath(); ctx.arc(end.x,end.y,3.0,0,Math.PI*2); ctx.fill();
    }
  }

  // ?섎떒 ?멸낸??湲곕컲 濡쒖뺄 吏???꾨줈?뚯씪??洹몃┛??
  const terrain=(profile.groundProfilePoints || []).filter(p => p.heightAboveLocalGround !== null);
  if (terrain.length) {
    ctx.strokeStyle='#00ffff'; ctx.lineWidth=3;
    ctx.beginPath();
    terrain.forEach((p,i) => {
      const q=profileXY(p.horizontalRange,p.heightAboveLocalGround,w,h,margin);
      if (i===0) ctx.moveTo(q.x,q.y); else ctx.lineTo(q.x,q.y);
    });
    ctx.stroke(); ctx.lineWidth=1;

    for (const p of terrain) {
      const q=profileXY(p.horizontalRange,p.heightAboveLocalGround,w,h,margin);
      ctx.fillStyle='#00ffff'; ctx.beginPath(); ctx.arc(q.x,q.y,3,0,Math.PI*2); ctx.fill();
    }
  }

  ctx.fillStyle='#4cc9f0'; ctx.beginPath(); ctx.arc(origin.x,origin.y,7,0,Math.PI*2); ctx.fill();

  ctx.fillStyle='#eee';
  ctx.fillText('LiDAR origin',origin.x+10,origin.y-8);
  ctx.fillText('forward horizontal range',w/2-55,h-10);
  ctx.fillText('height above local ground',8,18);
  ctx.fillText(
    'selected azimuth: '+(profile.selectedAngle == null ? 'n/a' : profile.selectedAngle.toFixed(1)+'째')
      +' | channels: '+(profile.channelCount || 0)
      +' | hits: '+(profile.hitCount || 0)
      +' | misses: '+(profile.missCount || 0),
    margin,34
  );
  ctx.fillText(
    'approx ground slope: '+(profile.approxGroundSlopeDeg == null ? 'n/a' : profile.approxGroundSlopeDeg.toFixed(1)+'째')
      +' | max uphill: '+(profile.maxUpSlopeDeg == null ? 'n/a' : profile.maxUpSlopeDeg.toFixed(1)+'째')
      +' | max downhill: '+(profile.maxDownSlopeDeg == null ? 'n/a' : profile.maxDownSlopeDeg.toFixed(1)+'째'),
    margin,50
  );
}

function draw(scan) {
  drawTop(scan);
  drawVerticalProfile(scan);
  updateFireControl(scan.fireControl);
  updateAimZero(scan);
  const alignment = scan.bodyAlignment || {};
  const alignmentMoveAD = alignment.moveAD || { command: '', weight: 0.0 };

  const provisionalLines=(scan.provisionalObjects||[]).slice(0,15).map(o =>
    '~'+o.candidateLabel+'? | '+o.candidateMeaning
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | width='+o.estimatedWidth.toFixed(2)+'m'
      +' | aboveGround='+(o.medianHeightAboveLocalGround == null ? 'n/a' : o.medianHeightAboveLocalGround.toFixed(2)+'m')
  );
  const trackLines=(scan.confirmedObjects||[]).map(o =>
    objectLabel(o)+' | '+o.candidateMeaning
      +' | yolo='+(o.recognizedClass || 'pending')
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | width='+o.estimatedWidth.toFixed(2)+'m'
      +' | persist='+o.persistenceHits+'/'+o.historySize
  );
  const priorityLines=(scan.fusionPriorityQueue||[]).slice(0,12).map(o =>
    '#'+o.fusionPriorityRank
      +' | '+objectLabel(o)
      +' | tier='+o.fusionPriorityTier
      +' | stage='+o.trackingStage
      +' | dist='+o.nearestDistance.toFixed(1)+'m'
      +' | angle='+o.centerAngle.toFixed(1)+'deg'
      +' | turretTarget='+o.recommendedTurretBodyRelativeAngleDeg.toFixed(1)+'deg'
  );
  const terrainLines=(scan.terrainSectors||[]).map(s =>
    s.centerAngle.toFixed(0)+'deg | state='+s.state+' | raw='+s.rawState+' | type='+s.hazardType+' | reason='+s.reason+' | range='+s.hazardBoundaryRange.toFixed(1)+'m | votes(B/C/P)='+s.blockedVoteCount+'/'+s.cautionVoteCount+'/'+s.passableVoteCount
  );
  const fusion=scan.lidarVisionFusion || {};
  const fusionLidar=fusion.lidar || {};
  const fusionVision=fusion.vision || {};
  const impactLines=(scan.impactMarkers||[]).slice().reverse().map(m =>
    '#'+m.id
      +' | dist='+(m.distance == null ? 'n/a' : Number(m.distance).toFixed(1)+'m')
      +' | angle='+(m.angle == null ? 'n/a' : Number(m.angle).toFixed(1)+'deg')
      +' | pos='+(m.position && m.position.x != null ? 'x '+Number(m.position.x).toFixed(1)+' / z '+Number(m.position.z).toFixed(1) : 'n/a')
      +' | turret='+(m.turret && m.turret.pitch != null ? 'pitch '+Number(m.turret.pitch).toFixed(2)+' / yaw '+Number(m.turret.yaw || 0).toFixed(2) : 'n/a')
      +' | RF='+((m.lastAction && m.lastAction.turretRF && m.lastAction.turretRF.command) ? m.lastAction.turretRF.command+' '+Number(m.lastAction.turretRF.weight || 0).toFixed(2) : 'STOP')
      +' | QE='+((m.lastAction && m.lastAction.turretQE && m.lastAction.turretQE.command) ? m.lastAction.turretQE.command+' '+Number(m.lastAction.turretQE.weight || 0).toFixed(2) : 'STOP')
      +' | zeroY='+(m.aimZero ? Number(m.aimZero.offsetY || 0).toFixed(3) : '0.000')
      +' | '+(m.objectName || 'impact')
  );
  const fusionLine=fusion.status
    ? 'LiDAR?봜OLO mapping: '+fusion.status
      +' | aligned='+fusion.alignedForFusion
      +' | lidar='+(fusionLidar.candidateLabel || '?')+'?'
      +' track='+(fusionLidar.trackId == null ? 'n/a' : fusionLidar.trackId)
      +' dist='+(fusionLidar.nearestDistance == null ? 'n/a' : Number(fusionLidar.nearestDistance).toFixed(1)+'m')
      +' angle='+(fusionLidar.bodyRelativeAngleErrorDeg == null ? 'n/a' : Number(fusionLidar.bodyRelativeAngleErrorDeg).toFixed(1)+'deg')
      +' | yolo='+(fusion.semanticClass || fusionVision.className || 'pending')
      +' conf='+(fusionVision.confidence == null ? 'n/a' : Number(fusionVision.confidence).toFixed(2))
      +' | attackTarget='+fusion.isAttackTarget
    : 'LiDAR?봜OLO mapping: none';
  statusText.textContent=[
    'simulationTime: '+scan.simulationTime,
    'rawRayCount: '+scan.rawRayCount,
    'rawDetectedPointCount: '+scan.rawDetectedPointCount,
    'frontPointCloudView: disabled for lightweight real-time monitoring',
    'frontVerticalProfile: selectedAngle='+(scan.frontVerticalProfile.selectedAngle == null ? 'n/a' : scan.frontVerticalProfile.selectedAngle+'deg')      +' | channels='+scan.frontVerticalProfile.channelCount      +' | hits='+scan.frontVerticalProfile.hitCount      +' | misses='+scan.frontVerticalProfile.missCount,
    'frontVerticalSlope: approx='+(scan.frontVerticalProfile.approxGroundSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.approxGroundSlopeDeg+'deg')      +' | maxUp='+(scan.frontVerticalProfile.maxUpSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.maxUpSlopeDeg+'deg')      +' | maxDown='+(scan.frontVerticalProfile.maxDownSlopeDeg == null ? 'n/a' : scan.frontVerticalProfile.maxDownSlopeDeg+'deg')      +' | maxDrop='+(scan.frontVerticalProfile.maxDrop == null ? 'n/a' : scan.frontVerticalProfile.maxDrop+'m'),
    'localGroundGridCellCount: '+scan.localGroundGridCellCount,
    'terrainDecision: '+scan.terrainDecision.state+' | action='+scan.terrainDecision.recommendedAction+' | reason='+scan.terrainDecision.reason,
    'deadEndDetected: '+scan.terrainDecision.deadEndDetected+' | blockedRatio='+scan.terrainDecision.deadEndBlockedRatio,
    'rawObjectCount: '+scan.rawObjectCount
      +' | suppressedTerrainObjects: '+scan.suppressedTerrainObjectCount
      +' | provisionalObjects: '+scan.provisionalObjectCount
      +' | trackedObjects: '+scan.trackedObjectCount
      +' | confirmedTracks: '+scan.confirmedObjectCount,
    '',
    'Automatic body alignment: '
      +(alignment.enabled ? 'enabled' : 'disabled')
      +' | reason='+(alignment.reason || 'waiting_for_target')
      +' | lockedTrackId='+(alignment.lockedTrackId == null ? 'none' : alignment.lockedTrackId)
      +' | moveAD='+(alignmentMoveAD.command || 'STOP')
      +' | weight='+alignmentMoveAD.weight
      +' | nextId>='+(alignment.nextTrackIdMin == null ? 'n/a' : alignment.nextTrackIdMin)
      +' | rejected='+(alignment.rejectedTrackIds ? alignment.rejectedTrackIds.join(',') : 'none')
      +' | lastJudge='+(alignment.lastJudgement || 'none'),
    'Body alignment target: '
      +(alignment.target
        ? objectLabel(alignment.target)
          +' | dist='+alignment.target.nearestDistance.toFixed(1)+'m'
          +' | angleError='+alignment.target.bodyRelativeAngleErrorDeg.toFixed(1)+'deg'
        : 'none'),
    'Sequential scan target: '
      +(scan.scanTarget
        ? 'ID '+(scan.scanTarget.trackId == null ? scan.scanTarget.id : scan.scanTarget.trackId)+' | '+objectLabel(scan.scanTarget)
          +' | dist='+scan.scanTarget.nearestDistance.toFixed(1)+'m'
          +' | angle='+scan.scanTarget.centerAngle.toFixed(1)+'deg'
        : 'none'),
    fusionLine,
    'Vision target: '
      +(scan.visionTarget
        ? scan.visionTarget.className+' | conf='+Number(scan.visionTarget.confidence).toFixed(2)
          +' | errorX='+Number(scan.visionTarget.errorX).toFixed(3)
          +' | errorY='+Number(scan.visionTarget.errorY).toFixed(3)
        : 'none'),
    'Fire control: '
      +(scan.fireControl
        ? 'ready='+scan.fireControl.ready
          +' | approved='+scan.fireControl.approved
          +' | nextFire='+scan.fireControl.fireOnNextAction
          +' | reason='+scan.fireControl.reason
        : 'none'),
    'Ballistic pitch: '
      +(scan.fireControl && scan.fireControl.ballisticPitch
        ? 'ready='+scan.fireControl.ballisticPitch.ready
          +' | dist='+(scan.fireControl.ballisticPitch.distance == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.distance).toFixed(1)+'m')
          +' | current='+(scan.fireControl.ballisticPitch.currentPitch == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.currentPitch).toFixed(2))
          +' | target='+(scan.fireControl.ballisticPitch.targetPitch == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.targetPitch).toFixed(2))
          +' | error='+(scan.fireControl.ballisticPitch.pitchError == null ? 'n/a' : Number(scan.fireControl.ballisticPitch.pitchError).toFixed(2))
          +' | RF='+(scan.fireControl.ballisticPitch.turretRF.command || 'STOP')
          +' '+Number(scan.fireControl.ballisticPitch.turretRF.weight || 0).toFixed(2)
          +' | reason='+scan.fireControl.ballisticPitch.reason
        : 'none'),
    'Aim zero: offsetY='+(scan.aimZero ? Number(scan.aimZero.offsetY || 0).toFixed(3) : '0.000')
      +' | rawY='+(scan.fireControl && scan.fireControl.aimError.rawY == null ? 'n/a' : (scan.fireControl ? Number(scan.fireControl.aimError.rawY).toFixed(3) : 'n/a'))
      +' | adjustedY='+(scan.fireControl && scan.fireControl.aimError.adjustedY == null ? 'n/a' : (scan.fireControl ? Number(scan.fireControl.aimError.adjustedY).toFixed(3) : 'n/a')),
    '',
    'YOLO fusion priority queue (BK first, <=50m next):', ...(priorityLines.length?priorityLines:['none']),
    '',
    'Primary fusion target: '
      +(scan.primaryFusionTarget
        ? objectLabel(scan.primaryFusionTarget)+' | dist='+scan.primaryFusionTarget.nearestDistance.toFixed(1)+'m | angle='+scan.primaryFusionTarget.centerAngle.toFixed(1)+'deg | tier='+scan.primaryFusionTarget.fusionPriorityTier
        : 'none'),
    '',
    'Bullet impacts (red X):', ...(impactLines.length?impactLines:['none']),
    '',
    'Provisional LiDAR candidates (~ hollow marker):', ...(provisionalLines.length?provisionalLines:['none']),
    '',
    'Confirmed LiDAR object tracks with YOLO class mapping (solid marker):', ...(trackLines.length?trackLines:['none']),
    '',
    'Terrain sectors:', ...(terrainLines.length?terrainLines:['none']),
    '',
    'Lightweight monitor API: /lidar_monitor_status',
    'Priority queue API: /priority_status',
    'Body alignment API: /body_align_status',
    'Enable: /body_align_enable?enabled=true | Disable: /body_align_enable?enabled=false',
    'Full debug JSON API: /lidar_status'
  ].join('\n');
}
async function refresh(){try{const r=await fetch('/lidar_monitor_status',{cache:'no-store'}); draw(await r.json());}catch(e){statusText.textContent=String(e);}}
async function confirmFire(){
  fireButton.disabled = true;
  fireText.textContent = 'Approving fire...';
  try {
    const r = await fetch('/fire_confirm', {method:'POST'});
    const payload = await r.json();
    updateFireControl(payload.fireControl);
  } catch (e) {
    fireText.textContent = 'Fire approval failed: '+String(e);
  }
}
fireButton.addEventListener('click', confirmFire);
async function postAimZero(action){
  try {
    const r = await fetch('/aim_zero', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})
    });
    const payload = await r.json();
    zeroText.textContent = payload.aimZero
      ? 'Zero Y '+Number(payload.aimZero.offsetY || 0).toFixed(3)
      : 'Zero update failed';
    await refresh();
  } catch (e) {
    zeroText.textContent = 'Zero update failed: '+String(e);
  }
}
document.getElementById('zeroDownButton').addEventListener('click', () => postAimZero('down'));
document.getElementById('zeroUpButton').addEventListener('click', () => postAimZero('up'));
document.getElementById('zeroCaptureButton').addEventListener('click', () => postAimZero('capture'));
document.getElementById('zeroResetButton').addEventListener('click', () => postAimZero('reset'));
setInterval(refresh,200); refresh();
</script>
</body>
</html>
"""


@app.route("/action_debug", methods=["GET"])
def action_debug():
    with state_lock:
        return jsonify(deepcopy(action_debug_state))


@app.route("/get_action", methods=["POST"])
def get_action():
    """
    Tank Challenge ?쒖뼱 ?붾뱶?ъ씤??

    moveWS:
      李⑥껜 ?뺣젹 以묒뿉??STOP.

    moveAD:
      A = 李⑥껜 醫뚰쉶??      D = 李⑥껜 ?고쉶??
    ?ы깙 ?붾뒗 ?쒖쟻 洹쇱쿂?먯꽌 吏??蹂댁긽 ?덉젙?붽린瑜??ъ슜?쒕떎.
    ?먮룞 諛쒖궗??鍮꾪솢?깊솕?섏뼱 ?덈떎. FIRE 踰꾪듉? 諛쒖궗瑜?臾댁옣?섍퀬,
    ?섎룞 ?뱀씤???쒖꽦?붾맂 ?숈븞 ?쒖쟻??異⑸텇??議곗? 媛?ν븷 ???ㅼ젣 諛쒖궗媛 ?섍컙??
    """
    request_body = request.get_json(silent=True) or {}

    with state_lock:
        if isinstance(request_body, dict):
            latest_raw_info.update(deepcopy(request_body))
        alignment = deepcopy(latest_state.get("bodyAlignment", {}))
        decision = deepcopy(latest_state.get("terrainDecision", {}))
        sectors = deepcopy(latest_state.get("terrainSectors", []))
        lidar_target = current_lidar_fusion_target(latest_state)
        vision_target = active_vision_target() if USE_YOLO_FOR_AIM else None

    turret_qe = {"command": "", "weight": 0.0}
    turret_rf = {"command": "", "weight": 0.0}

    if vision_target:
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad, turret_qe, turret_rf = vision_aim_commands(vision_target)
    elif alignment.get("enabled") and alignment.get("target"):
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = deepcopy(alignment.get("moveAD", {"command": "", "weight": 0.0}))
    elif not AUTO_DRIVE_ENABLED:
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = {"command": "", "weight": 0.0}
    elif decision.get("state") == "passable":
        move_ws = {"command": "W", "weight": 0.55}
        move_ad = {"command": "", "weight": 0.0}
    elif decision.get("state") == "caution":
        move_ws = {"command": "W", "weight": 0.25}
        move_ad = {"command": "", "weight": 0.0}
    else:
        left = [s for s in sectors if s["centerAngle"] < -15 and s["state"] == "passable"]
        right = [s for s in sectors if s["centerAngle"] > 15 and s["state"] == "passable"]
        move_ws = {"command": "STOP", "weight": 1.0}
        move_ad = (
            {"command": "A", "weight": 0.35}
            if left and not right
            else (
                {"command": "D", "weight": 0.35}
                if right
                else {"command": "", "weight": 0.0}
            )
        )

    # ?쒖쟻 洹쇱쿂?먯꽌??李⑥껜瑜?怨좎젙?섍퀬 ?ы깙留??뺣? 議곗??섍쾶 ?쒕떎.
    # ?쒖쟻 媛源뚯씠?먯꽌 李⑥껜瑜??뚮━硫???API 吏?곗씠 利앺룺?섏뼱
    # 議곗??좎씠 ?쒖쟻??諛섎났?댁꽌 吏?섏튂寃??쒕떎.
    if should_lock_body_near_lidar_target(lidar_target):
        move_ad = {"command": "", "weight": 0.0}

    if not USE_YOLO_FOR_AIM:
        lidar_turret_qe = world_target_turret_yaw_control(lidar_target)
        if lidar_turret_qe.get("command"):
            turret_qe = lidar_turret_qe

    pitch_status = ballistic_pitch_control(lidar_target)
    if pitch_status.get("enabled") and pitch_status.get("turretRF", {}).get("command"):
        turret_rf = deepcopy(pitch_status["turretRF"])

    should_fire = False
    with state_lock:
        fire_status_payload = fire_readiness_status()
        now_for_fire = monotonic()
        last_fired_iso = fire_control_state.get("lastFiredAt")
        # 荑⑤떎?댁? private helper key??monotonic ?쒓컙?쇰줈 異붿쟻?쒕떎.
        last_auto_fire_at = float(fire_control_state.get("lastAutoFireAt", 0.0) or 0.0)
        auto_fire_ready = (
            AUTO_FIRE_WHEN_STABLE
            and fire_status_payload.get("ready")
            and aim_stabilizer_ready()
            and now_for_fire - last_auto_fire_at >= AUTO_FIRE_COOLDOWN_SECONDS
        )
        manual_fire_ready = bool(fire_status_payload.get("fireOnNextAction"))

        if manual_fire_ready or auto_fire_ready:
            should_fire = True
            fire_control_state["approvedUntil"] = 0.0
            fire_control_state["lastAutoFireAt"] = now_for_fire
            fire_control_state["lastFiredAt"] = datetime.now().isoformat(timespec="milliseconds")
            fire_control_state["fireCount"] = int(fire_control_state.get("fireCount", 0) or 0) + 1
            mark_target_as_eliminated_after_fire(lidar_target)
            fire_status_payload = fire_readiness_status()
            fire_status_payload["autoFireTriggered"] = bool(auto_fire_ready)
            fire_status_payload["manualFireTriggered"] = bool(manual_fire_ready)

    response_body = {
        "moveWS": move_ws,
        "moveAD": move_ad,
        "turretQE": turret_qe,
        "turretRF": turret_rf,
        "fire": should_fire,
    }

    with state_lock:
        action_debug_state["getActionRequestCount"] += 1
        action_debug_state["lastRequestBody"] = deepcopy(request_body)
        action_debug_state["lastResponse"] = deepcopy(response_body)
        action_debug_state["lastFireStatus"] = deepcopy(fire_status_payload)
        action_debug_state["lastAimStabilizer"] = deepcopy(aim_stabilizer_state)
        action_debug_state["lastWorldTargetLock"] = deepcopy(world_target_lock_state)
        action_debug_state["lastRequestedAt"] = datetime.now().isoformat(timespec="milliseconds")

    print(
        "GET_ACTION | "
        f"moveWS={response_body['moveWS']} | "
        f"moveAD={response_body['moveAD']} | "
        f"turretQE={response_body['turretQE']} | "
        f"turretRF={response_body['turretRF']} | "
        f"fire={response_body['fire']} | "
        f"vision={vision_target.get('className') if vision_target else None} | "
        f"target={alignment.get('target', {}).get('candidateLabel') if alignment.get('target') else None} | "
        f"reason={alignment.get('reason')} | "
        f"aimStab={aim_stabilizer_state.get('reason')} "
        f"settled={aim_stabilizer_state.get('settled')}"
    )

    delay_seconds = 0.0 if should_fire else precision_aim_response_sleep_seconds(lidar_target)
    if delay_seconds > 0.0:
        sleep(delay_seconds)

    return jsonify(response_body)


@app.route("/update_bullet", methods=["POST"])
def update_bullet():
    data = request.get_json(force=True, silent=True) or {}
    marker = record_impact(data)
    return jsonify({"status": "OK", "message": "Bullet impact data received", "impact": marker})


@app.route("/update_obstacle", methods=["POST"])
def update_obstacle():
    data = request.get_json(force=True, silent=True) or {}
    if nested_position(data) or any(key in data for key in ("angle", "distance", "range")):
        marker = record_impact(data)
        return jsonify({"status": "OK", "message": "Impact-like obstacle data received", "impact": marker})
    return jsonify({"status": "success", "message": "Obstacle data received"})


@app.route("/reset_state", methods=["POST", "GET"])
def reset_state():
    with state_lock:
        reset_runtime_state()
    return jsonify({"status": "success", "message": "v8 temporal state cleared"})


@app.route("/init", methods=["GET"])
def init():
    with state_lock:
        reset_runtime_state()

    return jsonify({
        "startMode": "start", "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True, "detectMode": True, "logMode": True,
        "stereoCameraMode": False, "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000, "destoryObstaclesOnHit": True,
    })


@app.route("/start", methods=["GET"])
def start():
    return jsonify({"control": ""})


if __name__ == "__main__":
    print("Open v8.7 automatic body-alignment view: http://127.0.0.1:5000/lidar_view")
    print("Lightweight monitor JSON: http://127.0.0.1:5000/lidar_monitor_status")
    print("Full debug JSON: http://127.0.0.1:5000/lidar_status")
    print("YOLO fusion priority queue JSON: http://127.0.0.1:5000/priority_status")
    print("Automatic body alignment JSON: http://127.0.0.1:5000/body_align_status")
    print("GET_ACTION debug JSON: http://127.0.0.1:5000/action_debug")
    print("Disable body alignment: http://127.0.0.1:5000/body_align_enable?enabled=false")
    print("Front vertical profile JSON: http://127.0.0.1:5000/front_vertical_profile")
    print("Reset temporal state: http://127.0.0.1:5000/reset_state")
    app.run(host="0.0.0.0", port=5000, threaded=True)
