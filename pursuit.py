import math
import json

# ============================================================
# Pure Pursuit + PD 제어 설정
# ============================================================
WAYPOINT_FILE     = 'waypoints.json'
ARRIVAL_RADIUS    = 5.0
HEADING_THRESHOLD = 5.0
MOVE_WEIGHT       = 0.7
LOOKAHEAD_MIN     = 5.0
LOOKAHEAD_MAX     = 20.0

KP = 0.015
KD = 0.020

# 상태
waypoints        = []   # A*로 채워짐 - 초기엔 비어있음
current_wp_index = 0
wp_min_dist      = 9999.0
wp_miss_count    = 0
pid_prev_error   = 0.0


def load_waypoints(filepath=WAYPOINT_FILE):
    """선택적 로드 - 없어도 괜찮음"""
    global waypoints
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        waypoints = data.get('waypoints', [])
        print(f"✅ Waypoints loaded ({len(waypoints)}개)")
    except Exception:
        waypoints = []
        print("⚠️  waypoints.json 없음 - 대시보드에서 경로 계획 후 시작하세요")


def reset():
    global current_wp_index, wp_min_dist, wp_miss_count, pid_prev_error
    current_wp_index = 0
    wp_min_dist      = 9999.0
    wp_miss_count    = 0
    pid_prev_error   = 0.0


# ============================================================
# 헬퍼 함수
# ============================================================
def get_angle_to_target(pos_x, pos_z, tx, tz):
    dx = tx - pos_x
    dz = tz - pos_z
    return math.degrees(math.atan2(dx, dz)) % 360.0


def calc_heading_error(current, target):
    return (target - current + 180) % 360 - 180


def find_lookahead_target(pos_x, pos_z, wp_index, lookahead_dist):
    for i in range(wp_index, len(waypoints)):
        wp = waypoints[i]
        if math.hypot(wp['x'] - pos_x, wp['z'] - pos_z) >= lookahead_dist:
            return i, wp
    last = len(waypoints) - 1
    return last, waypoints[last]


# ============================================================
# Pure Pursuit 메인
# ============================================================
def pure_pursuit(pos_x, pos_z, heading_deg, wp_index):
    global current_wp_index, wp_min_dist, wp_miss_count, pid_prev_error

    # 경로 없으면 대기
    if not waypoints:
        print("⏸️  경로 없음 - 대시보드에서 목표를 설정하고 경로 계획하세요")
        return "STOP", "", 0, 0.0, 0.0

    def skip_wp(idx, reason):
        global current_wp_index, wp_min_dist, wp_miss_count
        print(f"⏭️  WP[{idx}] 스킵! ({reason}) → 다음 WP로")
        current_wp_index = idx + 1
        wp_min_dist      = 9999.0
        wp_miss_count    = 0
        return idx + 1

    # 도달 체크
    while wp_index < len(waypoints):
        wp   = waypoints[wp_index]
        dist = math.hypot(wp['x'] - pos_x, wp['z'] - pos_z)
        if dist < ARRIVAL_RADIUS:
            print(f"🏁 WP[{wp_index}] 도달! (x={wp['x']:.1f}, z={wp['z']:.1f}) → 다음 WP로")
            wp_index = skip_wp(wp_index, '도달')
        else:
            break

    if wp_index >= len(waypoints):
        print("🎯 === 모든 웨이포인트 완료! 정지 ===")
        return "STOP", "", wp_index, 1.0, 0.0

    cur_wp      = waypoints[wp_index]
    dist_to_wp  = math.hypot(cur_wp['x'] - pos_x, cur_wp['z'] - pos_z)

    # 최근접 거리 갱신
    if dist_to_wp < wp_min_dist:
        wp_min_dist   = dist_to_wp
        wp_miss_count = 0
    else:
        wp_miss_count += 1

    # 가변 Lookahead
    adaptive_lookahead = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, dist_to_wp * 0.5))
    _, target_wp   = find_lookahead_target(pos_x, pos_z, wp_index, adaptive_lookahead)
    target_heading = get_angle_to_target(pos_x, pos_z, target_wp['x'], target_wp['z'])
    error          = calc_heading_error(heading_deg, target_heading)

    # 거리 기반 속도 계수
    if dist_to_wp < 6.0:
        speed_factor = 0.5
    elif dist_to_wp < 10.0:
        speed_factor = 0.7
    else:
        speed_factor = 1.0

    print(f"📍 pos=({pos_x:.1f},{pos_z:.1f}) | WP[{wp_index}/{len(waypoints)-1}]=({cur_wp['x']:.1f},{cur_wp['z']:.1f}) | 거리:{dist_to_wp:.1f}m | lookahead:{adaptive_lookahead:.1f}m | 속도x{speed_factor:.1f}")
    print(f"🧭 heading:{heading_deg:.1f}° | 목표:{target_heading:.1f}° | 오차:{error:.1f}°")

    # PD 컨트롤러
    d_error      = error - pid_prev_error
    pid_prev_error = error
    pid_output   = KP * abs(error) + KD * abs(d_error)
    turn_weight  = min(1.0, max(0.0, pid_output))

    BASE_SPEED   = MOVE_WEIGHT
    abs_error    = abs(error)
    move_ws      = "W"

    if abs_error <= HEADING_THRESHOLD:
        move_ad     = ""
        move_weight = BASE_SPEED * speed_factor
        turn_weight = 0.0
    elif abs_error <= 45.0:
        move_ad     = "D" if error > 0 else "A"
        move_weight = BASE_SPEED * speed_factor
    elif abs_error <= 80.0:
        move_ad     = "D" if error > 0 else "A"
        move_weight = max(0.4, 0.6 * speed_factor)
    else:
        move_ad     = "D" if error > 0 else "A"
        move_weight = 0.3

    print(f"🎛️ PD | error:{error:.1f}° d_err:{d_error:.1f}° → turn:{turn_weight:.2f} move:{move_weight:.2f}")

    return move_ws, move_ad, wp_index, move_weight, turn_weight
