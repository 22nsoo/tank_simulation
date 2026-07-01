import math
import heapq
import json

# ============================================================
# A* 전역경로 계획
# ============================================================
GRID_RESOLUTION = 2.0
ROBOT_RADIUS    = 4.0
MAP_X_MIN, MAP_X_MAX = 40.0, 310.0
MAP_Z_MIN, MAP_Z_MAX = 10.0, 310.0

START_X, START_Z = 60.0, 27.23
GOAL_X,  GOAL_Z  = 142.47, 214.1


def world_to_grid(x, z):
    gx = int((x - MAP_X_MIN) / GRID_RESOLUTION)
    gz = int((z - MAP_Z_MIN) / GRID_RESOLUTION)
    return gx, gz


def grid_to_world(gx, gz):
    x = gx * GRID_RESOLUTION + MAP_X_MIN
    z = gz * GRID_RESOLUTION + MAP_Z_MIN
    return x, z


def build_obstacle_grid(obstacles, margin):
    cols = int((MAP_X_MAX - MAP_X_MIN) / GRID_RESOLUTION) + 1
    rows = int((MAP_Z_MAX - MAP_Z_MIN) / GRID_RESOLUTION) + 1
    grid = [[False] * rows for _ in range(cols)]
    for obs in obstacles:
        gx0, gz0 = world_to_grid(obs['x_min'] - margin, obs['z_min'] - margin)
        gx1, gz1 = world_to_grid(obs['x_max'] + margin, obs['z_max'] + margin)
        for gx in range(max(0, gx0), min(cols, gx1 + 1)):
            for gz in range(max(0, gz0), min(rows, gz1 + 1)):
                grid[gx][gz] = True
    return grid, cols, rows


def astar(grid, cols, rows, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        for dx, dz in dirs:
            nb = (cur[0]+dx, cur[1]+dz)
            if not (0 <= nb[0] < cols and 0 <= nb[1] < rows):
                continue
            if grid[nb[0]][nb[1]]:
                continue
            tg = g_score[cur] + math.hypot(dx, dz)
            if tg < g_score.get(nb, float('inf')):
                came_from[nb] = cur
                g_score[nb] = tg
                h = math.hypot(nb[0]-goal[0], nb[1]-goal[1])
                heapq.heappush(open_set, (tg + h, nb))
    return None


def simplify_path(path, grid, min_dist=4):
    """너무 가까운 점만 제거 (직선 가시성 대신 거리 기반)"""
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    for i in range(1, len(path)):
        last = simplified[-1]
        dist = math.hypot(path[i][0]-last[0], path[i][1]-last[1])
        if dist >= min_dist or i == len(path)-1:
            simplified.append(path[i])
    return simplified


def plan_path(obstacles, sx=START_X, sz=START_Z, gx=GOAL_X, gz=GOAL_Z, waypoint_file='waypoints.json'):
    """
    A*로 경로 계획 후 waypoints.json 저장.
    반환: waypoints 리스트 or None
    """
    grid, cols, rows = build_obstacle_grid(obstacles, ROBOT_RADIUS)
    start = world_to_grid(sx, sz)
    goal  = world_to_grid(gx, gz)

    # 목표가 장애물 안이면 마진 줄여서 재시도
    if grid[min(cols-1, goal[0])][min(rows-1, goal[1])]:
        print("⚠️  목표 격자가 장애물 안 - 마진 1m로 재시도")
        grid, cols, rows = build_obstacle_grid(obstacles, 1.0)

    path = astar(grid, cols, rows, start, goal)
    if path is None:
        print("❌ A* 경로 탐색 실패!")
        return None

    path = simplify_path(path, grid)

    waypoints = []
    for g in path[1:]:   # 출발점 제외
        wx, wz = grid_to_world(g[0], g[1])
        waypoints.append({'x': round(wx, 2), 'y': 9.3, 'z': round(wz, 2)})

    with open(waypoint_file, 'w', encoding='utf-8') as f:
        json.dump({'waypoints': waypoints}, f, indent=2)

    print(f"✅ A* 완료: {len(waypoints)}개 WP → {waypoint_file} 저장")
    return waypoints
