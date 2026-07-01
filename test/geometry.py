import math
from config import MAP_X_MIN, MAP_Z_MIN, GRID_SIZE, MAP_X_MAX, MAP_Z_MAX

def distance_2d(x1, z1, x2, z2):
    return math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def get_target_angle(pos_x, pos_z, target_x, target_z):
    angle = math.degrees(math.atan2(target_x - pos_x, target_z - pos_z))
    return angle + 360 if angle < 0 else angle


def world_to_grid(x, z):
    return int(round((x - MAP_X_MIN) / GRID_SIZE)), int(round((z - MAP_Z_MIN) / GRID_SIZE))


def grid_to_world(gx, gz):
    return gx * GRID_SIZE + MAP_X_MIN, gz * GRID_SIZE + MAP_Z_MIN


def is_inside_map(gx, gz):
    x, z = grid_to_world(gx, gz)
    return MAP_X_MIN <= x <= MAP_X_MAX and MAP_Z_MIN <= z <= MAP_Z_MAX