# -*- coding: utf-8 -*-
"""300x300 world-map parsing helpers for the RAG dashboard."""

from __future__ import annotations

import json
import os
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_MAP_FILE = "NewMap.map"
MAP_MIN_M = 0.0
MAP_MAX_M = 300.0

PREFAB_SIZE = {
    "Human002": 0.65,
    "Human003": 0.65,
    "Car001": 2.5,
    "Car002": 2.5,
    "Car003": 2.5,
    "Car004": 2.5,
    "Rock001": 3.0,
    "Rock002": 3.5,
    "Tree001": 1.5,
    "Tree003": 1.5,
    "House002": 6.0,
    "Wall002": 4.0,
    "Tank001": 3.0,
}
DEFAULT_SIZE = 2.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prefab_type(prefab_name: str) -> str:
    parts = str(prefab_name or "").split("_")
    if len(parts) <= 1:
        return str(prefab_name or "object")
    return "_".join(parts[:-1])


def resolve_map_path(filename: str | None = None) -> str:
    requested = filename or DEFAULT_MAP_FILE
    safe_name = os.path.basename(requested)
    candidate = os.path.abspath(os.path.join(ROOT, safe_name))
    if not candidate.startswith(ROOT):
        raise ValueError("map path escapes project root")
    return candidate


def list_map_files() -> list[str]:
    try:
        names = [name for name in os.listdir(ROOT) if name.lower().endswith(".map")]
    except OSError:
        return []
    return sorted(names)


def parse_world_map(filename: str | None = None) -> dict:
    path = resolve_map_path(filename)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    obstacles = []
    for obj in data.get("obstacles", []) or []:
        prefab_name = str(obj.get("prefabName") or "object")
        kind = prefab_type(prefab_name)
        position = obj.get("position") if isinstance(obj.get("position"), dict) else {}
        x = _safe_float(position.get("x"))
        y = _safe_float(position.get("y"))
        z = _safe_float(position.get("z"))
        radius = PREFAB_SIZE.get(kind, DEFAULT_SIZE)
        obstacles.append(
            {
                "name": prefab_name,
                "type": kind,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "radius": radius,
                "x_min": round(x - radius, 3),
                "x_max": round(x + radius, 3),
                "z_min": round(z - radius, 3),
                "z_max": round(z + radius, 3),
            }
        )

    return {
        "filename": os.path.basename(path),
        "terrainIndex": data.get("terrainIndex"),
        "bounds": {"x_min": MAP_MIN_M, "x_max": MAP_MAX_M, "z_min": MAP_MIN_M, "z_max": MAP_MAX_M},
        "obstacles": obstacles,
        "available_maps": list_map_files(),
    }
