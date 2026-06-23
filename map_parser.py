import json

# ============================================================
# prefab별 크기 (반경 m) - 시뮬레이터 실측 기준
# 없는 타입은 DEFAULT 사용
# ============================================================
PREFAB_SIZE = {
    'Human002': 0.65,
    'Human003': 0.65,
    'Car001':   2.5,
    'Car002':   2.5,
    'Car003':   2.5,
    'Car004':   2.5,
    'Rock001':  3.0,
    'Rock002':  3.5,
    'Tree001':  1.5,
    'Tree003':  1.5,
    'House002': 6.0,
    'Wall002':  4.0,
}
DEFAULT_SIZE = 2.0


def prefab_type(prefab_name: str) -> str:
    """'Human002_6' → 'Human002'"""
    return '_'.join(prefab_name.split('_')[:-1])


def parse_map(map_path: str) -> list:
    """
    .map 파일 파싱 → 정적 장애물 bbox 리스트 반환
    반환 형식: [{'x_min','x_max','z_min','z_max','name','type'}, ...]
    """
    with open(map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    obstacles = []
    for obj in data.get('obstacles', []):
        ptype  = prefab_type(obj['prefabName'])
        radius = PREFAB_SIZE.get(ptype, DEFAULT_SIZE)
        px     = obj['position']['x']
        pz     = obj['position']['z']

        obstacles.append({
            'name':  obj['prefabName'],
            'type':  ptype,
            'x_min': px - radius,
            'x_max': px + radius,
            'z_min': pz - radius,
            'z_max': pz + radius,
        })
        print(f"  📦 {obj['prefabName']:25s} ({px:.1f}, {pz:.1f}) r={radius}")

    print(f"\n✅ 정적 장애물 파싱 완료: {len(obstacles)}개")
    return obstacles


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'new_map3.map'
    obs  = parse_map(path)
    print(f"\n--- bbox 목록 ---")
    for o in obs:
        print(f"  {o['name']:25s} x:[{o['x_min']:.1f}~{o['x_max']:.1f}] z:[{o['z_min']:.1f}~{o['z_max']:.1f}]")
