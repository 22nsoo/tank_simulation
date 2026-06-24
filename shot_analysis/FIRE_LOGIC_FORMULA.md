# fire_logic.py 수식 기반 제어 정리

이 문서는 현재 `shot_analysis/fire_logic.py`에 구현된 자동 조준, 탄도 계산,
발사 판단 및 탄착 보정 로직을 코드 기준으로 정리한 문서다.

## 1. 현재 동작 개요

제어 흐름은 다음과 같다.

1. `/info`에서 플레이어와 적 탱크 상태를 받는다.
2. X-Z 평면에서 적의 방향과 수평 거리를 계산한다.
3. 포탑의 좌우 오차를 계산해 `Q/E` 명령을 만든다.
4. 포물선 탄도식으로 목표 포각을 계산한다.
5. 현재 포각과 목표 포각의 차이로 `R/F` 명령을 만든다.
6. 좌우 및 상하 오차가 허용 범위 안에 들어오면 발사한다.
7. 탄착점이 들어오면 실제 사거리 오차로 포각 보정값을 갱신한다.

현재 `STATIONARY_FIRE_MODE = True`이므로 플레이어 탱크는 전진하거나
후진하지 않고, 리스폰 위치에서 차체와 포탑만 회전해 조준한다.

## 2. 시작 위치와 적 생성 거리

플레이어 시작 위치:

```text
X = 150
Y = 10
Z = 150
```

적 탱크는 `/init`이 호출될 때마다 플레이어의 +Z 방향에 다음 거리로
순환 배치된다.

```text
120m → 100m → 80m → 60m → 반복
```

적 위치는 다음과 같다.

```text
enemy_x = player_x
enemy_y = player_y
enemy_z = player_z + enemy_distance
```

## 3. 좌표계와 입력값

Unity의 X-Z 평면을 수평면으로 사용하고 Y를 높이로 사용한다.

주요 입력값:

| 입력 | 의미 |
|---|---|
| `playerPos` | 플레이어 월드 좌표 `(x, y, z)` |
| `enemyPos` | 적 월드 좌표 `(x, y, z)` |
| `playerBodyX` | 플레이어 차체의 월드 yaw |
| `playerTurretX` | 포탑의 월드 yaw |
| `playerTurretY` | 포신의 현재 pitch |
| `enemyHealth` | 적 체력 |

## 4. 목표 방향과 수평 거리

플레이어와 적 위치의 차이는 다음과 같다.

```text
dx = enemy_x - player_x
dz = enemy_z - player_z
```

수평 거리:

```text
distance = sqrt(dx² + dz²)
```

월드 기준 목표 yaw:

```text
target_world_angle = degrees(atan2(dx, dz))
```

이 구현에서는 +Z 방향을 `0°`로 취급한다.

## 5. 좌우 조준 오차

모든 각도 오차는 `-180° ~ +180°`로 정규화한다.

차체 yaw 오차:

```text
body_error = normalize(target_world_angle - playerBodyX)
```

포탑 yaw 오차:

```text
turret_error = normalize(target_world_angle - playerTurretX)
```

### 포탑 Q/E 제어

포탑 오차의 절댓값이 `0.5°`보다 클 때만 회전한다.

거리별 최대 회전 weight:

| 거리 | 최대 weight |
|---|---:|
| 40m 미만 | 0.12 |
| 40m 이상 90m 미만 | 0.22 |
| 90m 이상 | 0.32 |

회전 weight:

```text
turret_weight =
    clamp(abs(turret_error) / 60 × max_turret_weight,
          0.04,
          max_turret_weight)
```

명령 방향:

```text
turret_error > 0  → E
turret_error < 0  → Q
```

## 6. 탄도 기반 목표 포각

현재 탄도 모델은 공기 저항이 없는 이상적인 포물선 운동을 사용한다.

상수:

```text
중력 가속도 g = 9.81 m/s²
추정 포구 속도 v = 45.0 m/s
```

수평 거리:

```text
R = sqrt((enemy_x - player_x)² + (enemy_z - player_z)²)
```

높이 차:

```text
dy = enemy_y - player_y
```

사용하는 탄도식:

```text
dy = R tan(θ) - gR² / (2v² cos²(θ))
```

저각 해:

```text
D = v⁴ - g(gR² + 2dyv²)

θ = atan((v² - sqrt(D)) / (gR))
```

여기서 `D`는 판별식이다.

- `D >= 0`: 계산 가능한 저각 탄도 해를 사용한다.
- `D < 0`: 현재 포구 속도로 도달 불가능한 거리로 판단하고
  `MAX_PITCH_DEG`를 사용한다.

## 7. 시뮬레이터 포각으로 변환

탄도식에서 계산한 물리 포각에 학습된 bias를 더한다.

```text
theta_with_bias = theta_raw + PITCH_BIAS_DEG
```

이 값은 우선 다음 범위로 제한된다.

```text
-10° <= theta_with_bias <= 35°
```

탄착 로그를 비교한 결과 `playerTurretY`가 더 음수가 될수록 실제 사거리가
짧아지고, 양수 방향으로 갈수록 사거리가 길어지는 것으로 관측되었다.

```text
desired_pitch_unclamped =
    PITCH_OFFSET_DEG + PITCH_CORRECTION_SIGN × theta_with_bias

PITCH_CORRECTION_SIGN = 1
PITCH_OFFSET_DEG = 0
```

마지막으로 관측된 실제 포탑 가동 범위에 맞춰 제한한다.

```text
desired_pitch =
    clamp(desired_pitch_unclamped, -5°, +10°)
```

현재 설정:

```text
SIM_MIN_TURRET_PITCH_DEG = -5
SIM_MAX_TURRET_PITCH_DEG = 10
```

포각 오차:

```text
pitch_error = desired_pitch - playerTurretY
```

## 8. 포신 R/F 제어

포각 오차가 `0.2°`보다 클 때 포신을 움직인다.

포각 변화에 대한 예상 사거리 민감도를 수치 미분으로 구한다.

평지 근사 사거리:

```text
range(θ) = v² / g × sin(2θ)
```

수치 미분:

```text
dR/dθ ≈ [range(θ + ε) - range(θ - ε)] / (2ε)
ε = 0.2°
```

민감도에 따른 최대 weight:

| 조건 | 최대 weight |
|---|---:|
| `abs(dR/dθ) > 80` | 0.22 |
| 그 외 | 0.18 |

포신 제어 weight:

```text
pitch_weight =
    clamp(abs(pitch_error) / 5 × max_pitch_weight,
          0.04,
          max_pitch_weight)
```

명령 방향:

```text
pitch_error > 0  → R
pitch_error < 0  → F
```

## 9. 차체 회전과 이동

차체 yaw 오차가 `35°`보다 크면 강하게 회전한다.

```text
body_weight =
    clamp(abs(body_error) / 90 × 0.35, 0.08, 0.35)
```

차체 yaw 오차가 `15°`보다 크고 거리가 `70m`보다 멀면 약하게 회전한다.

```text
body_weight =
    clamp(abs(body_error) / 90 × 0.2, 0.05, 0.2)
```

명령 방향:

```text
body_error > 0  → D
body_error < 0  → A
```

현재 정지 사격 모드에서는 전후 이동이 항상 정지다.

```text
moveWS = STOP
```

## 10. 동적 포각 허용 오차

발사 허용 포각 오차는 고정값이 아니라, 포각 변화가 사거리에 미치는
영향을 이용해 계산한다.

허용 사거리 오차:

```text
RANGE_ERROR_TOLERANCE = 3m
```

포각 허용 오차:

```text
pitch_tolerance_rad =
    RANGE_ERROR_TOLERANCE / abs(dR/dθ)

pitch_tolerance_deg =
    degrees(pitch_tolerance_rad)
```

최종 제한:

```text
0.08° <= pitch_tolerance <= 0.5°
```

## 11. 발사 조건

다음 조건을 모두 만족해야 `fire=True`가 된다.

```text
20m < distance < 200m
abs(turret_error) < 1.5°
abs(pitch_error) < pitch_tolerance
현재 시간 - 마지막 발사 시간 > 1초
```

적 체력이 `0` 이하이면 조준 및 발사를 하지 않는다.

## 12. 탄착 기반 포각 bias 보정

발사 순간의 플레이어 위치, 적 위치, 목표 포각과 제어 명령을
`pending_shots`에 저장한다.

`/update_bullet`로 탄착점이 들어오면 다음 사거리를 계산한다.

목표 사거리:

```text
target_range =
    sqrt((enemy_x - player_x)² + (enemy_z - player_z)²)
```

실제 탄착 사거리:

```text
impact_range =
    sqrt((impact_x - player_x)² + (impact_z - player_z)²)
```

사거리 오차:

```text
range_error = target_range - impact_range
```

- 양수: 포탄이 목표보다 짧게 떨어졌다.
- 음수: 포탄이 목표보다 멀리 날아갔다.

포각 보정량:

```text
delta_theta_rad =
    PITCH_TUNE_GAIN × range_error / (dR/dθ)

PITCH_TUNE_GAIN = 0.35
```

degree로 변환하고 시뮬레이터 포각 부호를 적용한다.

```text
delta_theta_deg =
    PITCH_CORRECTION_SIGN × degrees(delta_theta_rad)
```

한 발당 변경량:

```text
-2° <= delta_theta_deg <= +2°
```

누적 bias:

```text
PITCH_BIAS_DEG =
    clamp(PITCH_BIAS_DEG + delta_theta_deg, -10°, +15°)
```

## 13. 현재 구현의 중요한 한계

### 13.1 포각 가동 한계와 탄도 해의 불일치

예를 들어 120m에서 탄도식이 약 `18°`를 요구하면 시뮬레이터 값은
약 `+18°`가 된다. 그러나 현재 포탑은 `+10°`까지만 움직이는 것으로
관측되어 목표값이 `+10°`로 제한된다.

이 경우 코드상 포각 오차는 0에 가까워져 발사할 수 있지만, 실제 탄도식이
요구한 포각을 만들지 못하므로 포탄이 목표까지 도달하지 않을 수 있다.

즉, 포각을 단순히 가동 한계로 자르는 것은 “조준 완료” 판정에는 도움이
되지만 물리적으로 정확한 해결책은 아니다.

### 13.2 포구 속도는 추정값

`MUZZLE_SPEED = 45m/s`는 현재 추정값이다. 실제 게임의 포탄 속도와 다르면
거리별 목표 포각이 전부 달라진다. 탄착 로그를 이용해 실제 포구 속도를
역추정하거나 거리별 보정표를 만드는 것이 필요하다.

### 13.3 공기 저항 및 게임 내부 물리 미반영

현재 수식은 다음 요소를 반영하지 않는다.

- 공기 저항
- 게임 내부 중력 배율
- 포신 실제 발사점과 탱크 중심 좌표의 차이
- 적 탱크의 높이와 조준해야 할 실제 피격 지점
- 지형 높이 변화
- 게임 프레임과 명령 지연

### 13.4 탄착과 발사의 순서 매칭

현재는 가장 오래된 `pending_shot`과 다음 탄착 이벤트를 순서대로 연결한다.
연사 중 탄착 이벤트가 누락되거나 순서가 바뀌면 잘못된 발사와 연결될 수 있다.

## 14. 수식 기반 제어 개선 시 우선 확인할 값

1. 실제 `playerTurretY`의 최소·최대값과 부호
2. 같은 높이에서 거리별 최소 명중 포각
3. 실제 포탄 비행 시간
4. 실제 포구 속도 또는 게임 내부 중력값
5. `F`와 `R`이 각각 포신을 어느 방향으로 움직이는지
6. 포각이 가동 한계를 벗어나는 거리에서 발사를 금지할지 여부

특히 `desired_pitch_unclamped`가 `-5° ~ +10°` 범위를 벗어나면 현재
수식으로 계산한 탄도를 실제 포탑이 구현할 수 없는 상태다. 이때는 단순
발사보다 “도달 불가”로 처리하거나, 포구 속도 추정값을 먼저 보정하는
방식이 더 안전하다.

## 15. 이후 튜닝 및 변경 기록 원칙

앞으로 `fire_logic.py`의 조준 및 사격 로직은 다음 원칙으로 튜닝한다.

1. 가장 최근의 유효한 `shot_log_*.csv`를 기준 데이터로 사용한다.
2. 탄착 사진만 보고 임의의 상수를 정하지 않는다.
3. 목표 사거리, 실제 탄착 사거리, 포각, 비행시간 및 오차를 수치로 계산한다.
4. 포구 속도, 중력, 포각 bias 또는 제어 gain의 변경 근거를 수식으로 제시한다.
5. 변경 전후 결과를 같은 지표와 그래프로 비교한다.
6. 수식으로 설명할 수 없는 임시 보정값은 별도로 표시한다.
7. 코드가 수정될 때마다 이 문서에 변경 내용을 누적 기록한다.

### 변경 기록 형식

각 변경은 아래 항목을 기록한다.

```text
날짜:
대상 코드/상수:
변경 전:
변경 후:
사용한 CSV:
측정값:
사용한 수식:
계산 결과:
변경 이유:
검증 결과:
```

### 변경 이력

#### 2026-06-23 — 튜닝 정책 확정

- 이후 모든 포각 및 사격 튜닝은 CSV 측정값과 탄도식을 근거로 수행한다.
- 모든 코드 수정과 계산 근거를 이 문서에 계속 추가한다.
- 그래프는 데이터가 들어 있는 가장 최근 `shot_log_*.csv`를 사용한다.

#### 2026-06-23 — 거리별 실사격 검증 시작

- 사용한 CSV: `shot_log_6.csv`
- 검증 거리: 120m
- 발사 수: 1발
- 판정: `enemy` 명중
- 목표 사거리: 약 120.001m
- 실제 탄착 사거리: 약 117.073m
- 사거리 오차: 약 +2.928m
- 3D 탄착 오차: 약 3.106m
- 실제 발사 포각: 약 +9.51°
- 목표 시뮬레이터 포각: +10.0°
- 상태: 120m 1차 검증 완료, 100m·80m·60m 검증 대기

#### 2026-06-23 — 80m 검증 중 포구속도 및 발사 큐 보정

- 사용한 CSV: `shot_log_6.csv`
- 관측 거리: 80m
- 관측 실제 포각: 약 +9.98°
- 관측 탄착 사거리: 약 120.57m
- 결과: 목표보다 약 40.57m 초과

평지 탄도 근사식:

```text
R = v²/g × sin(2θ)
```

포구속도 역산:

```text
v = sqrt(Rg / sin(2θ))
v ≈ sqrt(120.57 × 9.81 / sin(19.96°))
v ≈ 58.9m/s
```

적용값:

```text
MUZZLE_SPEED: 45.0m/s → 59.0m/s
```

새 포구속도 기준 예상 저각 포각:

```text
120m ≈ 9.9°
100m ≈ 8.2°
 80m ≈ 6.5°
 60m ≈ 4.9°
```

추가로 기존 코드는 `FIRE_COOLDOWN = 1초`만 확인하여 포탄이 비행 중에도
다음 발을 발사했다. 이로 인해 `pending_shots`의 발사 기록과 늦게 들어오는
탄착 이벤트가 서로 다른 거리의 발사와 연결되었고, 잘못된 bias가 누적됐다.

수정:

```text
pending_shots가 비어 있을 때만 신규 발사 허용
```

따라서 앞으로는 한 발의 탄착이 기록된 후에만 다음 발을 발사한다.

#### 2026-06-23 — 80m 수식 보정 후 실사격 검증

- 사용한 CSV: `shot_log_7.csv`
- 목표 사거리: 약 80.007m
- 실제 탄착 사거리: 약 80.068m
- 사거리 오차: 약 -0.061m
- 3D 탄착 오차: 약 2.057m
- 실제 포각: 약 +6.82°
- 계산 목표 포각: 약 +6.95°
- 판정: `enemy` 명중
- 결론: `MUZZLE_SPEED = 59.0m/s` 수식은 80m에서 유효

거리별 검증 시 순환 인덱스로 원하는 거리를 건너뛰지 않도록 다음
검증 거리를 지정하는 API를 추가했다.

```text
GET 또는 POST /set_test_distance/120
GET 또는 POST /set_test_distance/100
GET 또는 POST /set_test_distance/80
GET 또는 POST /set_test_distance/60
```

API 호출 후 게임을 리셋하면 지정한 거리가 한 번 적용된다.

#### 2026-06-23 — 거리별 서로 다른 방향 순환 배치

현재 Unity `/init` 응답은 상대 탱크 시작 좌표 한 개만 제공하므로,
네 대를 동시에 생성하는 대신 리셋마다 상대 한 대를 다음 시나리오로
순환 배치한다.

| 순서 | 거리 | 방향각 | 플레이어 기준 방향 | 상대 좌표 |
|---:|---:|---:|---|---|
| 1 | 120m | 0° | 정면(+Z) | `(150, 270)` |
| 2 | 100m | 90° | 오른쪽(+X) | `(250, 150)` |
| 3 | 80m | 180° | 후방(-Z) | `(150, 70)` |
| 4 | 60m | -90° | 왼쪽(-X) | `(90, 150)` |

좌표 계산식:

```text
enemy_x = player_x + distance × sin(bearing)
enemy_z = player_z + distance × cos(bearing)
```

방향각은 기존 조준식과 동일하게 +Z를 `0°`, +X를 `90°`로 사용한다.

```text
target_yaw = atan2(enemy_x - player_x, enemy_z - player_z)
```

각 리셋에서 제어 순서는 다음과 같다.

```text
새 위치 생성
→ 차체 A/D 회전
→ 포탑 Q/E 회전
→ 포각 R/F 조절
→ 리셋 후 1.5초 경과
→ 조준 상태 0.5초 연속 유지
→ 한 발 사격
→ 다음 리셋에서 다음 시나리오로 이동
```

#### 2026-06-23 — 거리별 검증 발사 상태 분리

100m 리셋 후 조준 조건을 만족했지만 발사가 나오지 않는 현상을 확인했다.
원인은 이전 거리에서 발사된 포탄이 리셋으로 제거되면서
`/update_bullet`이 들어오지 않아 `pending_shots`가 남은 것이었다.

수정 사항:

```text
/init 호출 시 pending_shots 초기화
/init 호출 시 last_fire_time 초기화
/init 호출 시 현재 리스폰의 발사 여부 초기화
거리별 검증 모드에서는 리스폰당 한 발만 발사
```

설정:

```text
TEST_SINGLE_SHOT_PER_SPAWN = True
```

이제 각 거리의 한 발과 한 탄착이 독립적으로 연결되며, 리셋 전에 발생한
미완료 포탄이 다음 거리의 발사를 막거나 bias를 오염시키지 않는다.

#### 2026-06-23 — 리셋 직후 즉시 발사 방지

이전 거리의 포탑 각도가 새 거리의 발사 허용 범위와 우연히 겹치면,
리셋 직후 첫 제어 요청에서 바로 발사되는 현상이 있었다.

추가한 시간 조건:

```text
리셋 후 무장 대기 시간 = 1.5초
연속 조준 안정 시간 = 0.5초
```

새 발사 조건:

```text
20m < distance < 200m
abs(turret_error) < 1.5°
abs(pitch_error) < pitch_tolerance
리셋 후 경과 시간 >= 1.5초
조준 조건 연속 유지 시간 >= 0.5초
pending_shots가 비어 있음
현재 리스폰에서 아직 발사하지 않음
```

수식 표현:

```text
aligned(t) =
    distance 조건
    AND yaw 조건
    AND pitch 조건

fire(t) =
    aligned(t)
    AND (t - t_spawn >= 1.5s)
    AND (t - t_aligned_start >= 0.5s)
```

조준 조건이 한 번이라도 깨지면 `t_aligned_start`를 초기화한다. 따라서
새 거리에서 포탑이 실제로 안정된 상태를 0.5초 유지해야 발사한다.

#### 2026-06-23 — 상대 탱크 대신 네 개 장애물 순차 사격

요구사항 변경에 따라 상대 탱크는 조준 표적에서 제외하고 정적 장애물
네 개를 순서대로 사격하도록 변경했다.

`NewMap.map` 장애물 배치:

| 순서 | 장애물 | 거리 | 방향 | 좌표 `(x, y, z)` |
|---:|---|---:|---|---|
| 1 | `Rock002_120m_Front` | 120m | 정면 | `(150, 9.5022, 270)` |
| 2 | `Rock002_100m_Right` | 100m | 오른쪽 | `(250, 9.5022, 150)` |
| 3 | `Rock002_80m_Rear` | 80m | 후방 | `(150, 9.5022, 70)` |
| 4 | `Rock002_60m_Left` | 60m | 왼쪽 | `(90, 9.5022, 150)` |

동작 순서:

```text
120m 장애물 조준 및 한 발 발사
→ /update_bullet 탄착 로그 저장
→ 100m 장애물로 자동 전환
→ 조준 안정 후 한 발 발사
→ 80m
→ 60m
→ 다시 120m
```

설정:

```text
OBSTACLE_TARGET_MODE = True
TEST_SINGLE_SHOT_PER_SPAWN = False
```

각 탄착 이벤트가 들어와야 다음 장애물로 넘어가므로 발사와 탄착 로그의
순서가 유지된다. 상대 탱크는 맵 끝 `(290, 290)`으로 보내고 제어 로직은
`enemyPos` 대신 현재 장애물 좌표를 내부 표적 좌표로 사용한다.

CSV 호환성을 위해 기존 `enemy_x_fire`, `enemy_y_fire`, `enemy_z_fire` 열은
유지한다. 장애물 모드에서는 이 세 열이 실제로 현재 장애물 표적 좌표를
의미한다. `hit` 열에는 시뮬레이터가 전달한 장애물/지형 충돌 판정이 기록된다.

#### 2026-06-23 — 포각 제어 command 그래프 추가

`graph_plot.py`에 `10_pitch_command_diagnostics.png`를 추가했다.

포각 제어 명령을 다음과 같이 수치화한다.

```text
F    = -1
STOP =  0
R    = +1
```

signed weight:

```text
signed_pitch_weight = command_value × action_turretRF_weight
```

그래프 구성:

1. `pitch_error`와 발사 허용 `pitch_tolerance`
2. `action_turretRF_command` 방향
3. F는 음수, R은 양수로 표시한 command weight

현재 `shot_log_*.csv`의 command 값은 매 제어 프레임이 아니라 발사 시점의
스냅샷이다. 전체 조준 과정의 command 변화를 분석하려면 추후
`/get_action` 호출마다 별도의 제어 로그를 저장해야 한다.

#### 2026-06-23 — 전체 포각 제어 프레임 로그 추가

발사 순간 command만으로는 포탑이 목표 포각에 수렴하는 과정을 분석하기
어렵기 때문에 `/get_action` 호출마다 `control_log_*.csv`를 저장하도록
확장했다.

주요 기록 열:

```text
timestamp
target_type / target_name / target_index
target_distance / target_bearing
player_turret_pitch / desired_pitch
pitch_error / pitch_tolerance
turret_rf_command / turret_rf_weight
turret_error / turret_qe_command / turret_qe_weight
body_error / move_ad_command / move_ad_weight
aim_aligned / spawn_arm_ready / aim_stable_ready
fire
```

`graph_plot.py`의 `10_pitch_command_diagnostics.png`는 가장 최근의 유효한
`control_log_*.csv`를 우선 사용한다. 제어 로그가 없을 때만 기존
`shot_log_*.csv`의 발사 시점 스냅샷을 사용한다.

#### 2026-06-23 — 거리별 유효 포구속도 보정

`shot_log_1.csv`의 실제 포각과 탄착 좌표로 포구속도를 역산했다.

```text
R = sqrt((impact_x-player_x)² + (impact_z-player_z)²)
dy = impact_y-player_y
v = sqrt(gR² / (2cos²(θ) × (Rtan(θ)-dy)))
```

| 거리 | 실제 포각 | 역산 속도 |
|---:|---:|---:|
| 120m | 9.84° | 61.558m/s |
| 100m | 9.18° | 63.284m/s |
| 80m | 7.66° | 66.883m/s |
| 60m | 6.25° | 64.715m/s |

단일 속도 대신 다음 보정점을 거리 기준으로 선형 보간한다.

```text
DISTANCE_SPEED_CALIBRATION = (
    (60, 64.715),
    (80, 66.883),
    (100, 63.284),
    (120, 61.558),
)

v(d) = v0 + (d-d0)/(d1-d0) × (v1-v0)
```

예상 목표 포각은 120m `9.79°`, 100m `7.97°`, 80m `6.14°`,
60m `5.49°`다. 포각 허용치와 command gain 계산도 같은 보간 속도를
사용한다.

서로 다른 거리의 오차가 하나의 전역 bias에 누적되는 것을 방지하기 위해
검증 중 자동 bias 적용을 중지했다.

```text
IMPACT_BIAS_TUNING_ENABLED = False
PITCH_BIAS_DEG = 0.0
```

탄착 오차와 계산된 보정량은 계속 로그에 남지만 실제 전역 bias는 변경하지
않는다.

#### 2026-06-23 — 좌우 회전 오버슈팅 진단 그래프 추가

`graph_plot.py`에 `11_yaw_overshoot_diagnostics.png`를 추가했다.

그래프 구성:

1. 전체 제어 프레임의 `turret_error`
2. `Q/E` command 방향
3. `body_error`와 `A/D` 회전 기여
4. signed Q/E 및 A/D weight 비교

command 수치화:

```text
Q = -1, STOP = 0, E = +1
A = -1, STOP = 0, D = +1
```

signed weight:

```text
signed_qe_weight = qe_command_value × turret_qe_weight
signed_ad_weight = ad_command_value × move_ad_weight
```

오버슈팅 표시 조건:

```text
이전 turret_error × 현재 turret_error < 0
AND abs(현재 turret_error) >= 1.5°
```

즉 목표각 0°를 통과한 뒤 발사 yaw 허용 범위 밖으로 넘어간 지점을 빨간
`X`로 표시한다. 차체와 포탑이 같은 방향으로 동시에 회전하는 구간은 노란색
영역으로 표시하여 합산 회전에 의한 오버슈팅을 확인할 수 있다.

#### 2026-06-23 — 좌우 회전 PD 감쇠 및 차체 미세구간 정지

`control_log_2.csv`에서 100m 표적 전환 시 다음 현상을 확인했다.

```text
turret_error: +4.55° → +0.05° → -3.76°
body_error:   +43.76° → +40.17° → +36.78°
```

포탑이 목표에 도달했어도 차체가 계속 `D`로 회전해 다음 프레임에서 목표를
지나쳤다. 이를 줄이기 위해 Q/E를 PD 제어로 변경했다.

```text
yaw_rate = (error_now - error_prev) / dt
u = Kp × yaw_error + Kd × yaw_rate
```

설정:

```text
TURRET_YAW_KP = 0.0053
TURRET_YAW_KD = 0.0020
TURRET_YAW_DEADBAND_DEG = 0.5°
```

명령:

```text
u > 0 → E, weight = abs(u)
u < 0 → Q, weight = abs(u)
abs(error) <= 0.5° → STOP
```

기존 최소 Q/E weight `0.04`는 제거했다. 미분항은 목표에 빠르게 접근할 때
현재 회전 반대 방향의 감쇠를 만들어 0° 통과 전에 제동한다.

차체와 포탑의 회전 합산을 막기 위해 다음 조건도 추가했다.

```text
abs(turret_error) > 15° → 기존 조건에 따라 차체 A/D 허용
abs(turret_error) <= 15° → 차체 A/D STOP, 포탑만 미세 조준
```

설정:

```text
BODY_COARSE_TURRET_ERROR_DEG = 15.0°
```

새 `control_log_*.csv`에는 다음 진단값도 기록한다.

```text
turret_error_rate
turret_pd_effort
body_coarse_turn_enabled
```

#### 2026-06-23 — 차체 회전 상쇄형 포탑 제어 및 차체 정렬 발사

`control_log_2.csv`의 동일 표적 구간에서 실제 각속도를 역산했다.

```text
body_gain =
    median((-Δbody_error/Δt) / signed_AD_weight)
    = 37.254 deg/s/weight

turret_gain =
    median((-Δturret_error/Δt) / signed_QE_weight)
    = 43.498 deg/s/weight
```

차체 A/D 비례 제어:

```text
u_body = clamp(Kb × body_error, -0.35, +0.35)
Kb = 0.0039
```

차체 오차 `8°` 이내에서는 A/D를 정지한다.

```text
BODY_YAW_DEADBAND_DEG = 8°
```

차체가 만들 것으로 예상되는 월드 yaw 속도:

```text
ω_body = 37.254 × u_body
```

이를 Q/E weight로 환산한다.

```text
u_body_equivalent = ω_body / 43.498
```

최종 포탑 명령:

```text
u_pd = Kp × turret_error + Kd × error_rate
u_turret = clamp(
    u_pd - u_body_equivalent,
    -max_turret_weight,
    +max_turret_weight
)
```

차체가 D 방향으로 회전하면 포탑 명령에서 동일한 월드 회전량을 빼므로,
포탑은 필요할 경우 Q 방향으로 차체 회전을 상쇄하면서 표적을 유지한다.

발사 조건에도 차체 정렬을 추가했다.

```text
abs(body_error) < 10°
abs(turret_error) < 1.5°
abs(pitch_error) < pitch_tolerance
```

설정:

```text
TURRET_YAW_SPEED_PER_WEIGHT = 43.498
BODY_YAW_SPEED_PER_WEIGHT = 37.254
BODY_YAW_KP = 0.0039
BODY_YAW_DEADBAND_DEG = 8.0
BODY_FIRE_TOLERANCE_DEG = 10.0
```

새 제어 로그에는 원래 PD 출력과 차체 상쇄량도 기록한다.

```text
turret_pd_effort_raw
body_signed_effort
predicted_body_yaw_rate
body_equivalent_turret_weight
turret_pd_effort
```

#### 2026-06-23 — 그래프 포트폴리오 폴더 자동 누적

`graph_plot.py` 실행 시 이전 그래프를 덮어쓰지 않고 새 결과 폴더를 만든다.

생성 순서:

```text
shot_plots
shot_plots_1
shot_plots_2
shot_plots_3
...
```

선택 규칙:

```text
shot_plots가 없으면 shot_plots 사용
이미 있으면 존재하지 않는 가장 작은 다음 번호 사용
```

각 결과 폴더에는 그래프 PNG와 함께 `README.md`를 생성한다.

README 기록 내용:

```text
생성 시각
사용한 shot_log CSV
사용한 control_log CSV
발사 및 유효 탄착 수
평균 목표/탄착 사거리
평균 사거리 오차
평균 3D 탄착 오차
생성된 이미지 목록
```

따라서 제어 로직 개선 전후 그래프를 폴더별 포트폴리오로 비교할 수 있다.

#### 2026-06-23 — 동일 Rock002 장애물 20개 자동 검증

장애물 검증 시나리오를 `120m ~ 44m`, 4m 간격의 동일한 `Rock002`
장애물 20개로 고정했다. 각 장애물은 golden-angle 방식으로 서로 다른
방향에 배치된다.

```text
distance_i = 120 - 4i,  i = 0, 1, ..., 19
bearing_i = (137.507764 × i) mod 360
```

각 탄착 이벤트가 들어오면 다음 표적으로 한 번만 이동한다. 20번째 탄착이
기록되면 첫 표적으로 순환하지 않고 모든 조준·이동·사격 명령을 정지한다.
게임을 리셋하면 인덱스 0부터 새로운 20발 검증을 시작한다.

탄착 로그에는 장애물 중심에 대한 오차를 사격 방향 좌표계로 분해해 추가한다.

```text
f = normalize(target_xz - player_xz)
r = (f_z, -f_x)
e = impact - target

forward_error  = dot(e_xz, f)
lateral_error  = dot(e_xz, r)
vertical_error = impact_y - target_y
```

`forward_error < 0`은 장애물 중심보다 사수 쪽, `lateral_error`는 좌우 편향,
`vertical_error`는 상하 편향을 뜻한다. 동일 형상의 장애물 20개 결과에서
각 오차의 평균을 계산해 체계적인 pitch/yaw 편향과 무작위 탄 퍼짐을
구분한다. 이 값들은 장애물 중심 기준 진단값이며, 장애물 표면 반경을
추정하기 전에는 자동 포각 bias에 직접 누적하지 않는다.

#### 2026-06-24 정적 장애물 포각 제어 속도 튜닝

정적 장애물 검증에서는 표적이 움직이지 않으므로, 포각 제어가 너무 느려서
조준 안정화까지 오래 걸리는 문제가 더 크게 보인다. 그래서 `R/F` 포각 명령의
수식은 유지하되, 오차가 큰 구간에서 더 빠르게 목표 포각에 접근하도록
weight 상한과 오차 스케일을 조정했다.

기존 수식:

```text
max_pitch_weight = 0.22 if dR/dtheta > 80 else 0.18
pitch_weight = clamp(abs(pitch_error) / 5.0 × max_pitch_weight,
                     0.04,
                     max_pitch_weight)
```

변경 수식:

```text
PITCH_CONTROL_ERROR_SCALE_DEG = 4.0
PITCH_MIN_WEIGHT = 0.045
PITCH_MAX_WEIGHT_FLAT = 0.22
PITCH_MAX_WEIGHT_SENSITIVE = 0.28
PITCH_SENSITIVE_RANGE_DERIVATIVE = 80.0

max_pitch_weight =
    0.28  if abs(dR/dtheta) > 80
    0.22  otherwise

pitch_weight = clamp(abs(pitch_error) / 4.0 × max_pitch_weight,
                     0.045,
                     max_pitch_weight)
```

효과:

```text
큰 포각 오차  → 기존보다 더 빠르게 R/F 회전
작은 포각 오차 → 최소 weight만 소폭 증가하여 미세 접근 유지
발사 조건     → 기존 dynamic pitch tolerance 그대로 유지
```

주의점:

```text
오버슈팅이 보이면 PITCH_MAX_WEIGHT_SENSITIVE를 0.28 → 0.25로 낮춘다.
조준이 여전히 느리면 PITCH_CONTROL_ERROR_SCALE_DEG를 4.0 → 3.5로 낮춘다.
```

#### 2026-06-24 정적 장애물 목표 근처 포각 접근 속도 보정

목표 포각 근처에서 `pitch_error`가 작아질수록 weight가 최소값까지 낮아져
마지막 접근이 지나치게 느려지는 현상이 있었다. 큰 오차 구간의 최대 속도는
그대로 두고, 목표 근처 접근 속도만 소폭 올리기 위해 최소 R/F weight를 조정했다.

변경:

```text
PITCH_MIN_WEIGHT = 0.045 → 0.055
```

현재 포각 제어 수식:

```text
pitch_weight = clamp(abs(pitch_error) / 4.0 × max_pitch_weight,
                     0.055,
                     max_pitch_weight)
```

의도:

```text
큰 오차 구간     → 이전 튜닝과 동일
목표 근처 구간   → 너무 느리게 붙는 문제 완화
오버슈팅 위험    → 최대 weight는 그대로라 제한적
```

만약 목표 근처에서 지나쳤다가 되돌아오는 현상이 다시 보이면
`PITCH_MIN_WEIGHT`를 `0.055 → 0.050`으로 낮춘다.
