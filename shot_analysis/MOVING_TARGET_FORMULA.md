# 움직이는 적 전차 선행 사격 수식

`fire_logic_moving_target.py`는 기존 `fire_logic.py`를 수정하지 않고 만든
별도 실험 서버다. 기존 거리별 포구속도 보정, 탄도식, yaw PD 제어, 차체
회전 상쇄 및 발사 허용 조건을 그대로 사용한다.

## 1. 적 속도 추정

연속된 `/info` 프레임의 적 좌표로 순간 속도를 계산한다.

```text
v_raw = (p_now - p_prev) / dt
```

좌표 노이즈를 줄이기 위해 지수 이동 평균을 적용한다.

```text
v_filtered =
    alpha * v_raw + (1 - alpha) * v_filtered_previous

alpha = 0.35
```

`25m/s`를 넘는 위치 변화는 리스폰 또는 순간이동으로 판단하여 속도 추정값을
초기화한다. 유효 속도 표본이 3개 이상 모이기 전에는 발사하지 않는다.

## 2. 포탄 비행시간

예측 사거리 `R`에서 기존 저각 탄도식으로 포각 `theta`를 계산하고 수평
속도 성분으로 비행시간을 근사한다.

```text
t_flight = R / (v_muzzle * cos(theta))
```

거리별 보정 포구속도 `v_muzzle(R)`는 기존 보간식을 그대로 사용한다.

## 3. 선행 조준점

서버·게임 명령 지연을 `t_delay = 0.10s`로 두고 적 미래 위치를 계산한다.

```text
p_aim =
    p_enemy + v_enemy * (t_flight + t_delay)
```

예측 위치가 바뀌면 사거리와 비행시간도 달라지므로 위 계산을 4회 반복한다.
최대 예측시간은 4초, 최대 선행거리는 35m로 제한한다.

## 4. 발사 조건

```text
유효 속도 표본 >= 3
AND abs(body_error) < 10deg
AND abs(turret_error) < 1.5deg
AND abs(pitch_error) < dynamic_pitch_tolerance
AND 위 조건이 0.5초 이상 유지
AND 미처리 포탄 없음
```

## 5. 로그

기존 로그와 섞이지 않도록 다음 이름을 사용한다.

```text
moving_shot_log_N.csv
moving_control_log_N.csv
```

사격 로그에는 관측 적 좌표, 추정 속도, 예측 비행시간 및 선행거리를
추가한다. 기존 `enemy_*_fire` 좌표는 실제 관측 위치가 아니라 계산된
선행 조준점이다.

## 6. 제한사항

이 서버는 적 전차를 직접 움직이지 않는다. 적 이동은 게임 AI, 수동 조작
또는 별도 제어 클라이언트가 담당해야 한다. 서버는 `/info`로 전달된
움직이는 `enemyPos`를 추적하여 선행 사격한다.

## 7. 적 전차 좌우 대칭 이동 서버

`enemy_tank_movement.py`를 5100번 포트의 적 전차 제어 서버로 추가했다.
전진·후진 명령은 정지시키고 좌회전 `A`와 우회전 `D`에 동일한 weight와
시간을 적용한다.

```text
moveWS = STOP
left_turn_weight = right_turn_weight = 0.30
left_turn_time = right_turn_time = 4.0s
```

한 주기는 다음 순서이며 총 8초다.

```text
A 좌회전 4.0초
→ D 우회전 4.0초
→ 반복
```

이 API에서 `A/D`는 횡이동이 아니라 차체 회전이므로 탱크는 위치를 옆으로
평행 이동하지 않고 제자리에서 좌우로 회전한다. `/get_action` 호출 횟수가
아닌 `time.monotonic()` 경과시간을 사용하므로 요청 주기가 달라져도 좌우
명령 시간과 세기는 동일하다.

## 8. 이동 표적 로그 전용 폴더

이동 표적 로그는 정적 장애물 로그와 섞이지 않도록 다음 폴더에 저장한다.

```text
shot_analysis/moving_target_logs/
    moving_shot_log_1.csv
    moving_control_log_1.csv
    moving_shot_log_2.csv
    moving_control_log_2.csv
    ...
```

서버 시작 시 폴더가 없으면 자동 생성한다.

## 9. 90도 회전 후 전후 왕복 이동

적 전차는 리셋 직후 차체를 약 90도 회전한 다음 해당 축을 따라 `W/S`로
왕복한다. 차체 yaw가 `/get_action` 데이터에 없으므로 기존 실험에서 측정한
차체 회전 gain으로 회전시간을 계산한다.

```text
body_yaw_gain = 37.254 deg/s/weight
turn_weight = 0.30

turn_time
    = 90deg / (37.254 × 0.30)
    = 8.052s
```

이후 0.5초 정지하여 회전을 안정화하고 다음 이동을 반복한다.

```text
W 4.0초, weight 0.35
→ S 4.0초, weight 0.35
→ 반복
```

초기 방향이 사수와 마주 보는 축이라면 90도 회전 후 `W/S` 왕복은 사수
관점에서 좌우 횡방향 이동이 된다. 실제 회전 gain이 맵이나 물리 상태에
따라 달라질 수 있으므로 초기 회전 결과가 90도와 다르면
`BODY_YAW_SPEED_PER_WEIGHT`를 실측값으로 다시 조정한다.

### X축 전용 위치 피드백

시간 기준 `W/S` 전환 대신 실제 적 좌표를 사용한다. 리셋 직후 위치를
중심점으로 저장하고 X축에서 `±12m` 범위를 왕복한다.

```text
x_min = x_start - 12m
x_max = x_start + 12m

x >= x_max → 반대 이동 명령
x <= x_min → 반대 이동 명령
```

초기 90도 회전 후 첫 X 변위로 `W` 명령이 `+X/-X` 중 어느 방향인지 자동
판별한다. 이동 중 연속 위치 변화로 차체 방향을 역추정한다.

```text
yaw_velocity = atan2(delta_x, delta_z)

W 이동: body_yaw = yaw_velocity
S 이동: body_yaw = yaw_velocity + 180deg
```

목표 차체 방향은 `+90deg` 또는 `-90deg`이며, 오차가 2도보다 크면 작은
`A/D` 보정 명령을 적용한다.

```text
yaw_error = normalize(target_x_axis_yaw - estimated_body_yaw)
steer_weight = clamp(0.008 × abs(yaw_error), 0.03, 0.12)
```

따라서 적 전차는 Z 변화가 생기면 차체 방향을 다시 X축과 평행하게
보정하면서 월드 X축 범위 안에서 왕복한다.

#### 2026-06-23 — X축 왕복 정밀도 및 속도 개선

15발 검증에서 적의 Z 좌표가 약 25m 밀린 결과를 반영해 이동 설정을
다음처럼 변경했다.

```text
drive_weight: 0.35 → 0.20
X 왕복 범위: 시작점 ±12m → ±10m
차체 방향 허용오차: 2.0deg → 0.8deg
방향 보정 gain: 0.008 → 0.015
최대 A/D 보정 weight: 0.12 → 0.20
```

X 경계에 도달하면 즉시 반대 가속을 주지 않고 0.7초간 정지한 다음
`W/S`를 반전한다. 관성에 의한 경계 초과와 속도 추정 스파이크를 줄이기
위한 것이다.

```text
x 경계 도달
→ moveWS STOP 0.7초
→ W/S 반전
```

적 속도 벡터와 새 순간 속도 벡터의 내적이 음수이면 방향 반전으로
판단한다.

```text
v_raw dot v_filtered < 0
```

반전 시 기존 속도 EMA를 초기화하고 0.5초 동안 발사를 금지한 뒤 새 방향의
속도 표본을 다시 수집한다.

15발 로그에서 실제 비행시간이 예측값보다 평균 `1.096배` 길었으므로 다음
보정계수를 적용한다.

```text
t_flight_corrected = 1.096 × t_flight_ballistic
```

## 10. 포각·포탑 제어 지연을 포함한 선행 조준

기존 선행 조준은 포탄 비행시간만 반영했다.

```text
p_aim = p_enemy + v_enemy * (t_flight + t_delay)
```

포각과 포탑 회전이 적 이동보다 느린 경우에는 조준이 완료될 때까지 적이
움직이는 시간도 포함해야 한다. 따라서 총 요격시간을 다음처럼 계산한다.

```text
t_control = max(t_yaw, t_pitch) × 1.20

t_intercept =
    t_control
    + t_stable
    + t_system_delay
    + t_flight

p_aim = p_enemy + v_enemy × t_intercept
```

yaw와 pitch는 동시에 제어되므로 두 시간을 더하지 않고 더 오래 걸리는
축의 시간을 사용한다.

### Yaw 제어시간

```text
omega_yaw =
    43.498 × turret_weight
    + 37.254 × body_weight

t_yaw = abs(yaw_error) / omega_yaw
```

### Pitch 제어시간

`control_log_4.csv`의 65개 유효 제어 구간에서 실측한 pitch 응답 gain의
중앙값은 다음과 같다.

```text
pitch_speed_per_weight = 4.562 deg/s/weight
```

기존 pitch weight 수식을 0.05초 단위로 모의 계산하여 현재 포각에서 목표
포각 허용범위까지 걸리는 시간을 구한다.

```text
pitch_weight =
    clamp(
        abs(pitch_error) / 5 × max_pitch_weight,
        0.04,
        max_pitch_weight
    )

pitch_rate = 4.562 × pitch_weight
pitch_next = pitch + sign(error) × pitch_rate × 0.05
```

예측 조준점이 달라지면 yaw 오차, 목표 포각, 비행시간도 다시 달라지므로
전체 계산을 6회 반복한다. 제어시간은 최대 15초, 비행시간은 최대 4초,
선행거리는 최대 35m로 제한한다.

로그에는 다음 값을 추가한다.

```text
predicted_control_time
predicted_yaw_control_time
predicted_pitch_control_time
predicted_flight_time
predicted_total_intercept_time
lead_distance
```

## 11. 이동 표적 로그 그래프

`moving_target_graph_plot.py`는 `moving_target_logs`에서 번호가 가장 큰
유효한 `moving_shot_log_N.csv`와 `moving_control_log_N.csv` 쌍을 자동으로
선택한다.

결과는 기존 결과를 덮어쓰지 않고 다음 순서로 저장한다.

```text
moving_target_plots/
moving_target_plots_1/
moving_target_plots_2/
...
```

생성 그래프:

```text
00_moving_target_dashboard.png
01_intercept_geometry_xz.png
02_prediction_timeline.png
03_control_and_flight_time.png
04_aim_control_commands.png
05_shot_outcome_and_errors.png
06_intercept_formula.png
```

각 결과 폴더에는 사용 로그, 명중 수, 평균 적 속도, 선행거리, 제어시간,
비행시간 및 생성 이미지 목록을 기록한 `README.md`도 생성한다.

현재 로그에는 탄착 순간의 실제 적 좌표가 없으므로 관측 위치·예측 조준점·
탄착점은 비교할 수 있지만, 탄착 순간 실제 적 위치 대비 예측 오차는 직접
계산할 수 없다.

## 12. 이동 표적 15발 사격 제한

한 번의 게임 리셋에서 이동 표적을 최대 15발만 사격한다.

```text
MAX_MOVING_TARGET_SHOTS = 15

can_fire =
    기존 발사 조건
    AND shots_fired < 15
```

15번째 발사 명령 이후에는 아군 전차의 이동·차체·포탑·포각 명령을 모두
정지하고 추가 발사를 차단한다. 이미 발사된 15번째 포탄의 탄착 이벤트는
그대로 사격 로그에 기록된다. 게임을 리셋하면 발사 횟수는 0으로 돌아간다.

## 13. 2026-06-24 정적 장애물과 동일한 포각 제어 속도 튜닝 적용

정적 장애물 검증에서 사용한 `R/F` 포각 제어 파라미터를 움직이는 표적
조준에도 동일하게 적용했다. 움직이는 표적에서는 예측 조준점이 계속 변하므로,
실제 포각 명령 수식과 `predicted_pitch_control_time` 계산 수식이 서로 다르면
리드 시간이 틀어진다. 따라서 두 위치 모두 같은 파라미터를 사용한다.

현재 포각 제어 파라미터:

```text
PITCH_CONTROL_ERROR_SCALE_DEG = 4.0
PITCH_MIN_WEIGHT = 0.055
PITCH_MAX_WEIGHT_FLAT = 0.22
PITCH_MAX_WEIGHT_SENSITIVE = 0.28
PITCH_SENSITIVE_RANGE_DERIVATIVE = 80.0
```

현재 수식:

```text
max_pitch_weight =
    0.28  if abs(dR/dtheta) > 80
    0.22  otherwise

pitch_weight = clamp(abs(pitch_error) / 4.0 × max_pitch_weight,
                     0.055,
                     max_pitch_weight)
```

적용 위치:

```text
1. 실제 `/get_action`의 `turretRF` 명령 생성
2. 움직이는 표적 위치 예측에 쓰는 `predicted_pitch_control_time` 모의 계산
```

의도:

```text
목표 포각 근처 접근 속도 증가
포각 제어 지연 예측과 실제 명령 일치
움직이는 표적 리드 시간 오차 감소
```

오버슈팅이 보이면 우선 `PITCH_MIN_WEIGHT = 0.055 → 0.050`으로 낮추고,
큰 오차 구간에서 지나치게 튀면 `PITCH_MAX_WEIGHT_SENSITIVE = 0.28 → 0.25`로
낮춘다.

## 14. 2026-06-24 X축 이동 보정 시작 조건 개선

실제 테스트 상태에서 적 탱크 속도가 `vx≈0`, `vz≈-2m/s`로 관측되어,
의도한 X축 좌우 이동이 아니라 Z축 전후 이동으로 시작되는 문제가 확인됐다.
기존 이동 서버는 `position.x`가 충분히 변한 뒤에야 `forward_x_sign`을 정하고
그때부터 X축 heading 보정을 시작했다. 그래서 처음부터 Z축으로 움직이면
`forward_x_sign = None` 상태가 유지되고 X축 보정이 늦게 걸렸다.

수정 후에는 X 변위 방향이 아직 감지되지 않았더라도, 이동 속도 벡터로 추정한
차체 yaw가 있으면 우선 +X 방향을 목표로 보정한다.

수식:

```text
if forward_x_sign is None:
    desired_body_yaw = +90°
else if forward_x_sign > 0:
    desired_body_yaw = +90°
else:
    desired_body_yaw = -90°

x_axis_heading_error = normalize(desired_body_yaw - estimated_body_yaw)
steer_weight = clamp(abs(x_axis_heading_error) × 0.015,
                     0.03,
                     0.20)
```

의도:

```text
초기 Z축 드리프트를 빠르게 X축 이동으로 전환
X 변위 감지 전에도 A/D 보정 활성화
이후에는 기존 ±10m X축 왕복 로직 유지
```

## 15. 2026-06-24 X축 정렬 우선 이동 게이트

X축 보정이 켜진 뒤에도 W/S 전진과 A/D 회전을 동시에 주면 적 탱크가
원호를 그리면서 대각선으로 움직일 수 있다. 실제 상태에서 `vx=-1.29`,
`vz=+1.95`처럼 Z 성분이 더 크게 남는 현상이 확인됐다.

그래서 X축 heading 오차가 큰 동안에는 전진/후진을 멈추고 차체 정렬을 먼저
수행한다. 정렬 오차가 작아진 뒤에만 W/S 이동을 허용한다.

추가 파라미터:

```text
X_AXIS_DRIVE_HEADING_TOLERANCE_DEG = 6.0
```

수식:

```text
if abs(x_axis_heading_error) > 6°:
    moveWS = STOP
    moveAD = A/D steering
else:
    moveWS = W or S
    moveAD = small A/D correction if needed
```

의도:

```text
대각선 이동보다 X축 정렬을 우선
초기 Z 드리프트 감소
예측 그래프에서 enemy_velocity_z를 작게 만들기
```

## 16. 2026-06-24 15발 검증용 적 체력 조건 우회

움직이는 표적 테스트에서 명중이 잘 되면 적 체력이 0 이하가 되어
기존 `enemyHealth <= 0` 조건으로 사격 로직이 멈출 수 있다. 15발 검증은
포각·예측·탄착 오차를 충분히 모으는 것이 목적이므로, moving target 모드에서만
체력 0 조건을 우회한다.

추가 파라미터:

```text
IGNORE_MOVING_TARGET_HEALTH_FOR_SHOT_LIMIT_TEST = True
```

조건:

```text
if targetType == "moving_enemy" and test flag is True:
    enemyHealth <= 0 이어도 위치가 있으면 계속 조준/발사
else:
    기존처럼 enemyHealth <= 0이면 정지
```

영향:

```text
moving target 15발 검증 가능
정적 장애물/일반 enemy dead 처리에는 영향 없음
```

## 17. 2026-06-24 X축 정렬 중 제자리 회전 방지

`X_AXIS_DRIVE_HEADING_TOLERANCE_DEG`보다 heading 오차가 클 때 `moveWS=STOP`으로
완전히 막으면 적 탱크가 제자리에서 회전만 하고 이동하지 않는 문제가 생길 수
있다. 사용자가 관측한 “고개만 도는” 현상이 이 케이스다.

수정 후에는 정렬 오차가 큰 동안에도 아주 낮은 전후진 weight를 유지한다.

추가 파라미터:

```text
ALIGN_DRIVE_WEIGHT = 0.08
```

변경 수식:

```text
if abs(x_axis_heading_error) > 6°:
    moveWS = W or S, weight = 0.08
    moveAD = A/D steering
    phase = x_axis_align_while_slow_drive
else:
    moveWS = W or S, weight = 0.20
    moveAD = small A/D correction if needed
```

의도:

```text
제자리 도리도리 방지
정렬 중에도 위치 변화 생성
velocity 기반 heading 추정이 계속 갱신되게 유지
```

## 18. 2026-06-24 이동 중 발사 조건 완화

이동 표적을 계속 따라가다가 방향 전환이나 감속 순간에만 발사되는 현상이
있었다. 원인은 정적 표적 기준의 발사 안정화 조건을 그대로 사용했기 때문이다.
기존 조건은 조준 정렬 상태가 `0.5초` 유지되어야 했는데, 움직이는 표적은
예측 조준점이 계속 변하므로 이동 중에는 이 조건을 만족하기 어렵고, 속도가
낮아지는 방향 전환 순간에만 발사되기 쉽다.

이동 표적 전용 발사 조건:

```text
MOVING_AIM_STABLE_SECONDS = 0.12
MOVING_TURRET_FIRE_TOLERANCE_DEG = 2.5
MOVING_BODY_FIRE_TOLERANCE_DEG = 14.0
MOVING_PITCH_TOLERANCE_MULTIPLIER = 1.25
```

정적 표적:

```text
aim_stable_time >= 0.5초
abs(turret_error) < 1.5°
abs(body_error) < 10°
abs(pitch_error) < pitch_tolerance
```

이동 표적:

```text
aim_stable_time >= 0.12초
abs(turret_error) < 2.5°
abs(body_error) < 14°
abs(pitch_error) < pitch_tolerance × 1.25
```

의도:

```text
방향 전환/감속 순간에만 발사되는 현상 완화
예측 조준점 기준으로 이동 중에도 발사 가능
정적 장애물 발사 조건은 기존처럼 보수적으로 유지
```

오차가 커지면 우선 `MOVING_TURRET_FIRE_TOLERANCE_DEG`를 `2.5 → 2.0`으로
낮추고, 너무 늦게 쏘면 `MOVING_AIM_STABLE_SECONDS`를 `0.12 → 0.08`로 낮춘다.

## 19. 2026-06-24 예측 선행거리 기반 포탑/포각 속도 부스트

움직이는 표적의 예측 조준점이 현재 관측 위치보다 멀리 앞에 생기면,
포탑과 포각도 그 예측 위치까지 더 빠르게 따라가야 한다. 기존 구조는
예측 위치를 계산하더라도 실제 yaw/pitch 제어 상한은 정적 표적과 거의 같아서,
계속 뒤쫓다가 방향 전환/감속 순간에만 정렬되는 문제가 생길 수 있었다.

그래서 예측 선행거리 `lead_distance`가 클수록 yaw/pitch 최대 weight를 올린다.

추가 파라미터:

```text
MOVING_LEAD_BOOST_FULL_DISTANCE_M = 5.0
MOVING_YAW_MAX_WEIGHT_BOOST = 0.40
MOVING_PITCH_MAX_WEIGHT_BOOST = 0.30
```

공통 부스트:

```text
lead_boost = clamp(lead_distance / 5.0, 0.0, 1.0)
```

Yaw 제어:

```text
base_max_turret_weight =
    0.12  if distance < 40m
    0.22  if distance < 90m
    0.32  otherwise

max_turret_weight =
    base_max_turret_weight × (1.0 + 0.40 × lead_boost)
```

Pitch 제어:

```text
base_max_pitch_weight =
    0.28  if abs(dR/dtheta) > 80
    0.22  otherwise

max_pitch_weight =
    base_max_pitch_weight × (1.0 + 0.30 × lead_boost)
```

예:

```text
lead_distance = 0m  → 기존 속도
lead_distance = 2.5m → yaw 최대 +20%, pitch 최대 +15%
lead_distance ≥ 5m → yaw 최대 +40%, pitch 최대 +30%
```

예측 시간 모델도 동일한 부스트를 사용한다. 즉 `predicted_pitch_control_time`과
`predicted_yaw_control_time`은 실제 명령보다 느린 구식 속도로 계산하지 않고,
lead 거리 기반으로 빨라진 제어 속도를 반영한다.

또한 이동 표적 예측식의 안정화 시간은 정적 표적용 `0.5초`가 아니라
이동 표적용 값으로 맞춘다.

```text
predicted_total_intercept_time =
    predicted_control_time
  + MOVING_AIM_STABLE_SECONDS
  + FIRE_SYSTEM_DELAY_SECONDS
  + predicted_flight_time
```

의도:

```text
예측 조준점이 멀 때 더 빠르게 포탑/포각 이동
현재 위치를 뒤따라가는 현상 완화
방향 전환/감속 순간에만 발사되는 패턴 감소
실제 제어 속도와 예측 제어시간 일치
```

오버슈팅이 보이면 `MOVING_YAW_MAX_WEIGHT_BOOST`를 `0.40 → 0.25`로 낮추고,
포각이 튀면 `MOVING_PITCH_MAX_WEIGHT_BOOST`를 `0.30 → 0.15`로 낮춘다.

## 20. 2026-06-24 적 탱크 X축 왕복 거리 확대

이동 표적이 좌우로 움직이는 폭이 너무 짧아 방향 전환이 자주 발생했고,
그 결과 방향 전환/감속 구간 위주로 사격 데이터가 쌓일 수 있었다. 더 긴
등속 이동 구간을 만들기 위해 X축 왕복 반경을 키웠다.

변경:

```text
X_HALF_SPAN_METERS = 10.0 → 20.0
```

수식:

```text
x_min = movement_center_x - 20m
x_max = movement_center_x + 20m
```

효과:

```text
기존 총 왕복 폭: 20m
변경 총 왕복 폭: 40m
```

의도:

```text
좌우 이동 구간을 길게 만들어 등속 추적 데이터 확보
방향 전환 빈도 감소
이동 중 예측 조준/발사 성능 검증 강화
```

## 21. 2026-06-24 논문식 이동 표적 미래 위치 예측 적용

참고 논문 `Angular Orientation of Anti-Aircraft Gun for Interception of a Moving Air Target`의
핵심은 현재 표적 위치가 아니라 탄이 도착할 미래 위치를 조준하는 것이다.
논문은 표적의 위치, 속도, 가속도와 탄의 비행시간을 이용해 충돌 시점의
표적 위치를 계산하고, 그 위치로 방위각/yaw와 고각/pitch를 구한다.

전차 시뮬레이터 좌표계 대응:

```text
논문 수평 X/Y  → 시뮬레이터 X/Z
논문 수직 Z    → 시뮬레이터 Y
bearing        → turret yaw
elevation      → turret pitch
```

기존 단순 리드:

```text
p_future = p_now + v × t
```

논문식 확장:

```text
p_future = p_now + v × t + 0.5 × a × t²
```

시뮬레이터 X/Z 평면에 적용한 수식:

```text
x_future = x_now + vx × t + 0.5 × ax × t²
z_future = z_now + vz × t + 0.5 × az × t²
```

속도 추정:

```text
raw_vx = (x_now - x_prev) / dt
raw_vz = (z_now - z_prev) / dt

vx = alpha_v × raw_vx + (1 - alpha_v) × vx_prev
vz = alpha_v × raw_vz + (1 - alpha_v) × vz_prev

alpha_v = ENEMY_VELOCITY_EMA_ALPHA = 0.35
```

가속도 추정:

```text
raw_ax = (raw_vx - vx_prev) / dt
raw_az = (raw_vz - vz_prev) / dt

ax = alpha_a × raw_ax + (1 - alpha_a) × ax_prev
az = alpha_a × raw_az + (1 - alpha_a) × az_prev

alpha_a = ENEMY_ACCEL_EMA_ALPHA = 0.25
```

노이즈 제한:

```text
ENEMY_MAX_VALID_SPEED_MPS = 25.0
ENEMY_MAX_VALID_ACCEL_MPS2 = 8.0
MAX_ACCEL_LEAD_DISTANCE_M = 8.0
```

반복형 intercept solver:

```text
for i in 1..INTERCEPT_SOLVER_ITERATIONS:
    1. 현재 예측 위치까지 yaw/pitch 제어 시간 계산
    2. 현재 예측 위치까지 탄도 비행시간 계산
    3. total_time =
           predicted_control_time
         + MOVING_AIM_STABLE_SECONDS
         + FIRE_SYSTEM_DELAY_SECONDS
         + predicted_flight_time
    4. p_future = p_now + v × total_time + 0.5 × a × total_time²
    5. p_future 기준으로 다시 반복
```

현재 반복 횟수:

```text
INTERCEPT_SOLVER_ITERATIONS = 8
```

공기저항/탄 감속:

논문은 탄의 평균 감속도와 공기저항을 포함하지만, 현재 시뮬레이터에서는
정확한 탄 감속 모델을 직접 알 수 없다. 따라서 탄 감속은 물리식으로 억지로
추정하지 않고 기존 로그 기반 보정값과 거리별 탄속 보간을 유지한다.

```text
effective_muzzle_speed(distance)
FLIGHT_TIME_CORRECTION_FACTOR
```

로그 추가:

```text
enemy_accel_x
enemy_accel_z
enemy_accel
intercept_solver = iterative_p_vt_half_at2
intercept_solver_iterations
max_accel_lead_distance
```

의도:

```text
등속 이동뿐 아니라 방향 전환 전후의 가속/감속도 일부 반영
현재 위치 추적이 아니라 논문식 미래 위치 조준으로 구조 정리
예측 시간, 탄 비행시간, 포탑/포각 제어시간을 반복적으로 맞춤
```

주의:

가속도는 위치 미분의 미분이라 노이즈가 크다. 가속도 리드가 튀면 먼저
`ENEMY_ACCEL_EMA_ALPHA = 0.25 → 0.15`로 낮추거나,
`MAX_ACCEL_LEAD_DISTANCE_M = 8.0 → 4.0`으로 줄인다.
