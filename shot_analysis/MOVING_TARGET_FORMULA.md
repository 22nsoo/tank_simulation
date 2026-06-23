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
