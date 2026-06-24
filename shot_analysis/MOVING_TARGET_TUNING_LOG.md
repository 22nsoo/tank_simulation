# Moving Target Tuning Log

## 2026-06-24 — 예측 정렬 기반 선제 발사 게이트 추가

시뮬 화면에서 포탑이 예측점을 향해 조준하더라도, 발사 조건이 현재 오차만
기준으로 되어 있으면 실제 동작은 표적 앞을 계속 따라가다가 방향 전환이나
오차가 우연히 작아지는 순간에만 발사하는 것처럼 보인다.

이번 변경은 발사 조건에 포탑/포각의 현재 제어 속도를 반영한다. 즉, 현재
포탑이 완전히 정렬될 때까지 기다리지 않고, 짧은 시간 뒤 포탑과 포각이
예측 조준점에 들어올 것으로 계산되면 미리 발사할 수 있게 했다.

### 적용 수식

포탑 yaw의 예상 각속도:

```text
body_yaw_rate = BODY_YAW_SPEED_PER_WEIGHT * body_signed_effort

turret_yaw_rate =
    TURRET_YAW_SPEED_PER_WEIGHT * turret_pd_effort
  + body_yaw_rate
```

포각 pitch의 예상 각속도:

```text
pitch_rate = PITCH_SPEED_PER_WEIGHT * pitch_signed_effort
```

짧은 미래 시점의 예상 조준 오차:

```text
t_prefire = MOVING_PREDICTIVE_FIRE_LOOKAHEAD_SECONDS = 0.18

predicted_turret_yaw_at_fire =
    current_turret_yaw + turret_yaw_rate * t_prefire

predicted_pitch_at_fire =
    current_pitch + pitch_rate * t_prefire

predicted_turret_error_at_fire =
    target_world_angle - predicted_turret_yaw_at_fire

predicted_pitch_error_at_fire =
    desired_pitch - predicted_pitch_at_fire
```

기존 직접 정렬 조건:

```text
aim_aligned =
    abs(turret_error) < turret_fire_tolerance
and abs(pitch_error) < effective_pitch_tolerance
```

새 예측 정렬 조건:

```text
predictive_aim_aligned =
    abs(turret_error) < MOVING_PREFIRE_TURRET_WINDOW_DEG
and abs(predicted_turret_error_at_fire) < turret_fire_tolerance
and abs(predicted_pitch_error_at_fire)
    < pitch_tolerance * MOVING_PREFIRE_PITCH_TOLERANCE_MULTIPLIER
```

최종 발사 정렬 조건:

```text
fire_alignment_ready = aim_aligned or predictive_aim_aligned
```

예측 정렬로 발사할 때는 표적이 계속 움직이므로 안정 대기 시간을 짧게 잡는다.

```text
MOVING_PREDICTIVE_FIRE_STABLE_SECONDS = 0.04
```

### 같이 조정한 값

`moving_target_plots_5` 분석에서 실제 비행시간이 예측 비행시간보다 평균적으로
짧게 나왔기 때문에 비행시간 보정값을 낮췄다.

```text
FLIGHT_TIME_CORRECTION_FACTOR: 1.096 -> 1.06
```

또한 적 속도 추정에서 순간 튐이 리드샷을 과하게 만들 수 있어서 유효 속도
상한을 낮췄다.

```text
ENEMY_MAX_VALID_SPEED_MPS: 25.0 -> 8.0
```

### 로그 확인 포인트

다음 그래프/CSV에서 아래 값을 보면 실제로 예측 발사 게이트가 작동했는지
확인할 수 있다.

```text
predictive_aim_aligned
fire_alignment_ready
predictive_fire_lookahead_seconds
predicted_turret_yaw_rate
predicted_turret_error_at_fire
predicted_pitch_rate
predicted_pitch_error_at_fire
```

### 기대 효과

- 예측점만 계속 따라가다가 방향 전환 때만 쏘는 현상 감소
- 포탑/포각이 곧 조준점에 들어올 것으로 계산되면 선제 발사
- 리드샷 계산과 실제 발사 타이밍 사이의 지연을 줄임

## 2026-06-24 — 표적 각속도 기반 포탑 추적 지연 보정

문제: 표적이 좌우로 움직일 때 포탑 yaw가 표적 각속도보다 느리거나 비슷하면,
예측점을 계산해도 포탑은 계속 뒤따라가는 모양이 된다. 기존 제어시간 예측은
`yaw_time = abs(turret_error) / yaw_speed`로 계산했기 때문에, 표적 조준각이
포탑이 도는 동안에도 계속 변한다는 사실이 빠져 있었다.

### 표적 각속도

전차 기준 상대 위치를 `dx, dz`, 표적 속도를 `vx, vz`라고 하면 수평 조준각의
각속도는 다음처럼 근사한다.

```text
target_angle_rate_rad_s = (dz * vx - dx * vz) / (dx^2 + dz^2)
target_angle_rate_deg_s = degrees(target_angle_rate_rad_s)
```

### 포탑이 실제로 오차를 줄이는 속도

표적 조준각이 포탑 회전 방향과 같은 방향으로 도망가면 실제 오차 감소 속도는
느려진다.

```text
turret_error_sign = sign(turret_error)

yaw_closing_rate =
    yaw_speed - turret_error_sign * target_angle_rate_deg_s
```

너무 작은 값으로 나뉘어 예측 시간이 폭주하지 않게 최소값을 둔다.

```text
MOVING_MIN_YAW_CLOSING_RATE_DEG_S = 2.5
```

새 yaw 제어시간:

```text
yaw_time = abs(turret_error) / yaw_closing_rate
```

이 값이 `predicted_control_time`에 들어가고, 최종적으로 다음 선행시간을 늘린다.

```text
total_intercept_time =
    predicted_control_time
  + MOVING_AIM_STABLE_SECONDS
  + FIRE_SYSTEM_DELAY_SECONDS
  + predicted_flight_time
```

### 같이 조정한 포탑 속도 부스트

표적이 움직이는 상황에서 예측 조준점이 더 멀리 앞에 있으면 포탑 최대 weight를
더 적극적으로 쓰도록 조정했다.

```text
MOVING_LEAD_BOOST_FULL_DISTANCE_M: 5.0 -> 4.0
MOVING_YAW_MAX_WEIGHT_BOOST: 0.40 -> 0.65
```

### 로그 확인 포인트

다음 moving target 로그에서 아래 컬럼을 확인한다.

```text
target_angle_rate_deg_s
yaw_closing_rate_deg_s
predicted_yaw_control_time
lead_distance
turret_error_fire
predicted_turret_error_at_fire
```

`target_angle_rate_deg_s`가 크고 `yaw_closing_rate_deg_s`가 작으면 포탑이 표적
각속도를 따라가기 버거운 상태다. 이때는 lead distance와 predicted control time이
늘어나는 것이 정상이다.
