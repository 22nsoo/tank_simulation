from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "moving_target_logs"
PLOT_DPI = 180


def numbered_path(prefix, index, suffix=".csv"):
    return LOG_DIR / f"{prefix}_{index}{suffix}"


def latest_log_pair():
    indices = []
    for path in LOG_DIR.glob("moving_shot_log_*.csv"):
        match = re.fullmatch(r"moving_shot_log_(\d+)\.csv", path.name)
        if match:
            indices.append(int(match.group(1)))

    for index in sorted(indices, reverse=True):
        shot_path = numbered_path("moving_shot_log", index)
        control_path = numbered_path("moving_control_log", index)
        if not control_path.exists():
            continue
        try:
            shot = pd.read_csv(shot_path)
            control = pd.read_csv(control_path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not shot.empty and not control.empty:
            return index, shot_path, control_path, shot, control

    raise FileNotFoundError(
        f"No matching moving shot/control logs with data in {LOG_DIR}"
    )


def next_output_directory():
    base = BASE_DIR / "moving_target_plots"
    if not base.exists():
        return base
    index = 1
    while (BASE_DIR / f"moving_target_plots_{index}").exists():
        index += 1
    return BASE_DIR / f"moving_target_plots_{index}"


def numeric(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def command(frame, column, mapping):
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return (
        frame[column]
        .fillna("STOP")
        .astype(str)
        .str.upper()
        .map(mapping)
        .fillna(0.0)
    )


def mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def finite(value):
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def style(axis, title, xlabel=None, ylabel=None):
    axis.set_title(title, fontweight="bold")
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


run_index, shot_path, control_path, shots, control = latest_log_pair()
OUT_DIR = next_output_directory()
OUT_DIR.mkdir(parents=True, exist_ok=False)

shot_id = numeric(shots, "shot_id")
hit = shots.get("hit", pd.Series("unknown", index=shots.index)).fillna("unknown")
hit_lower = hit.astype(str).str.lower()
hit_mask = hit_lower.str.contains("enemy|tank", regex=True)

observed_x = numeric(shots, "observed_enemy_x_fire")
observed_z = numeric(shots, "observed_enemy_z_fire")
predicted_x = numeric(shots, "enemy_x_fire")
predicted_z = numeric(shots, "enemy_z_fire")
impact_x = numeric(shots, "impact_x")
impact_z = numeric(shots, "impact_z")
player_x = numeric(shots, "player_x_fire")
player_z = numeric(shots, "player_z_fire")

enemy_speed = numeric(shots, "enemy_speed_fire")
lead_distance = numeric(shots, "lead_distance_fire")
control_time = numeric(shots, "predicted_control_time_fire")
yaw_time = numeric(shots, "predicted_yaw_control_time_fire")
pitch_time = numeric(shots, "predicted_pitch_control_time_fire")
flight_time_predicted = numeric(shots, "predicted_flight_time_fire")
intercept_time = numeric(shots, "predicted_total_intercept_time_fire")
flight_time_actual = numeric(shots, "flight_time_sec")

impact_3d = numeric(shots, "impact_error_to_enemy_fire")
forward_error = numeric(shots, "forward_error_fire")
lateral_error = numeric(shots, "lateral_error_fire")
vertical_error = numeric(shots, "vertical_error_fire")
turret_error_fire = numeric(shots, "turret_error_fire")
pitch_error_fire = numeric(shots, "pitch_error_fire")

# 00. Dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Moving Target Dashboard — run {run_index}",
    fontsize=17,
    fontweight="bold",
)
ax = axes[0, 0]
colors = np.where(hit_mask, "#2ca02c", "#d62728")
ax.bar(shot_id, lead_distance, color=colors)
style(ax, "Lead distance at fire (green=enemy hit)", "Shot ID", "Lead (m)")

ax = axes[0, 1]
ax.plot(shot_id, enemy_speed, "o-", label="Estimated enemy speed")
style(ax, "Enemy speed at fire", "Shot ID", "Speed (m/s)")
ax.legend()

ax = axes[1, 0]
ax.plot(shot_id, control_time, "o-", label="Aim control")
ax.plot(shot_id, flight_time_predicted, "o-", label="Predicted flight")
ax.plot(shot_id, intercept_time, "o-", label="Total intercept")
style(ax, "Prediction time components", "Shot ID", "Time (s)")
ax.legend()

ax = axes[1, 1]
ax.axhline(0, color="black", linewidth=1)
ax.plot(shot_id, forward_error, "o-", label="Forward")
ax.plot(shot_id, lateral_error, "o-", label="Lateral")
ax.plot(shot_id, vertical_error, "o-", label="Vertical")
style(ax, "Impact error around predicted aim point", "Shot ID", "Error (m)")
ax.legend()
save(fig, "00_moving_target_dashboard.png")

# 01. XZ geometry
fig, ax = plt.subplots(figsize=(13, 10))
ax.scatter(player_x, player_z, marker="^", s=110, label="Shooter at fire")
ax.scatter(observed_x, observed_z, s=80, label="Observed enemy")
ax.scatter(predicted_x, predicted_z, marker="X", s=95, label="Predicted aim")
ax.scatter(impact_x, impact_z, marker="*", s=130, label="Impact")
for i in shots.index:
    if all(
        np.isfinite(value)
        for value in [observed_x[i], observed_z[i], predicted_x[i], predicted_z[i]]
    ):
        ax.arrow(
            observed_x[i],
            observed_z[i],
            predicted_x[i] - observed_x[i],
            predicted_z[i] - observed_z[i],
            width=0.02,
            head_width=0.45,
            length_includes_head=True,
            alpha=0.55,
            color="#9467bd",
        )
        ax.annotate(str(int(shot_id[i])), (predicted_x[i], predicted_z[i]))
style(ax, "Observed enemy → predicted aim → impact", "World X (m)", "World Z (m)")
ax.axis("equal")
ax.legend()
save(fig, "01_intercept_geometry_xz.png")

# 02. Control timeline
x = np.arange(len(control))
c_speed = numeric(control, "enemy_speed")
c_lead = numeric(control, "lead_distance")
c_total = numeric(control, "predicted_total_intercept_time")
c_control = numeric(control, "predicted_control_time")
c_flight = numeric(control, "predicted_flight_time")
c_fire = control.get("fire", pd.Series(False, index=control.index)).astype(str).str.lower().eq("true")

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle("Moving Target Prediction Timeline", fontsize=17, fontweight="bold")
axes[0].plot(x, c_speed, label="Enemy speed")
style(axes[0], "Estimated target speed", ylabel="m/s")
axes[0].legend()
axes[1].plot(x, c_lead, label="Lead distance", color="#9467bd")
style(axes[1], "Predicted lead distance", ylabel="m")
axes[1].legend()
axes[2].plot(x, c_control, label="Control time")
axes[2].plot(x, c_flight, label="Flight time")
axes[2].plot(x, c_total, label="Total intercept time", linewidth=2)
axes[2].scatter(x[c_fire], c_total[c_fire], color="red", marker="*", s=120, label="Fire")
style(axes[2], "Control + stabilization + flight", "Control sample", "s")
axes[2].legend()
save(fig, "02_prediction_timeline.png")

# 03. Predicted control decomposition
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
axes[0].plot(shot_id, yaw_time, "o-", label="Yaw control")
axes[0].plot(shot_id, pitch_time, "o-", label="Pitch control")
axes[0].plot(shot_id, control_time, "o-", linewidth=2, label="Selected control")
style(axes[0], "Predicted aim-control delay at fire", ylabel="Time (s)")
axes[0].legend()
axes[1].plot(shot_id, flight_time_predicted, "o-", label="Predicted flight")
axes[1].plot(shot_id, flight_time_actual, "o-", label="Measured flight")
style(axes[1], "Predicted vs measured flight time", "Shot ID", "Time (s)")
axes[1].legend()
save(fig, "03_control_and_flight_time.png")

# 04. Aim controller and command weights
c_turret_error = numeric(control, "turret_error")
c_pitch_error = numeric(control, "pitch_error")
c_qe_weight = numeric(control, "turret_qe_weight").fillna(0.0)
c_rf_weight = numeric(control, "turret_rf_weight").fillna(0.0)
c_qe_cmd = command(control, "turret_qe_command", {"Q": -1, "E": 1, "STOP": 0})
c_rf_cmd = command(control, "turret_rf_command", {"F": -1, "R": 1, "STOP": 0})

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].plot(x, c_turret_error, label="Turret yaw error")
style(axes[0], "Yaw tracking error", ylabel="deg")
axes[0].legend()
axes[1].axhline(0, color="black", linewidth=1)
axes[1].plot(x, c_pitch_error, label="Pitch error", color="#ff7f0e")
style(axes[1], "Pitch tracking error", ylabel="deg")
axes[1].legend()
axes[2].axhline(0, color="black", linewidth=1)
axes[2].plot(x, c_qe_cmd * c_qe_weight, label="Signed Q/E")
axes[2].plot(x, c_rf_cmd * c_rf_weight, label="Signed F/R")
style(axes[2], "Controller command weights", "Control sample", "Signed weight")
axes[2].legend()
save(fig, "04_aim_control_commands.png")

# 05. Shot outcome and impact errors
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
axes[0].bar(shot_id, impact_3d, color=colors)
style(axes[0], "3D impact error (green=enemy hit)", ylabel="Error (m)")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].plot(shot_id, forward_error, "o-", label="Forward")
axes[1].plot(shot_id, lateral_error, "o-", label="Lateral")
axes[1].plot(shot_id, vertical_error, "o-", label="Vertical")
style(axes[1], "Directional impact errors", "Shot ID", "Error (m)")
axes[1].legend()
save(fig, "05_shot_outcome_and_errors.png")

# 06. Formula sheet
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
formula_text = (
    "Moving-target intercept model\n\n"
    r"$v_k=\alpha(p_k-p_{k-1})/\Delta t+(1-\alpha)v_{k-1}$" "\n\n"
    r"$t_{control}=1.20\max(t_{yaw},t_{pitch})$" "\n\n"
    r"$t_{intercept}=t_{control}+t_{stable}+t_{delay}+t_{flight}$" "\n\n"
    r"$p_{aim}=p_{enemy}+v_{enemy}t_{intercept}$" "\n\n"
    r"$t_{flight}=R/(v_{muzzle}\cos\theta)$" "\n\n"
    r"$t_{yaw}=|e_{yaw}|/"
    r"(43.498w_{turret}+37.254w_{body})$" "\n\n"
    r"$\dot{\theta}_{pitch}=4.562w_{pitch}\ [deg/s]$"
)
ax.text(0.5, 0.54, formula_text, ha="center", va="center", fontsize=20)
ax.text(
    0.5,
    0.07,
    "Limits: control ≤ 15 s, flight ≤ 4 s, lead distance ≤ 35 m",
    ha="center",
    fontsize=13,
    color="#555555",
)
save(fig, "06_intercept_formula.png")

hit_count = int(hit_mask.sum())
miss_count = int(len(shots) - hit_count)
summary = [
    "# Moving Target Plot Portfolio",
    "",
    f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"- Run index: {run_index}",
    f"- Shot source: `{shot_path.name}`",
    f"- Control source: `{control_path.name}`",
    f"- Logged impacts: {len(shots)}",
    f"- Enemy/tank hits: {hit_count}",
    f"- Other impacts: {miss_count}",
    f"- Mean enemy speed at fire: {finite(mean(enemy_speed))} m/s",
    f"- Mean lead distance: {finite(mean(lead_distance))} m",
    f"- Mean predicted control time: {finite(mean(control_time))} s",
    f"- Mean predicted flight time: {finite(mean(flight_time_predicted))} s",
    f"- Mean total intercept time: {finite(mean(intercept_time))} s",
    f"- Mean 3D error to predicted aim point: {finite(mean(impact_3d))} m",
    "",
    "## Interpretation note",
    "",
    "The log does not contain enemy position at impact time. Therefore the plots "
    "can compare observed position, predicted aim point, and impact point, but "
    "cannot directly calculate prediction error against the enemy's actual "
    "impact-time position.",
    "",
    "## Generated images",
    "",
]
summary.extend(f"- `{path.name}`" for path in sorted(OUT_DIR.glob("*.png")))
(OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

print(f"[SHOT SOURCE] {shot_path}")
print(f"[CONTROL SOURCE] {control_path}")
print(f"[SHOTS] {len(shots)}")
print(f"[ENEMY HITS] {hit_count}")
print(f"[OUTPUT] {OUT_DIR}")
for path in sorted(OUT_DIR.glob("*.png")):
    print(path.name)
