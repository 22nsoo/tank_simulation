from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def next_plot_directory():
    base_name = "shot_plots"
    base_path = BASE_DIR / base_name

    if not base_path.exists():
        return base_path

    index = 1
    while True:
        candidate = BASE_DIR / f"{base_name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


OUT_DIR = next_plot_directory()
OUT_DIR.mkdir(parents=True, exist_ok=False)

PLOT_DPI = 180
SIM_MIN_PITCH = -5.0
SIM_MAX_PITCH = 10.0


def latest_csv_with_data():
    """Return the newest shot_log_*.csv containing at least one data row."""
    candidates = sorted(
        BASE_DIR.glob("shot_log_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue

        if not frame.empty:
            return path, frame

    raise FileNotFoundError(
        f"No shot_log_*.csv containing shot data was found in {BASE_DIR}"
    )


def latest_control_csv_with_data():
    candidates = sorted(
        BASE_DIR.glob("control_log_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if not frame.empty:
            return path, frame
    return None, None


def numeric(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def first_available(frame, *columns):
    for column in columns:
        if column in frame.columns:
            values = numeric(frame, column)
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype=float)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def style_axis(axis, title, xlabel="Shot ID", ylabel=None):
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)


def finite_mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


csv_path, df = latest_csv_with_data()

# Keep raw measurements untouched and build analysis columns separately.
shot_id = first_available(df, "shot_id")
player_x = first_available(df, "player_x_fire")
player_z = first_available(df, "player_z_fire")
enemy_x = first_available(df, "enemy_x_fire")
enemy_y = first_available(df, "enemy_y_fire")
enemy_z = first_available(df, "enemy_z_fire")
impact_x = first_available(df, "impact_x")
impact_y = first_available(df, "impact_y")
impact_z = first_available(df, "impact_z")

target_range = first_available(df, "tune_target_range", "distance_fire", "ballistic_R_fire")
impact_range_logged = first_available(df, "tune_impact_range")
impact_range_calculated = np.sqrt((impact_x - player_x) ** 2 + (impact_z - player_z) ** 2)
impact_range = impact_range_logged.where(
    impact_range_logged.notna(), impact_range_calculated
)

range_error = first_available(df, "range_error_fire", "tune_range_error")
range_error = range_error.where(range_error.notna(), target_range - impact_range)

z_shortfall = first_available(df, "z_shortfall_fire")
z_shortfall = z_shortfall.where(z_shortfall.notna(), enemy_z - impact_z)

impact_error_3d = first_available(df, "impact_error_to_enemy_fire")
impact_error_calculated = np.sqrt(
    (impact_x - enemy_x) ** 2
    + (impact_y - enemy_y) ** 2
    + (impact_z - enemy_z) ** 2
)
impact_error_3d = impact_error_3d.where(
    impact_error_3d.notna(), impact_error_calculated
)

actual_pitch = first_available(df, "player_turret_pitch_fire")
desired_pitch = first_available(df, "desired_pitch_fire")
pitch_error = first_available(df, "pitch_error_fire")
pitch_tolerance = first_available(df, "pitch_fire_tolerance")
theta_raw = first_available(
    df, "theta_raw_deg_fire", "theta_physical_deg_raw_fire"
)
theta_with_bias = first_available(
    df, "theta_with_bias_deg_fire", "theta_physical_deg_with_bias_fire"
)
pitch_bias = first_available(df, "pitch_bias_deg_fire", "tune_old_pitch_bias_deg")
new_pitch_bias = first_available(df, "tune_new_pitch_bias_deg")
delta_theta = first_available(df, "tune_delta_theta_deg")

body_error = first_available(df, "body_error_fire")
turret_error = first_available(df, "turret_error_fire")
flight_time = first_available(df, "flight_time_sec")

turret_qe_weight = first_available(df, "action_turretQE_weight")
turret_rf_weight = first_available(df, "action_turretRF_weight")
move_ad_weight = first_available(df, "action_moveAD_weight")
move_ws_weight = first_available(df, "action_moveWS_weight")


def command_series(frame, column, mapping):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return (
        frame[column]
        .fillna("STOP")
        .astype(str)
        .str.upper()
        .map(mapping)
        .astype(float)
    )


pitch_command = command_series(
    df,
    "action_turretRF_command",
    {"F": -1.0, "STOP": 0.0, "R": 1.0},
)
signed_pitch_weight = pitch_command * turret_rf_weight.fillna(0.0)

valid_impacts = impact_x.notna() & impact_z.notna()
valid_count = int(valid_impacts.sum())
hit_text = (
    df["hit"].fillna("unknown").astype(str).str.lower()
    if "hit" in df.columns
    else pd.Series("unknown", index=df.index)
)
enemy_hit_count = int(
    hit_text.isin(["enemy", "tank", "enemy_tank", "target", "hit"]).sum()
)

# ---------------------------------------------------------------------------
# 00. Summary dashboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Shot Analysis Dashboard — {csv_path.name}",
    fontsize=17,
    fontweight="bold",
)

ax = axes[0, 0]
ax.plot(shot_id, target_range, "o-", label="Target range", linewidth=2)
ax.plot(shot_id, impact_range, "o-", label="Impact range", linewidth=2)
style_axis(ax, "Target vs impact range", ylabel="Range (m)")
ax.legend()

ax = axes[0, 1]
ax.axhline(0, color="black", linewidth=1)
ax.plot(shot_id, range_error, "o-", color="#d62728", label="Range error")
style_axis(ax, "Range error (+ means short)", ylabel="Error (m)")
ax.legend()

ax = axes[1, 0]
ax.plot(shot_id, actual_pitch, "o-", label="Actual pitch")
ax.plot(shot_id, desired_pitch, "o-", label="Desired pitch")
ax.plot(shot_id, theta_with_bias, "o--", label="Ballistic theta + bias")
ax.axhline(SIM_MIN_PITCH, color="gray", linestyle=":", label="Pitch limits")
ax.axhline(SIM_MAX_PITCH, color="gray", linestyle=":")
style_axis(ax, "Pitch tracking", ylabel="Angle (deg)")
ax.legend()

ax = axes[1, 1]
ax.axis("off")
summary_lines = [
    f"Source CSV: {csv_path.name}",
    f"Rows / valid impacts: {len(df)} / {valid_count}",
    f"Enemy hits: {enemy_hit_count}",
    f"Mean target range: {finite_mean(target_range):.3f} m",
    f"Mean impact range: {finite_mean(impact_range):.3f} m",
    f"Mean range error: {finite_mean(range_error):.3f} m",
    f"Mean 3D impact error: {finite_mean(impact_error_3d):.3f} m",
    f"Mean flight time: {finite_mean(flight_time):.3f} s",
    f"Mean actual pitch: {finite_mean(actual_pitch):.3f} deg",
    f"Latest pitch bias: {new_pitch_bias.dropna().iloc[-1]:.3f} deg"
    if new_pitch_bias.notna().any()
    else "Latest pitch bias: n/a",
]
ax.text(
    0.03,
    0.97,
    "\n".join(summary_lines),
    va="top",
    family="monospace",
    fontsize=12,
    bbox={"boxstyle": "round", "facecolor": "#f4f6f8", "edgecolor": "#aab2bd"},
)
save_figure(fig, "00_shot_dashboard.png")

# ---------------------------------------------------------------------------
# 01. X-Z trajectory geometry
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(player_x, player_z, s=90, marker="s", label="Player at fire")
ax.scatter(enemy_x, enemy_z, s=100, marker="X", label="Enemy at fire")
ax.scatter(impact_x, impact_z, s=70, marker="o", label="Impact")
for idx in df.index[valid_impacts]:
    ax.plot(
        [player_x.loc[idx], impact_x.loc[idx]],
        [player_z.loc[idx], impact_z.loc[idx]],
        color="#1f77b4",
        alpha=0.25,
    )
style_axis(ax, "Top-down X-Z shot geometry", xlabel="World X (m)", ylabel="World Z (m)")
ax.axis("equal")
ax.legend()
save_figure(fig, "01_xz_trajectory.png")

# ---------------------------------------------------------------------------
# 02. Target range and actual impact range
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(shot_id, target_range, "o-", linewidth=2, label="Target range")
ax.plot(shot_id, impact_range, "o-", linewidth=2, label="Impact range")
style_axis(ax, "Target range vs actual impact range", ylabel="Range (m)")
ax.legend()
save_figure(fig, "02_range_target_vs_impact.png")

# ---------------------------------------------------------------------------
# 03. Range, Z and 3D errors
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
series = [
    (range_error, "Range error (+ means short)", "Range error (m)", "#d62728"),
    (z_shortfall, "Z shortfall (+ means short)", "Z shortfall (m)", "#ff7f0e"),
    (impact_error_3d, "3D impact error", "3D error (m)", "#9467bd"),
]
for ax, (values, title, ylabel, color) in zip(axes, series):
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(shot_id, values, "o-", color=color)
    style_axis(ax, title, ylabel=ylabel)
save_figure(fig, "03_impact_errors.png")

# ---------------------------------------------------------------------------
# 04. Pitch calculation and tracking
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
ax = axes[0]
ax.plot(shot_id, actual_pitch, "o-", label="Actual turret pitch")
ax.plot(shot_id, desired_pitch, "o-", label="Desired simulator pitch")
ax.plot(shot_id, theta_raw, "o--", label="Raw ballistic theta")
ax.plot(shot_id, theta_with_bias, "o--", label="Theta with bias")
ax.axhspan(
    SIM_MIN_PITCH,
    SIM_MAX_PITCH,
    color="gray",
    alpha=0.08,
    label="Observed simulator pitch range",
)
style_axis(ax, "Ballistic and simulator pitch", ylabel="Angle (deg)")
ax.legend(ncol=2)

ax = axes[1]
ax.axhline(0, color="black", linewidth=1)
ax.plot(shot_id, pitch_error, "o-", label="Pitch error")
ax.plot(shot_id, pitch_tolerance, "--", label="+ tolerance")
ax.plot(shot_id, -pitch_tolerance, "--", label="- tolerance")
style_axis(ax, "Pitch error vs fire tolerance", ylabel="Angle (deg)")
ax.legend()
save_figure(fig, "04_pitch_diagnostics.png")

# ---------------------------------------------------------------------------
# 05. Yaw/body aiming errors
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(0, color="black", linewidth=1)
ax.axhline(1.5, color="gray", linestyle=":", label="Fire yaw limit ±1.5°")
ax.axhline(-1.5, color="gray", linestyle=":")
ax.plot(shot_id, turret_error, "o-", label="Turret yaw error")
ax.plot(shot_id, body_error, "o-", label="Body yaw error")
style_axis(ax, "Yaw alignment at fire", ylabel="Angle error (deg)")
ax.legend()
save_figure(fig, "05_yaw_alignment.png")

# ---------------------------------------------------------------------------
# 06. Bias tuning history
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
ax = axes[0]
ax.plot(shot_id, pitch_bias, "o-", label="Bias used")
ax.plot(shot_id, new_pitch_bias, "o-", label="Bias after impact")
style_axis(ax, "Pitch bias update history", ylabel="Bias (deg)")
ax.legend()

ax = axes[1]
ax.axhline(0, color="black", linewidth=1)
ax.bar(shot_id, delta_theta, color="#2ca02c", label="Bias update per shot")
style_axis(ax, "Per-shot pitch correction", ylabel="Delta pitch (deg)")
ax.legend()
save_figure(fig, "06_bias_tuning.png")

# ---------------------------------------------------------------------------
# 07. Flight time and impact error
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(shot_id, flight_time, "o-", color="#1f77b4", label="Flight time")
ax1.set_xlabel("Shot ID")
ax1.set_ylabel("Flight time (s)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(True, alpha=0.28)

ax2 = ax1.twinx()
ax2.plot(
    shot_id,
    impact_error_3d,
    "s--",
    color="#d62728",
    label="3D impact error",
)
ax2.set_ylabel("3D impact error (m)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title("Flight time and impact error", fontweight="bold")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], loc="best")
save_figure(fig, "07_flight_time_and_error.png")

# ---------------------------------------------------------------------------
# 08. Control weights
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(shot_id, turret_qe_weight, "o-", label="Turret Q/E")
ax.plot(shot_id, turret_rf_weight, "o-", label="Turret R/F")
ax.plot(shot_id, move_ad_weight, "o-", label="Body A/D")
ax.plot(shot_id, move_ws_weight, "o-", label="Move W/S")
style_axis(ax, "Control command weights at fire", ylabel="Weight")
ax.legend(ncol=2)
save_figure(fig, "08_control_weights.png")

# ---------------------------------------------------------------------------
# 09. Formula reference image
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 11), facecolor="white")
fig.suptitle(
    "Formula Reference — fire_logic.py",
    fontsize=20,
    fontweight="bold",
    y=0.97,
)
formula_text = "\n".join(
    [
        r"$\Delta x=x_e-x_p,\quad \Delta z=z_e-z_p$",
        r"$R=\sqrt{(\Delta x)^2+(\Delta z)^2}$",
        r"$\psi_{target}=\mathrm{atan2}(\Delta x,\Delta z)$",
        "",
        r"$\Delta y=y_e-y_p$",
        r"$D=v^4-g(gR^2+2\Delta yv^2)$",
        r"$\theta=\tan^{-1}\left(\frac{v^2-\sqrt{D}}{gR}\right)$",
        r"$\theta_{bias}=\mathrm{clamp}(\theta+b,-10^\circ,35^\circ)$",
        r"$pitch_{target}=\mathrm{clamp}(offset+s\theta_{bias},-5^\circ,10^\circ)$",
        "",
        r"$R_{flat}(\theta)=\frac{v^2}{g}\sin(2\theta)$",
        r"$\frac{dR}{d\theta}\approx"
        r"\frac{R_{flat}(\theta+\epsilon)-R_{flat}(\theta-\epsilon)}{2\epsilon}$",
        r"$tol_{pitch}=\mathrm{clamp}\left("
        r"\mathrm{deg}\left(\frac{3\,m}{|dR/d\theta|}\right),0.08^\circ,0.5^\circ"
        r"\right)$",
        "",
        r"$e_R=R_{target}-R_{impact}$",
        r"$\Delta\theta= s\cdot\mathrm{deg}\left("
        r"K\frac{e_R}{dR/d\theta}\right)$",
        r"$b_{new}=\mathrm{clamp}(b+\Delta\theta,-10^\circ,15^\circ)$",
    ]
)
fig.text(0.07, 0.89, formula_text, va="top", fontsize=17, linespacing=1.5)

constants = [
    "Current constants",
    "g = 9.81 m/s²",
    "v = 45.0 m/s (estimated)",
    "s = PITCH_CORRECTION_SIGN = +1",
    "K = PITCH_TUNE_GAIN = 0.35",
    "Pitch limits = -5° to +10°",
    "Fire range = 20m to 200m",
    "Yaw fire tolerance = ±1.5°",
    "Cooldown = 1.0s",
]
fig.text(
    0.69,
    0.87,
    "\n".join(constants),
    va="top",
    fontsize=14,
    family="monospace",
    bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f4f6f8", "edgecolor": "#7f8c8d"},
)

fire_gate = [
    "Fire gate (all conditions required)",
    "20 < distance < 200",
    "|turret_error| < 1.5°",
    "|pitch_error| < pitch_tolerance",
    "elapsed time > cooldown",
]
fig.text(
    0.69,
    0.52,
    "\n".join(fire_gate),
    va="top",
    fontsize=14,
    family="monospace",
    bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fff4e6", "edgecolor": "#e67e22"},
)
save_figure(fig, "09_formula_reference.png")

# ---------------------------------------------------------------------------
# 10. Pitch control command diagnostics
# ---------------------------------------------------------------------------
control_csv_path, control_df = latest_control_csv_with_data()

if control_df is not None:
    control_x = np.arange(len(control_df))
    control_pitch_error = numeric(control_df, "pitch_error")
    control_pitch_tolerance = numeric(control_df, "pitch_tolerance")
    control_pitch_weight = numeric(control_df, "turret_rf_weight").fillna(0.0)
    control_pitch_command = command_series(
        control_df,
        "turret_rf_command",
        {"F": -1.0, "STOP": 0.0, "R": 1.0},
    )
    control_signed_weight = control_pitch_command * control_pitch_weight
    command_source_name = control_csv_path.name
    command_source_note = (
        "Every /get_action frame from control_log; x-axis is control sample index."
    )
else:
    control_x = shot_id
    control_pitch_error = pitch_error
    control_pitch_tolerance = pitch_tolerance
    control_pitch_command = pitch_command
    control_signed_weight = signed_pitch_weight
    command_source_name = csv_path.name
    command_source_note = (
        "Fallback: shot_log command snapshot at fire/impact logging time."
    )

fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
fig.suptitle(
    f"Pitch Control Command Timeline — {command_source_name}",
    fontsize=16,
    fontweight="bold",
)

ax = axes[0]
ax.axhline(0, color="black", linewidth=1)
ax.plot(control_x, control_pitch_error, linewidth=1.5, label="Pitch error")
ax.plot(control_x, control_pitch_tolerance, "--", label="+ fire tolerance")
ax.plot(control_x, -control_pitch_tolerance, "--", label="- fire tolerance")
style_axis(ax, "Pitch error used by the controller", ylabel="Angle (deg)")
ax.legend()

ax = axes[1]
ax.step(
    control_x,
    control_pitch_command,
    where="mid",
    linewidth=1.8,
    label="R/F command",
)
ax.set_yticks([-1, 0, 1], labels=["F (-1)", "STOP (0)", "R (+1)"])
ax.set_ylim(-1.4, 1.4)
style_axis(ax, "Pitch command direction", ylabel="Command")
ax.legend()

ax = axes[2]
ax.axhline(0, color="black", linewidth=1)
ax.plot(
    control_x,
    control_signed_weight,
    linewidth=1.5,
    color="#9467bd",
    label="Signed R/F weight",
)
style_axis(
    ax,
    "Signed pitch command weight (F negative, R positive)",
    ylabel="Signed weight",
)
ax.legend()

fig.text(
    0.5,
    0.01,
    command_source_note,
    ha="center",
    fontsize=10,
    color="#555555",
)
fig.subplots_adjust(bottom=0.06)
save_figure(fig, "10_pitch_command_diagnostics.png")

# ---------------------------------------------------------------------------
# 11. Yaw/body rotation overshoot diagnostics
# ---------------------------------------------------------------------------
if control_df is not None:
    yaw_x = np.arange(len(control_df))
    control_turret_error = numeric(control_df, "turret_error")
    control_body_error = numeric(control_df, "body_error")
    control_qe_weight = numeric(control_df, "turret_qe_weight").fillna(0.0)
    control_ad_weight = numeric(control_df, "move_ad_weight").fillna(0.0)
    control_qe_command = command_series(
        control_df,
        "turret_qe_command",
        {"Q": -1.0, "STOP": 0.0, "E": 1.0},
    )
    control_ad_command = command_series(
        control_df,
        "move_ad_command",
        {"A": -1.0, "STOP": 0.0, "D": 1.0},
    )
    signed_qe_weight = control_qe_command * control_qe_weight
    signed_ad_weight = control_ad_command * control_ad_weight

    previous_error = control_turret_error.shift(1)
    overshoot_mask = (
        control_turret_error.notna()
        & previous_error.notna()
        & (control_turret_error * previous_error < 0)
        & (control_turret_error.abs() >= 1.5)
    )
    overshoot_x = yaw_x[overshoot_mask.to_numpy()]
    overshoot_y = control_turret_error[overshoot_mask]

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(
        f"Yaw / Body Rotation Overshoot — {control_csv_path.name}",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0]
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(1.5, color="gray", linestyle=":", label="Fire yaw limit ±1.5°")
    ax.axhline(-1.5, color="gray", linestyle=":")
    ax.plot(yaw_x, control_turret_error, linewidth=1.6, label="Turret yaw error")
    ax.scatter(
        overshoot_x,
        overshoot_y,
        color="red",
        marker="X",
        s=75,
        label="Overshoot sign crossing",
        zorder=5,
    )
    style_axis(ax, "Turret yaw error and overshoot", ylabel="Yaw error (deg)")
    ax.legend()

    ax = axes[1]
    ax.step(
        yaw_x,
        control_qe_command,
        where="mid",
        linewidth=1.7,
        label="Q/E command",
    )
    ax.set_yticks([-1, 0, 1], labels=["Q (-1)", "STOP (0)", "E (+1)"])
    ax.set_ylim(-1.4, 1.4)
    style_axis(ax, "Turret rotation command", ylabel="Command")
    ax.legend()

    ax = axes[2]
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(yaw_x, control_body_error, linewidth=1.6, label="Body yaw error")
    ax.step(
        yaw_x,
        signed_ad_weight * 100.0,
        where="mid",
        linewidth=1.3,
        label="Signed A/D weight ×100",
    )
    style_axis(
        ax,
        "Body error and A/D rotation contribution",
        ylabel="Angle / scaled weight",
    )
    ax.legend()

    ax = axes[3]
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(
        yaw_x,
        signed_qe_weight,
        linewidth=1.6,
        label="Signed Q/E weight",
    )
    ax.plot(
        yaw_x,
        signed_ad_weight,
        linewidth=1.6,
        label="Signed A/D weight",
    )
    both_turning = (
        (control_qe_command != 0)
        & (control_ad_command != 0)
        & (np.sign(control_qe_command) == np.sign(control_ad_command))
    )
    ax.fill_between(
        yaw_x,
        -0.02,
        0.02,
        where=both_turning.to_numpy(),
        color="#ffcc00",
        alpha=0.45,
        label="Body + turret same direction",
    )
    style_axis(
        ax,
        "Combined yaw command weights",
        xlabel="Control sample index",
        ylabel="Signed weight",
    )
    ax.legend(ncol=3)

    fig.text(
        0.5,
        0.008,
        "Red X: turret yaw error crossed zero and landed outside ±1.5°. "
        "Yellow regions: body and turret rotated in the same direction.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.045)
    save_figure(fig, "11_yaw_overshoot_diagnostics.png")

print(f"[SOURCE] {csv_path}")
if control_csv_path is not None:
    print(f"[CONTROL SOURCE] {control_csv_path}")
print(f"[ROWS] {len(df)}")
print(f"[VALID IMPACTS] {valid_count}")
print(f"[OUTPUT] {OUT_DIR}")

portfolio_summary = [
    "# Shot Plot Portfolio Run",
    "",
    f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"- Shot source: `{csv_path.name}`",
    f"- Control source: `{control_csv_path.name if control_csv_path else 'none'}`",
    f"- Shot rows: {len(df)}",
    f"- Valid impacts: {valid_count}",
    f"- Enemy/target hits: {enemy_hit_count}",
    f"- Mean target range: {finite_mean(target_range):.4f} m",
    f"- Mean impact range: {finite_mean(impact_range):.4f} m",
    f"- Mean range error: {finite_mean(range_error):.4f} m",
    f"- Mean 3D impact error: {finite_mean(impact_error_3d):.4f} m",
    "",
    "## Generated images",
    "",
]
portfolio_summary.extend(
    f"- `{path.name}`"
    for path in sorted(OUT_DIR.glob("*.png"))
)
(OUT_DIR / "README.md").write_text(
    "\n".join(portfolio_summary) + "\n",
    encoding="utf-8",
)

for output_path in sorted(OUT_DIR.glob("*.png")):
    print(output_path.name)
