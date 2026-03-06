# Master Dataset: Motor Excitation Data for Rumi Sysid

This folder contains data collected from the physical Rumi quadruped by exciting individual motors with various input signals. The data is organized to enable systematic system identification (sysid) and model parameter fitting.

---

## Data Collection Protocol

### PD Gains (Fixed Throughout)
- **Kp = 6.0**
- **Kd = 0.0**

These gains are held constant across **both simulation and real robot** for all data collection runs.

### Starting Parameter Values (Sim)
Simulation parameters for armature damping and friction loss are initialized from the values used in:

> **Reference:** [WholeBodyBAM position_model/robot.xml](https://github.com/MarcDcls/WholeBodyBAM/blob/main/position_model/robot.xml)

---

## Filename Convention

```
<timestamp>_<task>_<signal>_<params>.npz
```

- **task**: `standup` — robot starts from standing position
- **signal**: `standup` (smooth interpolation to target position) or `sine` (sinusoidal excitation)
- **standup params**: `<center>c_<duration>s` — e.g. `40c_3s` = calf center 40 deg, interpolation duration 3 sec
- **sine params**: `<freq>hz_<center>-<amplitude><joint>` — e.g. `1hz_40-10c` = 1 Hz, center 40, amplitude 10, calf joint
- **Joint codes**: `c` = calf, `t` = thigh, `h` = hip

---

## NPZ File Contents

Each `.npz` file contains the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `t` | (N,) | Wall-clock time (s) |
| `target` | (N, 12) | Commanded joint positions (rad, offset-space) |
| `q_real` | (N, 12) | Measured joint positions (rad, offset-space) |
| `dq_real` | (N, 12) | Measured joint velocities (rad/s) |
| `tau_meas` | (N, 12) | Measured joint torques (N·m) |
| `q_sim` | (N, 12) | Simulated joint positions (rad) |
| `dq_sim` | (N, 12) | Simulated joint velocities (rad/s) |
| `control_hz` | (1,) | Control loop frequency (Hz) |
| `joint_names` | (12,) | Joint name strings |
| `kp_sim` | (1,) | Kp gain used in simulation |
| `kd_sim` | (1,) | Kd gain used in simulation |
| `kp_real` | (1,) | Kp gain used on real robot |
| `kd_real` | (1,) | Kd gain used on real robot |
| `signal_mode` | (1,) | Signal type string (e.g. `sine`, `standup`) |
| `hip_centre_deg` | (1,) | Hip joint center position (deg) |
| `thigh_centre_deg` | (1,) | Thigh joint center position (deg) |
| `calf_centre_deg` | (1,) | Calf joint center position (deg) |
| `hip_swing_deg` | (1,) | Hip joint swing amplitude (deg) |
| `thigh_swing_deg` | (1,) | Thigh joint swing amplitude (deg) |
| `calf_swing_deg` | (1,) | Calf joint swing amplitude (deg) |
| `frequency_hz` | (1,) | Sine signal frequency (Hz) |
| `step_period_s` | (1,) | Step signal period (s) |
| `chirp_f1_hz` | (1,) | Chirp start frequency (Hz) |
| `chirp_sweep_s` | (1,) | Chirp sweep duration (s) |
| `standup_duration_s` | (1,) | Standup interpolation duration (s) |

---

## Dataset Index

### Standup (Smooth Interpolation) Signals — Calf Joint

| Filename | Signal | Joint | Center (deg) | Duration | Date |
|----------|--------|-------|--------------|----------|------|
| `20260306_230735_standup_standup_40c_3s.npz` | Standup | Calf | 40 | 3 s | 2026-03-06 |
| `20260306_230754_standup_standup_40c_5s.npz` | Standup | Calf | 40 | 5 s | 2026-03-06 |
| `20260306_230808_standup_standup_40c_1dot5s.npz` | Standup | Calf | 40 | 1.5 s | 2026-03-06 |
| `20260306_230825_standup_standup_40c_8s.npz` | Standup | Calf | 40 | 8 s | 2026-03-06 |

### Sine Signals — Calf Joint (center=40, amplitude=10)

| Filename | Signal | Joint | Freq (Hz) | Center (deg) | Amplitude (deg) | Date |
|----------|--------|-------|-----------|--------------|-----------------|------|
| `20260306_231431_standup_sine_0dot25hz_40-10c.npz` | Sine | Calf | 0.25 | 40 | 10 | 2026-03-06 |
| `20260306_231318_standup_sine_0dot5hz_40-10c.npz` | Sine | Calf | 0.5 | 40 | 10 | 2026-03-06 |
| `20260306_231108_standup_sine_1hz_40-10c.npz` | Sine | Calf | 1.0 | 40 | 10 | 2026-03-06 |
| `20260306_231207_standup_sine_2hz_40-10c.npz` | Sine | Calf | 2.0 | 40 | 10 | 2026-03-06 |

### Sine Signals — Thigh Joint (center=0, amplitude=10)

| Filename | Signal | Joint | Freq (Hz) | Center (deg) | Amplitude (deg) | Date |
|----------|--------|-------|-----------|--------------|-----------------|------|
| `20260306_232122_standup_sine_0dot25hz_0-10t.npz` | Sine | Thigh | 0.25 | 0 | 10 | 2026-03-06 |
| `20260306_232024_standup_sine_0dot5hz_0-10t.npz` | Sine | Thigh | 0.5 | 0 | 10 | 2026-03-06 |
| `20260306_231710_standup_sine_1hz_0-10t.npz` | Sine | Thigh | 1.0 | 0 | 10 | 2026-03-06 |
| `20260306_231819_standup_sine_2hz_0-10t.npz` | Sine | Thigh | 2.0 | 0 | 10 | 2026-03-06 |

### Sine Signals — Hip Joint (center=0, amplitude=10)

| Filename | Signal | Joint | Freq (Hz) | Center (deg) | Amplitude (deg) | Date |
|----------|--------|-------|-----------|--------------|-----------------|------|
| `20260307_000329_standup_sine_0dot25hz_0-10h.npz` | Sine | Hip | 0.25 | 0 | 10 | 2026-03-07 |
| `20260307_000120_standup_sine_0dot5hz_0-10h.npz` | Sine | Hip | 0.5 | 0 | 10 | 2026-03-07 |
| `20260307_000430_standup_sine_1hz_0-10h.npz` | Sine | Hip | 1.0 | 0 | 10 | 2026-03-07 |
| `20260307_000831_standup_sine_2hz_0-10h.npz` | Sine | Hip | 2.0 | 0 | 10 | 2026-03-07 |

---

## Collection Summary

| Joint | Standup | Sine (0.25 Hz) | Sine (0.5 Hz) | Sine (1 Hz) | Sine (2 Hz) |
|-------|---------|----------------|---------------|-------------|-------------|
| Calf  | 4 runs (1.5s, 3s, 5s, 8s) | Done | Done | Done | Done |
| Thigh | — | Done | Done | Done | Done |
| Hip   | — | Done | Done | Done | Done |
