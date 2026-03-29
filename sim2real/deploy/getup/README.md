# Rumi Getup — Direct Sim2Real Deploy

Runs the trained getup policy directly on real hardware. No simulation, no mjlab at runtime.

**Pipeline:** Motors + IMU → 41-dim obs → MLP policy → motor position commands @ 50 Hz

---

## Files

```
deploy/getup/
├── checkpoint/
│   └── latest_getup.pt     — trained policy checkpoint (iter 3999)
├── mx64_controller.py      — Dynamixel MX-64 motor controller (GroupSync)
├── motor_config.json       — Kp=20, Kd=0, 50 Hz for all 12 motors
├── imu.py                  — BNO080 IMU wrapper (rotation vector only)
├── policy.py               — standalone MLP + obs normalizer (no mjlab)
├── observations.py         — builds the 41-dim observation vector
└── run_getup.py            — main 50 Hz control loop
```

---

## Raspberry Pi Setup

### 1. Enable I2C (for IMU)

```bash
sudo raspi-config
# Interface Options → I2C → Enable
# Reboot after
```

### 2. Install Python packages

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy adafruit-circuitpython-bno08x adafruit-blinka dynamixel-sdk
```

Package breakdown:

| Package | Purpose |
|---------|---------|
| `torch` (CPU build) | Policy inference |
| `numpy` | Observation math |
| `adafruit-circuitpython-bno08x` | BNO080 IMU driver |
| `adafruit-blinka` | Provides `board` and `busio` for RPi I2C |
| `dynamixel-sdk` | Dynamixel motor communication |

> **Note:** `dynamixel-sdk` can be skipped if `DynamixelSDK/` is already cloned at `../../DynamixelSDK/` — `mx64_controller.py` adds it to sys.path automatically.

### 3. Clone rumi-mjlab (for FK kinematics)

`observations.py` imports `estimate_body_height` from `rumi-mjlab/src` at runtime. The repo must exist at `../../rumi-mjlab/` relative to this folder (i.e. `sim2real/rumi-mjlab/`).

---

## Running

### Recommended sequence — don't skip steps

**Step 1 — dry run, no hardware:**
```bash
python run_getup.py --dry-run --no-imu --duration 3
```
Verifies policy loads and obs pipeline runs without any hardware connected.

**Step 2 — dry run with IMU only:**
```bash
python run_getup.py --dry-run --duration 3
```
Verifies IMU reads and quaternion is sane (projected gravity should be ~[0, 0, -1] when flat).

**Step 3 — motors only, no IMU:**
```bash
python run_getup.py --no-imu --duration 3
```
Verifies motors connect and hold position. Robot should barely move (identity quat = degraded obs but safe).

**Step 4 — full deploy:**
```bash
python run_getup.py --duration 5
```

### All flags

```
--checkpoint PATH   Path to .pt file (default: checkpoint/latest_getup.pt)
--duration SECS     Run duration in seconds (default: 5.0)
--dry-run           Do not send commands to motors — print actions only
--no-imu            Skip IMU, use identity quaternion (w=1, x=y=z=0)
```

---

## Hardware connections

- **Motors:** USB serial (`/dev/ttyUSB0`), 4 Mbaud, Protocol 2.0
- **IMU:** I2C, address `0x4b`, SDA/SCL pins

### Motor ID → joint mapping

| Motor ID | Joint |
|----------|-------|
| 1 | FL_hip_joint |
| 2 | FL_thigh_joint |
| 3 | FL_calf_joint |
| 4 | BL_hip_joint |
| 5 | BL_thigh_joint |
| 6 | BL_calf_joint |
| 7 | BR_hip_joint |
| 8 | BR_thigh_joint |
| 9 | BR_calf_joint |
| 10 | FR_hip_joint |
| 11 | FR_thigh_joint |
| 12 | FR_calf_joint |

---

## Observation vector (41-dim)

| Index | Term | Source |
|-------|------|--------|
| 0 | body_height | FK (encoders) + IMU quaternion, normalized to [0.10, 0.25] m range |
| 1 | target_height | Fixed: 1.0 (= 0.25 m normalized) |
| 2–4 | projected_gravity | IMU quaternion → gravity in body frame |
| 5–16 | joint_pos | Encoder positions relative to sit-pose zero reference |
| 17–28 | joint_vel | Finite-difference velocity from encoder positions |
| 29–40 | actions | Raw policy output from previous step (zeros at t=0) |

The obs normalizer (running mean/std) is baked into the checkpoint and applied automatically inside `policy.py`. Do not pre-normalize.

---

## Action

Policy outputs raw action `[12]`. Motor position targets are:

```
motor_target_rad = raw_action * 0.075
```

where `0.075 = 0.25 * effort_limit(6 Nm) / stiffness(20 Nm/rad)`.

Targets are **offsets from the sit-pose zero reference** captured at motor initialization.

---

## Important notes

- Place robot in **sitting position** before running (all joints at 0 rad). The controller captures this as the zero reference on startup.
- The IMU must be **mounted on the robot body** and oriented consistently with the simulation frame.
- If the robot jerks immediately, check the IMU quaternion convention — the BNO080 SDK returns `(x, y, z, w)` and `imu.py` reorders to `(w, x, y, z)`. Verify this is correct for your firmware version.
