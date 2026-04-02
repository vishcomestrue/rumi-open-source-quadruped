#!/usr/bin/env python3
"""Direct sim2real deployment — Rumi velocity task.

Reads observations from real hardware (motors + IMU), runs the MLP policy
on-device, and sends position commands to the Dynamixel motors at 50 Hz.

Startup sequence:
  1. Capture current motor positions as tick origin (same as getup).
  2. Treat that origin as the known sitting pose (SIT_POSE_RAD).
  3. Interpolate all joints from SIT_POSE_RAD → 0 rad over STANDUP_DURATION.
  4. Hold at 0 rad (standing) for STANDUP_HOLD seconds.
  5. Run RL policy for --duration seconds.

No parallel simulation. No mjlab dependency at runtime.

Usage:
    python run_velocity.py                              # default checkpoint, 20 s, zero command
    python run_velocity.py --duration 30               # longer run
    python run_velocity.py --target 0.5 0.0 0.0        # walk forward at 0.5 m/s
    python run_velocity.py --dry-run                   # motors silent, print actions only
    python run_velocity.py --record                    # save .npz after run
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
_REC_DIR = Path(__file__).resolve().parent / "recordings"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))             # deploy/velocity/
sys.path.insert(0, str(_HERE.parent))      # deploy/

from mx64_controller import MX64Controller
from imu import IMU
from policy import VelocityPolicy
from observations import build_obs, JOINT_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_HZ       = 50
DT               = 1.0 / CONTROL_HZ
ACTION_SCALE     = 0.075          # rad — 0.25 * effort_limit(6) / stiffness(20)
STANDUP_DURATION = 5.0            # seconds to interpolate sit → stand
STANDUP_HOLD     = 0.5            # seconds to hold at standing before policy
SIT_HOLD         = 0.5            # seconds to hold at sit pose before interpolation
DEFAULT_CKPT     = _HERE / "checkpoint" / "latest_velocity.pt"

# Sitting pose in radians (per joint, same JOINT_ORDER).
# The captured tick origin is treated as this configuration.
SIT_POSE_RAD = np.array([
    -0.02,  0.28,  0.66,   # FL_hip, FL_thigh, FL_calf
    -0.02, -0.28, -0.66,   # FR_hip, FR_thigh, FR_calf
    -0.02,  0.28,  0.66,   # BL_hip, BL_thigh, BL_calf
    -0.02, -0.28, -0.66,   # BR_hip, BR_thigh, BR_calf
], dtype=np.float32)

# Standing pose in radians — matches env init_state joint_pos.
# Policy outputs are offsets around this pose.
STAND_POSE_RAD = np.array([
     0.0,  -0.0705,  -0.113,   # FL_hip, FL_thigh, FL_calf
     0.0,   0.0705,   0.113,   # FR_hip, FR_thigh, FR_calf
     0.0,  -0.0705,  -0.113,   # BL_hip, BL_thigh, BL_calf
     0.0,   0.0705,   0.113,   # BR_hip, BR_thigh, BR_calf
], dtype=np.float32)

RADIANS_TO_POS = 4096.0 / (2.0 * np.pi)

# Motor ID → joint name (must match JOINT_ORDER indices)
JOINT_ID_MAP = {
    "FL_hip_joint": 1,  "FL_thigh_joint": 2,  "FL_calf_joint": 3,
    "BL_hip_joint": 4,  "BL_thigh_joint": 5,  "BL_calf_joint": 6,
    "BR_hip_joint": 7,  "BR_thigh_joint": 8,  "BR_calf_joint": 9,
    "FR_hip_joint": 10, "FR_thigh_joint": 11, "FR_calf_joint": 12,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rumi velocity direct sim2real")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                        help="Path to .pt checkpoint (default: checkpoint/latest_velocity.pt)")
    parser.add_argument("--duration",   type=float, default=20.0,
                        help="RL policy run duration in seconds (default: 20.0)")
    parser.add_argument("--target",     type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"),
                        help="Velocity command [lin_vel_x lin_vel_y ang_vel_z] (default: 0 0 0)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Do not send commands to motors (print actions only)")
    parser.add_argument("--record",     action="store_true",
                        help="Save a .npz recording to velocity/recordings/ after the run")
    args = parser.parse_args()

    command = np.array(args.target, dtype=np.float32)
    max_steps = int(args.duration * CONTROL_HZ)
    standup_steps = int(STANDUP_DURATION * CONTROL_HZ)
    sit_hold_steps = int(SIT_HOLD * CONTROL_HZ)
    stand_hold_steps = int(STANDUP_HOLD * CONTROL_HZ)

    print("\n" + "=" * 60)
    print("   Rumi Velocity — Direct Sim2Real")
    print("=" * 60)
    print(f"  Checkpoint : {Path(args.checkpoint).name}")
    print(f"  Duration   : {args.duration:.1f} s  ({max_steps} steps @ {CONTROL_HZ} Hz)")
    print(f"  Command    : vx={command[0]:.2f}  vy={command[1]:.2f}  wz={command[2]:.2f}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Record     : {args.record}")
    print()

    # ------------------------------------------------------------------
    # 1. Load policy
    # ------------------------------------------------------------------
    policy = VelocityPolicy(args.checkpoint)

    # ------------------------------------------------------------------
    # 2. Connect IMU
    # ------------------------------------------------------------------
    imu = IMU()

    # ------------------------------------------------------------------
    # 3. Connect motors
    # ------------------------------------------------------------------
    controller = None
    zero_pos = None

    if not args.dry_run:
        print("\nPlace robot in SITTING POSITION, then press Enter ...")
        input()

        ctrl_cfg = str(_HERE.parent / "motor_config.json")
        controller = MX64Controller(config_path=ctrl_cfg, auto_connect=False)
        motor_ids = controller.initialize(expected_motors=12)
        if motor_ids is None:
            print("[ERROR] Motor initialization failed. Exiting.")
            sys.exit(1)

        # Capture tick origin — treated as SIT_POSE_RAD
        zero_pos = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
        if zero_pos is None:
            print("[ERROR] Could not read initial motor positions. Exiting.")
            controller.disconnect()
            sys.exit(1)
        print("[INFO] Tick origin captured (= sitting pose reference).")
    else:
        zero_pos = {i: 2048 for i in range(1, 13)}
        print("[DRY RUN] Motors skipped.")

    # ------------------------------------------------------------------
    # Helpers: read joint states and write position targets.
    #
    # Joint position convention:
    #   pos_rad = (raw_ticks - zero_ticks) / RADIANS_TO_POS + SIT_POSE_RAD[i]
    #
    # When all joints are at zero_ticks, pos_rad == SIT_POSE_RAD (sitting).
    # When pos_rad == STAND_POSE_RAD, the robot is standing.
    # ------------------------------------------------------------------
    def read_joint_states() -> tuple:
        raw_pos, raw_vel = controller.sync_read_positions_and_velocities()
        if raw_pos is None:
            return None, None
        pos_dict = {}
        vel_dict = {}
        for i, joint in enumerate(JOINT_ORDER):
            mid = JOINT_ID_MAP[joint]
            pos_dict[joint] = (raw_pos[mid] - zero_pos[mid]) / RADIANS_TO_POS + SIT_POSE_RAD[i]
            vel_dict[joint] = raw_vel[mid]
        return pos_dict, vel_dict

    def write_joint_positions(pos_rad: np.ndarray) -> None:
        """Write target positions given as absolute rad (STAND_POSE_RAD = standing)."""
        raw_targets = {}
        for i, joint in enumerate(JOINT_ORDER):
            mid = JOINT_ID_MAP[joint]
            offset_ticks = int(round((pos_rad[i] - SIT_POSE_RAD[i]) * RADIANS_TO_POS))
            raw_targets[mid] = zero_pos[mid] + offset_ticks
        controller.sync_write_positions(raw_targets)

    # ------------------------------------------------------------------
    # 4. Warm up policy (first torch call is slow)
    # ------------------------------------------------------------------
    print("[WARMUP] Compiling policy (first call is slow) ...")
    _dummy_pos  = {j: 0.0 for j in JOINT_ORDER}
    _dummy_vel  = {j: 0.0 for j in JOINT_ORDER}
    _dummy_acc  = np.zeros(3, dtype=np.float32)
    _dummy_gyro = np.zeros(3, dtype=np.float32)
    _dummy_q    = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _dummy_obs  = build_obs(_dummy_pos, _dummy_vel, _dummy_acc, _dummy_gyro,
                            _dummy_q, np.zeros(12, dtype=np.float32), command)
    policy(_dummy_obs)
    print("[WARMUP] Done.\n")

    # ------------------------------------------------------------------
    # 5. Stand-up sequence
    # ------------------------------------------------------------------
    if not args.dry_run:
        print(f"[STANDUP] Holding sitting pose for {SIT_HOLD:.1f} s ...")
        for _ in range(sit_hold_steps):
            write_joint_positions(SIT_POSE_RAD)
            time.sleep(DT)

        print(f"[STANDUP] Interpolating sit → stand over {STANDUP_DURATION:.1f} s ...")
        for k in range(standup_steps):
            alpha = (k + 1) / standup_steps
            target = SIT_POSE_RAD + (STAND_POSE_RAD - SIT_POSE_RAD) * alpha
            write_joint_positions(target)
            time.sleep(DT)

        print(f"[STANDUP] Holding standing pose for {STANDUP_HOLD:.1f} s ...")
        for _ in range(stand_hold_steps):
            write_joint_positions(STAND_POSE_RAD)
            time.sleep(DT)

        print("[STANDUP] Standing. Starting RL policy.\n")
    else:
        print("[DRY RUN] Skipping stand-up sequence.\n")

    # ------------------------------------------------------------------
    # 6. RL policy control loop
    # ------------------------------------------------------------------
    last_action = np.zeros(12, dtype=np.float32)

    if args.record:
        rec_accel      = np.zeros((max_steps, 3),  dtype=np.float32)
        rec_gyro       = np.zeros((max_steps, 3),  dtype=np.float32)
        rec_quat       = np.zeros((max_steps, 4),  dtype=np.float32)
        rec_joint_pos  = np.zeros((max_steps, 12), dtype=np.float32)
        rec_joint_vel  = np.zeros((max_steps, 12), dtype=np.float32)
        rec_raw_action = np.zeros((max_steps, 12), dtype=np.float32)

    print("=" * 60)
    print("   Starting RL control loop — Ctrl+C to stop")
    print("=" * 60 + "\n")

    start_time = time.time()
    late_count = 0
    step = 0

    try:
        for step in range(max_steps):
            loop_start = time.time()

            # --- Read hardware ---
            if not args.dry_run:
                pos_dict, vel_dict = read_joint_states()
                if pos_dict is None:
                    print(f"[WARNING] Step {step}: motor read failed, skipping.")
                    time.sleep(DT)
                    continue
            else:
                pos_dict = {j: 0.0 for j in JOINT_ORDER}
                vel_dict = {j: 0.0 for j in JOINT_ORDER}

            quat, accel, gyro = imu.read_all()

            # --- Build obs ---
            obs = build_obs(pos_dict, vel_dict, accel, gyro, quat, last_action, command)

            # --- Policy inference ---
            raw_action = policy(obs)   # [12], unscaled

            # --- Record ---
            if args.record:
                rec_accel[step]      = accel
                rec_gyro[step]       = gyro
                rec_quat[step]       = quat
                rec_joint_pos[step]  = np.array([pos_dict[j] for j in JOINT_ORDER], dtype=np.float32)
                rec_joint_vel[step]  = np.array([vel_dict[j] for j in JOINT_ORDER], dtype=np.float32)
                rec_raw_action[step] = raw_action

            # --- Send to motors ---
            processed = raw_action * ACTION_SCALE   # rad offsets around standing pose
            if not args.dry_run:
                write_joint_positions(STAND_POSE_RAD + processed)

            # --- Store for next step ---
            last_action = raw_action

            # --- Status print every 50 steps (1 s) ---
            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  step {step:4d}/{max_steps} | t={elapsed:.2f}s | "
                      f"action_rms={float(np.sqrt(np.mean(processed**2))):.4f} rad")

            # --- Timing ---
            elapsed_loop = time.time() - loop_start
            sleep_time   = DT - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.002:
                late_count += 1

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received.")

    finally:
        total = time.time() - start_time
        steps_done = min(step + 1, max_steps)
        print(f"\n[DONE] {steps_done} steps in {total:.2f} s "
              f"(avg {steps_done/total:.1f} Hz, {late_count} late steps)")
        if controller is not None:
            controller.disconnect()
            print("[INFO] Motors disconnected.")

        if args.record:
            _REC_DIR.mkdir(parents=True, exist_ok=True)
            rec_path = _REC_DIR / f"velocity_{int(time.time())}.npz"
            np.savez(
                rec_path,
                accel      = rec_accel[:steps_done],
                gyro       = rec_gyro[:steps_done],
                quat       = rec_quat[:steps_done],
                joint_pos  = rec_joint_pos[:steps_done],
                joint_vel  = rec_joint_vel[:steps_done],
                raw_action = rec_raw_action[:steps_done],
                command    = command,
                control_hz = np.float32(CONTROL_HZ),
                action_scale = np.float32(ACTION_SCALE),
            )
            print(f"[REC] Saved {steps_done} steps → {rec_path}")


if __name__ == "__main__":
    main()
