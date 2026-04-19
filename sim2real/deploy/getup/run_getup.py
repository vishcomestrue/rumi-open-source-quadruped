#!/usr/bin/env python3
"""Direct sim2real deployment — Rumi getup task.

Reads observations from real hardware (motors + IMU), runs the MLP policy
on-device, and sends position commands to the Dynamixel motors at 50 Hz.

No parallel simulation. No mjlab dependency at runtime.

Usage:
    python run_getup.py                        # default checkpoint, 5 s
    python run_getup.py --duration 10          # longer run
    python run_getup.py --no-imu               # skip IMU (uses identity quat)
    python run_getup.py --dry-run              # motors silent, just print actions
    python run_getup.py --target 0.18          # use 0.18 m as target height
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
sys.path.insert(0, str(_HERE))                      # deploy/getup/
sys.path.insert(0, str(_HERE.parent))               # deploy/
sys.path.insert(0, str(_HERE.parent.parent))        # sim2real/ (DynamixelSDK path ref)

from mx64_controller import MX64Controller
from imu import IMU
from policy import GetupPolicy
from observations import build_obs, JOINT_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_HZ       = 50
DT               = 1.0 / CONTROL_HZ
ACTION_SCALE     = 0.075          # rad per unit — 0.25 * effort_limit(6) / stiffness(20)
SITDOWN_DURATION = 5.0            # seconds to interpolate → sit after policy ends
DEFAULT_CKPT     = _HERE / "checkpoint" / "latest_getup.pt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rumi getup direct sim2real")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                        help="Path to .pt checkpoint (default: checkpoint/latest_getup.pt)")
    parser.add_argument("--duration",   type=float, default=15.0,
                        help="Run duration in seconds (default: 5.0)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Do not send commands to motors (print actions only)")
    parser.add_argument("--no-imu",     action="store_true",
                        help="Skip IMU — use identity quaternion (for testing without IMU)")
    parser.add_argument("--target",     type=float, default=0.25,
                        help="Target standing height in metres (default: 0.25)")
    parser.add_argument("--record",     action="store_true",
                        help="Save a .npz recording to getup/recordings/ after the run")
    args = parser.parse_args()

    max_steps = int(args.duration * CONTROL_HZ)

    print("\n" + "=" * 60)
    print("   Rumi Getup — Direct Sim2Real")
    print("=" * 60)
    print(f"  Checkpoint : {Path(args.checkpoint).name}")
    print(f"  Duration   : {args.duration:.1f} s  ({max_steps} steps @ {CONTROL_HZ} Hz)")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  IMU        : {'disabled (identity quat)' if args.no_imu else 'enabled'}")
    print(f"  Target ht  : {args.target:.3f} m")
    print(f"  Record     : {args.record}")
    print()

    # ------------------------------------------------------------------
    # 1. Load policy
    # ------------------------------------------------------------------
    policy = GetupPolicy(args.checkpoint)

    # ------------------------------------------------------------------
    # 2. Connect IMU
    # ------------------------------------------------------------------
    imu = None
    if not args.no_imu:
        try:
            imu = IMU()
        except Exception as exc:
            print(f"[WARNING] IMU init failed: {exc}")
            print("[WARNING] Falling back to identity quaternion.")

    # ------------------------------------------------------------------
    # 3. Connect motors
    # ------------------------------------------------------------------
    controller = None
    if not args.dry_run:
        print("\nPlace robot in SITTING POSITION, then press Enter ...")
        input()

        ctrl_cfg = str(_HERE.parent / "motor_config.json")
        controller = MX64Controller(config_path=ctrl_cfg, auto_connect=False)
        motor_ids = controller.initialize(expected_motors=12)
        if motor_ids is None:
            print("[ERROR] Motor initialization failed. Exiting.")
            sys.exit(1)

        # Capture sit-pose zero reference
        zero_pos = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
        if zero_pos is None:
            print("[ERROR] Could not read initial motor positions. Exiting.")
            controller.disconnect()
            sys.exit(1)
        print("[INFO] Sit-pose zero reference captured.")
    else:
        zero_pos = {i: 2048 for i in range(1, 13)}   # dummy for dry-run
        print("[DRY RUN] Motors skipped.")

    # Motor ID → joint name (from JOINT_ORDER + JOINT_ID_MAP)
    JOINT_ID_MAP = {
        "FL_hip_joint": 1,  "FL_thigh_joint": 2,  "FL_calf_joint": 3,
        "BL_hip_joint": 4,  "BL_thigh_joint": 5,  "BL_calf_joint": 6,
        "BR_hip_joint": 7,  "BR_thigh_joint": 8,  "BR_calf_joint": 9,
        "FR_hip_joint": 10, "FR_thigh_joint": 11, "FR_calf_joint": 12,
    }
    RADIANS_TO_POS = 4096.0 / (2.0 * np.pi)

    def read_joint_states() -> tuple:
        """Read positions and velocities from hardware.

        Returns:
            (pos_dict, vel_dict) — joint-name keyed dicts, positions in rad
            relative to sit-pose zero, velocities in rad/s.
            Returns (None, None) on read failure.
        """
        raw_pos, raw_vel = controller.sync_read_positions_and_velocities()
        if raw_pos is None:
            return None, None
        pos_dict = {}
        vel_dict = {}
        for joint, mid in JOINT_ID_MAP.items():
            pos_dict[joint] = (raw_pos[mid] - zero_pos[mid]) / RADIANS_TO_POS
            vel_dict[joint] = raw_vel[mid]
        return pos_dict, vel_dict

    def write_joint_positions(pos_rad_dict: dict) -> None:
        """Write target positions (offsets from sit-pose zero) in radians."""
        raw_targets = {}
        for joint, mid in JOINT_ID_MAP.items():
            offset_ticks = int(round(pos_rad_dict[joint] * RADIANS_TO_POS))
            raw_targets[mid] = zero_pos[mid] + offset_ticks
        controller.sync_write_positions(raw_targets)

    # ------------------------------------------------------------------
    # 4. Warm up — force FK import and torch JIT before the timed loop
    # ------------------------------------------------------------------
    print("[WARMUP] Compiling FK and policy (first call is slow) ...")
    _dummy_pos = {j: 0.0 for j in JOINT_ORDER}
    _dummy_vel = {j: 0.0 for j in JOINT_ORDER}
    _dummy_q   = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _dummy_obs, _ = build_obs(_dummy_pos, _dummy_vel, _dummy_q, np.zeros(12, dtype=np.float32), args.target)
    policy(_dummy_obs)
    print("[WARMUP] Done.\n")

    # ------------------------------------------------------------------
    # 5. Control loop
    # ------------------------------------------------------------------
    last_action = np.zeros(12, dtype=np.float32)

    # Recording buffers (only allocated when --record is set)
    if args.record:
        rec_quat       = np.zeros((max_steps, 4),  dtype=np.float32)
        rec_joint_pos  = np.zeros((max_steps, 12), dtype=np.float32)
        rec_joint_vel  = np.zeros((max_steps, 12), dtype=np.float32)
        rec_raw_action = np.zeros((max_steps, 12), dtype=np.float32)
        rec_fk_height  = np.zeros((max_steps,),    dtype=np.float32)

    print("\n" + "=" * 60)
    print("   Starting control loop — Ctrl+C to stop")
    print("=" * 60 + "\n")

    start_time = time.time()
    late_count  = 0

    try:
        for step in range(max_steps):
            loop_start = time.time()
            t = loop_start - start_time

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

            if imu is not None:
                quat = imu.read_quaternion()
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # --- Build obs ---
            obs, fk_height = build_obs(pos_dict, vel_dict, quat, last_action, args.target)

            # --- Policy inference ---
            raw_action = policy(obs)   # [12], unscaled

            # --- Record ---
            if args.record:
                rec_quat[step]       = quat
                rec_joint_pos[step]  = np.array([pos_dict[j] for j in JOINT_ORDER], dtype=np.float32)
                rec_joint_vel[step]  = np.array([vel_dict[j] for j in JOINT_ORDER], dtype=np.float32)
                rec_raw_action[step] = raw_action
                rec_fk_height[step]  = fk_height

            # --- Send to motors ---
            processed = raw_action * ACTION_SCALE   # [12], rad offsets from sit pose
            if not args.dry_run:
                pos_targets = {j: float(processed[i]) for i, j in enumerate(JOINT_ORDER)}
                write_joint_positions(pos_targets)

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
        if late_count > steps_done * 0.05:
            print(f"[WARN] {late_count}/{steps_done} steps were late "
                  f"({100*late_count/steps_done:.1f}%) — loop may be overloaded.")
        if not args.dry_run and controller is not None:
            sitdown_steps = int(SITDOWN_DURATION * CONTROL_HZ)
            pos_dict, _ = read_joint_states()
            if pos_dict is not None:
                sitdown_start = np.array([pos_dict[j] for j in JOINT_ORDER], dtype=np.float32)
            else:
                sitdown_start = np.zeros(12, dtype=np.float32)
                print("[WARN] Could not read current positions; using sit-pose as sitdown start.")
            print(f"[SITDOWN] Interpolating → sit over {SITDOWN_DURATION:.1f} s ...")
            sit_target = np.zeros(12, dtype=np.float32)
            for k in range(sitdown_steps):
                alpha = (k + 1) / sitdown_steps
                target_arr = sitdown_start + (sit_target - sitdown_start) * alpha
                target_dict = {j: float(target_arr[i]) for i, j in enumerate(JOINT_ORDER)}
                write_joint_positions(target_dict)
                time.sleep(DT)

        if controller is not None:
            controller.disconnect()
            print("[INFO] Motors disconnected.")

        if args.record:
            ans = input("\nSave recording? [y/n]: ").strip().lower()
            if ans == "y":
                _REC_DIR.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                h_tag = f"h{str(args.target).replace('.', '')}"
                rec_path = _REC_DIR / f"getup_{ts}_{h_tag}.npz"
                np.savez(
                    rec_path,
                    # Only save the steps that actually ran
                    quat       = rec_quat[:steps_done],
                    joint_pos  = rec_joint_pos[:steps_done],
                    joint_vel  = rec_joint_vel[:steps_done],
                    raw_action = rec_raw_action[:steps_done],
                    fk_height  = rec_fk_height[:steps_done],
                    # Metadata
                    target_height = np.float32(args.target),
                    control_hz    = np.float32(CONTROL_HZ),
                    action_scale  = np.float32(ACTION_SCALE),
                    timestamp     = np.int64(ts),
                )
                print(f"[REC] Saved {steps_done} steps → {rec_path}")
            else:
                print("[REC] Recording discarded.")


if __name__ == "__main__":
    main()
