#!/usr/bin/env python3
"""Verify the sit → stand transition pipeline without running the RL policy.

Tests:
  1. Connect motors and capture tick origin.
  2. Verify captured ticks map to SIT_POSE_RAD correctly (print readback).
  3. Hold at sitting pose for SIT_HOLD seconds.
  4. Interpolate from SIT_POSE_RAD → 0 rad over STANDUP_DURATION seconds.
  5. Hold at 0 rad (standing) for STANDUP_HOLD seconds.
  6. Print joint positions throughout so you can verify they reach ~0 rad.

Usage:
    python test_standup.py
    python test_standup.py --standup-duration 8   # slower interpolation
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from mx64_controller import MX64Controller
from observations import JOINT_ORDER
from imu import IMU

# ---------------------------------------------------------------------------
# Constants — must match run_velocity.py exactly
# ---------------------------------------------------------------------------
CONTROL_HZ       = 50
DT               = 1.0 / CONTROL_HZ
SIT_HOLD         = 0.5
STANDUP_HOLD     = 20.0

SIT_POSE_RAD = np.array([
    -0.02,  0.28,  0.66,   # FL_hip, FL_thigh, FL_calf
    -0.02, -0.28, -0.66,   # FR_hip, FR_thigh, FR_calf
    -0.02,  0.28,  0.66,   # BL_hip, BL_thigh, BL_calf
    -0.02, -0.28, -0.66,   # BR_hip, BR_thigh, BR_calf
], dtype=np.float32)

RADIANS_TO_POS = 4096.0 / (2.0 * np.pi)

JOINT_ID_MAP = {
    "FL_hip_joint": 1,  "FL_thigh_joint": 2,  "FL_calf_joint": 3,
    "BL_hip_joint": 4,  "BL_thigh_joint": 5,  "BL_calf_joint": 6,
    "BR_hip_joint": 7,  "BR_thigh_joint": 8,  "BR_calf_joint": 9,
    "FR_hip_joint": 10, "FR_thigh_joint": 11, "FR_calf_joint": 12,
}


def main():
    parser = argparse.ArgumentParser(description="Test sit → stand transition")
    parser.add_argument("--standup-duration", type=float, default=5.0,
                        help="Seconds to interpolate sit → stand (default: 5.0)")
    args = parser.parse_args()

    standup_steps    = int(args.standup_duration * CONTROL_HZ)
    sit_hold_steps   = int(SIT_HOLD * CONTROL_HZ)
    stand_hold_steps = int(STANDUP_HOLD * CONTROL_HZ)

    print("\n" + "=" * 60)
    print("   Rumi Velocity — Stand-up Pipeline Test")
    print("=" * 60)
    print(f"  Standup duration : {args.standup_duration:.1f} s ({standup_steps} steps)")
    print(f"  Sit hold         : {SIT_HOLD:.1f} s")
    print(f"  Stand hold       : {STANDUP_HOLD:.1f} s")
    print()

    # ------------------------------------------------------------------
    # 1. Connect motors and capture tick origin
    # ------------------------------------------------------------------
    print("Place robot in SITTING POSITION, then press Enter ...")
    input()

    ctrl_cfg = str(_HERE.parent / "motor_config.json")
    controller = MX64Controller(config_path=ctrl_cfg, auto_connect=False)
    motor_ids = controller.initialize(expected_motors=12)
    if motor_ids is None:
        print("[ERROR] Motor initialization failed. Exiting.")
        sys.exit(1)

    zero_pos = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
    if zero_pos is None:
        print("[ERROR] Could not read initial motor positions. Exiting.")
        controller.disconnect()
        sys.exit(1)

    print("[INFO] Tick origin captured.\n")

    imu = IMU()
    time.sleep(0.5)   # let background thread fill initial values
    print(f"{'t (s)':>8}  {'ax':>10}  {'ay':>10}  {'az':>10}")
    print("-" * 46)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def read_joint_pos_rad() -> np.ndarray:
        raw_pos, _ = controller.sync_read_positions_and_velocities()
        if raw_pos is None:
            return None
        pos = np.array([
            (raw_pos[JOINT_ID_MAP[j]] - zero_pos[JOINT_ID_MAP[j]]) / RADIANS_TO_POS + SIT_POSE_RAD[i]
            for i, j in enumerate(JOINT_ORDER)
        ], dtype=np.float32)
        return pos

    def write_joint_positions(pos_rad: np.ndarray) -> None:
        raw_targets = {}
        for i, joint in enumerate(JOINT_ORDER):
            mid = JOINT_ID_MAP[joint]
            offset_ticks = int(round((pos_rad[i] - SIT_POSE_RAD[i]) * RADIANS_TO_POS))
            raw_targets[mid] = zero_pos[mid] + offset_ticks
        controller.sync_write_positions(raw_targets)

    def print_joint_pos(pos_rad: np.ndarray, label: str):
        print(f"\n  [{label}]")
        for i, joint in enumerate(JOINT_ORDER):
            print(f"    {joint:20s}  {pos_rad[i]:+.4f} rad  (target: {0.0:+.4f})")

    # ------------------------------------------------------------------
    # 2. Verify readback at sit pose
    # ------------------------------------------------------------------
    pos = read_joint_pos_rad()
    if pos is not None:
        print_joint_pos(pos, "Readback at capture — should be ~SIT_POSE_RAD")
        err = np.abs(pos - SIT_POSE_RAD)
        print(f"\n  Max error vs SIT_POSE_RAD: {err.max():.4f} rad  "
              f"({'OK' if err.max() < 0.05 else 'WARNING: large error'})")
    print()

    # ------------------------------------------------------------------
    # 3. Hold at sitting pose
    # ------------------------------------------------------------------
    start_time = time.time()
    print(f"[HOLD] Sitting for {SIT_HOLD:.1f} s ...")
    for _ in range(sit_hold_steps):
        write_joint_positions(SIT_POSE_RAD)
        _, accel, _ = imu.read_all()
        t = time.time() - start_time
        print(f"{t:8.2f}  {accel[0]:+10.4f}  {accel[1]:+10.4f}  {accel[2]:+10.4f}")
        time.sleep(DT)

    # ------------------------------------------------------------------
    # 4. Interpolate sit → stand
    # ------------------------------------------------------------------
    print(f"[INTERP] Interpolating to standing over {args.standup_duration:.1f} s ...")

    for k in range(standup_steps):
        alpha  = (k + 1) / standup_steps
        target = SIT_POSE_RAD * (1.0 - alpha)
        write_joint_positions(target)
        _, accel, _ = imu.read_all()
        t = time.time() - start_time
        print(f"{t:8.2f}  {accel[0]:+10.4f}  {accel[1]:+10.4f}  {accel[2]:+10.4f}")
        time.sleep(DT)

    # ------------------------------------------------------------------
    # 5. Hold at standing
    # ------------------------------------------------------------------
    print(f"\n[HOLD] Standing for {STANDUP_HOLD:.1f} s ...")
    for _ in range(stand_hold_steps):
        write_joint_positions(np.zeros(12, dtype=np.float32))
        _, accel, _ = imu.read_all()
        t = time.time() - start_time
        print(f"{t:8.2f}  {accel[0]:+10.4f}  {accel[1]:+10.4f}  {accel[2]:+10.4f}")
        time.sleep(DT)

    # ------------------------------------------------------------------
    # 6. Final readback — should be ~0 rad on all joints
    # ------------------------------------------------------------------
    pos = read_joint_pos_rad()
    if pos is not None:
        print_joint_pos(pos, "Final readback — should be ~0.0 rad")
        err = np.abs(pos)
        print(f"\n  Max error vs 0 rad: {err.max():.4f} rad  "
              f"({'OK' if err.max() < 0.05 else 'WARNING: did not reach standing'})")

    controller.disconnect()
    print("\n[DONE] Motors disconnected.")


if __name__ == "__main__":
    main()
