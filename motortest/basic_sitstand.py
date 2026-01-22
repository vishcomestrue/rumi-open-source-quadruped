#!/usr/bin/env python3
# rumi-custom-quadruped — Reinforcement-learning-based quadruped-robot control framework for custom quadruped
# Copyright (C) 2025  Vishwanath R
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Basic Sit-Stand Motion Test

Simple motion sequence:
1. Start at reference position (standing)
2. Move to sit position: motors 9,12 go to +500r, motors 3,6 go to -500r
3. Return to reference position (standing)

Motion Convention:
- Motors 3, 6: move to -500r (opposite of 9,12)
- Motors 9, 12: move to +500r
"""

import sys
import time
import argparse
from mx64_controller import MX64Controller


def run_sitstand(control_freq=5.0, step_size=30, target_offset=500, repeat=1):
    """
    Run basic sit-stand motion.

    Args:
        control_freq: Control loop frequency in Hz (default: 5 Hz)
        step_size: Position step size in raw units (default: 30)
        target_offset: Target offset for sit position (default: 500)
        repeat: Number of times to repeat the sit-stand cycle (default: 1)
    """
    print("\n" + "=" * 60)
    print("   Basic Sit-Stand Motion Test")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Control frequency: {control_freq} Hz ({1000/control_freq:.1f} ms period)")
    print(f"  Step size: {step_size} raw units")
    print(f"  Target: Motors 9,12 -> +{target_offset}r | Motors 3,6 -> -{target_offset}r")
    print(f"  Repeat: {repeat} cycle(s)")
    print("=" * 60)

    controller = MX64Controller()

    # Step 1: Connect
    print("\n[STEP 1] Scanning ports and connecting...")
    print("-" * 40)

    if not controller.connect():
        print("\n[FAILED] Could not connect to any motors.")
        print("Please check:")
        print("  - Motors are powered (12V)")
        print("  - USB cable is connected")
        print("  - Correct permissions on /dev/ttyUSB*")
        return

    # Step 2: Discover all motors
    print("\n[STEP 2] Discovering motors...")
    print("-" * 40)

    motors = controller.scan_motors(verbose=True)

    if not motors:
        print("[FAILED] No motors discovered.")
        controller.disconnect()
        return

    all_motor_ids = sorted(motors.keys())
    print(f"\nFound {len(all_motor_ids)} motor(s): {all_motor_ids}")

    # Check if all 12 motors are present
    if len(all_motor_ids) != 12:
        print(f"\n[WARNING] Expected 12 motors, but found {len(all_motor_ids)}.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("[INFO] Exiting as requested.")
            controller.disconnect()
            return

    # Filter to motors with IDs that are multiples of 3
    target_motor_ids = [mid for mid in all_motor_ids if mid % 3 == 0]

    if not target_motor_ids:
        print("\n[FAILED] No motors with IDs that are multiples of 3 found!")
        print(f"Available motor IDs: {all_motor_ids}")
        controller.disconnect()
        return

    # Split into groups for opposite motion
    group_a = [mid for mid in target_motor_ids if mid in [3, 6]]   # Go negative
    group_b = [mid for mid in target_motor_ids if mid in [9, 12]]  # Go positive

    print(f"\n[INFO] Target motors: {target_motor_ids}")
    print(f"[INFO]   Group A (3, 6) -> -{target_offset}r: {group_a}")
    print(f"[INFO]   Group B (9, 12) -> +{target_offset}r: {group_b}")
    print(f"[INFO] Other motors hold position: {[m for m in all_motor_ids if m not in target_motor_ids]}")

    # Step 3: Initialize GroupSync
    print("\n[STEP 3] Initializing GroupSync...")
    print("-" * 40)

    if not controller.init_sync(all_motor_ids):
        print("[FAILED] Could not initialize GroupSync.")
        controller.disconnect()
        return

    print("[OK] GroupSync initialized")

    # Step 4: Read initial positions as REFERENCE POINTS
    print("\n[STEP 4] Setting reference positions...")
    print("-" * 40)

    time.sleep(0.1)

    positions = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
    if positions is None:
        print("[WARNING] Sync read failed, trying individual reads...")
        positions = {}
        for mid in all_motor_ids:
            pos = controller.read_position(mid)
            if pos is not None:
                positions[mid] = pos
            else:
                print(f"[ERROR] Failed to read position for motor {mid}")
                controller.disconnect()
                return

    reference_positions = positions.copy()
    print("\nReference positions (STAND position):")
    for mid in all_motor_ids:
        ref_pos = reference_positions[mid]
        marker = ""
        if mid in group_a:
            marker = f" -> will go to {ref_pos - target_offset}r (-{target_offset}r)"
        elif mid in group_b:
            marker = f" -> will go to {ref_pos + target_offset}r (+{target_offset}r)"
        print(f"  Motor {mid}: {ref_pos} raw{marker}")

    # Step 5: Enable torque on all motors
    print("\n[STEP 5] Enabling torque on all motors...")
    print("-" * 40)

    for mid in all_motor_ids:
        if controller.set_torque(True, mid):
            print(f"[OK] Motor {mid}: Torque enabled")
        else:
            print(f"[ERROR] Motor {mid}: Failed to enable torque")
            controller.disconnect()
            return

    # Step 6: Run sit-stand sequence
    print("\n[STEP 6] Starting Sit-Stand Sequence")
    print("-" * 40)
    print("Sequence: STAND -> SIT -> STAND")
    print(f"Frequency: {control_freq} Hz")
    print(f"Step size: {step_size} raw units per cycle")
    print(f"Repeats: {repeat}")
    print("\nPress Ctrl+C to stop at any time")
    print("-" * 40)

    target_period = 1.0 / control_freq
    current_offset = 0  # Start at reference (standing)

    try:
        for cycle in range(1, repeat + 1):
            if repeat > 1:
                print(f"\n{'=' * 40}")
                print(f"  CYCLE {cycle} of {repeat}")
                print(f"{'=' * 40}")

            # Phase 1: STAND -> SIT (go to +target_offset for group B)
            print("\n[PHASE 1] Moving to SIT position...")
            print(f"  Motors 9, 12: 0r -> +{target_offset}r")
            print(f"  Motors 3, 6:  0r -> -{target_offset}r")

            while current_offset < target_offset:
                loop_start = time.time()

                current_offset = min(current_offset + step_size, target_offset)

                # Build target positions
                target_positions = {}
                for mid in all_motor_ids:
                    if mid in group_a:
                        target_positions[mid] = reference_positions[mid] - current_offset
                    elif mid in group_b:
                        target_positions[mid] = reference_positions[mid] + current_offset
                    else:
                        target_positions[mid] = reference_positions[mid]

                if not controller.sync_write_positions(target_positions):
                    print("\n[ERROR] Failed to write positions")
                    break

                # Status update
                print(f"\r  Offset: {current_offset:4d}r / {target_offset}r", end="", flush=True)

                # Maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"\r  Offset: {current_offset:4d}r / {target_offset}r - DONE")

            # Hold at sit position briefly
            print("\n[HOLD] At SIT position for 1 second...")
            time.sleep(1.0)

            # Phase 2: SIT -> STAND (return to reference)
            print("\n[PHASE 2] Moving to STAND position...")
            print(f"  Motors 9, 12: +{target_offset}r -> 0r")
            print(f"  Motors 3, 6:  -{target_offset}r -> 0r")

            while current_offset > 0:
                loop_start = time.time()

                current_offset = max(current_offset - step_size, 0)

                # Build target positions
                target_positions = {}
                for mid in all_motor_ids:
                    if mid in group_a:
                        target_positions[mid] = reference_positions[mid] - current_offset
                    elif mid in group_b:
                        target_positions[mid] = reference_positions[mid] + current_offset
                    else:
                        target_positions[mid] = reference_positions[mid]

                if not controller.sync_write_positions(target_positions):
                    print("\n[ERROR] Failed to write positions")
                    break

                # Status update
                print(f"\r  Offset: {current_offset:4d}r / 0r", end="", flush=True)

                # Maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"\r  Offset: {current_offset:4d}r / 0r - DONE")

            # Hold at stand position briefly between cycles (except last)
            if cycle < repeat:
                print("\n[HOLD] At STAND position for 1 second...")
                time.sleep(1.0)

        print("\n" + "=" * 60)
        print(f"[COMPLETE] Sit-Stand sequence finished! ({repeat} cycle(s))")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
        print("[INFO] Returning to reference position...")

        # Return to reference
        controller.sync_write_positions(reference_positions)
        time.sleep(0.5)

    finally:
        print("\n[CLEANUP] Disabling torque and disconnecting...")
        controller.disconnect()
        print("\n[DONE] Test complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Basic sit-stand motion test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 5 Hz, step 30, target 500r, 1 cycle
  uv run python basic_sitstand.py

  # Faster movement (10 Hz)
  uv run python basic_sitstand.py -f 10

  # Smaller steps for smoother motion
  uv run python basic_sitstand.py -s 15

  # Different target offset
  uv run python basic_sitstand.py -t 400

  # Repeat 5 times
  uv run python basic_sitstand.py -r 5

Motion:
  STAND (reference) -> SIT (+500r for 9,12 / -500r for 3,6) -> STAND (repeat x times)
        """
    )

    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=5.0,
        help="Control loop frequency in Hz (default: 5)"
    )

    parser.add_argument(
        "-s", "--step",
        type=int,
        default=30,
        help="Step size in raw units (default: 30)"
    )

    parser.add_argument(
        "-t", "--target",
        type=int,
        default=500,
        help="Target offset for sit position in raw units (default: 500)"
    )

    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the sit-stand cycle (default: 1)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.freq <= 0 or args.freq > 100:
        print("[ERROR] Frequency must be between 0 and 100 Hz")
        sys.exit(1)

    if args.step <= 0:
        print("[ERROR] Step size must be positive")
        sys.exit(1)

    if args.target <= 0:
        print("[ERROR] Target offset must be positive")
        sys.exit(1)

    if args.repeat <= 0:
        print("[ERROR] Repeat count must be positive")
        sys.exit(1)

    run_sitstand(
        control_freq=args.freq,
        step_size=args.step,
        target_offset=args.target,
        repeat=args.repeat
    )


if __name__ == "__main__":
    main()
