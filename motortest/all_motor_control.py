#!/usr/bin/env python3
"""
Automated Motor Oscillation Test (Multiples of 3)

Controls motors with IDs that are multiples of 3 (3, 6, 9, 12, ...).
Oscillates them between -300r and +300r from their reference positions.

Motion Convention (opposite phase):
- Group A (motors 3, 6): move in NORMAL direction
- Group B (motors 9, 12): move in OPPOSITE direction
- When Group A is at +300r, Group B is at -300r

Features:
1. Auto-discovers motors and filters to multiples of 3
2. Fixed frequency control loop (configurable, default 5 Hz)
3. Smooth oscillation with configurable step size
4. Uses GroupSync for efficient communication
5. Opposite phase motion for quadruped gait patterns
"""

import sys
import time
import argparse
from mx64_controller import MX64Controller


def run_oscillation_test(control_freq=5.0, step_size=30, min_pos=-300, max_pos=300):
    """
    Run automated oscillation test on motors with IDs that are multiples of 3.

    Args:
        control_freq: Control loop frequency in Hz (default: 5 Hz)
        step_size: Position step size in raw units (default: 30)
        min_pos: Minimum position offset from reference (default: -300)
        max_pos: Maximum position offset from reference (default: +300)
    """
    print("\n" + "=" * 60)
    print("   Motor Oscillation Test (Multiples of 3)")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Control frequency: {control_freq} Hz ({1000/control_freq:.1f} ms period)")
    print(f"  Step size: {step_size} raw units")
    print(f"  Range: {min_pos}r to {max_pos}r from reference")
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

    # Filter to motors with IDs that are multiples of 3
    target_motor_ids = [mid for mid in all_motor_ids if mid % 3 == 0]

    if not target_motor_ids:
        print("\n[FAILED] No motors with IDs that are multiples of 3 found!")
        print(f"Available motor IDs: {all_motor_ids}")
        controller.disconnect()
        return

    # Split target motors into two groups for opposite motion
    # Group A (3, 6): move with positive offset when offset is positive
    # Group B (9, 12): move with NEGATIVE offset when offset is positive (opposite phase)
    group_a = [mid for mid in target_motor_ids if mid in [3, 6]]
    group_b = [mid for mid in target_motor_ids if mid in [9, 12]]

    print(f"\n[INFO] Target motors (multiples of 3): {target_motor_ids}")
    print(f"[INFO]   Group A (motors 3, 6) - normal direction: {group_a}")
    print(f"[INFO]   Group B (motors 9, 12) - opposite direction: {group_b}")
    print(f"[INFO] Other motors will hold their positions: {[m for m in all_motor_ids if m not in target_motor_ids]}")

    # Step 3: Initialize GroupSync for ALL motors (needed for sync operations)
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

    # Wait a moment for stable communication
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
    print("\nReference positions (0 point):")
    for mid in all_motor_ids:
        ref_pos = reference_positions[mid]
        marker = " <-- TARGET" if mid in target_motor_ids else ""
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

    # Step 6: Run oscillation loop
    print("\n[STEP 6] Starting Oscillation Loop")
    print("-" * 40)
    print(f"Controlling {len(target_motor_ids)} motor(s): {target_motor_ids}")
    print(f"Frequency: {control_freq} Hz")
    print(f"Pattern: {min_pos}r <-> {max_pos}r (oscillating)")
    print(f"  Group A (3, 6): normal direction")
    print(f"  Group B (9, 12): opposite direction (when A is +, B is -)")
    print(f"Step size: {step_size} raw units per cycle")
    print("\nPress Ctrl+C to stop")
    print("-" * 40)

    # Calculate timing
    target_period = 1.0 / control_freq  # seconds per loop iteration

    # Oscillation state
    current_offset = 0  # Current position offset from reference
    direction = 1  # 1 = moving towards max, -1 = moving towards min

    # Statistics
    loop_count = 0
    loop_times = []
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # Calculate next offset
            next_offset = current_offset + (step_size * direction)

            # Check bounds and reverse direction
            if next_offset >= max_pos:
                next_offset = max_pos
                direction = -1
            elif next_offset <= min_pos:
                next_offset = min_pos
                direction = 1

            current_offset = next_offset

            # Build target positions:
            # - Group A (3, 6): reference + current_offset (normal direction)
            # - Group B (9, 12): reference - current_offset (opposite direction)
            # - Other motors: keep at reference (hold position)
            target_positions = {}
            for mid in all_motor_ids:
                if mid in group_a:
                    # Group A: normal direction
                    target_positions[mid] = reference_positions[mid] + current_offset
                elif mid in group_b:
                    # Group B: opposite direction
                    target_positions[mid] = reference_positions[mid] - current_offset
                else:
                    # Hold other motors at their reference
                    target_positions[mid] = reference_positions[mid]

            # Write positions using GroupSyncWrite
            if not controller.sync_write_positions(target_positions):
                print("\n[ERROR] Failed to write positions")
                break

            # Read current positions (optional, for monitoring)
            # Skip every few iterations to reduce overhead if needed
            if loop_count % 5 == 0:  # Read every 5th iteration
                actual_positions = controller.sync_read_positions(max_retries=2, retry_delay=0.005)
                if actual_positions is not None:
                    # Print status line showing both groups
                    status_a = []
                    status_b = []
                    for mid in group_a:
                        actual_offset = actual_positions[mid] - reference_positions[mid]
                        status_a.append(f"M{mid}:{actual_offset:+5d}r")
                    for mid in group_b:
                        actual_offset = actual_positions[mid] - reference_positions[mid]
                        status_b.append(f"M{mid}:{actual_offset:+5d}r")
                    grp_a_str = ' '.join(status_a) if status_a else "none"
                    grp_b_str = ' '.join(status_b) if status_b else "none"
                    print(f"\r[{loop_count:6d}] A(+{current_offset:+4d}r): {grp_a_str} | B({-current_offset:+4d}r): {grp_b_str}", end="", flush=True)

            loop_count += 1

            # Calculate elapsed time and sleep to maintain frequency
            loop_elapsed = time.time() - loop_start
            loop_times.append(loop_elapsed)

            sleep_time = target_period - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INFO] Stopping oscillation test...")

    finally:
        # Print statistics
        total_time = time.time() - start_time
        if loop_times:
            avg_loop_time = sum(loop_times) / len(loop_times)
            actual_freq = loop_count / total_time if total_time > 0 else 0

            print("\n" + "=" * 60)
            print("[STATISTICS]")
            print(f"  Total loops: {loop_count}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Target frequency: {control_freq:.1f} Hz")
            print(f"  Actual frequency: {actual_freq:.2f} Hz")
            print(f"  Avg loop time: {avg_loop_time*1000:.2f} ms")
            print(f"  Min loop time: {min(loop_times)*1000:.2f} ms")
            print(f"  Max loop time: {max(loop_times)*1000:.2f} ms")
            print("=" * 60)

        # Return motors to reference position before disconnecting
        print("\n[CLEANUP] Returning motors to reference positions...")
        controller.sync_write_positions(reference_positions)
        time.sleep(0.5)  # Wait for motors to reach position

        print("[CLEANUP] Disabling torque and disconnecting...")
        controller.disconnect()
        print("\n[DONE] Test complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Oscillate motors with IDs that are multiples of 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 5 Hz, step 30, range -300 to +300
  uv run python all_motor_control.py

  # 10 Hz control frequency
  uv run python all_motor_control.py -f 10

  # Smaller steps for smoother motion
  uv run python all_motor_control.py -s 15

  # Larger range
  uv run python all_motor_control.py --min -500 --max 500

  # Combined options
  uv run python all_motor_control.py -f 10 -s 20 --min -400 --max 400

Motion convention:
  - Motors 3, 6 (Group A): move in NORMAL direction
  - Motors 9, 12 (Group B): move in OPPOSITE direction
  - When Group A is at +300r, Group B is at -300r
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
        "--min",
        type=int,
        default=-300,
        dest="min_pos",
        help="Minimum position offset from reference in raw units (default: -300)"
    )

    parser.add_argument(
        "--max",
        type=int,
        default=300,
        dest="max_pos",
        help="Maximum position offset from reference in raw units (default: +300)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.freq <= 0 or args.freq > 100:
        print("[ERROR] Frequency must be between 0 and 100 Hz")
        sys.exit(1)

    if args.step <= 0:
        print("[ERROR] Step size must be positive")
        sys.exit(1)

    if args.min_pos >= args.max_pos:
        print("[ERROR] Min position must be less than max position")
        sys.exit(1)

    run_oscillation_test(
        control_freq=args.freq,
        step_size=args.step,
        min_pos=args.min_pos,
        max_pos=args.max_pos
    )


if __name__ == "__main__":
    main()
