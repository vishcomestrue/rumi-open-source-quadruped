#!/usr/bin/env python3
"""
Interactive Multi-Motor Test Script (GroupSync)

Automatically discovers and controls ALL connected motors.
Uses GroupSyncRead/Write for efficient communication (all motors in ONE packet).

Features:
1. Auto-discovers all connected motors via broadcast ping
2. Uses GroupSync for simultaneous read/write (faster, synchronized)
3. Sets each motor's initial position as its 0° reference
4. Accepts comma-separated degree inputs for all motors
5. All inputs are ABSOLUTE from each motor's reference

Input format: "deg1, deg2, deg3, ..." (comma-separated, one per motor)
Example with 3 motors: "90, 45, -30"
Use '_' to skip a motor: "90, _, -30"
"""

import sys
import time
from mx64_controller import MX64Controller


def main():
    print("\n" + "=" * 60)
    print("   MX-64 Multi-Motor Test (GroupSync)")
    print("=" * 60)

    controller = MX64Controller()

    # Step 1: Connect (auto-scans ports and baud rates)
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

    motor_ids = sorted(motors.keys())
    num_motors = len(motor_ids)
    print(f"\nFound {num_motors} motor(s): {motor_ids}")

    # Step 3: Initialize GroupSync for efficient multi-motor communication
    print("\n[STEP 3] Initializing GroupSync...")
    print("-" * 40)

    if not controller.init_sync(motor_ids):
        print("[FAILED] Could not initialize GroupSync.")
        controller.disconnect()
        return

    print("[OK] GroupSync initialized - all motors will be read/written in ONE packet")

    # Step 4: Read status for all motors
    print("\n[STEP 4] Reading motor status...")
    print("-" * 40)

    for mid in motor_ids:
        controller.print_status(mid)

    # Step 5: Choose operating mode (applied to all motors)
    print("\n[STEP 5] Select operating mode (for all motors)...")
    print("-" * 40)
    print("1. Position Control Mode (0-360 degrees)")
    print("2. Extended Position Mode (multi-turn)")
    print("3. Keep current mode")

    try:
        mode_input = input("Enter choice [3]: ").strip()
        mode_choice = int(mode_input) if mode_input else 3
    except (ValueError, KeyboardInterrupt):
        mode_choice = 3

    for mid in motor_ids:
        if mode_choice == 1:
            controller.set_position_control_mode(mid)
        elif mode_choice == 2:
            controller.set_extended_position_mode(mid)

    # Longer delay after mode changes for motors to stabilize
    # Mode changes require EEPROM writes which can take time
    if mode_choice in [1, 2]:
        print("Waiting for motors to stabilize after mode change...")
        time.sleep(0.3)  # 300ms for reliable stabilization

    # Check if extended mode (use first motor as reference)
    is_extended = controller.is_extended_position_mode(motor_ids[0])

    # Step 6: Read initial positions as REFERENCE POINTS (using sync read)
    print("\n[STEP 6] Setting reference positions...")
    print("-" * 40)

    # Try reading with more retries for initial read after mode change
    # Use longer delays and more retries since this is the first sync read
    positions = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
    if positions is None:
        print("[WARNING] Sync read failed, trying individual reads as fallback...")
        positions = {}
        for mid in motor_ids:
            pos = controller.read_position(mid)
            if pos is not None:
                positions[mid] = pos
            else:
                print(f"[ERROR] Failed to read position for motor {mid}")
                controller.disconnect()
                return

    reference_positions = positions.copy()
    for mid in motor_ids:
        ref_pos = reference_positions[mid]
        ref_degrees = ref_pos * controller.POSITION_TO_DEGREES
        print(f"Motor {mid}: Reference (0°) = {ref_pos} raw ({ref_degrees:.1f}° absolute)")

    print("\nAll degree inputs will be relative to these references.")

    # Step 7: Enable torque on all motors
    print("\n[STEP 7] Enabling torque on all motors...")
    print("-" * 40)

    for mid in motor_ids:
        if controller.set_torque(True, mid):
            print(f"[OK] Motor {mid}: Torque enabled")
        else:
            print(f"[ERROR] Motor {mid}: Failed to enable torque")
            controller.disconnect()
            return

    # Step 8: Interactive position control loop
    print("\n[STEP 8] Interactive Position Control (GroupSync)")
    print("-" * 40)
    print(f"Controlling {num_motors} motor(s): {motor_ids}")
    print("Communication: GroupSync (all motors in ONE packet)")

    if is_extended:
        print("Mode: Extended Position (Multi-turn)")
    else:
        print("Mode: Position Control (0-360°)")

    print(f"\nInput format: {num_motors} comma-separated degree values")
    if num_motors <= 4:
        example = ", ".join(["90"] * num_motors)
        print(f"  Example: {example}")
    else:
        print(f"  Example: 90, 45, -30, ... ({num_motors} values)")

    print("  Use '_' to skip a motor (e.g., '90, _, -30')")
    print("  Use 'r' suffix for raw values (e.g., '1024r, 2048r')")
    print("\nCommands:")
    print("  'q' - quit")
    print("  's' - show status of all motors")
    print("  'z' - re-zero all motors at current positions")
    print("  'h' - home all motors (return to 0°)")
    print("-" * 40)

    try:
        while True:
            # Read all positions in ONE packet using GroupSyncRead
            positions = controller.sync_read_positions()

            if positions is not None:
                print(f"\nCurrent positions ({num_motors} motors):")
                for mid in motor_ids:
                    pos = positions[mid]
                    relative_pos = pos - reference_positions[mid]
                    relative_degrees = relative_pos * controller.POSITION_TO_DEGREES
                    print(f"  Motor {mid}: {relative_degrees:8.1f}° [raw: {pos}]")
            else:
                print("\n[WARNING] Failed to read positions")

            # Get user input
            user_input = input(f"\nEnter {num_motors} positions (degrees) or command: ").strip()

            if not user_input:
                continue

            # Handle commands (case insensitive)
            cmd = user_input.lower()

            if cmd == 'q':
                print("Quitting...")
                break

            elif cmd == 's':
                for mid in motor_ids:
                    controller.print_status(mid)
                continue

            elif cmd == 'z':
                # Re-zero all motors at current position using sync read
                positions = controller.sync_read_positions()
                if positions is not None:
                    reference_positions = positions.copy()
                    for mid in motor_ids:
                        print(f"[OK] Motor {mid}: New reference = {positions[mid]} raw = 0°")
                else:
                    print("[ERROR] Failed to read positions for re-zero")
                continue

            elif cmd == 'h':
                # Home all motors (go to 0° / reference position) using sync write
                print("Homing all motors to 0°...")
                controller.sync_write_positions(reference_positions)

                # Wait for movement
                print("Moving", end="", flush=True)
                for _ in range(50):
                    time.sleep(0.1)
                    all_stopped = True
                    for mid in motor_ids:
                        if controller.is_moving(mid):
                            all_stopped = False
                            break
                    if all_stopped:
                        break
                    print(".", end="", flush=True)
                print(" Done!")
                continue

            # Parse position input (comma-separated)
            try:
                parts = [p.strip() for p in user_input.split(",")]

                if len(parts) != num_motors:
                    print(f"[ERROR] Expected {num_motors} values, got {len(parts)}")
                    print(f"Format: deg1, deg2, ... ({num_motors} values)")
                    continue

                target_positions = {}
                for mid, part in zip(motor_ids, parts):
                    # Skip if underscore
                    if part == '_':
                        continue

                    if part.lower().endswith('r'):
                        # Raw offset from reference
                        raw_offset = int(part[:-1])
                        target_positions[mid] = reference_positions[mid] + raw_offset
                    else:
                        # Degrees from reference
                        degrees_val = float(part)
                        target_positions[mid] = reference_positions[mid] + int(
                            degrees_val * controller.DEGREES_TO_POSITION
                        )

                if not target_positions:
                    print("[INFO] No motors to move (all skipped)")
                    continue

                # Display targets
                print("Targets:")
                for mid in motor_ids:
                    if mid in target_positions:
                        target_pos = target_positions[mid]
                        target_rel = target_pos - reference_positions[mid]
                        target_deg = target_rel * controller.POSITION_TO_DEGREES
                        print(f"  Motor {mid}: {target_deg:8.1f}° [raw: {target_pos}]")
                    else:
                        print(f"  Motor {mid}: (skipped)")

                # Write all positions in ONE packet using GroupSyncWrite
                if controller.sync_write_positions(target_positions):
                    # Wait for all motors to stop moving
                    print("Moving", end="", flush=True)
                    for _ in range(50):  # Max 5 seconds
                        time.sleep(0.1)
                        all_stopped = True
                        for mid in target_positions.keys():
                            if controller.is_moving(mid):
                                all_stopped = False
                                break
                        if all_stopped:
                            break
                        print(".", end="", flush=True)
                    print(" Done!")
                else:
                    print("[ERROR] Failed to send position command")

            except ValueError as e:
                print(f"[ERROR] Invalid input: {e}")
                print(f"Format: deg1, deg2, ... ({num_motors} comma-separated values)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup - disconnect handles torque disable and sync cleanup
        print("\n[CLEANUP] Disabling torque and disconnecting...")
        controller.disconnect()
        print("\n[DONE] Test complete.")


if __name__ == "__main__":
    main()
