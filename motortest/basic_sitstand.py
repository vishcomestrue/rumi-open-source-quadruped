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
import matplotlib.pyplot as plt
from collections import defaultdict


# Motor labels: FLH = Front Left Hip, FLT = Front Left Thigh, FLC = Front Left Calf, etc.
MOTOR_LABELS = {
    1: "FLH",   # Front Left Hip
    2: "FLT",   # Front Left Thigh
    3: "FLC",   # Front Left Calf
    4: "RLH",   # Rear Left Hip
    5: "RLT",   # Rear Left Thigh
    6: "RLC",   # Rear Left Calf
    7: "RRH",   # Rear Right Hip
    8: "RRT",   # Rear Right Thigh
    9: "RRC",   # Rear Right Calf
    10: "FRH",  # Front Right Hip
    11: "FRT",  # Front Right Thigh
    12: "FRC",  # Front Right Calf
}


def read_all_torques(controller, motor_ids):
    """
    Read current (torque) from all motors.
    
    Args:
        controller: MX64Controller instance
        motor_ids: List of motor IDs to read from
    
    Returns:
        dict: {motor_id: current_ma} or None if any read fails
    """
    torques = {}
    for mid in motor_ids:
        current_ma = controller.read_current_ma(mid)
        if current_ma is None:
            return None
        torques[mid] = current_ma
    return torques


def print_torques(torques):
    """
    Print torque values with short labels.
    
    Args:
        torques: dict {motor_id: current_ma}
    """
    if torques is None:
        print("  [ERROR] Failed to read torques")
        return
    
    # Print in a compact format: FLH:123mA FLT:456mA ...
    torque_strs = [f"{MOTOR_LABELS.get(mid, f'M{mid}')}:{current:.0f}mA" 
                   for mid, current in sorted(torques.items())]
    print(f"  Torques: {' '.join(torque_strs)}")


def print_torque_stats(torque_history):
    """
    Print min/max torque statistics for each motor.
    
    Args:
        torque_history: dict {motor_id: [list of torque values]}
    """
    print("\n" + "=" * 60)
    print("   Torque Statistics (Min/Max)")
    print("=" * 60)
    
    for mid in sorted(torque_history.keys()):
        values = torque_history[mid]
        if values:
            min_torque = min(values)
            max_torque = max(values)
            avg_torque = sum(values) / len(values)
            label = MOTOR_LABELS.get(mid, f"M{mid}")
            print(f"  {label:4s} (Motor {mid:2d}): Min={min_torque:6.0f}mA  Max={max_torque:6.0f}mA  Avg={avg_torque:6.0f}mA")


def plot_torque_history(torque_history, timestamps, control_freq):
    """
    Plot torque history for all motors in a single figure.
    
    Args:
        torque_history: dict {motor_id: [list of torque values]}
        timestamps: list of time points
        control_freq: control frequency for reference
    """
    if not torque_history or not timestamps:
        print("[INFO] No torque data to plot")
        return
    
    # Convert timestamps to relative time in seconds
    if timestamps:
        time_array = [(t - timestamps[0]) for t in timestamps]
    else:
        time_array = []
    
    plt.figure(figsize=(14, 10))
    
    # Plot all motors
    for mid in sorted(torque_history.keys()):
        values = torque_history[mid]
        label = MOTOR_LABELS.get(mid, f"M{mid}")
        plt.plot(time_array[:len(values)], values, label=f"{label} (M{mid})", linewidth=1.5)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Current (mA)', fontsize=12)
    plt.title(f'Motor Torque History - Control Frequency: {control_freq} Hz', fontsize=14)
    plt.legend(loc='best', ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show
    filename = f'torque_history_{int(time.time())}.png'
    plt.savefig(filename, dpi=150)
    print(f"\n[INFO] Torque plot saved to: {filename}")
    plt.show()


def run_sitstand(control_freq=5.0, duration=2.0, target_knee=500, target_hip=-200, repeat=1, plot=False):
    """
    Run basic sit-stand motion.

    Args:
        control_freq: Control loop frequency in Hz (default: 5 Hz)
        duration: Time in seconds to reach full sit position (default: 2.0)
        target_knee: Target offset for knee motors 3,6,9,12 (default: 500)
        target_hip: Target offset for hip motors - goes to 2,5; negative goes to 8,11 (default: -200)
        repeat: Number of times to repeat the sit-stand cycle (default: 1)
    """
    # Calculate step sizes based on duration and frequency
    num_steps = duration * control_freq
    sk = abs(target_knee) / num_steps  # Step size for knee
    sh = abs(target_hip) / num_steps   # Step size for hip

    print("\n" + "=" * 60)
    print("   Basic Sit-Stand Motion Test")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Control frequency: {control_freq} Hz ({1000/control_freq:.1f} ms period)")
    print(f"  Duration: {duration:.2f} seconds ({num_steps:.0f} steps)")
    print(f"  Step size knee (sk): {sk:.2f} raw units/step")
    print(f"  Step size hip (sh): {sh:.2f} raw units/step")
    print(f"  Target Knee: Motors 9,12 -> +{target_knee}r | Motors 3,6 -> -{target_knee}r")
    print(f"  Target Hip:  Motors 2,5 -> {target_hip:+d}r | Motors 8,11 -> {-target_hip:+d}r")
    print(f"  Repeat: {repeat} cycle(s)")
    print("=" * 60)

    controller = MX64Controller()

    # Initialize: connect, discover, configure, enable torque (all in one!)
    all_motor_ids = controller.initialize(expected_motors=12)
    if all_motor_ids is None:
        print("\n[FAILED] Initialization failed.")
        return

    # Get home positions and initial offsets from initialization
    home_positions = controller.get_home_positions()
    initial_offsets = controller.initial_offsets.copy()

    # Calculate reference positions (home + initial offsets)
    reference_positions = {mid: home_positions[mid] + initial_offsets[mid] for mid in all_motor_ids}

    # Filter to motors with IDs that are multiples of 3
    target_motor_ids = [mid for mid in all_motor_ids if mid % 3 == 0]

    if not target_motor_ids:
        print("\n[FAILED] No motors with IDs that are multiples of 3 found!")
        print(f"Available motor IDs: {all_motor_ids}")
        controller.disconnect()
        return

    # Split into groups for opposite motion
    group_a = [mid for mid in target_motor_ids if mid in [3, 6]]   # Go negative (offset 1)
    group_b = [mid for mid in target_motor_ids if mid in [9, 12]]  # Go positive (offset 1)
    group_c = [mid for mid in all_motor_ids if mid in [2, 5]]  # Go negative (offset 2)
    group_d = [mid for mid in all_motor_ids if mid in [8, 11]]  # Go positive (offset 2)

    print(f"\n[INFO] Target knee motors: {target_motor_ids}")
    print(f"[INFO]   Group A (3, 6) -> -{target_knee}r: {group_a}")
    print(f"[INFO]   Group B (9, 12) -> +{target_knee}r: {group_b}")
    print(f"[INFO] Target hip motors:")
    print(f"[INFO]   Group C (2, 5) -> {target_hip:+d}r: {group_c}")
    print(f"[INFO]   Group D (8, 11) -> {-target_hip:+d}r: {group_d}")
    print(f"[INFO] Other motors hold position: {[m for m in all_motor_ids if m not in target_motor_ids and m not in group_c and m not in group_d]}")

    print("\nReference positions (STAND position):")
    for mid in all_motor_ids:
        ref_pos = reference_positions[mid]
        marker = ""
        if mid in group_a:
            marker = f" -> will go to {ref_pos - target_knee}r (-{target_knee}r)"
        elif mid in group_b:
            marker = f" -> will go to {ref_pos + target_knee}r (+{target_knee}r)"
        elif mid in group_c:
            marker = f" -> will go to {ref_pos + target_hip}r ({target_hip:+d}r)"
        elif mid in group_d:
            marker = f" -> will go to {ref_pos - target_hip}r ({-target_hip:+d}r)"
        print(f"  Motor {mid}: {ref_pos} raw{marker}")

    # Start sit-stand sequence
    print("\n" + "=" * 60)
    print("   Starting Sit-Stand Sequence")
    print("=" * 60)
    print("Sequence: STAND -> SIT -> STAND")
    print(f"Frequency: {control_freq} Hz")
    print(f"Step sizes: sk={sk:.2f}, sh={sh:.2f} raw units per cycle")
    print(f"Repeats: {repeat}")
    print("\nPress Ctrl+C to stop at any time")
    print("-" * 40)

    target_period = 1.0 / control_freq
    current_knee = 0  # Start at reference (standing)
    current_hip = 0  # Start at reference for hip motors

    # Determine step directions and sizes
    knee_step = sk
    hip_step = sh

    # Initialize torque tracking
    torque_history = defaultdict(list)  # {motor_id: [torque1, torque2, ...]}
    timestamps = []  # Time points for each torque reading
    start_time = time.time()

    try:
        for cycle in range(1, repeat + 1):
            if repeat > 1:
                print(f"\n{'=' * 40}")
                print(f"  CYCLE {cycle} of {repeat}")
                print(f"{'=' * 40}")

            # Phase 1: STAND -> SIT (go to target offsets)
            print("\n[PHASE 1] Moving to SIT position...")
            print(f"  Motors 9, 12 (knee): 0r -> +{target_knee}r")
            print(f"  Motors 3, 6  (knee): 0r -> -{target_knee}r")
            print(f"  Motors 2, 5  (hip):  0r -> {target_hip:+d}r")
            print(f"  Motors 8, 11 (hip):  0r -> {-target_hip:+d}r")
            
            # Read and display initial torques
            torques = read_all_torques(controller, all_motor_ids)
            print_torques(torques)

            # Continue until both reach their targets
            while (current_knee < target_knee if target_knee > 0 else current_knee > target_knee) or \
                  (current_hip < target_hip if target_hip > 0 else current_hip > target_hip):
                loop_start = time.time()

                # Move knee toward target
                if target_knee > 0:
                    if current_knee < target_knee:
                        current_knee = min(current_knee + knee_step, target_knee)
                else:
                    if current_knee > target_knee:
                        current_knee = max(current_knee - knee_step, target_knee)

                # Move hip toward target
                if target_hip > 0:
                    if current_hip < target_hip:
                        current_hip = min(current_hip + abs(hip_step), target_hip)
                else:
                    if current_hip > target_hip:
                        current_hip = max(current_hip - abs(hip_step), target_hip)

                # Build target positions
                target_positions = {}
                for mid in all_motor_ids:
                    if mid in group_a:
                        target_positions[mid] = reference_positions[mid] - current_knee
                    elif mid in group_b:
                        target_positions[mid] = reference_positions[mid] + current_knee
                    elif mid in group_c:
                        target_positions[mid] = reference_positions[mid] + current_hip
                    elif mid in group_d:
                        target_positions[mid] = reference_positions[mid] - current_hip
                    else:
                        target_positions[mid] = reference_positions[mid]

                if not controller.sync_write_positions(target_positions):
                    print("\n[ERROR] Failed to write positions")
                    break

                # Read and store torques
                torques = read_all_torques(controller, all_motor_ids)
                if torques:
                    timestamps.append(time.time())
                    for mid, torque_val in torques.items():
                        torque_history[mid].append(torque_val)

                # Status update
                print(f"\r  Knee: {current_knee:5.0f}r / {target_knee}r | Hip: {current_hip:5.0f}r / {target_hip}r", end="", flush=True)

                # Maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            
            print(f"\r  Knee: {current_knee:5.0f}r / {target_knee}r | Hip: {current_hip:5.0f}r / {target_hip}r - DONE")
            
            # Print Phase 1 torque stats
            print("\n[PHASE 1 COMPLETE]")
            torques = read_all_torques(controller, all_motor_ids)
            print_torques(torques)

            # Hold at sit position briefly
            print("\n[HOLD] At SIT position for 1 second...")
            time.sleep(1.0)

            # Phase 2: SIT -> STAND (return to reference)
            print("\n[PHASE 2] Moving to STAND position...")
            print(f"  Motors 9, 12 (knee): {target_knee:+d}r -> 0r")
            print(f"  Motors 3, 6  (knee): {-target_knee:+d}r -> 0r")
            print(f"  Motors 2, 5  (hip):  {target_hip:+d}r -> 0r")
            print(f"  Motors 8, 11 (hip):  {-target_hip:+d}r -> 0r")

            # Return to zero
            while current_knee != 0 or current_hip != 0:
                loop_start = time.time()

                # Move knee back to zero
                if current_knee > 0:
                    current_knee = max(current_knee - knee_step, 0)
                elif current_knee < 0:
                    current_knee = min(current_knee + knee_step, 0)

                # Move hip back to zero
                if current_hip > 0:
                    current_hip = max(current_hip - abs(hip_step), 0)
                elif current_hip < 0:
                    current_hip = min(current_hip + abs(hip_step), 0)

                # Build target positions
                target_positions = {}
                for mid in all_motor_ids:
                    if mid in group_a:
                        target_positions[mid] = reference_positions[mid] - current_knee
                    elif mid in group_b:
                        target_positions[mid] = reference_positions[mid] + current_knee
                    elif mid in group_c:
                        target_positions[mid] = reference_positions[mid] + current_hip
                    elif mid in group_d:
                        target_positions[mid] = reference_positions[mid] - current_hip
                    else:
                        target_positions[mid] = reference_positions[mid]

                if not controller.sync_write_positions(target_positions):
                    print("\n[ERROR] Failed to write positions")
                    break

                # Read and store torques
                torques = read_all_torques(controller, all_motor_ids)
                if torques:
                    timestamps.append(time.time())
                    for mid, torque_val in torques.items():
                        torque_history[mid].append(torque_val)

                # Status update
                print(f"\r  Knee: {current_knee:5.0f}r / 0r | Hip: {current_hip:5.0f}r / 0r", end="", flush=True)

                # Maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"\r  Knee: {current_knee:5.0f}r / 0r | Hip: {current_hip:5.0f}r / 0r - DONE")

            # Print Phase 2 torque stats
            print("\n[PHASE 2 COMPLETE]")
            torques = read_all_torques(controller, all_motor_ids)
            print_torques(torques)

            # Hold at stand position briefly between cycles (except last)
            if cycle < repeat:
                print("\n[HOLD] At STAND position for 1 second...")
                time.sleep(1.0)

        print("\n" + "=" * 60)
        print(f"[COMPLETE] Sit-Stand sequence finished! ({repeat} cycle(s))")
        print("=" * 60)

        # Print torque statistics and plot
        if torque_history:
            print_torque_stats(torque_history)
            if plot:
                plot_torque_history(torque_history, timestamps, control_freq)

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
        print("[INFO] Returning to reference position...")

        # Return to reference
        controller.sync_write_positions(reference_positions)
        time.sleep(0.5)

    finally:
        print("\n[CLEANUP] Disabling torque and disconnecting...")
        controller.disconnect()
        
        # Show torque statistics if we have data (even on interrupt)
        if torque_history:
            print_torque_stats(torque_history)
            if plot:
                try:
                    plot_torque_history(torque_history, timestamps, control_freq)
                except Exception as e:
                    print(f"[WARNING] Could not generate plot: {e}")
        
        print("\n[DONE] Test complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Basic sit-stand motion test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 5 Hz, 2.0s duration, knee 500r, hip -200r, 1 cycle
  uv run python basic_sitstand.py

  # Faster control loop (10 Hz)
  uv run python basic_sitstand.py -f 10

  # Slower transition (5 seconds)
  uv run python basic_sitstand.py -d 5.0

  # Different target offsets
  uv run python basic_sitstand.py -tk 400 -th -150

  # Repeat 5 times with 3 second duration
  uv run python basic_sitstand.py -r 5 -d 3.0

Motion:
  STAND (reference) -> SIT (knee: +500r for 9,12 / -500r for 3,6 | hip: -200r for 2,5 / +200r for 8,11) -> STAND (repeat x times)

Step sizes (sk, sh) are automatically calculated as:
  sk = |target_knee| / (duration * frequency)
  sh = |target_hip| / (duration * frequency)
        """
    )

    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=5.0,
        help="Control loop frequency in Hz (default: 5)"
    )

    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=2.0,
        help="Duration in seconds to reach full sit position (default: 2.0)"
    )

    parser.add_argument(
        "-tk", "--target-knee",
        type=int,
        default=500,
        dest="target_knee",
        help="Target offset for knee motors 3,6,9,12 in raw units (default: 500)"
    )

    parser.add_argument(
        "-th", "--target-hip",
        type=int,
        default=-200,
        dest="target_hip",
        help="Target offset for hip motors - goes to 2,5; negative goes to 8,11 in raw units (default: -200)"
    )

    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the sit-stand cycle (default: 1)"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save torque plot after the run"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.freq <= 0 or args.freq > 100:
        print("[ERROR] Frequency must be between 0 and 100 Hz")
        sys.exit(1)

    if args.duration <= 0:
        print("[ERROR] Duration must be positive")
        sys.exit(1)

    if args.target_knee <= 0:
        print("[ERROR] Target knee offset must be positive")
        sys.exit(1)

    if args.target_hip == 0:
        print("[ERROR] Target hip offset must be non-zero")
        sys.exit(1)

    if args.repeat <= 0:
        print("[ERROR] Repeat count must be positive")
        sys.exit(1)

    run_sitstand(
        control_freq=args.freq,
        duration=args.duration,
        target_knee=args.target_knee,
        target_hip=args.target_hip,
        repeat=args.repeat,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
