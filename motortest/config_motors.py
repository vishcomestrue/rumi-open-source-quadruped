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
Motor Configuration Tool

Configures Position PID gains for all connected MX-64 motors.

Control Table Reference (Protocol 2.0):
    Address 80: Position D Gain (2 bytes), Default: 0, Range: 0-16383
    Address 82: Position I Gain (2 bytes), Default: 0, Range: 0-16383
    Address 84: Position P Gain (2 bytes), Default: 850, Range: 0-16383

Usage:
    # Set all gains
    uv run python config_motors.py -p 850 -i 0 -d 0

    # Set only P gain
    uv run python config_motors.py -p 1000

    # Read current gains (no arguments)
    uv run python config_motors.py
"""

import sys
import time
import argparse
from mx64_controller import MX64Controller
from dynamixel_sdk import COMM_SUCCESS


def debug_read_2_bytes(controller, address, motor_id):
    """Read 2 bytes with debug output."""
    try:
        value, comm_result, error = controller.packet_handler.read2ByteTxRx(
            controller.port_handler, motor_id, address
        )
        if comm_result != COMM_SUCCESS:
            err_msg = controller.packet_handler.getTxRxResult(comm_result)
            print(f"\n  [DEBUG] Motor {motor_id} addr {address}: comm_result={err_msg}")
            return None
        if error != 0:
            err_msg = controller.packet_handler.getRxPacketError(error)
            print(f"\n  [DEBUG] Motor {motor_id} addr {address}: error={err_msg}")
            return None
        return value
    except Exception as e:
        print(f"\n  [DEBUG] Motor {motor_id} addr {address}: exception={e}")
        return None


def format_gain(raw_value, divisor):
    """Format gain as 'actual (raw)' string."""
    if raw_value is None:
        return "ERROR"
    actual = raw_value / divisor
    return f"{actual:.4f} ({raw_value})"


def read_pid_gains(controller, motor_ids, debug=False):
    """
    Read and display current PID gains for all motors.

    Actual K values are computed from raw register values:
        Kp = P_Gain / 128
        Ki = I_Gain / 65536
        Kd = D_Gain / 16
    """
    print("\n" + "=" * 80)
    print("Current Position PID Gains")
    print("Format: actual_value (raw_register)")
    print("Kp = raw/128, Ki = raw/65536, Kd = raw/16")
    print("=" * 80)
    print(f"{'Motor':<7} {'Kp (P Gain)':<20} {'Ki (I Gain)':<20} {'Kd (D Gain)':<20}")
    print("-" * 80)

    # Clear port buffer before starting reads
    if controller.port_handler:
        controller.port_handler.clearPort()
    time.sleep(0.05)

    # Test first motor with debug output
    if debug and motor_ids:
        print(f"\n[DEBUG] Testing read on motor {motor_ids[0]}...")
        test_val = debug_read_2_bytes(controller, controller.ADDR_POSITION_P_GAIN, motor_ids[0])
        print(f"[DEBUG] Result: {test_val}\n")

    for mid in motor_ids:
        # Small delay between motors to avoid bus congestion
        time.sleep(0.01)

        # Retry reads up to 3 times
        p_gain = None
        i_gain = None
        d_gain = None

        for attempt in range(3):
            if p_gain is None:
                p_gain = controller.read_2_bytes(controller.ADDR_POSITION_P_GAIN, mid)
            time.sleep(0.005)
            if i_gain is None:
                i_gain = controller.read_2_bytes(controller.ADDR_POSITION_I_GAIN, mid)
            time.sleep(0.005)
            if d_gain is None:
                d_gain = controller.read_2_bytes(controller.ADDR_POSITION_D_GAIN, mid)

            if p_gain is not None and i_gain is not None and d_gain is not None:
                break
            time.sleep(0.01)

        # Format with actual values: Kp=raw/128, Ki=raw/65536, Kd=raw/16
        p_str = format_gain(p_gain, 128)
        i_str = format_gain(i_gain, 65536)
        d_str = format_gain(d_gain, 16)

        print(f"{mid:<7} {p_str:<20} {i_str:<20} {d_str:<20}")

    print("=" * 80)


def configure_motors(kp=None, ki=None, kd=None):
    """
    Configure Position PID gains for all connected motors.

    Args:
        kp: Actual Kp value (converted to raw: Kp × 128), None to skip
        ki: Actual Ki value (converted to raw: Ki × 65536), None to skip
        kd: Actual Kd value (converted to raw: Kd × 16), None to skip
    """
    # Convert actual K values to raw register values
    p_gain = int(round(kp * 128)) if kp is not None else None
    i_gain = int(round(ki * 65536)) if ki is not None else None
    d_gain = int(round(kd * 16)) if kd is not None else None

    # Clamp to valid range
    if p_gain is not None:
        p_gain = max(0, min(16383, p_gain))
    if i_gain is not None:
        i_gain = max(0, min(16383, i_gain))
    if d_gain is not None:
        d_gain = max(0, min(16383, d_gain))
    print("\n" + "=" * 60)
    print("   Motor PID Configuration Tool")
    print("=" * 60)

    controller = MX64Controller()

    # Step 1: Connect
    print("\n[STEP 1] Connecting to motors...")
    print("-" * 40)

    if not controller.connect():
        print("\n[FAILED] Could not connect to any motors.")
        print("Please check:")
        print("  - Motors are powered (12V)")
        print("  - USB cable is connected")
        print("  - Correct permissions on /dev/ttyUSB*")
        return False

    # Small delay to let connection settle
    time.sleep(0.1)

    # Step 2: Discover all motors
    print("\n[STEP 2] Discovering motors...")
    print("-" * 40)

    motors = controller.scan_motors(verbose=True)

    if not motors:
        print("[FAILED] No motors discovered.")
        controller.disconnect()
        return False

    all_motor_ids = sorted(motors.keys())
    print(f"\nFound {len(all_motor_ids)} motor(s): {all_motor_ids}")

    # Step 3: Read current gains
    print("\n[STEP 3] Reading current PID gains...")
    print("-" * 40)
    read_pid_gains(controller, all_motor_ids, debug=True)

    # If no gains specified, just display and exit
    if p_gain is None and i_gain is None and d_gain is None:
        print("\n[INFO] No gains specified. Use --kp, --ki, --kd to set gains.")
        controller.disconnect()
        return True

    # Step 4: Configure gains
    print("\n[STEP 4] Configuring PID gains...")
    print("-" * 40)

    gains_to_set = []
    if p_gain is not None:
        gains_to_set.append(f"Kp={kp} (raw={p_gain})")
    if i_gain is not None:
        gains_to_set.append(f"Ki={ki} (raw={i_gain})")
    if d_gain is not None:
        gains_to_set.append(f"Kd={kd} (raw={d_gain})")

    print(f"Setting: {', '.join(gains_to_set)}")
    print()

    success_count = 0
    error_count = 0

    for mid in all_motor_ids:
        print(f"Motor {mid}: ", end="")
        motor_success = True

        # Small delay between motors
        time.sleep(0.01)

        if p_gain is not None:
            if controller.write_2_bytes(controller.ADDR_POSITION_P_GAIN, p_gain, mid):
                print(f"P={p_gain} ", end="")
            else:
                print(f"P=FAIL ", end="")
                motor_success = False
            time.sleep(0.005)

        if i_gain is not None:
            if controller.write_2_bytes(controller.ADDR_POSITION_I_GAIN, i_gain, mid):
                print(f"I={i_gain} ", end="")
            else:
                print(f"I=FAIL ", end="")
                motor_success = False
            time.sleep(0.005)

        if d_gain is not None:
            if controller.write_2_bytes(controller.ADDR_POSITION_D_GAIN, d_gain, mid):
                print(f"D={d_gain} ", end="")
            else:
                print(f"D=FAIL ", end="")
                motor_success = False
            time.sleep(0.005)

        if motor_success:
            print("[OK]")
            success_count += 1
        else:
            print("[ERROR]")
            error_count += 1

    # Step 5: Verify configuration
    print("\n[STEP 5] Verifying configuration...")
    print("-" * 40)
    read_pid_gains(controller, all_motor_ids)

    # Summary
    print("\n" + "=" * 60)
    print(f"[COMPLETE] Configured {success_count}/{len(all_motor_ids)} motors successfully")
    if error_count > 0:
        print(f"[WARNING] {error_count} motor(s) had errors")
    print("=" * 60)

    controller.disconnect()
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Configure Position PID gains for all connected MX-64 motors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read current gains (no changes)
  uv run python config_motors.py

  # Set Kp only (default is 6.64)
  uv run python config_motors.py --kp 6.64

  # Set all gains
  uv run python config_motors.py --kp 6.64 --ki 0 --kd 0

  # Higher Kp for stiffer response
  uv run python config_motors.py --kp 10.0

PID Gain Reference:
  Kp (Proportional): Default 6.64 (raw 850)
    - Range: 0 to 127.99 (raw 0-16383)
    - Higher = stiffer, faster response

  Ki (Integral): Default 0 (raw 0)
    - Range: 0 to 0.25 (raw 0-16383)
    - Eliminates steady-state error
    - Too high can cause oscillation

  Kd (Derivative): Default 0 (raw 0)
    - Range: 0 to 1023.94 (raw 0-16383)
    - Dampens oscillation
    - Too high can cause vibration

Conversion: raw_P = Kp*128, raw_I = Ki*65536, raw_D = Kd*16
        """
    )

    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Kp (proportional gain), default: 6.64, range: 0-127.99"
    )

    parser.add_argument(
        "--ki",
        type=float,
        default=None,
        help="Ki (integral gain), default: 0, range: 0-0.25"
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Kd (derivative gain), default: 0, range: 0-1023.94"
    )

    args = parser.parse_args()

    # Validate arguments (check against max raw values converted back)
    if args.kp is not None:
        if args.kp < 0 or args.kp > 127.99:
            print(f"[ERROR] Kp must be between 0 and 127.99")
            sys.exit(1)

    if args.ki is not None:
        if args.ki < 0 or args.ki > 0.25:
            print(f"[ERROR] Ki must be between 0 and 0.25")
            sys.exit(1)

    if args.kd is not None:
        if args.kd < 0 or args.kd > 1023.94:
            print(f"[ERROR] Kd must be between 0 and 1023.94")
            sys.exit(1)

    success = configure_motors(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
