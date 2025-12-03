#!/usr/bin/env python3
"""
Multi-Motor Dynamixel Controller
Controls multiple Dynamixel motors simultaneously at high frequency.
Easily scalable from 3 to 12+ motors.

Architecture:
- GroupSyncWrite: Write all motor positions in ONE packet
- GroupSyncRead: Read all motor positions in ONE packet
- Achieves 80-120Hz control frequency

Usage:
    # For 3 motors
    controller = MultiMotorController(motor_ids=[1, 2, 3])

    # For 12 motors
    controller = MultiMotorController(motor_ids=list(range(1, 13)))
"""

import math
import time
from dynamixel_sdk import *


class MultiMotorController:
    """High-performance controller for multiple Dynamixel motors using Protocol 2.0."""

    # Control Table Addresses (MX-64/X-Series with Protocol 2.0)
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132

    # Data Byte Length
    LEN_GOAL_POSITION = 4
    LEN_PRESENT_POSITION = 4

    # Protocol and Communication Settings
    PROTOCOL_VERSION = 2.0
    BAUDRATE = 2000000  # 2Mbps for high-speed communication

    # Motor position range
    DXL_MINIMUM_POSITION = 0
    DXL_MAXIMUM_POSITION = 4095
    DXL_CENTER_POSITION = 2048  # Center position (0 radians)

    # Communication result values
    COMM_SUCCESS = 0
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    def __init__(self, port="/dev/ttyUSB0", num_motors=3, baudrate=2000000):
        """
        Initialize Multi-Motor controller with automatic motor discovery.

        Args:
            port: Serial port path (e.g., "/dev/ttyUSB0")
            num_motors: Number of motors to control (will select first N motors by ID)
            baudrate: Communication baudrate (default: 2Mbps)
        """
        self.port = port
        self.num_motors = num_motors
        self.motor_ids = []  # Will be populated by connect()
        self.BAUDRATE = baudrate
        self.connected = False

        # Initialize port and packet handlers
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        # Initialize GroupSyncRead for efficient multi-motor reading
        self.group_sync_read = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            self.ADDR_PRESENT_POSITION,
            self.LEN_PRESENT_POSITION
        )

        # Initialize GroupSyncWrite for efficient multi-motor writing
        self.group_sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            self.ADDR_GOAL_POSITION,
            self.LEN_GOAL_POSITION
        )

        # Statistics
        self.stats = {
            'read_count': 0,
            'write_count': 0,
            'read_errors': 0,
            'write_errors': 0,
        }

        print(f"[Multi-Motor Controller] Initialized for {self.num_motors} motor(s)")
        print(f"[Multi-Motor Controller] Port: {port}")
        print(f"[Multi-Motor Controller] Baudrate: {self.BAUDRATE} bps")
        print(f"[Multi-Motor Controller] Motors will be auto-discovered on connect()")

    def scan_motors(self, verbose=True):
        """
        Scan for all connected Dynamixel motors using broadcast ping.

        Args:
            verbose: If True, print detailed scan results

        Returns:
            list: Sorted list of discovered motor IDs, or empty list on error
        """
        try:
            # Perform broadcast ping to discover motors
            data_list, comm_result = self.packet_handler.broadcastPing(self.port_handler)

            if comm_result != COMM_SUCCESS:
                print(f"[ERROR] Broadcast ping failed: {self.packet_handler.getTxRxResult(comm_result)}")
                return []

            # Extract and sort motor IDs
            motor_ids = sorted(data_list.keys())

            # Display results
            if verbose and len(motor_ids) > 0:
                print(f"[Motor Scan] Found {len(motor_ids)} motor(s):")
                for motor_id in motor_ids:
                    model_num, fw_version = data_list[motor_id]
                    print(f"  Motor ID {motor_id}: Model {model_num}, Firmware v{fw_version}")

            return motor_ids

        except Exception as e:
            print(f"[ERROR] Exception during motor scan: {e}")
            return []

    def connect(self):
        """
        Open serial port, set baudrate, and enable torque for all motors.

        Returns:
            bool: True if successful, False otherwise
        """
        # Open port
        if not self.port_handler.openPort():
            print(f"[ERROR] Failed to open port {self.port}")
            return False
        print(f"[SUCCESS] Opened port {self.port}")

        # Set baudrate
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            print(f"[ERROR] Failed to set baudrate to {self.BAUDRATE}")
            return False
        print(f"[SUCCESS] Set baudrate to {self.BAUDRATE}")

        # Scan for motors
        print(f"\n[INFO] Scanning for connected motors...")
        discovered_motors = self.scan_motors(verbose=True)

        # Handle no motors found
        if len(discovered_motors) == 0:
            print(f"[ERROR] No motors found on {self.port}")
            print(f"[ERROR] Please check:")
            print(f"  - Motors are powered")
            print(f"  - Connections are secure")
            print(f"  - Baudrate matches motor configuration ({self.BAUDRATE} bps)")
            self.port_handler.closePort()
            return False

        # Handle insufficient motors (prompt user)
        if len(discovered_motors) < self.num_motors:
            print(f"\n[WARNING] Found {len(discovered_motors)} motor(s), but {self.num_motors} requested")
            print(f"[WARNING] Discovered motor IDs: {discovered_motors}")

            try:
                response = input(f"[CONFIRM] Proceed with {len(discovered_motors)} motor(s)? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"[INFO] User cancelled. Exiting...")
                    self.port_handler.closePort()
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\n[INFO] User cancelled")
                self.port_handler.closePort()
                return False

            self.num_motors = len(discovered_motors)
            print(f"[INFO] Proceeding with {self.num_motors} motor(s)")

        # Select first n motors (lowest IDs)
        self.motor_ids = discovered_motors[:self.num_motors]
        print(f"\n[INFO] Selected motors: {self.motor_ids}")

        # Enable torque for all motors
        print(f"\n[INFO] Enabling torque for {self.num_motors} motors...")
        for motor_id in self.motor_ids:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.ADDR_TORQUE_ENABLE,
                self.TORQUE_ENABLE
            )

            if result != COMM_SUCCESS:
                print(f"[ERROR] Motor ID {motor_id}: {self.packet_handler.getTxRxResult(result)}")
                return False
            elif error != 0:
                print(f"[ERROR] Motor ID {motor_id}: {self.packet_handler.getRxPacketError(error)}")
                return False

            print(f"  Motor ID {motor_id}: Torque enabled ✓")

        # Add all motors to sync read group (only done once)
        print(f"\n[INFO] Adding motors to sync read group...")
        for motor_id in self.motor_ids:
            if not self.group_sync_read.addParam(motor_id):
                print(f"[ERROR] Failed to add motor {motor_id} to sync read group")
                return False
        print(f"[SUCCESS] All {self.num_motors} motors added to sync read group")

        self.connected = True
        print(f"\n{'='*50}")
        print(f"[READY] All {self.num_motors} motors connected and ready!")
        print(f"{'='*50}\n")
        return True

    def write_positions(self, positions_dict):
        """
        Write goal positions to all motors simultaneously in ONE packet.

        Args:
            positions_dict: Dictionary mapping motor_id to position in Dynamixel units
                           Example: {1: 2048, 2: 3000, 3: 1500}

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("[ERROR] Not connected to motors")
            return False

        self.stats['write_count'] += 1

        # Add parameters for each motor
        for motor_id in self.motor_ids:
            if motor_id not in positions_dict:
                print(f"[WARNING] Position not provided for motor {motor_id}, skipping")
                continue

            position = positions_dict[motor_id]

            # Clamp position to valid range
            position = max(self.DXL_MINIMUM_POSITION,
                          min(self.DXL_MAXIMUM_POSITION, position))

            # Convert position to 4-byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position)),
                DXL_HIBYTE(DXL_LOWORD(position)),
                DXL_LOBYTE(DXL_HIWORD(position)),
                DXL_HIBYTE(DXL_HIWORD(position))
            ]

            # Add to sync write buffer
            if not self.group_sync_write.addParam(motor_id, param_goal_position):
                print(f"[ERROR] Failed to add motor {motor_id} to sync write")
                self.group_sync_write.clearParam()
                self.stats['write_errors'] += 1
                return False

        # Send all commands in ONE packet
        result = self.group_sync_write.txPacket()

        # Clear parameters for next write
        self.group_sync_write.clearParam()

        if result != COMM_SUCCESS:
            print(f"[ERROR] Sync write failed: {self.packet_handler.getTxRxResult(result)}")
            self.stats['write_errors'] += 1
            return False

        return True

    def read_positions(self, max_retries=3):
        """
        Read current positions from all motors simultaneously in ONE packet.

        Args:
            max_retries: Maximum number of retry attempts on failure (default: 3)

        Returns:
            dict: {motor_id: position_in_dynamixel_units} or None if error
                  Example: {1: 2048, 2: 3000, 3: 1500}
        """
        if not self.connected:
            print("[ERROR] Not connected to motors")
            return None

        self.stats['read_count'] += 1

        # Retry loop for transient errors
        for attempt in range(max_retries):
            # Send read request and receive responses in ONE transaction
            result = self.group_sync_read.txRxPacket()

            if result != COMM_SUCCESS:
                if attempt < max_retries - 1:
                    # Retry on failure
                    time.sleep(0.001)  # Small delay before retry
                    continue
                else:
                    # Final attempt failed
                    print(f"[ERROR] Sync read failed after {max_retries} attempts: {self.packet_handler.getTxRxResult(result)}")
                    self.stats['read_errors'] += 1
                    return None

            # Extract position data for each motor
            positions = {}
            all_available = True

            for motor_id in self.motor_ids:
                # Check if data is available
                if not self.group_sync_read.isAvailable(
                    motor_id,
                    self.ADDR_PRESENT_POSITION,
                    self.LEN_PRESENT_POSITION
                ):
                    if attempt < max_retries - 1:
                        # Data not available, retry
                        all_available = False
                        time.sleep(0.001)
                        break
                    else:
                        print(f"[ERROR] Data not available for motor {motor_id} after {max_retries} attempts")
                        self.stats['read_errors'] += 1
                        return None

                # Get position data
                position = self.group_sync_read.getData(
                    motor_id,
                    self.ADDR_PRESENT_POSITION,
                    self.LEN_PRESENT_POSITION
                )
                positions[motor_id] = position

            # If all data available, return success
            if all_available:
                return positions

        # Should not reach here, but handle gracefully
        self.stats['read_errors'] += 1
        return None

    def radians_to_dynamixel(self, radians):
        """
        Convert radians to Dynamixel position units.

        Args:
            radians: Position in radians

        Returns:
            int: Position in Dynamixel units (0-4095)
        """
        # Center position (2048) = 0 radians
        # Full range: 0-4095 = -π to +π radians
        position = int((radians * 2048 / math.pi) + self.DXL_CENTER_POSITION)

        # Clamp to valid range
        position = max(self.DXL_MINIMUM_POSITION,
                      min(self.DXL_MAXIMUM_POSITION, position))

        return position

    def dynamixel_to_radians(self, dxl_position):
        """
        Convert Dynamixel position units to radians.

        Args:
            dxl_position: Position in Dynamixel units (0-4095)

        Returns:
            float: Position in radians
        """
        radians = (dxl_position - self.DXL_CENTER_POSITION) * math.pi / 2048
        return radians

    def write_positions_radians(self, positions_rad_dict):
        """
        Write positions in radians to all motors.

        Args:
            positions_rad_dict: Dictionary mapping motor_id to position in radians
                               Example: {1: 0.0, 2: 0.5, 3: -0.3}

        Returns:
            bool: True if successful, False otherwise
        """
        # Convert radians to Dynamixel units
        positions_dxl = {
            motor_id: self.radians_to_dynamixel(rad_pos)
            for motor_id, rad_pos in positions_rad_dict.items()
        }

        return self.write_positions(positions_dxl)

    def read_positions_radians(self):
        """
        Read current positions from all motors in radians.

        Returns:
            dict: {motor_id: position_in_radians} or None if error
                  Example: {1: 0.0, 2: 0.5, 3: -0.3}
        """
        positions_dxl = self.read_positions()

        if positions_dxl is None:
            return None

        # Convert Dynamixel units to radians
        positions_rad = {
            motor_id: self.dynamixel_to_radians(dxl_pos)
            for motor_id, dxl_pos in positions_dxl.items()
        }

        return positions_rad

    def disconnect(self):
        """Disable torque for all motors and close port."""
        if self.connected:
            print(f"\n[INFO] Disabling torque for {self.num_motors} motors...")

            # Disable torque for all motors
            for motor_id in self.motor_ids:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler,
                    motor_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_DISABLE
                )
            print("[SUCCESS] Torque disabled for all motors")

        # Clear sync read parameters
        self.group_sync_read.clearParam()

        # Close port
        self.port_handler.closePort()
        self.connected = False
        print("[SUCCESS] Port closed")

        # Print statistics
        if self.stats['read_count'] > 0 or self.stats['write_count'] > 0:
            print(f"\n{'='*50}")
            print(f"[STATISTICS] Session Summary:")
            print(f"  Total reads:  {self.stats['read_count']}")
            print(f"  Total writes: {self.stats['write_count']}")
            print(f"  Read errors:  {self.stats['read_errors']}")
            print(f"  Write errors: {self.stats['write_errors']}")
            if self.stats['read_count'] > 0:
                print(f"  Read success rate:  {100*(1-self.stats['read_errors']/self.stats['read_count']):.2f}%")
            if self.stats['write_count'] > 0:
                print(f"  Write success rate: {100*(1-self.stats['write_errors']/self.stats['write_count']):.2f}%")
            print(f"{'='*50}\n")


def test_motors(num_motors=3, control_rate=100, port="/dev/ttyUSB0", baudrate=2000000,
                write_read_delay=0.0005, step_size=40):
    """
    Test script for N motors with incremental position control.
    Pattern: Read current position → Add step_size → Write new position → Repeat
    Motors oscillate between 0 and 4000 units, reversing direction at limits.

    Args:
        num_motors: Number of motors to control (default: 3)
        control_rate: Target control frequency in Hz (default: 100)
        port: Serial port (default: "/dev/ttyUSB0")
        baudrate: Communication baudrate (default: 2000000)
        write_read_delay: Delay between write and read in seconds (default: 0.0005 = 0.5ms)
        step_size: Position step size in Dynamixel units (default: 40)
    """
    print("\n" + "="*70)
    print(f"MULTI-MOTOR CONTROLLER TEST - {num_motors} MOTORS")
    print("="*70)
    print(f"Requested motors: {num_motors}")
    print(f"Motors will be auto-discovered on connect")
    print(f"Target control rate: {control_rate} Hz")
    print(f"Port: {port}")
    print(f"Baudrate: {baudrate} bps")
    print("Make sure motors are connected and powered!")
    print("="*70 + "\n")

    # Create controller for specified number of motors
    controller = MultiMotorController(
        port=port,
        num_motors=num_motors,
        baudrate=baudrate
    )

    # Connect to motors
    if not controller.connect():
        print("\n[FAILED] Could not connect to motors. Exiting...")
        return

    print(f"[INFO] Starting motor movement test...")
    print(f"[INFO] {num_motors} motors with incremental position control")
    print(f"[INFO] Pattern: Read position → Add {step_size} units → Write → Repeat")
    print(f"[INFO] Step size: {step_size} units (~{step_size*0.088:.2f}°)")
    print(f"[INFO] Range: 0 to 4000 units (reverses at limits)")
    print(f"[INFO] Target frequency: {control_rate} Hz")

    # Warm up communication with a few test reads
    print(f"[INFO] Warming up communication...")
    for i in range(3):
        test_pos = controller.read_positions()
        if test_pos is None:
            print(f"[WARNING] Warm-up read {i+1}/3 failed, retrying...")
            time.sleep(0.01)
        else:
            print(f"  Warm-up read {i+1}/3: OK")
            break

    print(f"[INFO] Press Ctrl+C to stop\n")

    # Calculate sleep time to achieve target control rate
    # Sleep time = (1/control_rate) - expected_loop_time
    # Assume ~5ms for communication overhead
    target_period = 1.0 / control_rate
    comm_overhead = 0.005  # 5ms estimated
    sleep_time = max(0.001, target_period - comm_overhead)  # Minimum 1ms

    try:
        loop_times = []
        iteration = 0

        # Position limits
        min_position = 0
        max_position = 4000

        # Direction: 1 = ascending, -1 = descending
        direction = 1

        print(f"[INFO] Incremental position control: Read current → Add step → Write new position")
        print(f"[INFO] Step size: {step_size} units (~{step_size*0.088:.2f}°)")
        print(f"[INFO] Range: {min_position} to {max_position} units")
        print(f"[INFO] Motors will oscillate between min and max positions\n")

        # Read initial positions
        print("[INFO] Reading initial positions...")
        current_positions = controller.read_positions()
        if current_positions is None:
            print("[ERROR] Could not read initial positions!")
            return

        print("[INFO] Initial positions:", {motor_id: pos for motor_id, pos in current_positions.items()})

        while True:
            loop_start = time.time()

            # Calculate next positions (current + step in current direction)
            next_positions = {}
            for motor_id in controller.motor_ids:
                current_pos = current_positions[motor_id]
                next_pos = current_pos + (step_size * direction)

                # Check bounds and reverse direction if needed
                if next_pos >= max_position:
                    next_pos = max_position
                    direction = -1  # Start going down
                elif next_pos <= min_position:
                    next_pos = min_position
                    direction = 1   # Start going up

                next_positions[motor_id] = next_pos

            # Write positions to all motors (ONE packet)
            success = controller.write_positions(next_positions)
            if not success:
                print("\n[ERROR] Write failed!")
                break

            # Small delay between write and read (helps with packet timing)
            if write_read_delay > 0:
                time.sleep(write_read_delay)

            # Read current positions from all motors (ONE packet)
            new_positions = controller.read_positions()
            if new_positions is None:
                print(f"\n[WARNING] Read failed at iteration {iteration}, continuing with last known positions...")
                # Continue with previous positions instead of breaking
                iteration += 1
                continue
            else:
                current_positions = new_positions

            # Calculate loop time and frequency
            loop_end = time.time()
            loop_time_ms = (loop_end - loop_start) * 1000
            loop_times.append(loop_time_ms)
            frequency = 1000.0 / loop_time_ms if loop_time_ms > 0 else 0

            # Control loop timing - sleep to achieve target rate
            time.sleep(sleep_time)

            iteration += 1

    except KeyboardInterrupt:
        print("\n\n[INFO] Stopping test...")

        # Calculate and print performance statistics
        if loop_times:
            avg_time = sum(loop_times) / len(loop_times)
            min_time = min(loop_times)
            max_time = max(loop_times)
            avg_freq = 1000.0 / avg_time

            print(f"\n{'='*70}")
            print("[PERFORMANCE] Statistics:")
            print(f"  Target frequency:  {control_rate:.1f} Hz")
            print(f"  Actual frequency:  {avg_freq:.1f} Hz")
            print(f"  Average loop time: {avg_time:.2f} ms")
            print(f"  Min loop time:     {min_time:.2f} ms ({1000.0/min_time:.1f} Hz)")
            print(f"  Max loop time:     {max_time:.2f} ms ({1000.0/max_time:.1f} Hz)")
            print(f"  Total loops:       {len(loop_times)}")
            print(f"  Frequency error:   {avg_freq - control_rate:+.1f} Hz ({100*(avg_freq - control_rate)/control_rate:+.1f}%)")
            print(f"{'='*70}")

    finally:
        controller.disconnect()
        print("\n[COMPLETE] Test finished\n")


if __name__ == "__main__":
    """Main entry point with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Motor Dynamixel Controller - Test multiple motors at desired frequency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and use first 3 motors at 100Hz (default)
  python3 multi_motor_controller.py

  # Auto-discover and use first 12 motors at 100Hz
  python3 multi_motor_controller.py -n 12

  # Auto-discover 6 motors at 50Hz
  python3 multi_motor_controller.py -n 6 -r 50

  # Auto-discover with custom port and baudrate
  python3 multi_motor_controller.py -p /dev/ttyUSB1 -b 3000000

  # Auto-discover with custom step size (larger steps, faster cycle)
  python3 multi_motor_controller.py -s 100

  # Auto-discover with small steps for fine control
  python3 multi_motor_controller.py -s 20 -r 50

  # Test with no delay between write/read (maximum speed)
  python3 multi_motor_controller.py -d 0

Note: Motors are auto-discovered on startup using broadcast ping.
      The first N motors (by ID number) will be selected and controlled.
        """
    )

    parser.add_argument(
        "-n", "--num-motors",
        type=int,
        default=3,
        help="Number of motors to control (default: 3). Motors will be auto-discovered and sorted by ID."
    )

    parser.add_argument(
        "-r", "--rate",
        type=int,
        default=100,
        help="Target control frequency in Hz (default: 100)"
    )

    parser.add_argument(
        "-p", "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port path (default: /dev/ttyUSB0)"
    )

    parser.add_argument(
        "-b", "--baudrate",
        type=int,
        default=2000000,
        choices=[57600, 115200, 1000000, 2000000, 3000000, 4000000],
        help="Communication baudrate in bps (default: 2000000)"
    )

    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=0.0005,
        help="Delay between write and read in seconds (default: 0.0005 = 0.5ms). Use 0 for no delay."
    )

    parser.add_argument(
        "-s", "--step-size",
        type=int,
        default=40,
        help="Position step size in Dynamixel units (default: 40). Range: 0-4095."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_motors < 1 or args.num_motors > 253:
        print("[ERROR] Number of motors must be between 1 and 253")
        exit(1)

    if args.rate < 1 or args.rate > 1000:
        print("[ERROR] Control rate must be between 1 and 1000 Hz")
        exit(1)

    if args.step_size < 1 or args.step_size > 4095:
        print("[ERROR] Step size must be between 1 and 4095")
        exit(1)

    # Run test with specified parameters
    test_motors(
        num_motors=args.num_motors,
        control_rate=args.rate,
        port=args.port,
        baudrate=args.baudrate,
        write_read_delay=args.delay,
        step_size=args.step_size
    )
