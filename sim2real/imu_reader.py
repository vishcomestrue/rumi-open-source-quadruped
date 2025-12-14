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
MPU6050 IMU Reader for Quadruped Robot

Reads accelerometer, gyroscope, and computed orientation data from MPU6050 IMU
at high frequency (default 40Hz) for real-time robot control.

Can be used standalone for testing or as a library class in other applications.

Hardware:
    - MPU6050 IMU connected via I2C
    - Default I2C address: 0x68 (AD0 pin low)
    - I2C bus: 1 (default for Raspberry Pi)

Data Output:
    - Accelerometer: 3-axis (x, y, z) in m/s²
    - Gyroscope: 3-axis (x, y, z) in rad/s
    - Orientation: Roll, pitch, yaw in radians (computed via complementary filter)
"""

import sys
import time
import math
import struct
from typing import Dict, Tuple, Optional

try:
    import smbus2
except ImportError:
    print("ERROR: smbus2 not installed. Install with: uv pip install smbus2")
    sys.exit(1)


# ============================================================================
# MPU6050 Register Definitions
# ============================================================================

# Identity
WHO_AM_I = 0x75

# Power Management
PWR_MGMT_1 = 0x6B

# Configuration
SMPLRT_DIV = 0x19      # Sample Rate Divider
CONFIG = 0x1A          # DLPF Configuration
GYRO_CONFIG = 0x1B     # Gyroscope Configuration
ACCEL_CONFIG = 0x1C    # Accelerometer Configuration

# Data Registers
ACCEL_XOUT_H = 0x3B    # Accelerometer data start (6 bytes)
GYRO_XOUT_H = 0x43     # Gyroscope data start (6 bytes)


# ============================================================================
# MPU6050Reader Class
# ============================================================================

class MPU6050Reader:
    """
    Read data from MPU6050 IMU at high frequency.

    Features:
        - Configurable sample rate (default 40 Hz)
        - Automatic gyroscope calibration
        - Complementary filter for orientation estimation
        - I2C communication via smbus2

    Usage:
        # Initialize
        imu = MPU6050Reader(sample_rate=40)
        imu.calibrate()

        # Read data
        data = imu.read_data()
        print(data['orientation']['roll'])

        # Close
        imu.close()
    """

    def __init__(self, bus: int = 1, address: int = 0x68, sample_rate: int = 40):
        """
        Initialize MPU6050 reader.

        Args:
            bus: I2C bus number (1 for Raspberry Pi)
            address: MPU6050 I2C address (0x68 or 0x69)
            sample_rate: Target sample rate in Hz (default: 40)

        Raises:
            RuntimeError: If MPU6050 is not found at specified address
        """
        self.bus_num = bus
        self.address = address
        self.target_sample_rate = sample_rate
        self.dt = 1.0 / sample_rate  # Time step for integration

        # Initialize I2C bus
        self.bus = smbus2.SMBus(bus)

        # Verify sensor presence
        try:
            who_am_i = self.bus.read_byte_data(self.address, WHO_AM_I)
            if who_am_i != 0x68:
                raise RuntimeError(
                    f"MPU6050 not found at address {hex(self.address)}. "
                    f"WHO_AM_I returned {hex(who_am_i)} (expected 0x68)"
                )
        except OSError as e:
            raise RuntimeError(
                f"Failed to communicate with MPU6050 at address {hex(self.address)}. "
                f"Check I2C connection and permissions. Error: {e}"
            )

        # Wake up MPU6050 (it starts in sleep mode)
        self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Configure sensor
        self._configure_sensor()

        # Calibration offsets (will be set during calibrate())
        self.gyro_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Orientation state for complementary filter
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = time.time()

        print(f"✓ MPU6050 initialized at address {hex(self.address)}")

    def _configure_sensor(self):
        """Configure MPU6050 sensor parameters."""

        # Set gyroscope range: ±500 deg/s (GYRO_FS_SEL = 1)
        self.bus.write_byte_data(self.address, GYRO_CONFIG, 0x08)
        self.gyro_scale = 65.5  # LSB / (deg/s)

        # Set accelerometer range: ±4g (ACCEL_FS_SEL = 1)
        self.bus.write_byte_data(self.address, ACCEL_CONFIG, 0x08)
        self.accel_scale = 8192.0  # LSB / g

        # Configure DLPF: 42 Hz bandwidth
        # This gives internal sample rate of 1kHz
        self.bus.write_byte_data(self.address, CONFIG, 0x03)

        # Set sample rate divider for target Hz output
        # Sample Rate = 1000 Hz / (1 + SMPLRT_DIV)
        # For 40 Hz: SMPLRT_DIV = (1000/40) - 1 = 24
        divider = int(1000 / self.target_sample_rate) - 1
        self.bus.write_byte_data(self.address, SMPLRT_DIV, divider)

        time.sleep(0.1)  # Let settings take effect

        print(f"  Accel range: ±4g")
        print(f"  Gyro range: ±500 deg/s")
        print(f"  DLPF: 42 Hz")
        print(f"  Sample rate: {self.target_sample_rate} Hz")

    def read_raw_data(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Read raw accelerometer and gyroscope data.

        Returns:
            (accel_raw, gyro_raw): Tuples of (x, y, z) raw values (signed 16-bit)
        """
        # Read 6 bytes of accel data starting from ACCEL_XOUT_H
        accel_data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 6)

        # Read 6 bytes of gyro data starting from GYRO_XOUT_H
        gyro_data = self.bus.read_i2c_block_data(self.address, GYRO_XOUT_H, 6)

        # Convert to signed 16-bit integers (big-endian)
        accel_x = struct.unpack('>h', bytes(accel_data[0:2]))[0]
        accel_y = struct.unpack('>h', bytes(accel_data[2:4]))[0]
        accel_z = struct.unpack('>h', bytes(accel_data[4:6]))[0]

        gyro_x = struct.unpack('>h', bytes(gyro_data[0:2]))[0]
        gyro_y = struct.unpack('>h', bytes(gyro_data[2:4]))[0]
        gyro_z = struct.unpack('>h', bytes(gyro_data[4:6]))[0]

        return (accel_x, accel_y, accel_z), (gyro_x, gyro_y, gyro_z)

    def calibrate(self, num_samples: int = 100):
        """
        Calibrate gyroscope by measuring bias while stationary.

        IMPORTANT: Robot must be completely stationary during calibration!

        Args:
            num_samples: Number of samples to average (default: 100)
        """
        print(f"\nCalibrating gyroscope (keep robot stationary)...")
        print(f"Collecting {num_samples} samples...")

        gyro_x_sum = 0
        gyro_y_sum = 0
        gyro_z_sum = 0

        for i in range(num_samples):
            _, gyro_raw = self.read_raw_data()
            gyro_x_sum += gyro_raw[0]
            gyro_y_sum += gyro_raw[1]
            gyro_z_sum += gyro_raw[2]
            time.sleep(0.01)  # 10ms between samples

        # Calculate average bias
        self.gyro_offset['x'] = gyro_x_sum / num_samples
        self.gyro_offset['y'] = gyro_y_sum / num_samples
        self.gyro_offset['z'] = gyro_z_sum / num_samples

        print(f"✓ Calibration complete")
        print(f"  Gyro offsets: X={self.gyro_offset['x']:.1f}, "
              f"Y={self.gyro_offset['y']:.1f}, Z={self.gyro_offset['z']:.1f}")

    def read_data(self) -> Dict:
        """
        Read and process IMU data with orientation computation.

        Returns:
            Dictionary containing:
                - accel: {'x', 'y', 'z'} in m/s²
                - gyro: {'x', 'y', 'z'} in rad/s
                - orientation: {'roll', 'pitch', 'yaw'} in radians
                - timestamp: Unix timestamp
        """
        # Read raw data
        accel_raw, gyro_raw = self.read_raw_data()

        # Convert accelerometer to m/s²
        accel_x = (accel_raw[0] / self.accel_scale) * 9.81
        accel_y = (accel_raw[1] / self.accel_scale) * 9.81
        accel_z = (accel_raw[2] / self.accel_scale) * 9.81

        # Convert gyroscope to rad/s (subtract bias)
        gyro_x = ((gyro_raw[0] - self.gyro_offset['x']) / self.gyro_scale) * (math.pi / 180.0)
        gyro_y = ((gyro_raw[1] - self.gyro_offset['y']) / self.gyro_scale) * (math.pi / 180.0)
        gyro_z = ((gyro_raw[2] - self.gyro_offset['z']) / self.gyro_scale) * (math.pi / 180.0)

        # Compute orientation using complementary filter
        self._update_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

        # Return formatted data
        return {
            'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
            'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
            'orientation': {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw
            },
            'timestamp': time.time()
        }

    def _update_orientation(self, ax: float, ay: float, az: float,
                          gx: float, gy: float, gz: float):
        """
        Update orientation using complementary filter.

        Combines accelerometer (stable but noisy) with gyroscope (smooth but drifts)
        using weighted average: 98% gyro + 2% accel.

        Note: Roll/pitch from accelerometer assume no linear acceleration (only gravity).
        Yaw integrates from gyroscope only (no magnetometer for absolute heading).

        Args:
            ax, ay, az: Accelerometer in m/s²
            gx, gy, gz: Gyroscope in rad/s
        """
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Compute roll and pitch from accelerometer (gravity vector)
        # Note: These are only valid when robot is not accelerating!
        accel_roll = math.atan2(ay, math.sqrt(ax*ax + az*az))
        accel_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

        # Integrate gyroscope
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        # Complementary filter: 98% gyro (fast, drifts) + 2% accel (slow, stable)
        alpha = 0.98
        self.roll = alpha * self.roll + (1 - alpha) * accel_roll
        self.pitch = alpha * self.pitch + (1 - alpha) * accel_pitch

        # Normalize yaw to [-pi, pi]
        while self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        while self.yaw < -math.pi:
            self.yaw += 2 * math.pi

    def close(self):
        """Close I2C bus connection."""
        if hasattr(self, 'bus'):
            self.bus.close()
            print("\n✓ IMU reader closed")


# ============================================================================
# Standalone Main Function
# ============================================================================

def main():
    """Main function for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='MPU6050 IMU Reader - Read accelerometer, gyroscope, and orientation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--bus', type=int, default=1,
                       help='I2C bus number')
    parser.add_argument('--address', type=lambda x: int(x, 0), default=0x68,
                       help='I2C address (0x68 or 0x69)')
    parser.add_argument('--rate', type=int, default=40,
                       help='Sample rate in Hz')
    parser.add_argument('--no-calibrate', action='store_true',
                       help='Skip gyro calibration')
    args = parser.parse_args()

    print("=" * 80)
    print("MPU6050 IMU READER")
    print("=" * 80)

    try:
        # Initialize reader
        imu = MPU6050Reader(
            bus=args.bus,
            address=args.address,
            sample_rate=args.rate
        )

        # Calibrate if requested
        if not args.no_calibrate:
            imu.calibrate()

        print("\n" + "=" * 80)
        print("STREAMING DATA (Ctrl+C to stop)")
        print("=" * 80)
        print()

        # Main loop
        loop_count = 0
        start_time = time.time()

        while True:
            loop_start = time.time()

            # Read data
            data = imu.read_data()

            # Print every 10 samples (4 Hz display rate)
            if loop_count % 10 == 0:
                print(f"Accel (m/s²): X={data['accel']['x']:7.3f}  "
                      f"Y={data['accel']['y']:7.3f}  Z={data['accel']['z']:7.3f}  | "
                      f"Gyro (rad/s): X={data['gyro']['x']:7.3f}  "
                      f"Y={data['gyro']['y']:7.3f}  Z={data['gyro']['z']:7.3f}  | "
                      f"Orient (deg): Roll={math.degrees(data['orientation']['roll']):6.1f}  "
                      f"Pitch={math.degrees(data['orientation']['pitch']):6.1f}  "
                      f"Yaw={math.degrees(data['orientation']['yaw']):6.1f}")

            loop_count += 1

            # Maintain sample rate
            elapsed = time.time() - loop_start
            sleep_time = imu.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("STOPPED BY USER")
        print("=" * 80)

        # Print statistics
        total_time = time.time() - start_time
        actual_rate = loop_count / total_time if total_time > 0 else 0
        print(f"\nStatistics:")
        print(f"  Total samples: {loop_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average rate: {actual_rate:.1f} Hz (target: {args.rate} Hz)")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        imu.close()


if __name__ == "__main__":
    main()
