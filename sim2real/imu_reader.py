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
    - Gyroscope: 3-axis (x, y, z) in rad/s (angular velocity)
    - Orientation: Roll, pitch, yaw in radians (computed via complementary filter)
    - Velocity: 3-axis world-frame linear velocity (m/s) with gravity compensation
"""

import sys
import time
import math
from typing import Dict

try:
    from mpu6050 import mpu6050
except ImportError:
    print("ERROR: mpu6050-raspberrypi not installed. Install with: pip install mpu6050-raspberrypi")
    sys.exit(1)


# ============================================================================
# MPU6050Reader Class
# ============================================================================

class MPU6050Reader:
    """Read data from MPU6050 IMU at high frequency with velocity tracking."""

    def __init__(self, address: int = 0x68, sample_rate: int = 40):
        """Initialize MPU6050 reader."""
        self.address = address
        self.target_sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

        try:
            self.sensor = mpu6050(address)
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with MPU6050 at {hex(address)}: {e}")

        # Calibration offsets
        self.gyro_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.accel_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Orientation state - Initialize from accelerometer to eliminate startup transient
        accel_data = self.sensor.get_accel_data()
        ax, ay, az = accel_data['x'], accel_data['y'], accel_data['z']
        self.roll = math.atan2(ay, math.sqrt(ax*ax + az*az))
        self.pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        self.yaw = 0.0  # No absolute reference for yaw (no magnetometer)
        self.last_time = time.time()

        # Velocity state
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.stationary_threshold = 0.2  # Tighter threshold for better zero detection
        self.velocity_decay = 0.90  # More aggressive decay

        print(f"[OK] MPU6050 initialized at {hex(self.address)}, {self.target_sample_rate} Hz")

    def calibrate(self, num_samples: int = 500):
        """Calibrate gyroscope and accelerometer by measuring bias while stationary."""
        print(f"\nCalibrating sensors ({num_samples} samples, keep stationary and level)...")

        gyro_x_sum = 0
        gyro_y_sum = 0
        gyro_z_sum = 0
        accel_x_sum = 0
        accel_y_sum = 0
        accel_z_sum = 0

        for i in range(num_samples):
            gyro_data = self.sensor.get_gyro_data()
            accel_data = self.sensor.get_accel_data()

            gyro_x_sum += gyro_data['x']
            gyro_y_sum += gyro_data['y']
            gyro_z_sum += gyro_data['z']
            accel_x_sum += accel_data['x']
            accel_y_sum += accel_data['y']
            accel_z_sum += accel_data['z']
            time.sleep(0.01)

        # Gyroscope offsets (should be zero when stationary)
        self.gyro_offset['x'] = gyro_x_sum / num_samples
        self.gyro_offset['y'] = gyro_y_sum / num_samples
        self.gyro_offset['z'] = gyro_z_sum / num_samples

        # Accelerometer offsets (X,Y should be zero, Z should be gravity when level)
        gravity = 9.81
        self.accel_offset['x'] = (accel_x_sum / num_samples) - 0.0
        self.accel_offset['y'] = (accel_y_sum / num_samples) - 0.0
        self.accel_offset['z'] = (accel_z_sum / num_samples) - gravity

        print(f"[OK] Gyro offsets: X={self.gyro_offset['x']:.2f}, "
              f"Y={self.gyro_offset['y']:.2f}, Z={self.gyro_offset['z']:.2f} deg/s")
        print(f"[OK] Accel offsets: X={self.accel_offset['x']:.3f}, "
              f"Y={self.accel_offset['y']:.3f}, Z={self.accel_offset['z']:.3f} m/s²")

    def get_raw_data(self):
        """Get raw 16-bit sensor values before scaling."""
        raw_accel_x = self.sensor.read_i2c_word(self.sensor.ACCEL_XOUT0)
        raw_accel_y = self.sensor.read_i2c_word(self.sensor.ACCEL_YOUT0)
        raw_accel_z = self.sensor.read_i2c_word(self.sensor.ACCEL_ZOUT0)
        raw_gyro_x = self.sensor.read_i2c_word(self.sensor.GYRO_XOUT0)
        raw_gyro_y = self.sensor.read_i2c_word(self.sensor.GYRO_YOUT0)
        raw_gyro_z = self.sensor.read_i2c_word(self.sensor.GYRO_ZOUT0)

        return {
            'accel': {'x': raw_accel_x, 'y': raw_accel_y, 'z': raw_accel_z},
            'gyro': {'x': raw_gyro_x, 'y': raw_gyro_y, 'z': raw_gyro_z}
        }

    def read_data(self) -> Dict:
        """Read and process all IMU data including velocity, orientation, and raw values."""
        # Read raw sensor values
        raw = self.get_raw_data()

        # Read scaled accelerometer data (m/s²) and apply calibration
        accel_data = self.sensor.get_accel_data()
        accel_x = accel_data['x'] - self.accel_offset['x']
        accel_y = accel_data['y'] - self.accel_offset['y']
        accel_z = accel_data['z'] - self.accel_offset['z']

        # Read gyroscope data (deg/s) and convert to rad/s
        gyro_data = self.sensor.get_gyro_data()
        gyro_x = (gyro_data['x'] - self.gyro_offset['x']) * (math.pi / 180.0)
        gyro_y = (gyro_data['y'] - self.gyro_offset['y']) * (math.pi / 180.0)
        gyro_z = (gyro_data['z'] - self.gyro_offset['z']) * (math.pi / 180.0)

        # Read temperature
        temperature = self.sensor.get_temp()

        # Update orientation using complementary filter (returns dt for velocity calc)
        dt = self._update_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

        # Update velocity with drift compensation
        self._update_velocity(accel_x, accel_y, accel_z, dt)

        return {
            'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},  # m/s²
            'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},      # rad/s (angular velocity)
            'velocity': {'x': self.vx, 'y': self.vy, 'z': self.vz},  # m/s (linear velocity)
            'orientation': {'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw},  # radians
            'raw_accel': raw['accel'],   # 16-bit signed integers
            'raw_gyro': raw['gyro'],     # 16-bit signed integers
            'temperature': temperature,   # °C
            'timestamp': time.time()      # Unix timestamp
        }

    def _update_orientation(self, ax: float, ay: float, az: float,
                          gx: float, gy: float, gz: float):
        """Update orientation using complementary filter (98% gyro + 2% accel). Returns dt."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Compute roll and pitch from accelerometer (gravity vector)
        accel_roll = math.atan2(ay, math.sqrt(ax*ax + az*az))
        accel_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

        # Integrate gyroscope
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        # Complementary filter
        alpha = 0.98
        self.roll = alpha * self.roll + (1 - alpha) * accel_roll
        self.pitch = alpha * self.pitch + (1 - alpha) * accel_pitch

        # Normalize yaw to [-pi, pi]
        while self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        while self.yaw < -math.pi:
            self.yaw += 2 * math.pi

        return dt

    def _compute_rotation_matrix(self) -> tuple:
        """
        Compute rotation matrix from sensor frame to world frame.
        Uses ZYX Euler angle convention (yaw-pitch-roll).

        Returns:
            tuple: 9 values (R00, R01, R02, R10, R11, R12, R20, R21, R22)
        """
        # Cache trig values for efficiency
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)

        # Compute rotation matrix elements (ZYX convention)
        R00 = cp * cy
        R01 = sr * sp * cy - cr * sy
        R02 = cr * sp * cy + sr * sy
        R10 = cp * sy
        R11 = sr * sp * sy + cr * cy
        R12 = cr * sp * sy - sr * cy
        R20 = -sp
        R21 = sr * cp
        R22 = cr * cp

        return (R00, R01, R02, R10, R11, R12, R20, R21, R22)

    def _remove_gravity(self, ax: float, ay: float, az: float) -> tuple:
        """
        Remove gravity from measured acceleration using current orientation.

        The accelerometer measures total acceleration (linear + gravity).
        This method subtracts the gravity component based on IMU orientation.

        Args:
            ax, ay, az: Measured acceleration in sensor frame (m/s²)

        Returns:
            tuple: (ax_linear, ay_linear, az_linear) in sensor frame (m/s²)
        """
        # Get rotation matrix
        R00, R01, R02, R10, R11, R12, R20, R21, R22 = self._compute_rotation_matrix()

        # Gravity in sensor frame = +9.81 * third column of R^T
        gravity = 9.81
        gx_sensor = gravity * R20  # = gravity * sin(pitch)
        gy_sensor = gravity * R21  # = gravity * sin(roll) * cos(pitch)
        gz_sensor = gravity * R22  # = gravity * cos(roll) * cos(pitch)

        # Remove gravity component
        ax_linear = ax - gx_sensor
        ay_linear = ay - gy_sensor
        az_linear = az - gz_sensor

        return (ax_linear, ay_linear, az_linear)

    def _transform_to_world_frame(self, ax: float, ay: float, az: float) -> tuple:
        """
        Transform acceleration from sensor frame to world frame.

        Args:
            ax, ay, az: Acceleration in sensor frame (m/s²)

        Returns:
            tuple: (ax_world, ay_world, az_world) in world frame (m/s²)
        """
        # Get rotation matrix
        R00, R01, R02, R10, R11, R12, R20, R21, R22 = self._compute_rotation_matrix()

        # Matrix-vector multiplication: a_world = R * a_sensor
        ax_world = R00 * ax + R01 * ay + R02 * az
        ay_world = R10 * ax + R11 * ay + R12 * az
        az_world = R20 * ax + R21 * ay + R22 * az

        return (ax_world, ay_world, az_world)

    def _update_velocity(self, ax: float, ay: float, az: float, dt: float):
        """
        Update linear velocity with gravity compensation and world-frame integration.

        Process:
        1. Remove gravity from measured acceleration (sensor frame)
        2. Transform linear acceleration to world frame
        3. Integrate to get world-frame velocity
        4. Apply zero-velocity update when stationary

        Args:
            ax, ay, az: Measured acceleration in sensor frame (m/s²)
            dt: Time step (seconds)
        """
        # Step 1: Remove gravity component
        ax_linear, ay_linear, az_linear = self._remove_gravity(ax, ay, az)

        # Step 2: Transform to world frame
        ax_world, ay_world, az_world = self._transform_to_world_frame(
            ax_linear, ay_linear, az_linear
        )

        # Step 3: Integrate to get velocity (in world frame)
        self.vx += ax_world * dt
        self.vy += ay_world * dt
        self.vz += az_world * dt

        # Step 4: Drift compensation - Zero-velocity update when stationary
        # Check if linear acceleration magnitude is small (near zero motion)
        linear_accel_magnitude = math.sqrt(
            ax_linear*ax_linear + ay_linear*ay_linear + az_linear*az_linear
        )

        # If linear acceleration is small, assume stationary
        if linear_accel_magnitude < self.stationary_threshold:
            # Apply velocity decay (gradual zeroing)
            self.vx *= self.velocity_decay
            self.vy *= self.velocity_decay
            self.vz *= self.velocity_decay

            # Hard zero if very small
            if abs(self.vx) < 0.01:
                self.vx = 0.0
            if abs(self.vy) < 0.01:
                self.vy = 0.0
            if abs(self.vz) < 0.01:
                self.vz = 0.0

    def close(self):
        """Close connection."""
        print("\n[OK] IMU reader closed")


# ============================================================================
# Standalone Main Function
# ============================================================================

def main():
    """Main function for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(description='MPU6050 IMU Reader')
    parser.add_argument('--address', type=lambda x: int(x, 0), default=0x68)
    parser.add_argument('--rate', type=int, default=40)
    parser.add_argument('--no-calibrate', action='store_true')
    args = parser.parse_args()

    imu = MPU6050Reader(address=args.address, sample_rate=args.rate)
    if not args.no_calibrate:
        imu.calibrate()

    print("\n--- Streaming IMU data (Ctrl+C to stop) ---\n")

    try:
        while True:
            loop_start = time.time()
            data = imu.read_data()

            print(f"Vel: [{data['velocity']['x']:6.2f}, {data['velocity']['y']:6.2f}, {data['velocity']['z']:6.2f}] m/s | "
                  f"Gyro: [{data['gyro']['x']:6.3f}, {data['gyro']['y']:6.3f}, {data['gyro']['z']:6.3f}] rad/s | "
                  f"Temp: {data['temperature']:5.1f}°C")

            elapsed = time.time() - loop_start
            sleep_time = imu.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n--- Stopped ---")
    finally:
        imu.close()


if __name__ == "__main__":
    main()
