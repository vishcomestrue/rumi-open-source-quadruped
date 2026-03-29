#!/usr/bin/env python3
"""
Example: BNO080 Velocity Estimation

Demonstrates real-time velocity estimation from BNO080 IMU data.
Reads linear acceleration and orientation, then estimates velocity in world frame.
"""

import time
import numpy as np
from bare import BNO080Reader
from estimate import VelocityEstimator


def calibrate_drift(imu, estimator, duration=5.0):
    """
    Calibrate velocity drift by measuring accumulated velocity while stationary.

    The IMU should be kept completely stationary during calibration.
    Any velocity accumulated during this period represents drift/bias that
    should be compensated for in subsequent measurements.

    Args:
        imu: BNO080Reader instance
        estimator: VelocityEstimator instance (will be reset after calibration)
        duration: Calibration duration in seconds (default: 5.0)

    Returns:
        drift_rate: Drift rate in m/s per second (3D vector)
    """
    print("\n" + "="*80)
    print("DRIFT CALIBRATION")
    print("="*80)
    print(f"\n⚠️  IMPORTANT: Keep the IMU completely stationary for {duration:.0f} seconds!")
    print("   This will measure and compensate for velocity drift.\n")

    input("Press ENTER when ready to start calibration...")

    print(f"\nCalibrating for {duration:.0f} seconds", end="", flush=True)

    # Reset estimator before calibration
    estimator.reset()

    start_time = time.time()
    prev_time = None
    sample_count = 0

    while time.time() - start_time < duration:
        # Read IMU data
        data = imu.read_data()
        current_time = data['timestamp']

        # Calculate time step
        if prev_time is None:
            prev_time = current_time
            time.sleep(0.025)
            continue

        dt = current_time - prev_time
        prev_time = current_time

        # Extract linear acceleration and quaternion
        accel_body = np.array([
            data['linear_accel']['x'],
            data['linear_accel']['y'],
            data['linear_accel']['z']
        ])
        quat = data['quaternion']

        # Update estimator during calibration
        estimator.update(accel_body, quat, dt)
        sample_count += 1

        # Print progress indicator
        elapsed = time.time() - start_time
        if sample_count % 20 == 0:  # Print dot every ~0.5s
            print(".", end="", flush=True)

        time.sleep(0.025)

    # Measure drift: velocity accumulated while stationary
    drift_velocity = estimator.get_velocity()
    drift_rate = drift_velocity / duration

    # Save the bias learned during calibration
    calibrated_bias = estimator.get_bias()

    print(" Done!")
    print(f"\nCalibration completed ({sample_count} samples):")
    print(f"  Total drift:         [{drift_velocity[0]:7.4f}, {drift_velocity[1]:7.4f}, {drift_velocity[2]:7.4f}] m/s")
    print(f"  Drift rate:          [{drift_rate[0]:7.4f}, {drift_rate[1]:7.4f}, {drift_rate[2]:7.4f}] m/s/s")
    print(f"  Drift magnitude:     {np.linalg.norm(drift_velocity):7.4f} m/s")
    print(f"  Calibrated bias:     [{calibrated_bias[0]:7.4f}, {calibrated_bias[1]:7.4f}, {calibrated_bias[2]:7.4f}] m/s²")
    print("="*80)

    # Reset velocity but preserve the bias learned during calibration
    estimator.velocity = np.zeros(3)
    # Note: We keep estimator.bias as-is (don't reset it)

    return drift_rate


def main():
    """Run velocity estimation example."""
    print("="*80)
    print("BNO080 Velocity Estimation Example")
    print("="*80)

    try:
        # Initialize IMU reader
        print("\n[1/2] Initializing BNO080 sensor...")
        imu = BNO080Reader()

        # Initialize velocity estimator
        print("[2/2] Initializing velocity estimator...")
        estimator = VelocityEstimator(
            lambda_=0.005,      # Moderate drift control
            bias_alpha=0.001,   # Slow bias adaptation
            zero_vz=False       # Allow vertical velocity
        )

        print("\n" + "="*80)
        print("Configuration:")
        print(f"  Leakage factor (λ):  {estimator.lambda_}")
        print(f"  Bias adaptation (α): {estimator.bias_alpha}")
        print(f"  Zero vertical vel:   {estimator.zero_vz}")
        print(f"  Sample rate:         ~40 Hz")
        print("="*80)

        # Run drift calibration
        drift_rate = calibrate_drift(imu, estimator, duration=5.0)

        print("\nStarting drift-corrected velocity estimation...")
        print("Press Ctrl+C to stop\n")

        prev_time = None
        iteration = 0
        calibration_start_time = None

        while True:
            # Read IMU data
            data = imu.read_data()
            current_time = data['timestamp']

            # Set calibration reference time on first iteration
            if calibration_start_time is None:
                calibration_start_time = current_time

            # Calculate time step
            if prev_time is None:
                prev_time = current_time
                continue

            dt = current_time - prev_time
            prev_time = current_time
            iteration += 1

            # Extract linear acceleration (body frame, gravity removed)
            accel_body = np.array([
                data['linear_accel']['x'],
                data['linear_accel']['y'],
                data['linear_accel']['z']
            ])

            # Extract orientation quaternion
            quat = data['quaternion']

            # Update velocity estimate
            velocity_raw = estimator.update(accel_body, quat, dt)

            # Apply drift correction
            # Subtract accumulated drift since calibration
            elapsed_since_calibration = current_time - calibration_start_time
            drift_correction = drift_rate * elapsed_since_calibration
            velocity = velocity_raw - drift_correction

            # Get bias estimate
            bias = estimator.get_bias()

            # Calculate speed (magnitude)
            speed = np.linalg.norm(velocity)

            # Print results
            print(f"\n{'─'*80}")
            print(f"Iteration: {iteration:5d} | Time: {current_time:8.3f}s | dt: {dt*1000:5.1f}ms")
            print(f"{'─'*80}")

            print(f"\n  Linear Accel (body frame):  "
                  f"X:{accel_body[0]:7.3f}  Y:{accel_body[1]:7.3f}  Z:{accel_body[2]:7.3f} m/s²")

            print(f"\n  Velocity Raw (world):       "
                  f"X:{velocity_raw[0]:7.3f}  Y:{velocity_raw[1]:7.3f}  Z:{velocity_raw[2]:7.3f} m/s")

            print(f"  Drift Correction:           "
                  f"X:{drift_correction[0]:7.3f}  Y:{drift_correction[1]:7.3f}  Z:{drift_correction[2]:7.3f} m/s")

            print(f"  Velocity Corrected:         "
                  f"X:{velocity[0]:7.3f}  Y:{velocity[1]:7.3f}  Z:{velocity[2]:7.3f} m/s")

            print(f"  Speed (magnitude):          {speed:7.3f} m/s")

            print(f"\n  Accel Bias (world frame):   "
                  f"X:{bias[0]:7.3f}  Y:{bias[1]:7.3f}  Z:{bias[2]:7.3f} m/s²")

            # Sleep to maintain approximate sample rate
            time.sleep(0.025)  # Target ~40Hz

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Stopped by user")
        print("="*80)

        if 'velocity' in locals():
            print(f"\nFinal State:")
            print(f"  Velocity: [{velocity[0]:7.3f}, {velocity[1]:7.3f}, {velocity[2]:7.3f}] m/s")
            print(f"  Speed:    {np.linalg.norm(velocity):7.3f} m/s")
            print(f"  Bias:     [{bias[0]:7.3f}, {bias[1]:7.3f}, {bias[2]:7.3f}] m/s²")
            print(f"  Samples:  {iteration}")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Error occurred: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
