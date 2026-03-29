#!/usr/bin/env python3
"""
Body Linear Velocity Estimation from BNO080 IMU

This module estimates linear velocity in the world frame from body-mounted IMU data.
The BNO080 provides gravity-compensated linear acceleration and orientation (quaternion).

Drift Control:
    IMU velocity estimation suffers from unbounded drift due to:
    - Sensor noise accumulation during integration
    - Small bias errors that compound over time
    - Numerical integration errors

    We use velocity leakage (damping) to prevent drift:
        v_new = (1 - λ) * v_old + a * dt

    This exponentially decays velocity toward zero when no acceleration is present,
    which is a reasonable assumption for short-horizon velocity estimation without
    external reference signals (no GPS, no ZUPT).
"""

import numpy as np


class VelocityEstimator:
    """
    Estimates linear velocity in world frame from body-mounted IMU.

    Uses quaternion-based frame transformation and velocity leakage to control drift.
    Optionally estimates and removes acceleration bias.

    Args:
        lambda_: Velocity leakage factor (0 to 1). Higher values = more damping.
                 Typical range: 0.001 - 0.01. Default: 0.005
        bias_alpha: Low-pass filter coefficient for bias estimation (0 to 1).
                    Lower values = slower adaptation. Default: 0.001
        zero_vz: If True, constrains vertical velocity to zero (ground robot assumption).
                 Default: False

    Example:
        >>> estimator = VelocityEstimator(lambda_=0.005, bias_alpha=0.001)
        >>> # In your data loop:
        >>> data = imu.read_data()
        >>> accel_body = np.array([data['linear_accel']['x'],
        ...                         data['linear_accel']['y'],
        ...                         data['linear_accel']['z']])
        >>> quat = data['quaternion']  # dict with keys: x, y, z, w
        >>> velocity = estimator.update(accel_body, quat, dt=0.025)
    """

    def __init__(self, lambda_=0.005, bias_alpha=0.001, zero_vz=False):
        """Initialize velocity estimator with drift control parameters."""
        self.lambda_ = lambda_  # Velocity leakage factor
        self.bias_alpha = bias_alpha  # Bias adaptation rate
        self.zero_vz = zero_vz  # Vertical velocity constraint

        # State variables
        self.velocity = np.zeros(3)  # [vx, vy, vz] in world frame
        self.bias = np.zeros(3)      # Acceleration bias in world frame

    @staticmethod
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        """
        Convert quaternion to 3x3 rotation matrix.

        Quaternion convention: (qx, qy, qz, qw) represents rotation from body to world frame.
        The resulting rotation matrix R satisfies: v_world = R @ v_body

        Args:
            qx, qy, qz, qw: Quaternion components (scalar qw, vector [qx, qy, qz])

        Returns:
            R: 3x3 rotation matrix (NumPy array)

        Reference:
            Standard quaternion to rotation matrix conversion.
            Numerically stable for unit quaternions.
        """
        # Pre-compute repeated terms
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz
        qw2 = qw * qw

        qxqy = qx * qy
        qxqz = qx * qz
        qxqw = qx * qw
        qyqz = qy * qz
        qyqw = qy * qw
        qzqw = qz * qw

        # Construct rotation matrix
        # Row-major order: R[i, j] = R_ij
        R = np.array([
            [qw2 + qx2 - qy2 - qz2,  2*(qxqy - qzqw),        2*(qxqz + qyqw)],
            [2*(qxqy + qzqw),        qw2 - qx2 + qy2 - qz2,  2*(qyqz - qxqw)],
            [2*(qxqz - qyqw),        2*(qyqz + qxqw),        qw2 - qx2 - qy2 + qz2]
        ])

        return R

    def update(self, accel_body, quat, dt):
        """
        Update velocity estimate with new IMU measurement.

        Processing pipeline:
            1. Convert quaternion to rotation matrix
            2. Transform acceleration from body to world frame
            3. Update bias estimate (slow low-pass filter)
            4. Integrate velocity with leakage (drift control)
            5. Apply vertical constraint if enabled

        Args:
            accel_body: Linear acceleration in body frame, shape (3,) or [ax, ay, az].
                        Units: m/s² (gravity already removed by BNO080)
            quat: Orientation quaternion as dict with keys 'x', 'y', 'z', 'w'
                  or as array-like [qx, qy, qz, qw]
            dt: Time step in seconds

        Returns:
            velocity: Estimated velocity in world frame [vx, vy, vz], units: m/s
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(accel_body, (list, tuple)):
            accel_body = np.array(accel_body)

        # Extract quaternion components
        if isinstance(quat, dict):
            qx, qy, qz, qw = quat['x'], quat['y'], quat['z'], quat['w']
        else:
            qx, qy, qz, qw = quat

        # Step 1: Convert quaternion to rotation matrix (body → world)
        R = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

        # Step 2: Transform acceleration to world frame
        accel_world = R @ accel_body

        # Step 3: Update bias estimate using exponential moving average
        # Bias adapts slowly to capture DC offsets in acceleration
        # bias_new = (1 - α) * bias_old + α * accel_world
        self.bias = (1 - self.bias_alpha) * self.bias + self.bias_alpha * accel_world

        # Step 4: Velocity integration with leakage (drift control)
        # Leakage prevents unbounded drift by exponentially decaying velocity toward zero
        # This is necessary because:
        #   - We have no external velocity reference (no GPS, no ZUPT)
        #   - Integration errors accumulate over time
        #   - Small bias errors lead to velocity drift
        # The leakage factor λ controls the trade-off between:
        #   - Responsiveness to real acceleration (low λ)
        #   - Drift suppression (high λ)
        accel_corrected = accel_world - self.bias
        self.velocity = (1 - self.lambda_) * self.velocity + accel_corrected * dt

        # Step 5: Apply vertical constraint if enabled (for ground robots)
        if self.zero_vz:
            self.velocity[2] = 0.0

        return self.velocity.copy()

    def reset(self):
        """Reset estimator state (velocity and bias) to zero."""
        self.velocity = np.zeros(3)
        self.bias = np.zeros(3)

    def get_velocity(self):
        """Get current velocity estimate without updating."""
        return self.velocity.copy()

    def get_bias(self):
        """Get current acceleration bias estimate."""
        return self.bias.copy()


def main():
    """
    Example usage: Integrate velocity estimator with BNO080 reader.

    This demonstrates how to use VelocityEstimator with data from bare.py.
    """
    import time
    from bare import BNO080Reader

    print("="*80)
    print("BNO080 Velocity Estimation Demo")
    print("="*80)

    try:
        # Initialize IMU reader
        imu = BNO080Reader()

        # Initialize velocity estimator
        # lambda_=0.005: moderate drift control (velocity halves in ~139 samples @ 40Hz)
        # bias_alpha=0.001: slow bias adaptation (~1000 samples to converge)
        # zero_vz=False: allow vertical velocity (set True for ground robots)
        estimator = VelocityEstimator(lambda_=0.005, bias_alpha=0.001, zero_vz=False)

        print("\nStarting velocity estimation...")
        print("Lambda (leakage): {:.4f}".format(estimator.lambda_))
        print("Bias alpha: {:.4f}".format(estimator.bias_alpha))
        print("Vertical constraint: {}".format(estimator.zero_vz))
        print("\nPress Ctrl+C to stop\n")

        prev_time = None

        while True:
            # Read IMU data
            data = imu.read_data()
            current_time = data['timestamp']

            # Calculate dt
            if prev_time is None:
                prev_time = current_time
                continue
            dt = current_time - prev_time
            prev_time = current_time

            # Extract acceleration and quaternion
            accel_body = np.array([
                data['linear_accel']['x'],
                data['linear_accel']['y'],
                data['linear_accel']['z']
            ])
            quat = data['quaternion']  # Already in dict format

            # Update velocity estimate
            velocity = estimator.update(accel_body, quat, dt)

            # Display results
            print(f"\n{'='*80}")
            print(f"Time: {current_time:.3f}s | dt: {dt*1000:.1f}ms")
            print(f"{'='*80}")
            print(f"Accel (body):  [{accel_body[0]:7.3f}, {accel_body[1]:7.3f}, {accel_body[2]:7.3f}] m/s²")
            print(f"Velocity (world): [{velocity[0]:7.3f}, {velocity[1]:7.3f}, {velocity[2]:7.3f}] m/s")
            print(f"Bias estimate: [{estimator.bias[0]:7.3f}, {estimator.bias[1]:7.3f}, {estimator.bias[2]:7.3f}] m/s²")

            # Sleep to maintain ~40Hz rate
            time.sleep(0.025)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        print(f"\nFinal velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] m/s")
        print(f"Final bias: [{estimator.bias[0]:.3f}, {estimator.bias[1]:.3f}, {estimator.bias[2]:.3f}] m/s²")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
