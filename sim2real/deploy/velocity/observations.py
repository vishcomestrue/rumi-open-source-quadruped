"""Build the 48-dim observation vector from real hardware readings.

Obs layout (matches rumi_velocity training exactly):
  [0:3]   base_ang_vel      (3,)  — IMU gyroscope, body frame (rad/s)
  [3:6]   projected_gravity (3,)  — gravity in body frame via IMU quat
  [6:18]  joint_pos         (12,) — encoder positions rel. to standing zero
  [18:30] joint_vel         (12,) — encoder velocities (rad/s)
  [30:42] actions           (12,) — last raw policy output (zeros at t=0)
  [42:45] command           (3,)  — [lin_vel_x, lin_vel_y, ang_vel_z]
  [45:48] imu_lin_acc       (3,)  — IMU accelerometer, body frame (m/s²)

No observation normalization (obs_normalization=False in rl_cfg).
All values are float32.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint order must match rumi_velocity training order exactly.
# ---------------------------------------------------------------------------
JOINT_ORDER = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "BL_hip_joint",  "BL_thigh_joint",  "BL_calf_joint",
    "BR_hip_joint",  "BR_thigh_joint",  "BR_calf_joint",
]


def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into body frame.

    Uses the inverse quaternion rotation:
        t = 2 * (xyz × v)
        v' = v + w*t + (xyz × t)

    Args:
        quat_wxyz: [4] (w, x, y, z).

    Returns:
        [3] gravity vector in body frame. When upright: [~0, ~0, ~-1].
    """
    w, x, y, z = quat_wxyz.astype(np.float64)
    xyz = np.array([x, y, z])
    v   = np.array([0.0, 0.0, -1.0])
    t   = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


def build_obs(
    joint_pos_dict: dict,
    joint_vel_dict: dict,
    accel: np.ndarray,
    gyro: np.ndarray,
    quat_wxyz: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    """Assemble the 48-dim observation vector.

    Args:
        joint_pos_dict: {joint_name: position_rad} relative to standing zero (STAND_POSE_RAD).
        joint_vel_dict: {joint_name: velocity_rad_s}.
        accel:          IMU linear acceleration [3] (x, y, z) in m/s², body frame.
        gyro:           IMU angular velocity [3] (x, y, z) in rad/s, body frame.
        quat_wxyz:      IMU quaternion [4] in (w, x, y, z) convention.
        last_action:    Raw policy output from previous step [12]. Zeros at t=0.
        command:        Velocity command [3] = [lin_vel_x, lin_vel_y, ang_vel_z].

    Returns:
        obs: float32 np.ndarray of shape [48].
    """
    joint_pos = np.array([joint_pos_dict[j] for j in JOINT_ORDER], dtype=np.float32)
    joint_vel = np.array([joint_vel_dict[j] for j in JOINT_ORDER], dtype=np.float32)

    proj_grav = _projected_gravity(quat_wxyz)

    obs = np.concatenate([
        gyro.astype(np.float32),             # [3]  base_ang_vel
        proj_grav,                           # [3]  projected_gravity
        joint_pos,                           # [12] joint_pos
        joint_vel,                           # [12] joint_vel
        last_action.astype(np.float32),      # [12] actions
        command.astype(np.float32),          # [3]  command
        accel.astype(np.float32),            # [3]  imu_lin_acc
    ])
    assert obs.shape == (48,), f"Expected (48,), got {obs.shape}"
    return obs.astype(np.float32)
