"""Build the 41-dim observation vector from real hardware readings.

Obs layout (matches training exactly):
  [0]      body_height       (1,)  — FK + IMU, normalized
  [1]      target_height     (1,)  — fixed goal, normalized
  [2:5]    projected_gravity (3,)  — gravity in body frame via IMU
  [5:17]   joint_pos         (12,) — encoder positions rel. to sit pose
  [17:29]  joint_vel         (12,) — encoder velocities
  [29:41]  actions           (12,) — last raw policy output (zeros at t=0)

All values are float32. The obs normalizer lives inside the policy (applied
automatically on each forward pass) — do NOT pre-normalize here.
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Joint order must match MJLAB training order exactly.
# ---------------------------------------------------------------------------
JOINT_ORDER = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "BL_hip_joint",  "BL_thigh_joint",  "BL_calf_joint",
    "BR_hip_joint",  "BR_thigh_joint",  "BR_calf_joint",
]

# ---------------------------------------------------------------------------
# Height normalization constants (must match training: min=0.10, max=0.25).
# ---------------------------------------------------------------------------
_H_MIN = 0.10
_H_MAX = 0.25

# Target standing height in metres. Normalized value fed to the policy.
# Training curriculum play range: [0.20, 0.30] m → use 0.25 (normalizes to 1.0).
TARGET_HEIGHT_M = 0.25
_TARGET_HEIGHT_OBS = float((TARGET_HEIGHT_M - _H_MIN) / (_H_MAX - _H_MIN))   # = 1.0


# ---------------------------------------------------------------------------
# FK import (copied kinematics are in the rumi-mjlab source tree).
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_RUMI_MJLAB = _Path(__file__).resolve().parent.parent.parent / "rumi-mjlab" / "src"
if str(_RUMI_MJLAB) not in _sys.path:
    _sys.path.insert(0, str(_RUMI_MJLAB))
from rumi_getup.rumi.kinematics import estimate_body_height as _estimate_body_height_fk


def _estimate_body_height(joint_pos_rad: np.ndarray, quat_wxyz: np.ndarray) -> float:
    """Body height in metres via FK + IMU."""
    jp = torch.from_numpy(joint_pos_rad).float().unsqueeze(0)   # [1, 12]
    q  = torch.from_numpy(quat_wxyz).float().unsqueeze(0)       # [1,  4]
    return float(_estimate_body_height_fk(jp, q).item())


# ---------------------------------------------------------------------------
# Gravity projection.
# ---------------------------------------------------------------------------
def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into body frame.

    Uses the inverse quaternion rotation:
        t = 2 * (xyz × v)
        v' = v + w*t + (xyz × t)

    Args:
        quat_wxyz: [4] (w, x, y, z).

    Returns:
        [3] gravity vector in body frame.  When upright: [~0, ~0, ~-1].
    """
    w, x, y, z = quat_wxyz.astype(np.float64)
    xyz = np.array([x, y, z])
    v   = np.array([0.0, 0.0, -1.0])
    t   = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def build_obs(
    joint_pos_dict: dict,
    joint_vel_dict: dict,
    quat_wxyz: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Assemble the 41-dim observation vector.

    Args:
        joint_pos_dict: {joint_name: position_rad} from DynamixelController.
                        Values are already relative to sit-pose zero reference.
        joint_vel_dict: {joint_name: velocity_rad_s} same order/convention.
        quat_wxyz:      IMU quaternion [4] in (w, x, y, z) convention.
        last_action:    Raw policy output from the previous step [12].
                        Pass np.zeros(12) at t=0.

    Returns:
        obs: float32 np.ndarray of shape [41].
    """
    joint_pos = np.array([joint_pos_dict[j] for j in JOINT_ORDER], dtype=np.float32)
    joint_vel = np.array([joint_vel_dict[j] for j in JOINT_ORDER], dtype=np.float32)

    # body_height
    raw_h = _estimate_body_height(joint_pos, quat_wxyz)
    body_height_obs = float(np.clip((raw_h - _H_MIN) / (_H_MAX - _H_MIN), -0.5, 1.5))

    # projected gravity
    proj_grav = _projected_gravity(quat_wxyz)

    obs = np.concatenate([
        [body_height_obs],           # [1]
        [_TARGET_HEIGHT_OBS],        # [1]
        proj_grav,                   # [3]
        joint_pos,                   # [12]
        joint_vel,                   # [12]
        last_action.astype(np.float32),  # [12]
    ])
    assert obs.shape == (41,), f"Expected (41,), got {obs.shape}"
    return obs.astype(np.float32)
