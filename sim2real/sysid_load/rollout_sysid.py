"""Rollout validation — replay a recording with custom physics params and plot.

Loads a load_recorder .npz, replays the target sequence through MuJoCo with
the given armature/damping/frictionloss, and plots the result against real data.
Prints RMSE_pos, RMSE_vel, and the combined loss.

Usage:
  python rollout_sysid.py data/sine.npz --armature 0.005 --damping 0.2 --frictionloss 0.01
  python rollout_sysid.py data/sine.npz --armature 0.005 --damping 0.2 --frictionloss 0.01 --vel-weight 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

_HERE      = Path(__file__).parent
_XML       = _HERE / "motor_assembly.xml"
PHYSICS_DT = 0.001

_JOINT_NAME = "load_joint"


# ── MuJoCo helpers ──────────────────────────────────────────────────────────────

def _apply_params(m: mujoco.MjModel, armature: float, damping: float, frictionloss: float) -> None:
    jid = m.joint(_JOINT_NAME).id
    dof = int(m.jnt_dofadr[jid])
    m.dof_armature[dof]     = armature
    m.dof_damping[dof]      = damping
    m.dof_frictionloss[dof] = frictionloss


def _get_ids(m: mujoco.MjModel) -> tuple[int, int, int]:
    jid     = m.joint(_JOINT_NAME).id
    qpos_id = int(m.jnt_qposadr[jid])
    qvel_id = int(m.jnt_dofadr[jid])
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == jid:
            return qpos_id, qvel_id, i
    raise ValueError(f"No actuator found for joint '{_JOINT_NAME}'")


# ── Replay ───────────────────────────────────────────────────────────────────────

def replay(
    targets:      np.ndarray,
    q0:           float,
    control_hz:   int,
    kp_sim:       float,
    kd_sim:       float,
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    m = mujoco.MjModel.from_xml_path(str(_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, armature, damping, frictionloss)

    # Overwrite XML kp/kv with values from the dataset
    m.actuator_gainprm[0, 0] = kp_sim
    m.actuator_biasprm[0, 1] = -kp_sim
    m.actuator_biasprm[0, 2] = -kd_sim

    d = mujoco.MjData(m)
    qpos_id, qvel_id, ctrl_id = _get_ids(m)

    d.qpos[qpos_id] = q0
    d.qvel[qvel_id] = 0.0
    mujoco.mj_forward(m, d)

    N        = len(targets)
    substeps = max(1, round(1.0 / (control_hz * PHYSICS_DT)))
    q_sim    = np.empty(N)
    dq_sim   = np.empty(N)

    for i in range(N):
        q_sim[i]  = d.qpos[qpos_id]
        dq_sim[i] = d.qvel[qvel_id]
        d.ctrl[ctrl_id] = targets[i]
        for _ in range(substeps):
            mujoco.mj_step(m, d)

    return q_sim, dq_sim


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recording with custom params and plot vs real."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--armature",     type=float, required=True, help="e.g. 0.05")
    parser.add_argument("--damping",      type=float, required=True, help="e.g. 0.2")
    parser.add_argument("--frictionloss", type=float, required=True, help="e.g. 0.01")
    parser.add_argument("--vel-weight",   type=float, default=None,
                        help="Velocity loss weight (auto = std_q/std_dq if omitted).")
    parser.add_argument("--save", action="store_true",
                        help="Save plot as <recording_stem>_rollout.png next to the data file.")
    args = parser.parse_args()

    # ── Load recording ───────────────────────────────────────────────────────────
    data = np.load(args.recording, allow_pickle=True)

    t          = data["t"]
    target     = data["target"]
    q_real     = data["q_real"]
    dq_real    = data["dq_real"]
    control_hz = int(data["control_hz"][0])
    signal_mode = str(data["signal_mode"].flat[0]) if "signal_mode" in data else "?"

    # kp/kd from dataset — warn if changed mid-recording
    if "kp_sim" in data:
        kp_arr = data["kp_sim"]
        if np.any(kp_arr != kp_arr[0]):
            print(f"[WARNING] kp_sim changed during recording "
                  f"(min={kp_arr.min():.4f} max={kp_arr.max():.4f}). "
                  f"Using first value {kp_arr[0]:.4f}.")
        kp_sim = float(kp_arr[0])
    else:
        print("[WARNING] kp_sim not found in recording, defaulting to 6.0")
        kp_sim = 6.0

    if "kd_sim" in data:
        kd_arr = data["kd_sim"]
        if np.any(kd_arr != kd_arr[0]):
            print(f"[WARNING] kd_sim changed during recording "
                  f"(min={kd_arr.min():.4f} max={kd_arr.max():.4f}). "
                  f"Using first value {kd_arr[0]:.4f}.")
        kd_sim = float(kd_arr[0])
    else:
        print("[WARNING] kd_sim not found in recording, defaulting to 0.0")
        kd_sim = 0.0

    kp_real = float(data["kp_real"][0]) if "kp_real" in data else float("nan")
    kd_real = float(data["kd_real"][0]) if "kd_real" in data else float("nan")

    N   = len(t)
    dur = t[-1] - t[0]
    print(f"Loaded  {args.recording.name}  N={N}  hz={control_hz}  duration={dur:.1f}s  "
          f"mode={signal_mode}  kp_sim={kp_sim}  kd_sim={kd_sim}")

    # ── Velocity weight ──────────────────────────────────────────────────────────
    std_q  = float(np.std(q_real))
    std_dq = float(np.std(dq_real))
    if args.vel_weight is not None:
        vel_weight = args.vel_weight
        print(f"vel_weight (manual) = {vel_weight:.6f}")
    else:
        vel_weight = std_q / std_dq if std_dq > 1e-12 else 0.0
        print(f"vel_weight (auto)   = {vel_weight:.6f}  "
              f"[std_q={std_q:.4f}  std_dq={std_dq:.4f}]")

    # ── Replay ───────────────────────────────────────────────────────────────────
    print(f"Replaying with armature={args.armature}  damping={args.damping}  "
          f"frictionloss={args.frictionloss} …")
    q_sim, dq_sim = replay(
        target, float(q_real[0]), control_hz, kp_sim, kd_sim,
        args.armature, args.damping, args.frictionloss,
    )

    # ── Losses ───────────────────────────────────────────────────────────────────
    rmse_pos = float(np.sqrt(np.mean((q_sim  - q_real)  ** 2)))
    rmse_vel = float(np.sqrt(np.mean((dq_sim - dq_real) ** 2)))
    loss     = rmse_pos + vel_weight * rmse_vel

    print(f"\nRMSE_pos     = {rmse_pos:.6f} rad")
    print(f"RMSE_vel     = {rmse_vel:.6f} rad/s")
    print(f"vel_weight   = {vel_weight:.6f}")
    print(f"loss         = {loss:.6f}  (RMSE_pos + vel_weight * RMSE_vel)")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    C_TARGET = "#f5a623"
    C_REAL   = "#2171b5"
    C_SIM    = "#d62728"

    title = (
        f"{args.recording.name}   mode={signal_mode}   N={N}   hz={control_hz}   dur={dur:.1f}s\n"
        f"armature={args.armature}   damping={args.damping}   frictionloss={args.frictionloss}\n"
        f"kp_sim={kp_sim}   kd_sim={kd_sim}   kp_real={kp_real}   kd_real={kd_real}\n"
        f"RMSE_pos={rmse_pos:.5f} rad   RMSE_vel={rmse_vel:.5f} rad/s   "
        f"vel_weight={vel_weight:.4f}   loss={loss:.5f}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(title, fontsize=8.5, family="monospace",
                 verticalalignment="top", y=1.02)

    # Row 0 — position
    ax = axes[0]
    ax.plot(t, target, color=C_TARGET, lw=1.2, ls="--", label="target")
    ax.plot(t, q_real, color=C_REAL,   lw=1.5, ls="--", label="real")
    ax.plot(t, q_sim,  color=C_SIM,    lw=1.5,           label="sim")
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 — position error (real - sim)
    ax = axes[1]
    pos_err = q_real - q_sim
    ax.plot(t, pos_err, color=C_SIM, lw=1.0)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Pos error (rad)\n(real − sim)")
    ax.set_title(f"RMSE_pos = {rmse_pos:.5f} rad", fontsize=8, loc="right")
    ax.grid(True, alpha=0.3)

    # Row 2 — velocity
    ax = axes[2]
    ax.plot(t, dq_real, color=C_REAL, lw=1.5, ls="--", label="real")
    ax.plot(t, dq_sim,  color=C_SIM,  lw=1.5,           label="sim")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"RMSE_vel = {rmse_vel:.5f} rad/s", fontsize=8, loc="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        out = args.recording.parent / f"{args.recording.stem}_rollout.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
