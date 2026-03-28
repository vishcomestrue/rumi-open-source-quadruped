"""Rollout and plot only — no optimisation.

Replays a standup recording with two sets of params (init and optimized)
and plots position, position error, and velocity for the selected joints.

Usage:
  python rollout_plot.py data/<recording>.npz --calf \
      --init 0.0005 0.5 0.1 \
      --optimized 0.0005 0.5845 0.02

Joint selection flags (same as sysid_standup.py):
  --calf / --calf-front / --calf-back
  --thigh / --thigh-front / --thigh-back
  --hip / --hip-front / --hip-back
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import mujoco
import numpy as np

_HERE      = Path(__file__).parent
_SCENE_XML = _HERE / "scene.xml"
PHYSICS_DT = 0.004


# ── MuJoCo helpers (copied from sysid_standup.py) ───────────────────────────

def _joint_part(name: str) -> str:
    return name.split("_")[1]


def _get_joint_ids(m: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    jid     = m.joint(joint_name).id
    qpos_id = int(m.jnt_qposadr[jid])
    qvel_id = int(m.jnt_dofadr[jid])
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == jid:
            return qpos_id, qvel_id, i
    raise ValueError(f"No actuator for joint '{joint_name}'")


def _apply_params(
    m: mujoco.MjModel,
    joint_names: list[str],
    armature: float,
    damping: float,
    frictionloss: float,
) -> None:
    for jname in joint_names:
        dof_id = m.jnt_dofadr[m.joint(jname).id]
        m.dof_armature[dof_id]     = armature
        m.dof_damping[dof_id]      = damping
        m.dof_frictionloss[dof_id] = frictionloss


def replay_all(
    targets:      np.ndarray,
    q0:           np.ndarray,
    control_hz:   int,
    joint_names:  list[str],
    sel_names:    list[str],
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    m = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, sel_names, armature, damping, frictionloss)

    d   = mujoco.MjData(m)
    ids = [_get_joint_ids(m, jn) for jn in joint_names]

    for col, (qpos_id, _, _) in enumerate(ids):
        d.qpos[qpos_id] = float(q0[col])
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    N        = len(targets)
    substeps = max(1, round(1.0 / (control_hz * PHYSICS_DT)))
    q_sim    = np.empty((N, 12))
    dq_sim   = np.empty((N, 12))

    for i in range(N):
        for col, (qpos_id, qvel_id, ctrl_id) in enumerate(ids):
            q_sim[i, col]  = d.qpos[qpos_id]
            dq_sim[i, col] = d.qvel[qvel_id]
            d.ctrl[ctrl_id] = targets[i, col]
        for _ in range(substeps):
            mujoco.mj_step(m, d)

    return q_sim, dq_sim


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollout and plot: compare init vs optimized params, no optimisation."
    )
    parser.add_argument("recording", type=Path, help="Path to standup .npz recording.")
    # Joint selection
    parser.add_argument("--calf",        action="store_true")
    parser.add_argument("--calf-front",  action="store_true")
    parser.add_argument("--calf-back",   action="store_true")
    parser.add_argument("--thigh",       action="store_true")
    parser.add_argument("--thigh-front", action="store_true")
    parser.add_argument("--thigh-back",  action="store_true")
    parser.add_argument("--hip",         action="store_true")
    parser.add_argument("--hip-front",   action="store_true")
    parser.add_argument("--hip-back",    action="store_true")
    # Params
    parser.add_argument("--init", type=float, nargs=3,
                        metavar=("ARMATURE", "DAMPING", "FRICTIONLOSS"),
                        default=[0.0005, 0.5, 0.1],
                        help="Init params (default: 0.0005 0.5 0.1).")
    parser.add_argument("--optimized", type=float, nargs=3,
                        metavar=("ARMATURE", "DAMPING", "FRICTIONLOSS"),
                        required=True,
                        help="Optimized params to compare against init.")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    data        = np.load(args.recording, allow_pickle=True)
    t           = data["t"]
    targets     = data["target"]             # (N, 12)
    q_real      = data["q_real"]             # (N, 12)
    dq_real     = data["dq_real"]            # (N, 12)
    joint_names = data["joint_names"].tolist()
    control_hz  = int(data["control_hz"][0])
    N           = len(t)
    kp_sim  = float(data["kp_sim"][0])  if "kp_sim"  in data else float("nan")
    kd_sim  = float(data["kd_sim"][0])  if "kd_sim"  in data else float("nan")
    kp_real = float(data["kp_real"][0]) if "kp_real" in data else float("nan")
    kd_real = float(data["kd_real"][0]) if "kd_real" in data else float("nan")

    print(f"Loaded  N={N}  hz={control_hz}  duration={t[-1]-t[0]:.1f}s")

    # ── Select joints ────────────────────────────────────────────────────────
    _FRONT = {"FL", "FR"}
    _BACK  = {"BL", "BR"}

    for part in ("calf", "thigh", "hip"):
        if getattr(args, f"{part}_front") and getattr(args, f"{part}_back"):
            parser.error(f"--{part}-front + --{part}-back is not allowed; use --{part} instead.")

    sel_set:   set[int]  = set()
    tag_parts: list[str] = []

    def _add(part: str, locs: set[str] | None, tag: str) -> None:
        for i, n in enumerate(joint_names):
            if _joint_part(n) == part and (locs is None or n.split("_")[0] in locs):
                sel_set.add(i)
        tag_parts.append(tag)

    if args.calf:        _add("calf",  None,   "calf")
    if args.calf_front:  _add("calf",  _FRONT, "calf_front")
    if args.calf_back:   _add("calf",  _BACK,  "calf_back")
    if args.thigh:       _add("thigh", None,   "thigh")
    if args.thigh_front: _add("thigh", _FRONT, "thigh_front")
    if args.thigh_back:  _add("thigh", _BACK,  "thigh_back")
    if args.hip:         _add("hip",   None,   "hip")
    if args.hip_front:   _add("hip",   _FRONT, "hip_front")
    if args.hip_back:    _add("hip",   _BACK,  "hip_back")

    if not sel_set:
        sel_set   = set(range(len(joint_names)))
        tag_parts = ["all"]

    sel_cols  = sorted(sel_set)
    sel_names = [joint_names[i] for i in sel_cols]
    parts_tag = "_".join(tag_parts)

    print(f"Selected joints ({len(sel_names)}): {sel_names}")

    q0 = q_real[0].copy()

    # ── Rollout ──────────────────────────────────────────────────────────────
    arm_i, dmp_i, fri_i = args.init
    arm_o, dmp_o, fri_o = args.optimized

    print(f"Init params:      armature={arm_i}  damping={dmp_i}  frictionloss={fri_i}")
    print(f"Optimized params: armature={arm_o}  damping={dmp_o}  frictionloss={fri_o}")
    print("Rolling out init…")
    q_sim_init, dq_sim_init = replay_all(
        targets, q0, control_hz, joint_names, sel_names, arm_i, dmp_i, fri_i,
    )
    print("Rolling out optimized…")
    q_sim_opt, dq_sim_opt = replay_all(
        targets, q0, control_hz, joint_names, sel_names, arm_o, dmp_o, fri_o,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_sel  = len(sel_names)
    n_rows = 3
    fig, axes = plt.subplots(
        n_rows, n_sel,
        figsize=(max(13, 4.0 * n_sel), 3.5 * n_rows),
        sharex=True, squeeze=False,
    )
    fig.suptitle(
        f"Rollout  joints={sel_names}\n"
        f"init:      armature={arm_i}  damping={dmp_i}  frictionloss={fri_i}\n"
        f"optimized: armature={arm_o}  damping={dmp_o}  frictionloss={fri_o}\n"
        f"N={N}  hz={control_hz}  kp_sim={kp_sim}  kd_sim={kd_sim}  "
        f"kp_real={kp_real}  kd_real={kd_real}",
        fontsize=9,
    )

    for j, (col, jname) in enumerate(zip(sel_cols, sel_names)):
        short = jname.replace("_joint", "")

        # Position
        ax = axes[0][j]
        ax.plot(t, np.rad2deg(targets[:, col]),     color="orange",    lw=1.2, label="target")
        ax.plot(t, np.rad2deg(q_real[:, col]),       color="red",       lw=1.5, ls="--", label="real")
        ax.plot(t, np.rad2deg(q_sim_init[:, col]),   color="gray",      lw=1.0, ls=":",  label="sim init")
        ax.plot(t, np.rad2deg(q_sim_opt[:, col]),    color="steelblue", lw=1.5, label="sim optimized")
        ax.set_title(short, fontsize=9)
        ax.set_ylabel("deg" if j == 0 else "")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Position error
        ax = axes[1][j]
        ax.plot(t, np.rad2deg(q_real[:, col] - q_sim_opt[:, col]),
                color="steelblue", lw=1.0, label="optimized")
        ax.plot(t, np.rad2deg(q_real[:, col] - q_sim_init[:, col]),
                color="gray",      lw=1.0, ls=":", label="init")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("pos err (deg)" if j == 0 else "")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Velocity
        ax = axes[2][j]
        ax.plot(t, np.rad2deg(dq_real[:, col]),     color="red",       lw=1.5, ls="--", label="real")
        ax.plot(t, np.rad2deg(dq_sim_init[:, col]), color="gray",      lw=1.0, ls=":",  label="sim init")
        ax.plot(t, np.rad2deg(dq_sim_opt[:, col]),  color="steelblue", lw=1.5, label="sim optimized")
        ax.set_ylabel("deg/s" if j == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_dir = args.recording.parent / f"{args.recording.stem}_rollout"
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"{parts_tag}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
