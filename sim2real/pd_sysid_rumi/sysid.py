"""Rumi sysid — fits armature, damping, frictionloss for one joint.

Loads a recording produced by rumi_sync.py, replays the target sequence
through MuJoCo for candidate parameters, and uses CMA-ES to minimise the
MSE between sim and real position/velocity trajectories.

The joint name is read from the recording file (saved by rumi_sync.py).
Override with --joint if needed.

Usage:
  python sysid.py data/<stamp>_<joint>_<mode>_recording.npz
  python sysid.py data/<stamp>_recording.npz --joint FL_hip_joint --popsize 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_HERE      = Path(__file__).parent
_SCENE_XML = _HERE / "scene.xml"
PHYSICS_DT = 0.004


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────

def _get_indices(m: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    jid     = m.joint(joint_name).id
    qpos_id = int(m.jnt_qposadr[jid])
    qvel_id = int(m.jnt_dofadr[jid])
    # find actuator that drives this joint
    for i in range(m.nu):
        trnid = m.actuator_trnid[i, 0]
        if trnid == jid:
            return qpos_id, qvel_id, i
    raise ValueError(f"No actuator found for joint '{joint_name}'")


def _apply_params(m: mujoco.MjModel, joint_name: str, armature: float,
                  damping: float, frictionloss: float) -> None:
    dof_id = m.jnt_dofadr[m.joint(joint_name).id]
    m.dof_armature[dof_id]     = armature
    m.dof_damping[dof_id]      = damping
    m.dof_frictionloss[dof_id] = frictionloss


# ── Simulation replay ──────────────────────────────────────────────────────────

def replay(
    targets:      np.ndarray,   # (N,) rad
    q0:           float,        # initial position rad
    control_hz:   int,
    joint_name:   str,
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay target sequence in MuJoCo and return (q_sim, dq_sim), each (N,)."""
    m = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, joint_name, armature, damping, frictionloss)

    d = mujoco.MjData(m)
    qpos_id, qvel_id, ctrl_id = _get_indices(m, joint_name)

    d.qpos[qpos_id] = q0
    d.qvel[:]       = 0.0
    mujoco.mj_forward(m, d)

    substeps = max(1, round(1.0 / (control_hz * PHYSICS_DT)))
    q_sim  = np.empty(len(targets))
    dq_sim = np.empty(len(targets))

    for i, tgt in enumerate(targets):
        q_sim[i]        = d.qpos[qpos_id]
        dq_sim[i]       = d.qvel[qvel_id]
        d.ctrl[ctrl_id] = tgt
        for _ in range(substeps):
            mujoco.mj_step(m, d)

    return q_sim, dq_sim


# ── Loss ───────────────────────────────────────────────────────────────────────

def loss(
    theta:       np.ndarray,   # [log_armature, log_damping, log_frictionloss]
    targets:     np.ndarray,
    q_real:      np.ndarray,
    dq_real:     np.ndarray,
    q0:          float,
    control_hz:  int,
    joint_name:  str,
    vel_weight:  float,
) -> float:
    armature, damping, frictionloss = np.exp(theta)
    q_sim, dq_sim = replay(targets, q0, control_hz, joint_name,
                           armature, damping, frictionloss)
    pos_loss = float(np.sqrt(np.mean((q_sim  - q_real)  ** 2)))
    vel_loss = float(np.sqrt(np.mean((dq_sim - dq_real) ** 2)))
    return pos_loss + vel_weight * vel_loss


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MX-64 sysid via CMA-ES.")
    parser.add_argument("recording", type=Path, help="Path to .npz recording.")
    parser.add_argument("--popsize",  type=int,   default=12,
                        help="CMA-ES population size (default 12).")
    parser.add_argument("--max-iter", type=int,   default=300,
                        help="Max CMA-ES iterations (default 300).")
    parser.add_argument("--sigma0",     type=float, default=0.5,
                        help="Initial CMA-ES step size in log-space (default 0.5).")
    parser.add_argument("--vel-weight", type=float, default=None,
                        help="Velocity loss weight. Omit for auto-normalization "
                             "(var(q_real)/var(dq_real)).")
    parser.add_argument("--joint", type=str, default=None,
                        help="Joint name to fit. Reads from recording if not set.")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    data = np.load(args.recording, allow_pickle=True)
    t           = data["t"]
    targets     = data["target"]       # rad
    q_real      = data["q_real"]       # rad
    dq_real     = data["dq_real"]      # rad/s
    control_hz  = int(data["control_hz"][0])
    N           = len(t)
    q0          = float(q_real[0])
    kp_sim  = float(data["kp_sim"][0])  if "kp_sim"  in data else float("nan")
    kd_sim  = float(data["kd_sim"][0])  if "kd_sim"  in data else float("nan")
    kp_real = float(data["kp_real"][0]) if "kp_real" in data else float("nan")
    kd_real = float(data["kd_real"][0]) if "kd_real" in data else float("nan")

    # joint name: CLI override > stored in file > error
    if args.joint is not None:
        joint_name = args.joint
    elif "joint" in data:
        joint_name = str(data["joint"][0])
    else:
        print("[error] No joint name in recording. Use --joint <name>.")
        sys.exit(1)

    print(f"Loaded {N} samples  joint={joint_name}  control_hz={control_hz}  "
          f"duration={t[-1]:.1f}s  q0={np.rad2deg(q0):.1f}°")

    # ── Velocity weight ────────────────────────────────────────────────────────
    std_q  = float(np.std(q_real))
    std_dq = float(np.std(dq_real))
    if args.vel_weight is not None:
        vel_weight = args.vel_weight
        print(f"vel_weight (manual) = {vel_weight:.6f}")
    else:
        vel_weight = std_q / std_dq if std_dq > 1e-12 else 0.0
        print(f"vel_weight (auto)   = {vel_weight:.6f}  "
              f"[std_q={std_q:.4f}  std_dq={std_dq:.4f}]")

    # ── Initial guess (rumi.xml defaults) ─────────────────────────────────────
    # armature=0.012, damping=0.66, frictionloss=0.09
    theta0 = np.log([0.012, 0.66, 0.09])

    # ── CMA-ES ─────────────────────────────────────────────────────────────────
    es = cma.CMAEvolutionStrategy(
        theta0,
        args.sigma0,
        {
            "popsize":  args.popsize,
            "maxiter":  args.max_iter,
            "tolx":     1e-6,
            "tolfun":   1e-8,
            "verbose":  1,
        },
    )

    print("\nRunning CMA-ES optimisation…")
    t0 = time.time()
    iter_i = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            loss(theta, targets, q_real, dq_real, q0, control_hz, joint_name, vel_weight)
            for theta in solutions
        ]
        es.tell(solutions, fitnesses)
        iter_i += 1
        if iter_i % 10 == 0:
            best = np.exp(es.result.xbest)
            print(f"  iter {iter_i:4d}  loss={es.result.fbest:.6f}  "
                  f"armature={best[0]:.5f}  damping={best[1]:.5f}  "
                  f"frictionloss={best[2]:.5f}")

    elapsed = time.time() - t0
    best_theta = np.exp(es.result.xbest)
    armature, damping, frictionloss = best_theta

    print(f"\nOptimisation done in {elapsed:.1f}s  ({iter_i} iters)")
    print(f"  armature    = {armature:.6f}")
    print(f"  damping     = {damping:.6f}")
    print(f"  frictionloss= {frictionloss:.6f}")
    print(f"  final loss  = {es.result.fbest:.8f}")

    # ── Replay best params ─────────────────────────────────────────────────────
    q_sim_best, dq_sim_best = replay(targets, q0, control_hz, joint_name,
                                     armature, damping, frictionloss)
    q_sim_init, dq_sim_init = replay(targets, q0, control_hz, joint_name,
                                     0.012, 0.66, 0.09)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(t, np.rad2deg(targets),    color="orange",    lw=1.2, label="target")
    ax.plot(t, np.rad2deg(q_real),     color="red",       lw=1.5, ls="--", label="real")
    ax.plot(t, np.rad2deg(q_sim_init), color="gray",      lw=1.0, ls=":",  label="sim (init)")
    ax.plot(t, np.rad2deg(q_sim_best), color="steelblue", lw=1.5, label="sim (fitted)")
    ax.set_ylabel("Position (deg)")
    ax.legend(loc="upper right")
    ax.set_title(
        f"Sysid — {joint_name}  armature={armature:.5f}  "
        f"damping={damping:.5f}  frictionloss={frictionloss:.5f}\n"
        f"loss={es.result.fbest:.2e}  N={N}  hz={control_hz}  vel_weight={vel_weight:.4f}\n"
        f"kp_sim={kp_sim}  kd_sim={kd_sim}  kp_real={kp_real}  kd_real={kd_real}"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, np.rad2deg(q_real)     - np.rad2deg(q_sim_best), color="steelblue",
            lw=1.0, label="pos error: real − sim (fitted)")
    ax.plot(t, np.rad2deg(q_real)     - np.rad2deg(q_sim_init), color="gray",
            lw=1.0, ls=":", label="pos error: real − sim (init)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Pos error (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, np.rad2deg(dq_real),     color="red",       lw=1.5, ls="--", label="real")
    ax.plot(t, np.rad2deg(dq_sim_init), color="gray",      lw=1.0, ls=":",  label="sim (init)")
    ax.plot(t, np.rad2deg(dq_sim_best), color="steelblue", lw=1.5, label="sim (fitted)")
    ax.set_ylabel("Velocity (deg/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, np.rad2deg(dq_real)     - np.rad2deg(dq_sim_best), color="steelblue",
            lw=1.0, label="vel error: real − sim (fitted)")
    ax.plot(t, np.rad2deg(dq_real)     - np.rad2deg(dq_sim_init), color="gray",
            lw=1.0, ls=":", label="vel error: real − sim (init)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Vel error (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    stamp = "_".join(args.recording.stem.split("_")[:2])
    plot_path   = args.recording.parent / f"{stamp}_result.png"
    result_path = args.recording.parent / f"{stamp}_result.npz"

    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()

    # ── Save results ───────────────────────────────────────────────────────────
    np.savez(
        str(result_path),
        armature      = np.array([armature]),
        damping       = np.array([damping]),
        frictionloss  = np.array([frictionloss]),
        loss          = np.array([es.result.fbest]),
        vel_weight    = np.array([vel_weight]),
        q_sim_fitted  = q_sim_best,
        dq_sim_fitted = dq_sim_best,
        q_sim_init    = q_sim_init,
        dq_sim_init   = dq_sim_init,
    )
    print(f"Results saved → {result_path}")


if __name__ == "__main__":
    main()
