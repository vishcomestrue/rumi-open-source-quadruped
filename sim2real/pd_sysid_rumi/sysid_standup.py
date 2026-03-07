"""Rumi sysid — fits armature, damping, frictionloss from a standup recording.

Loads a recording produced by rumi_standup.py (q_real[N×12], target[N×12]),
replays the target sequence through MuJoCo for all selected joints in one sim,
and uses CMA-ES to minimise the sum of MSE across those joints.

One shared (armature, damping, frictionloss) triplet is fitted — all selected
joints are assumed mechanically identical (same motor model, MX-64).

Joint selection (default: all 12):
  --calf    fit only the 4 calf  joints (FL/BL/BR/FR_calf_joint)
  --thigh   fit only the 4 thigh joints
  --hip     fit only the 4 hip   joints
  --joint <name>  fit a specific joint by full name (repeatable)
  Flags can be combined: --calf --joint BL_hip_joint fits 5 joints.

Usage:
  python sysid_standup.py data/<timestamp>_standup_standup.npz --calf
  python sysid_standup.py data/<timestamp>_standup_sine.npz --calf --thigh
  python sysid_standup.py data/<timestamp>_standup_sine.npz   # all 12
  python sysid_standup.py data/<timestamp>_standup_sine.npz --joint BL_calf_joint
  python sysid_standup.py data/<timestamp>_standup_sine.npz --joint BL_calf_joint --joint FR_calf_joint
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_HERE      = Path(__file__).parent
_SCENE_XML = _HERE / "scene.xml"
PHYSICS_DT = 0.004

_PARTS = ["hip", "thigh", "calf"]


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────

def _joint_part(name: str) -> str:
    return name.split("_")[1]


def _get_joint_ids(m: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    """Return (qpos_id, qvel_id, ctrl_id) for a joint."""
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
    """Apply one shared (armature, damping, frictionloss) to all selected joints."""
    for jname in joint_names:
        dof_id = m.jnt_dofadr[m.joint(jname).id]
        m.dof_armature[dof_id]     = armature
        m.dof_damping[dof_id]      = damping
        m.dof_frictionloss[dof_id] = frictionloss


# ── Simulation replay ───────────────────────────────────────────────────────────

def replay_all(
    targets:      np.ndarray,    # (N, 12) rad — full robot targets
    q0:           np.ndarray,    # (12,)   rad — initial positions (offset-space, ~0)
    control_hz:   int,
    joint_names:  list[str],     # all 12 joint names in data column order
    sel_names:    list[str],     # subset to apply params to
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay all-joint target sequence in MuJoCo.

    Returns q_sim (N×12) and dq_sim (N×12).
    Only sel_names get the candidate params; others keep XML defaults.
    """
    m = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, sel_names, armature, damping, frictionloss)

    d = mujoco.MjData(m)

    # Build index arrays for all 12 joints
    ids = [_get_joint_ids(m, jn) for jn in joint_names]   # list of (qpos_id, qvel_id, ctrl_id)

    # Set initial positions
    for col, (qpos_id, qvel_id, ctrl_id) in enumerate(ids):
        d.qpos[qpos_id] = float(q0[col])
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    N         = len(targets)
    substeps  = max(1, round(1.0 / (control_hz * PHYSICS_DT)))
    q_sim     = np.empty((N, 12))
    dq_sim    = np.empty((N, 12))

    for i in range(N):
        for col, (qpos_id, qvel_id, ctrl_id) in enumerate(ids):
            q_sim[i, col]  = d.qpos[qpos_id]
            dq_sim[i, col] = d.qvel[qvel_id]
            d.ctrl[ctrl_id] = targets[i, col]
        for _ in range(substeps):
            mujoco.mj_step(m, d)

    return q_sim, dq_sim


# ── Loss ────────────────────────────────────────────────────────────────────────

def loss(
    theta:       np.ndarray,   # [log_armature, log_damping, log_frictionloss]
    targets:     np.ndarray,   # (N, 12)
    q_real:      np.ndarray,   # (N, 12)
    dq_real:     np.ndarray,   # (N, 12)
    q0:          np.ndarray,   # (12,)
    control_hz:  int,
    joint_names: list[str],    # all 12
    sel_names:   list[str],    # subset to fit
    sel_cols:    list[int],    # column indices of sel_names in data
    vel_weight:  float,
) -> float:
    armature, damping, frictionloss = np.exp(theta)
    q_sim, dq_sim = replay_all(
        targets, q0, control_hz, joint_names, sel_names,
        armature, damping, frictionloss,
    )
    # Loss only over selected joints
    pos_loss = float(np.sqrt(np.mean((q_sim[:, sel_cols] - q_real[:, sel_cols]) ** 2)))
    vel_loss = float(np.sqrt(np.mean((dq_sim[:, sel_cols] - dq_real[:, sel_cols]) ** 2)))
    return pos_loss + vel_weight * vel_loss


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rumi standup sysid — fits shared MX-64 params via CMA-ES."
    )
    parser.add_argument("recording", type=Path, help="Path to standup .npz recording.")
    # ── Joint selection flags ──────────────────────────────────────────────────
    parser.add_argument("--calf",        action="store_true", help="Fit all 4 calf joints.")
    parser.add_argument("--calf-front",  action="store_true", help="Fit FL+FR calf joints.")
    parser.add_argument("--calf-back",   action="store_true", help="Fit BL+BR calf joints.")
    parser.add_argument("--thigh",       action="store_true", help="Fit all 4 thigh joints.")
    parser.add_argument("--thigh-front", action="store_true", help="Fit FL+FR thigh joints.")
    parser.add_argument("--thigh-back",  action="store_true", help="Fit BL+BR thigh joints.")
    parser.add_argument("--hip",         action="store_true", help="Fit all 4 hip joints.")
    parser.add_argument("--hip-front",   action="store_true", help="Fit FL+FR hip joints.")
    parser.add_argument("--hip-back",    action="store_true", help="Fit BL+BR hip joints.")
    parser.add_argument("--popsize",    type=int,   default=12)
    parser.add_argument("--max-iter",   type=int,   default=300)
    parser.add_argument("--sigma0",     type=float, default=0.5,
                        help="CMA-ES initial step size in log-space.")
    parser.add_argument("--vel-weight", type=float, default=None,
                        help="Velocity loss weight (auto if omitted).")
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────────────────────
    data        = np.load(args.recording, allow_pickle=True)
    t           = data["t"]                      # (N,)
    targets     = data["target"]                 # (N, 12) rad
    q_real      = data["q_real"]                 # (N, 12) rad offset-space
    dq_real     = data["dq_real"]                # (N, 12) rad/s
    joint_names = data["joint_names"].tolist()   # len 12
    control_hz  = int(data["control_hz"][0])
    N           = len(t)
    kp_sim  = float(data["kp_sim"][0])  if "kp_sim"  in data else float("nan")
    kd_sim  = float(data["kd_sim"][0])  if "kd_sim"  in data else float("nan")
    kp_real = float(data["kp_real"][0]) if "kp_real" in data else float("nan")
    kd_real = float(data["kd_real"][0]) if "kd_real" in data else float("nan")

    print(f"Loaded  N={N}  hz={control_hz}  duration={t[-1]-t[0]:.1f}s")
    print(f"Joints: {joint_names}")

    # ── Select joints to fit ────────────────────────────────────────────────────
    _FRONT = {"FL", "FR"}
    _BACK  = {"BL", "BR"}

    # Validate: --calf-front + --calf-back must use --calf instead
    for part in ("calf", "thigh", "hip"):
        if getattr(args, f"{part}_front") and getattr(args, f"{part}_back"):
            parser.error(f"--{part}-front + --{part}-back is not allowed; use --{part} instead.")

    # Build selected joint set and tag
    sel_set:  set[int] = set()
    tag_parts: list[str] = []

    def _add(part: str, locs: set[str] | None, tag: str) -> None:
        for i, n in enumerate(joint_names):
            if _joint_part(n) == part and (locs is None or n.split("_")[0] in locs):
                sel_set.add(i)
        tag_parts.append(tag)

    if args.calf:        _add("calf",  None,    "calf")
    if args.calf_front:  _add("calf",  _FRONT,  "calf_front")
    if args.calf_back:   _add("calf",  _BACK,   "calf_back")
    if args.thigh:       _add("thigh", None,    "thigh")
    if args.thigh_front: _add("thigh", _FRONT,  "thigh_front")
    if args.thigh_back:  _add("thigh", _BACK,   "thigh_back")
    if args.hip:         _add("hip",   None,    "hip")
    if args.hip_front:   _add("hip",   _FRONT,  "hip_front")
    if args.hip_back:    _add("hip",   _BACK,   "hip_back")

    if not sel_set:
        # default: all 12
        sel_set   = set(range(len(joint_names)))
        tag_parts = ["all"]

    sel_cols  = sorted(sel_set)
    sel_names = [joint_names[i] for i in sel_cols]
    parts_tag = "_".join(tag_parts)

    print(f"Selected joints ({len(sel_names)}): {sel_names}")

    # Initial positions — recording is in offset-space so q0 ≈ 0
    q0 = q_real[0].copy()   # (12,) — close to zero

    # ── Velocity weight ─────────────────────────────────────────────────────────
    std_q  = float(np.std(q_real[:, sel_cols]))
    std_dq = float(np.std(dq_real[:, sel_cols]))
    if args.vel_weight is not None:
        vel_weight = args.vel_weight
        print(f"vel_weight (manual) = {vel_weight:.6f}")
    else:
        vel_weight = std_q / std_dq if std_dq > 1e-12 else 0.0
        print(f"vel_weight (auto)   = {vel_weight:.6f}  "
              f"[std_q={std_q:.4f}  std_dq={std_dq:.4f}]")

    # ── Initial guess — taken directly from rumi.xml defaults ──────────────────
    _INIT = [0.012, 0.66, 0.09]
    theta0 = np.log(_INIT)
    loss0  = loss(theta0, targets, q_real, dq_real, q0,
                  control_hz, joint_names, sel_names, sel_cols, vel_weight)
    print(f"\nInitial guess:  armature={_INIT[0]}  damping={_INIT[1]}  "
          f"frictionloss={_INIT[2]}  loss={loss0:.6f}")

    # ── CMA-ES ──────────────────────────────────────────────────────────────────
    es = cma.CMAEvolutionStrategy(
        theta0,
        args.sigma0,
        {
            "popsize": args.popsize,
            "maxiter": args.max_iter,
            "tolx":    1e-6,
            "tolfun":  1e-8,
            "verbose": 1,
        },
    )

    print("\nRunning CMA-ES…")
    t0     = time.time()
    iter_i = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            loss(theta, targets, q_real, dq_real, q0,
                 control_hz, joint_names, sel_names, sel_cols, vel_weight)
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
    armature, damping, frictionloss = np.exp(es.result.xbest)

    print(f"\nDone in {elapsed:.1f}s  ({iter_i} iters)")
    print(f"  armature     = {armature:.6f}")
    print(f"  damping      = {damping:.6f}")
    print(f"  frictionloss = {frictionloss:.6f}")
    print(f"  final loss   = {es.result.fbest:.8f}")

    # ── Replay best and initial params ──────────────────────────────────────────
    q_sim_best,  dq_sim_best  = replay_all(
        targets, q0, control_hz, joint_names, sel_names,
        armature, damping, frictionloss,
    )
    q_sim_init, dq_sim_init = replay_all(
        targets, q0, control_hz, joint_names, sel_names,
        0.012, 0.66, 0.09,
    )

    # ── Plot — one column per selected joint ────────────────────────────────────
    n_sel  = len(sel_names)
    n_rows = 3   # position, pos-error, velocity
    fig, axes = plt.subplots(
        n_rows, n_sel,
        figsize=(max(13, 4.0 * n_sel), 3.5 * n_rows),
        sharex=True, squeeze=False,
    )
    fig.suptitle(
        f"Sysid  joints={sel_names}  "
        f"armature={armature:.5f}  damping={damping:.5f}  frictionloss={frictionloss:.5f}\n"
        f"loss={es.result.fbest:.2e}  N={N}  hz={control_hz}  vel_weight={vel_weight:.4f}\n"
        f"kp_sim={kp_sim}  kd_sim={kd_sim}  kp_real={kp_real}  kd_real={kd_real}",
        fontsize=10,
    )

    for j, (col, jname) in enumerate(zip(sel_cols, sel_names)):
        short = jname.replace("_joint", "")

        # Position
        ax = axes[0][j]
        ax.plot(t, np.rad2deg(targets[:, col]),       color="orange",    lw=1.2, label="target")
        ax.plot(t, np.rad2deg(q_real[:, col]),         color="red",       lw=1.5, ls="--", label="real")
        ax.plot(t, np.rad2deg(q_sim_init[:, col]),     color="gray",      lw=1.0, ls=":",  label="sim init")
        ax.plot(t, np.rad2deg(q_sim_best[:, col]),     color="steelblue", lw=1.5, label="sim fitted")
        ax.set_title(short, fontsize=9)
        ax.set_ylabel("deg" if j == 0 else "")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Position error
        ax = axes[1][j]
        ax.plot(t, np.rad2deg(q_real[:, col] - q_sim_best[:, col]),
                color="steelblue", lw=1.0, label="fitted")
        ax.plot(t, np.rad2deg(q_real[:, col] - q_sim_init[:, col]),
                color="gray",      lw=1.0, ls=":", label="init")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("pos err (deg)" if j == 0 else "")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Velocity
        ax = axes[2][j]
        ax.plot(t, np.rad2deg(dq_real[:, col]),       color="red",       lw=1.5, ls="--", label="real")
        ax.plot(t, np.rad2deg(dq_sim_init[:, col]),   color="gray",      lw=1.0, ls=":",  label="sim init")
        ax.plot(t, np.rad2deg(dq_sim_best[:, col]),   color="steelblue", lw=1.5, label="sim fitted")
        ax.set_ylabel("deg/s" if j == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    stamp       = "_".join(args.recording.stem.split("_")[:2])
    plot_path   = args.recording.parent / f"{stamp}_sysid_{parts_tag}.png"
    result_path = args.recording.parent / f"{stamp}_sysid_{parts_tag}.npz"

    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved   → {plot_path}")
    plt.show()

    # ── Save results ─────────────────────────────────────────────────────────────
    np.savez(
        str(result_path),
        armature      = np.array([armature]),
        damping       = np.array([damping]),
        frictionloss  = np.array([frictionloss]),
        loss          = np.array([es.result.fbest]),
        vel_weight    = np.array([vel_weight]),
        fit_tag       = np.array([parts_tag]),
        sel_joints    = np.array(sel_names),
        q_sim_fitted  = q_sim_best[:, sel_cols],
        dq_sim_fitted = dq_sim_best[:, sel_cols],
        q_sim_init    = q_sim_init[:, sel_cols],
        dq_sim_init   = dq_sim_init[:, sel_cols],
    )
    print(f"Results saved → {result_path}")


if __name__ == "__main__":
    main()
