"""Load sysid — fits armature, damping, frictionloss from load_recorder recordings.

Loads one or two recordings produced by load_recorder.py and uses CMA-ES to
minimise the position + velocity RMSE between real and simulated trajectories.

Single-file mode:
  loss = RMSE_pos + vel_weight * RMSE_vel

Joint mode (two files):
  loss = L1 + lambda * L2
  where L1 is the primary file loss, L2 is the secondary file loss.
  Each file gets its own auto vel_weight unless --vel-weight is specified.

Parameters fitted (in log-space):
  armature, damping, frictionloss  (one shared triplet)

Usage:
  # single
  python sysid_load.py data/sine.npz
  python sysid_load.py data/sine.npz --init 0.005 0.5 0.1 --sigma0 0.4

  # joint
  python sysid_load.py data/sine.npz data/triangle.npz
  python sysid_load.py data/sine.npz data/triangle.npz --lambda 0.5

  # freeze one param
  python sysid_load.py data/sine.npz --freeze damping

  # full options
  python sysid_load.py data/sine.npz data/triangle.npz \\
      --lambda 0.5 --sigma0 0.4 --popsize 16 --max-iter 400 \\
      --init 0.005 0.5 0.1 --vel-weight 0.1 --freeze armature
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
_XML       = _HERE / "motor_assembly.xml"
PHYSICS_DT = 0.001   # matches motor_assembly.xml timestep

_JOINT_NAME = "load_joint"
_PARAM_LBL  = ["armature", "damping", "frictionloss"]

# ── Parameter bounds ────────────────────────────────────────────────────────────
#                      armature   damping   frictionloss
_BOUNDS_LO = np.array([0.0005,    0.05,     0.001])
_BOUNDS_HI = np.array([0.5,       5.0,      1.0  ])


# ── MuJoCo helpers ──────────────────────────────────────────────────────────────

def _apply_params(
    m: mujoco.MjModel,
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> None:
    jid = m.joint(_JOINT_NAME).id
    dof = int(m.jnt_dofadr[jid])
    m.dof_armature[dof]     = armature
    m.dof_damping[dof]      = damping
    m.dof_frictionloss[dof] = frictionloss


def _get_ids(m: mujoco.MjModel) -> tuple[int, int, int]:
    """Return (qpos_id, qvel_id, ctrl_id) for load_joint."""
    jid     = m.joint(_JOINT_NAME).id
    qpos_id = int(m.jnt_qposadr[jid])
    qvel_id = int(m.jnt_dofadr[jid])
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == jid:
            return qpos_id, qvel_id, i
    raise ValueError(f"No actuator found for joint '{_JOINT_NAME}'")


# ── Simulation replay ────────────────────────────────────────────────────────────

def replay(
    targets:      np.ndarray,   # (N,) rad
    q0:           float,        # initial position (rad)
    control_hz:   int,
    kp_sim:       float,
    kd_sim:       float,
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay target sequence in MuJoCo. Returns q_sim (N,) and dq_sim (N,)."""
    m = mujoco.MjModel.from_xml_path(str(_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, armature, damping, frictionloss)

    # Overwrite XML kp/kv with values from the dataset (do NOT use XML defaults)
    m.actuator_gainprm[0, 0] = kp_sim    # kp  (XML kp  is ignored)
    m.actuator_biasprm[0, 1] = -kp_sim   # -kp (position feedback)
    m.actuator_biasprm[0, 2] = -kd_sim   # -kd (velocity feedback; XML kv is ignored)

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


# ── Per-recording data loader ────────────────────────────────────────────────────

def load_recording(path: Path) -> dict:
    """Load and validate a load_recorder .npz. Returns a dict of arrays."""
    data = np.load(path, allow_pickle=True)

    # kp_sim / kd_sim are per-timestep arrays; warn if they changed mid-recording
    if "kp_sim" in data:
        kp_arr = data["kp_sim"]
        if np.any(kp_arr != kp_arr[0]):
            print(f"[WARNING] {path.name}: kp_sim changed during recording "
                  f"(min={kp_arr.min():.4f} max={kp_arr.max():.4f}). "
                  f"Using first value {kp_arr[0]:.4f}.")
        kp_sim = float(kp_arr[0])
    else:
        print(f"[WARNING] {path.name}: kp_sim not found, defaulting to 6.0")
        kp_sim = 6.0

    if "kd_sim" in data:
        kd_arr = data["kd_sim"]
        if np.any(kd_arr != kd_arr[0]):
            print(f"[WARNING] {path.name}: kd_sim changed during recording "
                  f"(min={kd_arr.min():.4f} max={kd_arr.max():.4f}). "
                  f"Using first value {kd_arr[0]:.4f}.")
        kd_sim = float(kd_arr[0])
    else:
        print(f"[WARNING] {path.name}: kd_sim not found, defaulting to 0.0")
        kd_sim = 0.0

    rec = {
        "path":        path,
        "t":           data["t"],
        "target":      data["target"],
        "q_real":      data["q_real"],
        "dq_real":     data["dq_real"],
        "control_hz":  int(data["control_hz"][0]),
        "kp_sim":      kp_sim,
        "kd_sim":      kd_sim,
        "kp_real":     float(data["kp_real"][0]) if "kp_real" in data else float("nan"),
        "kd_real":     float(data["kd_real"][0]) if "kd_real" in data else float("nan"),
        "signal_mode": str(data["signal_mode"].flat[0]) if "signal_mode" in data else "?",
    }
    return rec


def compute_vel_weight(rec: dict, sel: slice = slice(None)) -> float:
    std_q  = float(np.std(rec["q_real"][sel]))
    std_dq = float(np.std(rec["dq_real"][sel]))
    return std_q / std_dq if std_dq > 1e-12 else 0.0


# ── Loss ─────────────────────────────────────────────────────────────────────────

def single_loss(
    theta_full:  np.ndarray,   # [log_arm, log_damp, log_fric] always len 3
    rec:         dict,
    vel_weight:  float,
) -> float:
    armature, damping, frictionloss = np.exp(theta_full)
    q_sim, dq_sim = replay(
        rec["target"], float(rec["q_real"][0]),
        rec["control_hz"], rec["kp_sim"], rec["kd_sim"],
        armature, damping, frictionloss,
    )
    pos_loss = float(np.sqrt(np.mean((q_sim - rec["q_real"]) ** 2)))
    vel_loss = float(np.sqrt(np.mean((dq_sim - rec["dq_real"]) ** 2)))
    return pos_loss + vel_weight * vel_loss


def joint_loss(
    theta_full:   np.ndarray,
    rec1:         dict,
    rec2:         dict,
    vw1:          float,
    vw2:          float,
    lam:          float,
) -> float:
    l1 = single_loss(theta_full, rec1, vw1)
    l2 = single_loss(theta_full, rec2, vw2)
    return l1 + lam * l2


# ── Plot ─────────────────────────────────────────────────────────────────────────

def make_plot(
    recordings:   list[dict],
    results:      list[dict],   # one per recording, keys: q_sim_best, dq_sim_best, q_sim_init, dq_sim_init
    armature:     float,
    damping:      float,
    frictionloss: float,
    final_loss:   float,
    args:         argparse.Namespace,
    out_dir:      Path,
) -> None:
    n_rec  = len(recordings)
    n_rows = 3   # position, pos-error, velocity
    fig, axes = plt.subplots(
        n_rows, n_rec,
        figsize=(max(10, 6.0 * n_rec), 3.5 * n_rows),
        sharex="col", squeeze=False,
    )

    init_str = f"arm={args.init[0]}  damp={args.init[1]}  fric={args.init[2]}"
    best_str = f"arm={armature:.5f}  damp={damping:.5f}  fric={frictionloss:.5f}"
    mode_str = "joint  λ=" + str(args.lam) if n_rec > 1 else "single"

    fig.suptitle(
        f"Load sysid  [{mode_str}]  loss={final_loss:.4e}\n"
        f"init:   {init_str}\n"
        f"fitted: {best_str}",
        fontsize=9,
    )

    C_TARGET = "#f5a623"
    C_REAL   = "#2171b5"
    C_INIT   = "#aaaaaa"
    C_BEST   = "#d62728"

    for col, (rec, res) in enumerate(zip(recordings, results)):
        t     = rec["t"]
        label = f"{rec['path'].name}\n({rec['signal_mode']}  hz={rec['control_hz']})"

        # Row 0 — position
        ax = axes[0][col]
        ax.plot(t, rec["target"],        color=C_TARGET, lw=1.2, ls="--", label="target")
        ax.plot(t, rec["q_real"],        color=C_REAL,   lw=1.5, ls="--", label="real")
        ax.plot(t, res["q_sim_init"],    color=C_INIT,   lw=1.0, ls=":",  label="sim init")
        ax.plot(t, res["q_sim_best"],    color=C_BEST,   lw=1.5,          label="sim fitted")
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("Position (rad)" if col == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        # Row 1 — position error
        ax = axes[1][col]
        ax.plot(t, rec["q_real"] - res["q_sim_best"], color=C_BEST, lw=1.0, label="fitted")
        ax.plot(t, rec["q_real"] - res["q_sim_init"], color=C_INIT, lw=1.0, ls=":", label="init")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Pos error (rad)" if col == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        # Row 2 — velocity
        ax = axes[2][col]
        ax.plot(t, rec["dq_real"],       color=C_REAL, lw=1.5, ls="--", label="real")
        ax.plot(t, res["dq_sim_init"],   color=C_INIT, lw=1.0, ls=":",  label="sim init")
        ax.plot(t, res["dq_sim_best"],   color=C_BEST, lw=1.5,          label="sim fitted")
        ax.set_ylabel("Velocity (rad/s)" if col == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    tag = "_".join(r["signal_mode"] for r in recordings)
    plot_path = out_dir / f"{tag}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved   → {plot_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load sysid — fits armature, damping, frictionloss via CMA-ES."
    )
    parser.add_argument("recordings", type=Path, nargs="+",
                        help="One or two load_recorder .npz files. "
                             "First is primary, second (if given) is secondary.")

    # ── Optimiser ────────────────────────────────────────────────────────────────
    parser.add_argument("--sigma0",   type=float, default=0.5,
                        help="CMA-ES initial step size in log-space (default 0.5).")
    parser.add_argument("--popsize",  type=int,   default=12)
    parser.add_argument("--max-iter", type=int,   default=300)

    # ── Initial guess ─────────────────────────────────────────────────────────────
    parser.add_argument("--init", type=float, nargs=3,
                        metavar=("ARMATURE", "DAMPING", "FRICTIONLOSS"),
                        default=[0.005, 0.5, 0.1],
                        help="Initial parameter guess (default: 0.005 0.5 0.1).")

    # ── Loss ─────────────────────────────────────────────────────────────────────
    parser.add_argument("--vel-weight", type=float, default=None,
                        help="Velocity loss weight (applied to all files). "
                             "Auto (std_q/std_dq) if omitted.")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="λ weight for secondary file loss in joint mode (default 1.0).")

    # ── Freeze ───────────────────────────────────────────────────────────────────
    parser.add_argument("--freeze", type=str, default=None,
                        choices=_PARAM_LBL,
                        help="Lock one parameter at its --init value.")

    args = parser.parse_args()

    if len(args.recordings) > 2:
        parser.error("At most two recording files are supported.")

    is_joint = len(args.recordings) == 2

    # ── Load recordings ──────────────────────────────────────────────────────────
    recordings = [load_recording(p) for p in args.recordings]
    for rec in recordings:
        N  = len(rec["t"])
        hz = rec["control_hz"]
        dur = rec["t"][-1] - rec["t"][0]
        print(f"Loaded  {rec['path'].name}  N={N}  hz={hz}  duration={dur:.1f}s  "
              f"mode={rec['signal_mode']}  kp_sim={rec['kp_sim']}  kd_sim={rec['kd_sim']}")

    # ── Joint mode: check kp_sim / kd_sim match ──────────────────────────────────
    if is_joint:
        r1, r2 = recordings
        if r1["kp_sim"] != r2["kp_sim"] or r1["kd_sim"] != r2["kd_sim"]:
            parser.error(
                f"Joint optimisation requires identical kp_sim and kd_sim in both files.\n"
                f"  {r1['path'].name}: kp_sim={r1['kp_sim']}  kd_sim={r1['kd_sim']}\n"
                f"  {r2['path'].name}: kp_sim={r2['kp_sim']}  kd_sim={r2['kd_sim']}"
            )

    # ── Velocity weights ─────────────────────────────────────────────────────────
    vel_weights = []
    for rec in recordings:
        if args.vel_weight is not None:
            vw = args.vel_weight
            print(f"vel_weight (manual) = {vw:.6f}  [{rec['path'].name}]")
        else:
            vw = compute_vel_weight(rec)
            print(f"vel_weight (auto)   = {vw:.6f}  "
                  f"[std_q={np.std(rec['q_real']):.4f}  "
                  f"std_dq={np.std(rec['dq_real']):.4f}]  [{rec['path'].name}]")
        vel_weights.append(vw)

    # ── Frozen / free params ─────────────────────────────────────────────────────
    _INIT    = np.array(args.init, dtype=float)
    frozen   = {_PARAM_LBL.index(args.freeze)} if args.freeze else set()
    free_idx = [i for i in range(3) if i not in frozen]

    theta0_full    = np.log(_INIT)
    theta0_reduced = theta0_full[free_idx]
    lo_reduced     = np.log(_BOUNDS_LO)[free_idx].tolist()
    hi_reduced     = np.log(_BOUNDS_HI)[free_idx].tolist()

    def _expand(theta_reduced: np.ndarray) -> np.ndarray:
        full = theta0_full.copy()
        for k, fi in enumerate(free_idx):
            full[fi] = theta_reduced[k]
        return full

    print(f"\nBounds lo: armature={_BOUNDS_LO[0]}  damping={_BOUNDS_LO[1]}  frictionloss={_BOUNDS_LO[2]}")
    print(f"Bounds hi: armature={_BOUNDS_HI[0]}  damping={_BOUNDS_HI[1]}  frictionloss={_BOUNDS_HI[2]}")
    if frozen:
        print(f"Frozen:    {[_PARAM_LBL[i] for i in sorted(frozen)]}  (held at init values)")
        print(f"Free:      {[_PARAM_LBL[i] for i in free_idx]}")
    else:
        print("Frozen:    none  (optimising all three)")

    # ── Initial loss ─────────────────────────────────────────────────────────────
    if is_joint:
        loss0 = joint_loss(theta0_full, recordings[0], recordings[1],
                           vel_weights[0], vel_weights[1], args.lam)
        print(f"\nInitial guess: arm={_INIT[0]}  damp={_INIT[1]}  fric={_INIT[2]}  "
              f"joint_loss={loss0:.6f}  (L1 + {args.lam}·L2)")
    else:
        loss0 = single_loss(theta0_full, recordings[0], vel_weights[0])
        print(f"\nInitial guess: arm={_INIT[0]}  damp={_INIT[1]}  fric={_INIT[2]}  "
              f"loss={loss0:.6f}")

    # ── CMA-ES ───────────────────────────────────────────────────────────────────
    es = cma.CMAEvolutionStrategy(
        theta0_reduced,
        args.sigma0,
        {
            "popsize": args.popsize,
            "maxiter": args.max_iter,
            "tolx":    1e-6,
            "tolfun":  1e-8,
            "verbose": 1,
            "bounds":  [lo_reduced, hi_reduced],
        },
    )

    print("\nRunning CMA-ES…")
    t0     = time.time()
    iter_i = 0

    while not es.stop():
        solutions = es.ask()
        if is_joint:
            fitnesses = [
                joint_loss(_expand(th), recordings[0], recordings[1],
                           vel_weights[0], vel_weights[1], args.lam)
                for th in solutions
            ]
        else:
            fitnesses = [
                single_loss(_expand(th), recordings[0], vel_weights[0])
                for th in solutions
            ]
        es.tell(solutions, fitnesses)
        iter_i += 1
        if iter_i % 10 == 0:
            best = np.exp(_expand(es.result.xbest))
            print(f"  iter {iter_i:4d}  loss={es.result.fbest:.6f}  "
                  f"armature={best[0]:.5f}  damping={best[1]:.5f}  "
                  f"frictionloss={best[2]:.5f}")

    elapsed = time.time() - t0
    armature, damping, frictionloss = np.exp(_expand(es.result.xbest))
    final_loss = es.result.fbest

    print(f"\nDone in {elapsed:.1f}s  ({iter_i} iters)")
    print(f"  armature     = {armature:.6f}")
    print(f"  damping      = {damping:.6f}")
    print(f"  frictionloss = {frictionloss:.6f}")
    print(f"  final loss   = {final_loss:.8f}")

    # ── Replay best and init for all recordings ───────────────────────────────────
    best_theta = _expand(es.result.xbest)
    results = []
    for rec in recordings:
        q_best, dq_best = replay(
            rec["target"], float(rec["q_real"][0]),
            rec["control_hz"], rec["kp_sim"], rec["kd_sim"],
            *np.exp(best_theta),
        )
        q_init, dq_init = replay(
            rec["target"], float(rec["q_real"][0]),
            rec["control_hz"], rec["kp_sim"], rec["kd_sim"],
            *_INIT,
        )
        results.append({
            "q_sim_best":  q_best,
            "dq_sim_best": dq_best,
            "q_sim_init":  q_init,
            "dq_sim_init": dq_init,
        })

    # ── Output directory ─────────────────────────────────────────────────────────
    # Place next to first recording's parent / <stem>_sysid/
    out_dir = recordings[0]["path"].parent / f"{recordings[0]['path'].stem}_sysid"
    out_dir.mkdir(exist_ok=True)

    # ── Plot ─────────────────────────────────────────────────────────────────────
    make_plot(recordings, results, armature, damping, frictionloss,
              final_loss, args, out_dir)

    # ── Save results ─────────────────────────────────────────────────────────────
    tag         = "_".join(r["signal_mode"] for r in recordings)
    result_path = out_dir / f"{tag}.npz"

    save_dict = dict(
        armature      = np.array([armature]),
        damping       = np.array([damping]),
        frictionloss  = np.array([frictionloss]),
        loss          = np.array([final_loss]),
        is_joint      = np.array([is_joint]),
        lam           = np.array([args.lam]),
        init          = np.array(args.init),
        sigma0        = np.array([args.sigma0]),
        freeze        = np.array([args.freeze or "none"]),
    )
    for i, (rec, res) in enumerate(zip(recordings, results)):
        suffix = f"_{i+1}"
        save_dict[f"recording{suffix}"]   = np.array([str(rec["path"])])
        save_dict[f"q_sim_fitted{suffix}"] = res["q_sim_best"]
        save_dict[f"dq_sim_fitted{suffix}"] = res["dq_sim_best"]
        save_dict[f"q_sim_init{suffix}"]   = res["q_sim_init"]
        save_dict[f"dq_sim_init{suffix}"]  = res["dq_sim_init"]
        save_dict[f"vel_weight{suffix}"]   = np.array([vel_weights[i]])

    np.savez(str(result_path), **save_dict)
    print(f"Results saved → {result_path}")


if __name__ == "__main__":
    main()
