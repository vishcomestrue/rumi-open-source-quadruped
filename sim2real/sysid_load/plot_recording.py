"""Plot a load_recorder .npz file.

Two rows: joint position (rad) and joint velocity (rad/s).
Sim and real are overlaid in different colours.
The figure title contains all slider settings and physics params.

Usage:
  python plot_recording.py data/<timestamp>_load_<mode>.npz
  python plot_recording.py data/<timestamp>_load_<mode>.npz --save   # save PNG next to npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot load_recorder recording.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--save", action="store_true",
                        help="Save PNG next to the .npz file instead of showing.")
    args = parser.parse_args()

    d = np.load(args.recording, allow_pickle=True)

    # ── Time ──────────────────────────────────────────────────────────────────────
    t = d["t"]

    # ── Signals ───────────────────────────────────────────────────────────────────
    target   = d["target"]
    q_sim    = d["q_sim"]
    dq_sim   = d["dq_sim"]
    q_real   = d["q_real"]
    dq_real  = d["dq_real"]

    # ── Scalar metadata ───────────────────────────────────────────────────────────
    def _s(key, fmt=".4f"):
        if key not in d:
            return "—"
        v = d[key]
        val = v.flat[0]
        return f"{val:{fmt}}"

    def _sarr(key, fmt=".4f"):
        """For per-timestep arrays just show the first unique value."""
        if key not in d:
            return "—"
        v = d[key]
        vals = np.unique(np.round(v, 6))
        if len(vals) == 1:
            return f"{vals[0]:{fmt}}"
        return f"{v[0]:{fmt}}…{v[-1]:{fmt}}"

    mode        = str(d["signal_mode"].flat[0])
    control_hz  = int(d["control_hz"].flat[0])
    N           = len(t)
    duration    = t[-1] - t[0]

    # Build title string
    title_lines = [
        f"file: {args.recording.name}   mode: {mode}   N={N}   hz={control_hz}   duration={duration:.1f}s",
        (
            f"signal — offset={_s('sig_offset_rad')} rad   "
            f"amp={_s('sig_amplitude_rad')} rad   "
            f"freq={_s('sig_frequency_hz')} Hz   "
            f"chirp_f_end={_s('sig_chirp_f_end')} Hz   "
            f"chirp_sweep={_s('sig_chirp_sweep_s')} s"
        ),
        (
            f"physics — kp_sim={_sarr('kp_sim')}   kd_sim={_sarr('kd_sim')}   "
            f"kp_real={_s('kp_real')}   kd_real={_s('kd_real')}   "
            f"damping={_sarr('damping')}   armature={_sarr('armature')}   "
            f"frictionloss={_sarr('frictionloss')}"
        ),
    ]
    title = "\n".join(title_lines)

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=8.5, family="monospace",
                 verticalalignment="top", y=1.01)

    C_TARGET = "#f5a623"   # orange
    C_SIM    = "#444444"   # dark grey
    C_REAL   = "#2171b5"   # blue

    # Row 0 — position
    ax = axes[0]
    ax.plot(t, target,  color=C_TARGET, lw=1.2, ls="--", label="target",   zorder=2)
    ax.plot(t, q_sim,   color=C_SIM,   lw=1.5,           label="sim",      zorder=3)
    ax.plot(t, q_real,  color=C_REAL,  lw=1.5, alpha=0.85, label="real",   zorder=4)
    ax.set_ylabel("Position (rad)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 — velocity
    ax = axes[1]
    ax.plot(t, dq_sim,  color=C_SIM,  lw=1.5,           label="sim",  zorder=3)
    ax.plot(t, dq_real, color=C_REAL, lw=1.5, alpha=0.85, label="real", zorder=4)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Velocity (rad/s)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        out = args.recording.with_suffix(".png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
