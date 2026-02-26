"""Plot a raw recording produced by mx64_sync.py.

Generates a 4-panel figure:
  1. Position — target, sim, real
  2. Position error — real − sim
  3. Velocity — sim, real
  4. Velocity error — real − sim

Saves a .png alongside the .npz (same name, different extension).

Usage:
  python plot_recording.py data/recording_<timestamp>.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(npz_path: Path) -> None:
    data = np.load(npz_path)

    t        = data["t"]
    targets  = data["target"]    # rad
    q_real   = data["q_real"]    # rad
    dq_real  = data["dq_real"]   # rad/s
    q_sim    = data["q_sim"]     # rad
    dq_sim   = data["dq_sim"]    # rad/s
    control_hz = int(data["control_hz"][0])
    N = len(t)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"{npz_path.name}  —  N={N}  hz={control_hz}  "
        f"duration={t[-1] - t[0]:.1f}s",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(t, np.rad2deg(targets), color="orange",    lw=1.2, label="target")
    ax.plot(t, np.rad2deg(q_sim),   color="steelblue", lw=1.5, label="sim")
    ax.plot(t, np.rad2deg(q_real),  color="red",       lw=1.5, ls="--", label="real")
    ax.set_ylabel("Position (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, np.rad2deg(q_real) - np.rad2deg(q_sim),
            color="steelblue", lw=1.0, label="real − sim")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Pos error (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, np.rad2deg(dq_sim),  color="steelblue", lw=1.5, label="sim")
    ax.plot(t, np.rad2deg(dq_real), color="red",       lw=1.5, ls="--", label="real")
    ax.set_ylabel("Velocity (deg/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, np.rad2deg(dq_real) - np.rad2deg(dq_sim),
            color="steelblue", lw=1.0, label="real − sim")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Vel error (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = npz_path.parent / f"{npz_path.stem}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a raw mx64_sync recording.")
    parser.add_argument("recording", type=Path, help="Path to .npz recording file.")
    args = parser.parse_args()

    path = args.recording
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".npz" or path.stem.endswith("_result"):
        raise ValueError(f"Expected a raw recording .npz, got: {path.name}")

    plot(path)


if __name__ == "__main__":
    main()
