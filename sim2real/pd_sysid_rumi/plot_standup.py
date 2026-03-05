"""Plot a standup recording produced by rumi_standup.py.

Layout: 4 columns (FL | BL | BR | FR) × 6 rows (hip-pos, hip-tau,
thigh-pos, thigh-tau, calf-pos, calf-tau).
Each position cell shows: target (orange), sim (blue), real (red dashed).
Each torque cell shows:   measured torque (green).

Usage:
  python plot_standup.py data/<timestamp>_standup_<mode>.npz
  python plot_standup.py data/<timestamp>_standup_<mode>.npz --group calf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_GROUPS   = ["hip", "thigh", "calf"]
_LOCATIONS = ["FL", "BL", "BR", "FR"]   # column order


def _joint_part(name: str) -> str:
    return name.split("_")[1]


def _joint_loc(name: str) -> str:
    return name.split("_")[0]


def _joint_sign(name: str) -> int:
    return -1 if _joint_loc(name) in ("FL", "BL") else +1


def plot(npz_path: Path, group_filter: str | None = None) -> None:
    data = np.load(npz_path, allow_pickle=True)

    t           = data["t"]                     # (N,)
    targets     = data["target"]                # (N, 12) rad offset-space
    q_real      = data["q_real"]                # (N, 12) rad offset-space
    q_sim       = data["q_sim"]                 # (N, 12) rad
    tau_meas    = data["tau_meas"]              # (N, 12) N·m
    joint_names = data["joint_names"].tolist()  # len 12
    control_hz  = int(data["control_hz"][0])
    N = len(t)
    kp_sim  = float(data["kp_sim"][0])  if "kp_sim"  in data else float("nan")
    kd_sim  = float(data["kd_sim"][0])  if "kd_sim"  in data else float("nan")
    kp_real = float(data["kp_real"][0]) if "kp_real" in data else float("nan")
    kd_real = float(data["kd_real"][0]) if "kd_real" in data else float("nan")

    groups = [group_filter] if group_filter else _GROUPS
    n_groups = len(groups)
    n_cols   = len(_LOCATIONS)          # 4
    n_rows   = n_groups * 2             # 2 rows per group (pos + tau)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"{npz_path.name}   N={N}   hz={control_hz}   duration={t[-1]-t[0]:.1f}s\n"
        f"kp_sim={kp_sim}  kd_sim={kd_sim}  kp_real={kp_real}  kd_real={kd_real}",
        fontsize=11,
    )

    # Build lookup: (part, loc) → column index in joint_names / data arrays
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    for row_group, g in enumerate(groups):
        row_pos = row_group * 2       # position row
        row_tau = row_group * 2 + 1  # torque row

        for col, loc in enumerate(_LOCATIONS):
            jname = f"{loc}_{g}_joint"
            if jname not in name_to_idx:
                # joint not present in this recording — blank axes
                axes[row_pos][col].set_visible(False)
                axes[row_tau][col].set_visible(False)
                continue

            mi = name_to_idx[jname]

            # ── Position ──────────────────────────────────────────────────────
            ax = axes[row_pos][col]
            ax.plot(t, np.rad2deg(targets[:, mi]),
                    color="orange", lw=1.8, label="target", zorder=4)
            ax.plot(t, np.rad2deg(q_sim[:,  mi]),
                    color="steelblue", lw=1.4, label="sim")
            ax.plot(t, np.rad2deg(q_real[:, mi]),
                    color="red", lw=1.4, ls="--", label="real")
            ax.set_title(jname.replace("_joint", ""), fontsize=9)
            ax.set_ylabel("deg" if col == 0 else "")
            ax.grid(True, alpha=0.3)
            if row_group == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)

            # ── Torque ────────────────────────────────────────────────────────
            ax = axes[row_tau][col]
            ax.plot(t, tau_meas[:, mi], color="seagreen", lw=1.3)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("N·m" if col == 0 else "")
            ax.grid(True, alpha=0.3)

        # Row labels on the left
        axes[row_pos][0].set_ylabel(f"{g}\npos (deg)", fontsize=8)
        axes[row_tau][0].set_ylabel(f"{g}\ntau (N·m)", fontsize=8)

    # x-axis label on bottom row only
    for col in range(n_cols):
        axes[-1][col].set_xlabel("Time (s)", fontsize=8)

    # Column headers (location names) on top row
    for col, loc in enumerate(_LOCATIONS):
        axes[0][col].set_title(
            f"{loc}  —  {groups[0]}_{loc}_joint".replace(f"_{groups[0]}_", "_"),
            fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = npz_path.parent / f"{npz_path.stem}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rumi_standup recording.")
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--group", choices=_GROUPS, default=None,
        help="Plot only one group row (default: all three).",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        raise FileNotFoundError(args.recording)

    plot(args.recording, group_filter=args.group)


if __name__ == "__main__":
    main()
