"""Plot all recordings in a dataset folder — one figure per joint group per file.

For each .npz in the dataset folder, generates three plots:
  <stem>_hip.png   — FL/BL/BR/FR hip   position + torque
  <stem>_thigh.png — FL/BL/BR/FR thigh position + torque
  <stem>_calf.png  — FL/BL/BR/FR calf  position + torque

All plots are saved into data/<dataset_name>_plots/ created automatically.

Each figure title includes all scalar metadata:
  kp_sim, kd_sim, kp_real, kd_real, signal_mode,
  hip/thigh/calf centre & swing, frequency, step period,
  chirp f1 & sweep, standup duration, control_hz, N, duration.

Layout per figure: 2 rows (position, torque) × 4 columns (FL, BL, BR, FR).

Usage:
  python plot_dataset.py data/<dataset_folder>
  python plot_dataset.py data/my_dataset/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_GROUPS    = ["hip", "thigh", "calf"]
_LOCATIONS = ["FL", "BL", "BR", "FR"]


def _joint_part(name: str) -> str:
    return name.split("_")[1]


def _load_scalar(data: np.lib.npyio.NpzFile, key: str, fmt: str = "{}") -> str:
    """Load a scalar key from npz, return formatted string or 'n/a'."""
    if key not in data:
        return "n/a"
    val = data[key]
    # string arrays
    if val.dtype.kind in ("U", "S", "O"):
        return str(val.flat[0])
    return fmt.format(float(val.flat[0]))


def plot_group(
    npz_path: Path,
    group: str,
    out_path: Path,
) -> None:
    data        = np.load(npz_path, allow_pickle=True)
    t           = data["t"]
    targets     = data["target"]
    q_real      = data["q_real"]
    q_sim       = data["q_sim"]
    tau_meas    = data["tau_meas"]
    joint_names = data["joint_names"].tolist()
    N           = len(t)

    # ── Scalar metadata for title ────────────────────────────────────────────
    control_hz  = _load_scalar(data, "control_hz", "{:.0f}")
    kp_sim      = _load_scalar(data, "kp_sim",      "{:.3g}")
    kd_sim      = _load_scalar(data, "kd_sim",      "{:.3g}")
    kp_real     = _load_scalar(data, "kp_real",     "{:.3g}")
    kd_real     = _load_scalar(data, "kd_real",     "{:.3g}")
    mode        = _load_scalar(data, "signal_mode")
    hip_c       = _load_scalar(data, "hip_centre_deg",   "{:.1f}")
    thigh_c     = _load_scalar(data, "thigh_centre_deg", "{:.1f}")
    calf_c      = _load_scalar(data, "calf_centre_deg",  "{:.1f}")
    hip_s       = _load_scalar(data, "hip_swing_deg",    "{:.1f}")
    thigh_s     = _load_scalar(data, "thigh_swing_deg",  "{:.1f}")
    calf_s      = _load_scalar(data, "calf_swing_deg",   "{:.1f}")
    freq        = _load_scalar(data, "frequency_hz",     "{:.3g}")
    step_T      = _load_scalar(data, "step_period_s",    "{:.3g}")
    chirp_f1    = _load_scalar(data, "chirp_f1_hz",      "{:.3g}")
    chirp_sw    = _load_scalar(data, "chirp_sweep_s",    "{:.3g}")
    standup_dur = _load_scalar(data, "standup_duration_s", "{:.3g}")
    duration    = f"{t[-1] - t[0]:.1f}"

    title = (
        f"{npz_path.name}  —  {group.upper()}  —  mode={mode}  "
        f"N={N}  hz={control_hz}  duration={duration}s\n"
        f"kp_sim={kp_sim}  kd_sim={kd_sim}  kp_real={kp_real}  kd_real={kd_real}\n"
        f"hip  centre={hip_c}°  swing={hip_s}°  |  "
        f"thigh  centre={thigh_c}°  swing={thigh_s}°  |  "
        f"calf  centre={calf_c}°  swing={calf_s}°\n"
        f"freq={freq}Hz  step_T={step_T}s  "
        f"chirp_f1={chirp_f1}Hz  chirp_sweep={chirp_sw}s  "
        f"standup_dur={standup_dur}s"
    )

    # ── Figure: 2 rows (pos, tau) × 4 cols (FL BL BR FR) ────────────────────
    fig, axes = plt.subplots(
        2, 4,
        figsize=(18, 7),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=8.5)

    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    for col, loc in enumerate(_LOCATIONS):
        jname = f"{loc}_{group}_joint"
        ax_pos = axes[0][col]
        ax_tau = axes[1][col]

        if jname not in name_to_idx:
            ax_pos.set_visible(False)
            ax_tau.set_visible(False)
            continue

        mi = name_to_idx[jname]

        # Position
        ax_pos.plot(t, np.rad2deg(targets[:, mi]),
                    color="orange",    lw=1.8, label="target", zorder=4)
        ax_pos.plot(t, np.rad2deg(q_sim[:,   mi]),
                    color="steelblue", lw=1.4, label="sim")
        ax_pos.plot(t, np.rad2deg(q_real[:,  mi]),
                    color="red",       lw=1.4, ls="--", label="real")
        ax_pos.set_title(jname.replace("_joint", ""), fontsize=9)
        ax_pos.set_ylabel("deg" if col == 0 else "")
        ax_pos.grid(True, alpha=0.3)
        if col == 0:
            ax_pos.legend(loc="upper right", fontsize=7)

        # Torque
        ax_tau.plot(t, tau_meas[:, mi], color="seagreen", lw=1.3)
        ax_tau.axhline(0, color="black", lw=0.5)
        ax_tau.set_ylabel("N·m" if col == 0 else "")
        ax_tau.set_xlabel("Time (s)", fontsize=8)
        ax_tau.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("pos (deg)", fontsize=8)
    axes[1][0].set_ylabel("tau (N·m)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def process_dataset(dataset_dir: Path) -> None:
    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {dataset_dir}")
        return

    for npz_path in npz_files:
        print(f"\n{npz_path.name}")
        out_dir = dataset_dir / npz_path.stem
        out_dir.mkdir(exist_ok=True)
        for group in _GROUPS:
            out_path = out_dir / f"{npz_path.stem}_{group}.png"
            plot_group(npz_path, group, out_path)

    print(f"\nDone. {len(npz_files)} files × 3 groups = {len(npz_files)*3} plots.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all recordings in a dataset folder, one fig per group per file."
    )
    parser.add_argument("dataset", type=Path,
                        help="Path to dataset folder inside data/ (e.g. data/my_dataset)")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    if args.dataset.is_dir():
        process_dataset(args.dataset)
    else:
        # Single file — create output folder next to the file
        npz_path = args.dataset
        out_dir  = npz_path.parent / npz_path.stem
        out_dir.mkdir(exist_ok=True)
        print(f"Output folder: {out_dir}")
        for group in _GROUPS:
            out_path = out_dir / f"{npz_path.stem}_{group}.png"
            plot_group(npz_path, group, out_path)


if __name__ == "__main__":
    main()
