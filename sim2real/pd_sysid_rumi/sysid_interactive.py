"""Interactive sysid tuner — manually sweep armature/damping/frictionloss in viser.

Loads a standup recording, lets you pick one joint, adjust the three MuJoCo
params via sliders, click "Run Replay" and instantly see a target/sim/real plot.

Usage:
  python sysid_interactive.py data/<timestamp>_standup_sine.npz
  python sysid_interactive.py data/<timestamp>_standup_sine.npz --joint BL_calf_joint
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import mujoco
import numpy as np
import viser
import viser.uplot as uplot

_HERE      = Path(__file__).parent
_SCENE_XML = _HERE / "scene.xml"
PHYSICS_DT = 0.004

# XML defaults (rumi.xml)
_XML_ARMATURE    = 0.012
_XML_DAMPING     = 0.66
_XML_FRICTION    = 0.09


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────

def _get_joint_ids(m: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    jid     = m.joint(joint_name).id
    qpos_id = int(m.jnt_qposadr[jid])
    qvel_id = int(m.jnt_dofadr[jid])
    for i in range(m.nu):
        if m.actuator_trnid[i, 0] == jid:
            return qpos_id, qvel_id, i
    raise ValueError(f"No actuator for joint '{joint_name}'")


def _joint_part(name: str) -> str:
    return name.split("_")[1]   # "hip" | "thigh" | "calf"


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


# ── Simulation replay ───────────────────────────────────────────────────────────

def replay(
    targets:      np.ndarray,   # (N, 12)
    q0:           np.ndarray,   # (12,)
    control_hz:   int,
    joint_names:  list[str],    # all 12
    sel_joint:    str,          # selected joint — all joints of same part get params
    armature:     float,
    damping:      float,
    frictionloss: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay full 12-joint sequence; all joints of the same part get candidate params."""
    part = _joint_part(sel_joint)
    same_part = [jn for jn in joint_names if _joint_part(jn) == part]

    m = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    m.opt.timestep = PHYSICS_DT
    _apply_params(m, same_part, armature, damping, frictionloss)

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
            q_sim[i, col]   = d.qpos[qpos_id]
            dq_sim[i, col]  = d.qvel[qvel_id]
            d.ctrl[ctrl_id] = targets[i, col]
        for _ in range(substeps):
            mujoco.mj_step(m, d)

    return q_sim, dq_sim


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive sysid slider tuner."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--joint", default=None, metavar="JOINT_NAME",
        help="Joint to tune (default: first joint in recording).",
    )
    args = parser.parse_args()

    # ── Load recording ──────────────────────────────────────────────────────────
    data        = np.load(args.recording, allow_pickle=True)
    t           = data["t"]                      # (N,)
    targets     = data["target"]                 # (N, 12)
    q_real      = data["q_real"]                 # (N, 12)
    dq_real     = data["dq_real"]                # (N, 12)
    joint_names = data["joint_names"].tolist()   # len 12
    control_hz  = int(data["control_hz"][0])
    N           = len(t)
    q0          = q_real[0].copy()

    print(f"Loaded  N={N}  hz={control_hz}  duration={t[-1]-t[0]:.1f}s")
    print(f"Joints: {joint_names}")

    # ── Resolve joint ───────────────────────────────────────────────────────────
    if args.joint is not None:
        if args.joint not in joint_names:
            parser.error(f"--joint '{args.joint}' not in recording. "
                         f"Available: {joint_names}")
        sel_joint = args.joint
    else:
        sel_joint = joint_names[0]

    sel_col = joint_names.index(sel_joint)
    print(f"Selected joint: {sel_joint}  (col {sel_col})")

    # ── Viser server ────────────────────────────────────────────────────────────
    server = viser.ViserServer(port=8081)
    server.scene.world_axes.visible = False

    with server.gui.add_folder("Joint"):
        joint_dd = server.gui.add_dropdown(
            "Joint", options=joint_names, initial_value=sel_joint
        )

    with server.gui.add_folder("Parameters"):
        sl_arm  = server.gui.add_slider(
            "Armature",     min=0.0,  max=10.0, step=0.1,  initial_value=_XML_ARMATURE
        )
        sl_damp = server.gui.add_slider(
            "Damping",      min=0.0,  max=10.0, step=0.1,  initial_value=_XML_DAMPING
        )
        sl_fric = server.gui.add_slider(
            "Frictionloss", min=0.0, max=2.0,  step=0.01, initial_value=_XML_FRICTION
        )

    with server.gui.add_folder("Actions"):
        btn_run   = server.gui.add_button("Run Replay")
        btn_reset = server.gui.add_button("Reset to XML defaults")

    status_md = server.gui.add_markdown("*Adjust sliders then click Run Replay.*")

    _zeros = np.zeros(N)

    # Plot panel — position, pos-error, velocity
    # data tuple: (x, y1, y2, ...) — series tuple must match len(data)
    plot_pos = server.gui.add_uplot(
        data=(
            t,
            np.rad2deg(targets[:, sel_col]),
            _zeros,                               # sim placeholder
            np.rad2deg(q_real[:, sel_col]),
        ),
        series=(
            uplot.Series(label=""),               # x-axis entry (required)
            uplot.Series(label="target", stroke="#f97316", width=1.5),
            uplot.Series(label="sim",    stroke="#3b82f6", width=1.5),
            uplot.Series(label="real",   stroke="#ef4444", width=1.5),
        ),
        title="Position (deg)",
        aspect=2.5,
    )
    plot_err = server.gui.add_uplot(
        data=(t, _zeros),
        series=(
            uplot.Series(label=""),
            uplot.Series(label="error real-sim", stroke="#6366f1", width=1.2),
        ),
        title="Pos error real–sim (deg)",
        aspect=2.0,
    )
    plot_vel = server.gui.add_uplot(
        data=(
            t,
            _zeros,                               # sim placeholder
            np.rad2deg(dq_real[:, sel_col]),
        ),
        series=(
            uplot.Series(label=""),
            uplot.Series(label="sim",  stroke="#3b82f6", width=1.5),
            uplot.Series(label="real", stroke="#ef4444", width=1.5),
        ),
        title="Velocity (deg/s)",
        aspect=2.0,
    )

    # ── Shared mutable state ────────────────────────────────────────────────────
    _lock      = threading.Lock()
    _running   = [False]

    def _do_replay() -> None:
        with _lock:
            if _running[0]:
                return
            _running[0] = True

        jname    = joint_dd.value
        arm      = float(sl_arm.value)
        damp     = float(sl_damp.value)
        fric     = float(sl_fric.value)
        j_col    = joint_names.index(jname)

        part = _joint_part(jname)
        same_part = [jn for jn in joint_names if _joint_part(jn) == part]
        status_md.content = f"*Running replay for **{jname}** (params applied to all {part}: {same_part}) …*"

        q_sim, dq_sim = replay(
            targets, q0, control_hz, joint_names, jname,
            arm, damp, fric,
        )

        pos_loss = float(np.sqrt(np.mean((q_sim[:, j_col] - q_real[:, j_col]) ** 2)))
        vel_loss = float(np.sqrt(np.mean((dq_sim[:, j_col] - dq_real[:, j_col]) ** 2)))

        # Update plots
        plot_pos.data = (
            t,
            np.rad2deg(targets[:, j_col]),
            np.rad2deg(q_sim[:,   j_col]),
            np.rad2deg(q_real[:,  j_col]),
        )
        plot_err.data = (
            t,
            np.rad2deg(q_real[:, j_col] - q_sim[:, j_col]),
        )
        plot_vel.data = (
            t,
            np.rad2deg(dq_sim[:,  j_col]),
            np.rad2deg(dq_real[:, j_col]),
        )

        pos_rms_deg = float(pos_loss * 180.0 / np.pi)
        vel_rms_deg = float(vel_loss * 180.0 / np.pi)
        status_md.content = (
            f"**{jname}** (all {part})  "
            f"armature={arm:.4f}  damping={damp:.4f}  frictionloss={fric:.5f}  \n"
            f"pos RMSE={pos_rms_deg:.2f}°  vel RMSE={vel_rms_deg:.2f}°/s"
        )

        with _lock:
            _running[0] = False

    @btn_run.on_click
    def _on_run(_) -> None:
        threading.Thread(target=_do_replay, daemon=True).start()

    @btn_reset.on_click
    def _on_reset(_) -> None:
        sl_arm.value  = _XML_ARMATURE
        sl_damp.value = _XML_DAMPING
        sl_fric.value = _XML_FRICTION

    print(f"\nViser running at http://localhost:8081")
    print(f"Adjust sliders → click 'Run Replay' to evaluate.")

    # Keep main thread alive
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
