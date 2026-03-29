"""Rumi quadruped sim2real with mink IK — multiprocessing edition.

User controls 1 body target in viser.  mink solves IK once per step (from the
current sim qpos) to produce 12 joint-position targets that are sent to both
the MuJoCo sim and the real Dynamixel motors.

Three visualised robots
───────────────────────
  target (green) : pure IK solution — no physics, just FK at the IK joint angles.
                   This is the ideal/reference pose the controller is asking for.
  sim    (grey)  : MuJoCo physics simulation driven by the same joint targets.
                   Shows how the sim robot responds under the tuned physics params.
  real   (blue)  : ghost of the actual hardware joint angles read back each cycle.
                   Shows how the real robot tracks the target.

Floor reference for ghosts
───────────────────────────
Feet are assumed not to slip.  Each ghost's body is shifted so that its foot
centroid matches the sim foot centroid (3-D translation only, no rotation).

Processes / threads
───────────────────
  Real process  (CONTROL_HZ): reads shm_q_target, sends to hardware, reads back
  Main process:
      Sim  thread (CONTROL_HZ): single mink IK solve → shm_q_target + mj_step
      Viz  thread (VIZ_HZ)    : viser update for all three robots
      Main thread (60 Hz)     : viser target poll + GUI update

Run
───
  python rumi_sim2real.py
  python rumi_sim2real.py --control-hz 50 --device /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import mujoco
import mink
import numpy as np
import viser

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SCENE_XML = _HERE / "scene.xml"

_SIM2REAL = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SIM2REAL))
from dynamixel_controller import DynamixelController  # noqa: E402
from viewer import ViserMujocoScene                   # noqa: E402

# ── Joint ordering (matches mujoco model q[7:]) ────────────────────────────────
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "BL_hip_joint", "BL_thigh_joint", "BL_calf_joint",
    "BR_hip_joint", "BR_thigh_joint", "BR_calf_joint",
]
N_JOINTS = len(JOINT_NAMES)

# ── Timing ─────────────────────────────────────────────────────────────────────
CONTROL_HZ = 50
VIZ_HZ     = 50

# ── IK settings ────────────────────────────────────────────────────────────────
IK_SOLVER     = "daqp"
IK_MAX_ITERS  = 20
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# Default mink task costs
DEFAULT_BODY_POS_COST = 1.0
DEFAULT_BODY_ORI_COST = 1.0
DEFAULT_POSTURE_COST  = 1e-3

# ── Default physics parameters (read from rumi.xml defaults) ───────────────────
DEFAULT_KP           = 20.0
DEFAULT_KV           = 0.0
# armature="0.01623" damping="0.59436" frictionloss="0.001"
DEFAULT_DAMPING      = 0.59436
DEFAULT_ARMATURE     = 0.01623
DEFAULT_FRICTIONLOSS = 0.001

# Thread-shared physics params — written by main thread, read by sim thread.
# Plain dict + lock: both share the same process, no serialisation needed.
_phys_lock   = threading.Lock()
_phys_params = {
    "kp":           DEFAULT_KP,
    "kv":           DEFAULT_KV,
    "damping":      DEFAULT_DAMPING,
    "armature":     DEFAULT_ARMATURE,
    "frictionloss": DEFAULT_FRICTIONLOSS,
}

# ── Shared memory layout ───────────────────────────────────────────────────────
# shm_targets  : body_pos(3) + body_wxyz(4)  = 7 doubles
# shm_q_target : 12 IK-solution joint angles (absolute) — written by sim thread,
#                read by real process (send to motors) and viz thread (target ghost)
# shm_q_real   : 12 absolute joint angles read back from hardware
# shm_q0       : 12 settled sim joint angles — the zero reference offset
#                written by main after sim settles; real process waits for it
# shm_real_hz / shm_sim_hz : scalar rate monitors

_N_TARGET = 7   # body pos(3) + wxyz(4)


def _np(shm: mp.Array) -> np.ndarray:
    """Zero-copy numpy view of a shared double Array."""
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── IK helpers ─────────────────────────────────────────────────────────────────

def build_ik_tasks(model: mujoco.MjModel):
    """Create body + posture + fixed-foot tasks."""
    base_task = mink.FrameTask(
        frame_name="body_mid",
        frame_type="site",
        position_cost=DEFAULT_BODY_POS_COST,
        orientation_cost=DEFAULT_BODY_ORI_COST,
    )
    posture_task = mink.PostureTask(model, cost=DEFAULT_POSTURE_COST)
    foot_tasks = [
        mink.FrameTask(
            frame_name=f,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        for f in ["FL", "FR", "BL", "BR"]
    ]
    return base_task, posture_task, foot_tasks


def solve_ik(
    configuration: mink.Configuration,
    base_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    foot_tasks: list,
    data: mujoco.MjData,
    dt: float,
    body_mocap_id: int,
    foot_mocap_ids: list = [],
) -> None:
    """Body target from mocap; foot targets already set at init, never changed."""
    base_task.set_target(mink.SE3.from_mocap_id(data, body_mocap_id))
    tasks = [base_task, posture_task, *foot_tasks]
    for _ in range(IK_MAX_ITERS):
        vel = mink.solve_ik(configuration, tasks, dt, IK_SOLVER, 1e-3)
        configuration.integrate_inplace(vel, dt)
        pos_ok = np.linalg.norm(base_task.compute_error(configuration)[:3]) <= POS_THRESHOLD
        ori_ok = np.linalg.norm(base_task.compute_error(configuration)[3:]) <= ORI_THRESHOLD
        if pos_ok and ori_ok:
            break


# ── Real-robot process ─────────────────────────────────────────────────────────

def real_process_fn(
    control_hz: int,
    device: str,
    shm_q_real: mp.Array,
    shm_q_target: mp.Array,
    shm_q0: mp.Array,
    shm_real_hz: mp.Array,
    connect_event: mp.Event,
    stop_event: mp.Event,
):
    """Waits for q0 offset and GUI connect, then forwards shm_q_target to hardware."""
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    _sys.path.insert(0, str(_SIM2REAL))
    import numpy as _np_mod

    dt = 1.0 / control_hz

    # Wait for main to write the settled sim q0 offset
    print("[real] Waiting for sim to settle (q0)…")
    while not stop_event.is_set():
        q0 = _np(shm_q0).copy()
        if _np_mod.any(q0 != 0.0):
            break
        time.sleep(0.05)
    if stop_event.is_set():
        return

    print("[real] Waiting for Connect button in GUI…")
    connect_event.wait()
    if stop_event.is_set():
        return

    hw = DynamixelController(port=device)
    print("[real] Initializing motors…")
    if not hw.initialize(expected_motors=N_JOINTS):
        print("[real] Failed to initialize motors — exiting.")
        return

    print("[real] Motors ready. Starting control loop.")
    iters  = 0
    t_rate = time.time()

    try:
        while not stop_event.is_set():
            t = time.time()

            # Read back hardware positions → absolute angles for real ghost
            pos_dict = hw.get_joint_positions_rad()
            if pos_dict is not None:
                q_hw  = _np_mod.array([pos_dict.get(n, 0.0) for n in JOINT_NAMES])
                q_abs = q_hw + q0           # absolute = hw_relative + offset
                _np(shm_q_real)[:] = q_abs

            # Forward IK target from sim thread → motors
            # shm_q_target holds absolute angles; motors expect relative to q0
            q_cmd_abs = _np(shm_q_target).copy()
            q_cmd_hw  = q_cmd_abs - q0      # relative to Dynamixel zero
            hw.set_joint_positions_rad({n: float(q_cmd_hw[i]) for i, n in enumerate(JOINT_NAMES)})

            iters += 1
            elapsed_rate = time.time() - t_rate
            if elapsed_rate >= 1.0:
                _np(shm_real_hz)[0] = iters / elapsed_rate
                iters  = 0
                t_rate = time.time()

            elapsed = time.time() - t
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        hw.move_to_zero()
        hw.disconnect()
        print("[real] Done.")


# ── Sim thread ─────────────────────────────────────────────────────────────────

def sim_thread_fn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: mink.Configuration,
    base_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    foot_tasks: list,
    body_mocap_id: int,
    shm_targets: mp.Array,
    shm_q_target: mp.Array,
    shm_sim_hz: mp.Array,
    stop_event: threading.Event,
):
    dt          = 1.0 / CONTROL_HZ
    n_substeps  = max(1, round(dt / model.opt.timestep))
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        # Apply live physics params from GUI sliders
        with _phys_lock:
            p = _phys_params.copy()
        model.dof_damping[6:]        = p["damping"]
        model.dof_armature[6:]       = p["armature"]
        model.dof_frictionloss[6:]   = p["frictionloss"]
        model.actuator_gainprm[:, 0] = p["kp"]
        model.actuator_biasprm[:, 1] = -p["kp"]
        model.actuator_biasprm[:, 2] = -p["kv"]

        configuration.update(data.qpos)

        tgt = _np(shm_targets).copy()
        data.mocap_pos[body_mocap_id]  = tgt[0:3]
        data.mocap_quat[body_mocap_id] = tgt[3:7]

        solve_ik(configuration, base_task, posture_task, foot_tasks,
                 data, dt, body_mocap_id, [])

        # Publish IK result — shared with real process and viz thread
        q_ik = configuration.q[7:].copy()
        _np(shm_q_target)[:] = q_ik

        data.ctrl[:] = q_ik
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        iters += 1
        elapsed_rate = time.time() - t_rate
        if elapsed_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / elapsed_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Viz thread ─────────────────────────────────────────────────────────────────

def _place_ghost_by_foot_centroid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_joints: np.ndarray,
    sim_foot_centroid: np.ndarray,
    foot_ids: list,
) -> None:
    """Set ghost joint angles and shift body so its foot centroid matches sim."""
    data.qpos[7:]  = q_joints
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [0.0, 0.0, 0.2]          # trial position
    mujoco.mj_kinematics(model, data)
    ghost_centroid = np.mean([data.site(sid).xpos for sid in foot_ids], axis=0)
    data.qpos[0:3] += sim_foot_centroid - ghost_centroid
    mujoco.mj_kinematics(model, data)


def viz_thread_fn(
    sim_view:    object,
    target_view: object,
    real_view:   object,
    model: mujoco.MjModel,
    data_sim:    mujoco.MjData,
    data_target: mujoco.MjData,
    data_real:   mujoco.MjData,
    shm_q_target: mp.Array,
    shm_q_real:   mp.Array,
    stop_event: threading.Event,
):
    """Updates viser at VIZ_HZ for all three robots.

    All ghosts use foot-centroid matching: body is shifted so their foot
    centroid aligns with the sim foot centroid (translation only, no rotation).

      target (green) : IK solution — ideal pose the controller requests
      sim    (grey)  : MuJoCo physics response to those targets
      real   (blue)  : actual hardware joint angles read back
    """
    period   = 1.0 / VIZ_HZ
    foot_ids = [model.site(f).id for f in ["FL", "FR", "BL", "BR"]]

    while not stop_event.is_set():
        t = time.time()

        # ── Sim (grey) ─────────────────────────────────────────────────────────
        mujoco.mj_kinematics(model, data_sim)
        sim_view.update(data_sim)

        sim_foot_centroid = np.mean(
            [data_sim.site(sid).xpos for sid in foot_ids], axis=0
        )

        # ── Target (green) ─────────────────────────────────────────────────────
        q_target = _np(shm_q_target).copy()
        _place_ghost_by_foot_centroid(model, data_target, q_target,
                                      sim_foot_centroid, foot_ids)
        target_view.update(data_target)

        # ── Real (blue) ────────────────────────────────────────────────────────
        q_real = _np(shm_q_real).copy()
        _place_ghost_by_foot_centroid(model, data_real, q_real,
                                      sim_foot_centroid, foot_ids)
        real_view.update(data_real)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rumi sim2real with mink IK")
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    parser.add_argument("--device", type=str, default=None,
                        help="Serial port for Dynamixel (e.g. /dev/ttyUSB0). Auto-scans if omitted.")
    args = parser.parse_args()

    # ── Shared memory ──────────────────────────────────────────────────────────
    shm_targets  = mp.Array(ctypes.c_double, _N_TARGET)
    shm_q_target = mp.Array(ctypes.c_double, N_JOINTS)  # IK solution → real + viz
    shm_q_real   = mp.Array(ctypes.c_double, N_JOINTS)  # hardware readback → viz
    shm_q0       = mp.Array(ctypes.c_double, N_JOINTS)  # settled sim offset
    shm_real_hz  = mp.Array(ctypes.c_double, 1)
    shm_sim_hz   = mp.Array(ctypes.c_double, 1)

    connect_mp = mp.Event()
    stop_mp    = mp.Event()

    # ── Spawn real process FIRST ───────────────────────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(args.control_hz, args.device,
              shm_q_real, shm_q_target, shm_q0, shm_real_hz, connect_mp, stop_mp),
        daemon=True,
    )
    real_proc.start()

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    print("Loading MuJoCo model…")
    model        = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data_sim     = mujoco.MjData(model)
    data_target  = mujoco.MjData(model)
    data_real    = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data_sim,    model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data_target, model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data_real,   model.key("home").id)
    mujoco.mj_forward(model, data_sim)
    mujoco.mj_forward(model, data_target)
    mujoco.mj_forward(model, data_real)

    body_mocap_id = model.body("body_target").mocapid[0]

    # ── mink setup ────────────────────────────────────────────────────────────
    print("Setting up mink IK…")
    configuration = mink.Configuration(model)
    configuration.update(data_sim.qpos)

    base_task, posture_task, foot_tasks = build_ik_tasks(model)
    posture_task.set_target_from_configuration(configuration)

    # Init body mocap to current body_mid site position
    mink.move_mocap_to_frame(model, data_sim, "body_target", "body_mid", "site")
    mujoco.mj_forward(model, data_sim)

    # Set foot targets from keyframe FK (initial estimate before sim settles)
    for task, fname in zip(foot_tasks, ["FL", "FR", "BL", "BR"]):
        pos = data_sim.site(model.site(fname).id).xpos.copy()
        task.set_target(mink.SE3.from_translation(pos))

    # Init shm_targets from current mocap
    tgt = _np(shm_targets)
    tgt[0:3] = data_sim.mocap_pos[body_mocap_id]
    tgt[3:7] = data_sim.mocap_quat[body_mocap_id]

    # ── Start sim thread early, let it settle, then snapshot q0 ───────────────
    th_stop = threading.Event()

    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(model, data_sim, configuration,
              base_task, posture_task, foot_tasks,
              body_mocap_id,
              shm_targets, shm_q_target, shm_sim_hz, th_stop),
    ).start()

    print("Waiting for sim to settle…")
    time.sleep(1.0)

    # Snapshot settled sim joint angles as q0 offset
    q0 = data_sim.qpos[7:].copy()
    _np(shm_q0)[:] = q0           # unblocks real process
    _np(shm_q_target)[:] = q0     # target ghost starts at settled sim pose
    _np(shm_q_real)[:] = q0       # real ghost starts at settled sim pose

    # Now set foot targets from the settled sim foot positions
    mujoco.mj_kinematics(model, data_sim)
    for task, fname in zip(foot_tasks, ["FL", "FR", "BL", "BR"]):
        pos = data_sim.site(model.site(fname).id).xpos.copy()
        task.set_target(mink.SE3.from_translation(pos))

    print(f"q0 = {np.round(q0, 4)}")

    # ── Viser ─────────────────────────────────────────────────────────────────
    print("Starting viser…")
    server      = viser.ViserServer(label="Rumi Sim2Real")
    scene       = ViserMujocoScene.create(server, model)
    sim_view    = scene.add_robot("sim",    color=(0.75, 0.75, 0.75, 1.00))
    target_view = scene.add_robot("target", color=(0.20, 0.80, 0.20, 0.80))
    real_view   = scene.add_robot("real",   color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(
        camera_distance=1.2, camera_azimuth=135.0, camera_elevation=25.0
    )

    body_pos0  = data_sim.mocap_pos[body_mocap_id].copy()
    body_quat0 = data_sim.mocap_quat[body_mocap_id].copy()

    body_ctrl = server.scene.add_transform_controls(
        "/ik/body",
        position=tuple(float(v) for v in body_pos0),
        wxyz=tuple(float(v) for v in body_quat0),
        scale=0.12,
    )

    # GUI panels
    with server.gui.add_folder("Sim2Real"):
        chk_connect = server.gui.add_checkbox("Connect robot", initial_value=False)
        txt_connect = server.gui.add_text("Robot status", initial_value="Disconnected")
        txt_sim_hz  = server.gui.add_text("Sim rate",  initial_value="— Hz")
        txt_real_hz = server.gui.add_text("Real rate", initial_value="— Hz")

    @chk_connect.on_update
    def _on_connect_toggle(_):
        if chk_connect.value and not connect_mp.is_set():
            connect_mp.set()
            txt_connect.value = "Connecting…"
        elif not chk_connect.value and connect_mp.is_set():
            stop_mp.set()
            txt_connect.value = "Disconnected"

    with server.gui.add_folder("IK costs"):
        sl_body_pos = server.gui.add_slider("Body pos", min=0.0, max=5.0,  step=0.1,  initial_value=DEFAULT_BODY_POS_COST)
        sl_body_ori = server.gui.add_slider("Body ori", min=0.0, max=5.0,  step=0.1,  initial_value=DEFAULT_BODY_ORI_COST)
        sl_posture  = server.gui.add_slider("Posture",  min=0.0, max=0.01, step=1e-4, initial_value=DEFAULT_POSTURE_COST)

    with server.gui.add_folder("Physics params"):
        sl_kp           = server.gui.add_slider("kp",            min=0.0,   max=30.0,  step=0.1,   initial_value=DEFAULT_KP)
        sl_kv           = server.gui.add_slider("kv",            min=0.0,   max=5.0,   step=0.05,  initial_value=DEFAULT_KV)
        sl_damping      = server.gui.add_slider("joint_damping", min=0.0,   max=5.0,   step=0.01,  initial_value=DEFAULT_DAMPING)
        sl_armature     = server.gui.add_slider("armature",      min=0.0,   max=0.1,   step=0.001, initial_value=DEFAULT_ARMATURE)
        sl_frictionloss = server.gui.add_slider("frictionloss",  min=0.0,   max=1.0,   step=0.01,  initial_value=DEFAULT_FRICTIONLOSS)

    # ── Viz thread ────────────────────────────────────────────────────────────
    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(sim_view, target_view, real_view,
              model, data_sim, data_target, data_real,
              shm_q_target, shm_q_real, th_stop),
    ).start()

    # ── Main loop (60 Hz) ─────────────────────────────────────────────────────
    print("Running. Ctrl+C to stop.")
    try:
        while True:
            tgt = _np(shm_targets)
            p = body_ctrl.position
            w = body_ctrl.wxyz
            tgt[0:3] = [p[0], p[1], p[2]]
            tgt[3:7] = [w[0], w[1], w[2], w[3]]

            base_task.set_position_cost(sl_body_pos.value)
            base_task.set_orientation_cost(sl_body_ori.value)
            posture_task.set_cost(sl_posture.value)

            with _phys_lock:
                _phys_params["kp"]           = sl_kp.value
                _phys_params["kv"]           = sl_kv.value
                _phys_params["damping"]      = sl_damping.value
                _phys_params["armature"]     = sl_armature.value
                _phys_params["frictionloss"] = sl_frictionloss.value

            txt_sim_hz.value = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            real_hz = _np(shm_real_hz)[0]
            txt_real_hz.value = f"{real_hz:.0f} Hz"
            if connect_mp.is_set() and real_hz > 0:
                txt_connect.value = f"Connected ({real_hz:.0f} Hz)"

            time.sleep(1.0 / 60)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        th_stop.set()
        stop_mp.set()
        real_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
