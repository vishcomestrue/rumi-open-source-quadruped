"""Rumi getup validation — policy running on sim and real in parallel.

Three robots visualised in viser simultaneously:

  target (green) : raw policy output scaled by ACTION_SCALE — the position
                   command that is being sent.  No physics.
  sim    (grey)  : MuJoCo physics driven by those same position commands.
                   Observations are built from the sim state each step.
  real   (blue)  : Actual Dynamixel encoder readback after sending commands.
                   Observations are built from hardware exactly as in run_getup.py.

Both sim and real run the same policy with their own independent observations,
so any divergence between them is visible immediately.

Processes / threads
───────────────────
  Real process  (50 Hz) — separate mp.Process:
      init motors → capture sit-pose zero → wait for connect_event
      loop: read encoders + IMU → build_obs → policy → send commands
            write shm_q_real (encoder readback, radians rel. to sit-pose zero)

  Main process:
      sim_thread   (50 Hz): mj_forward → build_obs_sim → policy → mj_step
                            writes shm_q_target (scaled action) and shm_q_sim
      viz_thread   (50 Hz): reads all three shm arrays → viser updates
      main_loop    (60 Hz): GUI polling

Run
───
  python run_validation.py                         # full run
  python run_validation.py --dry-run               # no motors, sim + viz only
  python run_validation.py --duration 10           # 10 s then stop
  python run_validation.py --target 0.18           # initial target height (adjust live via GUI slider)
  python run_validation.py --device /dev/ttyUSB0   # explicit serial port
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
import numpy as np
import viser

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_DEPLOY    = _HERE.parent          # deploy/
_SIM2REAL  = _DEPLOY.parent        # sim2real/
_RUMI_MJLAB_SRC = _SIM2REAL / "rumi-mjlab" / "src"

sys.path.insert(0, str(_HERE))          # validation/  (viewer module)
sys.path.insert(0, str(_DEPLOY))        # deploy/      (mx64_controller, imu, …)
sys.path.insert(0, str(_RUMI_MJLAB_SRC))# rumi-mjlab   (FK kinematics)

_SCENE_XML   = _HERE / "scene.xml"
_GETUP_DIR   = _DEPLOY / "getup"
_DEFAULT_CKPT = _GETUP_DIR / "checkpoint" / "latest_getup.pt"

sys.path.insert(0, str(_GETUP_DIR))    # getup/ (policy, observations)

from viewer import ViserMujocoScene    # noqa: E402

# ── Joint ordering — MuJoCo XML order: FL FR BL BR ────────────────────────────
# (matches observations.py JOINT_ORDER)
JOINT_NAMES = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "BL_hip_joint",  "BL_thigh_joint",  "BL_calf_joint",
    "BR_hip_joint",  "BR_thigh_joint",  "BR_calf_joint",
]
N_JOINTS = len(JOINT_NAMES)

# Motor ID → joint index in JOINT_NAMES
# Motor IDs from run_getup.py: 1-3 FL, 4-6 BL, 7-9 BR, 10-12 FR
# JOINT_NAMES order:           0-2 FL, 3-5 FR, 6-8 BL, 9-11 BR
_MOTOR_ID_TO_JOINT_IDX = {
    1:  0,   # FL_hip
    2:  1,   # FL_thigh
    3:  2,   # FL_calf
    4:  6,   # BL_hip
    5:  7,   # BL_thigh
    6:  8,   # BL_calf
    7:  9,   # BR_hip
    8:  10,  # BR_thigh
    9:  11,  # BR_calf
    10: 3,   # FR_hip
    11: 4,   # FR_thigh
    12: 5,   # FR_calf
}

# ── Timing / scaling ───────────────────────────────────────────────────────────
CONTROL_HZ   = 50
VIZ_HZ       = 50
DT           = 1.0 / CONTROL_HZ
ACTION_SCALE = 0.075          # rad per unit — matches run_getup.py

# ── Shared memory helpers ──────────────────────────────────────────────────────
def _np(shm: mp.Array) -> np.ndarray:
    """Zero-copy numpy view of a shared double Array."""
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── Obs building from MuJoCo state (sim side) ─────────────────────────────────

def _projected_gravity_from_xmat(xmat_body: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0,0,-1] into body frame from a 3×3 rotation matrix.

    xmat_body is the body-to-world rotation (row-major from mj_data.xmat).
    We need world-to-body, so we transpose.
    """
    R_w2b = xmat_body.T                        # world → body
    g_world = np.array([0.0, 0.0, -1.0])
    return (R_w2b @ g_world).astype(np.float32)


def build_obs_sim(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    last_action: np.ndarray,
    target_height_m: float,
) -> np.ndarray:
    """Build 41-dim observation vector from MuJoCo simulation state.

    Mirrors the layout in getup/observations.py exactly, but uses
    ground-truth sim state rather than encoder + IMU readings.

    Obs layout:
      [0]      body_height       — qpos[2] (direct z position), normalised
      [1]      target_height     — fixed goal, normalised
      [2:5]    projected_gravity — gravity in body frame from sim orientation
      [5:17]   joint_pos         — qpos[7:19]
      [17:29]  joint_vel         — qvel[6:18]
      [29:41]  last_action       — previous raw policy output
    """
    _H_MIN = 0.10
    _H_MAX = 0.25

    body_height_raw  = float(data.qpos[2])
    body_height_obs  = float(np.clip(
        (body_height_raw - _H_MIN) / (_H_MAX - _H_MIN), -0.5, 1.5
    ))
    target_height_obs = float((target_height_m - _H_MIN) / (_H_MAX - _H_MIN))

    # body index 1 is the floating "body" link (index 0 = world)
    body_id    = model.body("body").id
    xmat_body  = data.xmat[body_id].reshape(3, 3).copy()
    proj_grav  = _projected_gravity_from_xmat(xmat_body)

    joint_pos = data.qpos[7:19].astype(np.float32)
    joint_vel = data.qvel[6:18].astype(np.float32)

    obs = np.concatenate([
        [body_height_obs],
        [target_height_obs],
        proj_grav,
        joint_pos,
        joint_vel,
        last_action.astype(np.float32),
    ])
    assert obs.shape == (41,), f"Expected (41,), got {obs.shape}"
    return obs.astype(np.float32)


# ── Foot-centroid placement (same as sysid_rumi_visual) ───────────────────────

def _place_ghost_by_foot_centroid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_joints: np.ndarray,
    sim_foot_centroid: np.ndarray,
    foot_ids: list,
) -> None:
    """Set ghost joint angles and shift body so foot centroid matches sim."""
    data.qpos[7:]  = q_joints
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [0.0, 0.0, 0.2]
    mujoco.mj_kinematics(model, data)
    ghost_centroid = np.mean([data.site(sid).xpos for sid in foot_ids], axis=0)
    data.qpos[0:3] += sim_foot_centroid - ghost_centroid
    mujoco.mj_kinematics(model, data)


# ── Real-robot process ─────────────────────────────────────────────────────────

def real_process_fn(
    checkpoint_path:    str,
    shm_target_height:  mp.Array,  # in:  live target height in metres (GUI slider)
    shm_q_real:         mp.Array,  # out: encoder readback, rel. to sit-pose zero [rad]
    shm_q_target:       mp.Array,  # out: scaled action offsets from sit-pose zero [rad]
    shm_imu_quat:       mp.Array,  # out: IMU quaternion (w,x,y,z)
    shm_real_hz:        mp.Array,  # out: measured loop rate
    connect_event:      mp.Event,
    stop_event:         mp.Event,
    dry_run:            bool,
):
    """Runs on real hardware: IMU + motors → obs → policy → motor commands.

    Mirrors run_getup.py exactly, but shares results via shm for the visualiser.
    Waits for connect_event before touching motors.
    """
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    _sys.path.insert(0, str(_DEPLOY))
    _sys.path.insert(0, str(_GETUP_DIR))
    _sys.path.insert(0, str(_RUMI_MJLAB_SRC))

    import numpy as _np_mod
    from policy import GetupPolicy
    from observations import build_obs, JOINT_ORDER

    RADIANS_TO_POS = 4096.0 / (2.0 * _np_mod.pi)

    JOINT_ID_MAP = {
        "FL_hip_joint": 1,  "FL_thigh_joint": 2,  "FL_calf_joint": 3,
        "BL_hip_joint": 4,  "BL_thigh_joint": 5,  "BL_calf_joint": 6,
        "BR_hip_joint": 7,  "BR_thigh_joint": 8,  "BR_calf_joint": 9,
        "FR_hip_joint": 10, "FR_thigh_joint": 11, "FR_calf_joint": 12,
    }

    print("[real] Loading policy …")
    policy = GetupPolicy(checkpoint_path)

    # ── IMU (background thread, always running) ──────────────────────────────
    from imu import IMU
    print("[real] Initializing IMU …")
    imu = IMU()
    print("[real] IMU ready.")

    # ── Wait for connect ─────────────────────────────────────────────────────
    print("[real] Waiting for Connect in GUI …")
    connect_event.wait()
    if stop_event.is_set():
        return

    if dry_run:
        print("[real] DRY RUN — motors skipped.")
        zero_pos = {i: 2048 for i in range(1, 13)}
        controller = None
    else:
        from mx64_controller import MX64Controller
        ctrl_cfg = str(_DEPLOY / "motor_config.json")
        controller = MX64Controller(config_path=ctrl_cfg, auto_connect=False)
        motor_ids = controller.initialize(expected_motors=12)
        if motor_ids is None:
            print("[real] Motor initialization failed — exiting.")
            return

        # Capture sit-pose zero reference (same as run_getup.py)
        zero_pos = controller.sync_read_positions(max_retries=10, retry_delay=0.02)
        if zero_pos is None:
            print("[real] Could not read initial motor positions — exiting.")
            controller.disconnect()
            return
        print("[real] Sit-pose zero captured.")

    def read_joint_states():
        if dry_run:
            return (
                {j: 0.0 for j in JOINT_ORDER},
                {j: 0.0 for j in JOINT_ORDER},
            )
        raw_pos, raw_vel = controller.sync_read_positions_and_velocities()
        if raw_pos is None:
            return None, None
        pos_dict = {}
        vel_dict = {}
        for joint, mid in JOINT_ID_MAP.items():
            pos_dict[joint] = (raw_pos[mid] - zero_pos[mid]) / RADIANS_TO_POS
            vel_dict[joint] = raw_vel[mid]
        return pos_dict, vel_dict

    def write_joint_positions(pos_rad_dict: dict) -> None:
        if dry_run or controller is None:
            return
        raw_targets = {}
        for joint, mid in JOINT_ID_MAP.items():
            offset_ticks = int(round(pos_rad_dict[joint] * RADIANS_TO_POS))
            raw_targets[mid] = zero_pos[mid] + offset_ticks
        controller.sync_write_positions(raw_targets)

    # ── Warmup ───────────────────────────────────────────────────────────────
    _dummy_obs = build_obs(
        {j: 0.0 for j in JOINT_ORDER},
        {j: 0.0 for j in JOINT_ORDER},
        _np_mod.array([1.0, 0.0, 0.0, 0.0], dtype=_np_mod.float32),
        _np_mod.zeros(12, dtype=_np_mod.float32),
        float(_np(shm_target_height)[0]),
    )
    policy(_dummy_obs)
    print("[real] Warmup done. Starting control loop.")

    last_action = _np_mod.zeros(12, dtype=_np_mod.float32)
    iters  = 0
    t_rate = time.time()

    try:
        while not stop_event.is_set():
            loop_start = time.time()

            pos_dict, vel_dict = read_joint_states()
            if pos_dict is None:
                time.sleep(DT)
                continue

            quat = imu.read_quaternion()
            _np(shm_imu_quat)[:] = quat.astype(np.float64)

            target_h = float(_np(shm_target_height)[0])
            obs = build_obs(pos_dict, vel_dict, quat, last_action, target_h)
            raw_action = policy(obs)

            # Write scaled action for visualiser (target ghost)
            scaled = raw_action * ACTION_SCALE
            _np(shm_q_target)[:] = scaled.astype(np.float64)

            # Write encoder readback (JOINT_NAMES order) for real ghost
            q_real = _np_mod.array(
                [pos_dict[j] for j in JOINT_ORDER], dtype=_np_mod.float64
            )
            _np(shm_q_real)[:] = q_real

            # Send to motors
            pos_targets = {j: float(scaled[i]) for i, j in enumerate(JOINT_ORDER)}
            write_joint_positions(pos_targets)

            last_action = raw_action

            iters += 1
            elapsed_rate = time.time() - t_rate
            if elapsed_rate >= 1.0:
                _np(shm_real_hz)[0] = iters / elapsed_rate
                iters  = 0
                t_rate = time.time()

            elapsed = time.time() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

    finally:
        if controller is not None:
            controller.disconnect()
        print("[real] Done.")


# ── Sim thread ─────────────────────────────────────────────────────────────────

def sim_thread_fn(
    model:              mujoco.MjModel,
    data_sim:           mujoco.MjData,
    policy,                              # GetupPolicy instance
    shm_target_height:  mp.Array,        # in: live target height in metres
    shm_q_sim:          mp.Array,        # out: MuJoCo qpos[7:19]
    shm_sim_hz:         mp.Array,        # out: loop rate
    stop_event:         threading.Event,
):
    """Runs policy on sim observations and steps MuJoCo at CONTROL_HZ."""
    n_substeps = max(1, round(DT / model.opt.timestep))
    last_action = np.zeros(12, dtype=np.float32)
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        loop_start = time.time()

        mujoco.mj_forward(model, data_sim)

        target_h = float(_np(shm_target_height)[0])
        obs      = build_obs_sim(data_sim, model, last_action, target_h)
        raw_action = policy(obs)                    # [12] unscaled

        # Motor position targets: offsets from sit-pose zero
        data_sim.ctrl[:] = raw_action * ACTION_SCALE

        for _ in range(n_substeps):
            mujoco.mj_step(model, data_sim)

        last_action = raw_action

        # Publish sim joint angles for viz thread
        _np(shm_q_sim)[:] = data_sim.qpos[7:19].astype(np.float64)

        iters += 1
        elapsed_rate = time.time() - t_rate
        if elapsed_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / elapsed_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - loop_start
        if elapsed < DT:
            time.sleep(DT - elapsed)


# ── Viz thread ─────────────────────────────────────────────────────────────────

def viz_thread_fn(
    sim_view:    object,
    target_view: object,
    real_view:   object,
    model:         mujoco.MjModel,
    data_sim:      mujoco.MjData,
    data_target:   mujoco.MjData,
    data_real:     mujoco.MjData,
    shm_q_sim:     mp.Array,
    shm_q_target:  mp.Array,
    shm_q_real:    mp.Array,
    stop_event:    threading.Event,
):
    """Updates viser at VIZ_HZ for all three robots using foot-centroid placement."""
    period   = 1.0 / VIZ_HZ
    foot_ids = [model.site(f).id for f in ["FL", "FR", "BL", "BR"]]

    while not stop_event.is_set():
        t = time.time()

        # ── Sim (grey) ─────────────────────────────────────────────────────────
        q_sim = _np(shm_q_sim).copy()
        data_sim.qpos[7:] = q_sim
        mujoco.mj_kinematics(model, data_sim)
        sim_view.update(data_sim)

        sim_foot_centroid = np.mean(
            [data_sim.site(sid).xpos for sid in foot_ids], axis=0
        )

        # ── Target (green) — policy action scaled ─────────────────────────────
        q_target = _np(shm_q_target).copy()
        _place_ghost_by_foot_centroid(model, data_target, q_target,
                                      sim_foot_centroid, foot_ids)
        target_view.update(data_target)

        # ── Real (blue) — encoder readback ────────────────────────────────────
        q_real = _np(shm_q_real).copy()
        _place_ghost_by_foot_centroid(model, data_real, q_real,
                                      sim_foot_centroid, foot_ids)
        real_view.update(data_real)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rumi getup validation")
    parser.add_argument("--checkpoint", default=str(_DEFAULT_CKPT),
                        help="Path to .pt checkpoint")
    parser.add_argument("--duration",   type=float, default=0.0,
                        help="Run duration in seconds (0 = unlimited)")
    parser.add_argument("--target",     type=float, default=0.25,
                        help="Target standing height in metres (default: 0.25)")
    parser.add_argument("--device",     type=str,   default=None,
                        help="Serial port for Dynamixel (auto-scans if omitted)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Do not send commands to motors (sim + viz only)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   Rumi Getup — Validation (Sim + Real)")
    print("=" * 60)
    print(f"  Checkpoint : {Path(args.checkpoint).name}")
    print(f"  Target ht  : {args.target:.3f} m")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Duration   : {'unlimited' if args.duration == 0 else f'{args.duration:.1f} s'}")
    print()

    # ── Shared memory ──────────────────────────────────────────────────────────
    shm_q_sim          = mp.Array(ctypes.c_double, N_JOINTS)   # sim qpos[7:]
    shm_q_target       = mp.Array(ctypes.c_double, N_JOINTS)   # policy action * scale
    shm_q_real         = mp.Array(ctypes.c_double, N_JOINTS)   # encoder readback
    shm_imu_quat       = mp.Array(ctypes.c_double, 4)          # (w,x,y,z)
    shm_sim_hz         = mp.Array(ctypes.c_double, 1)
    shm_real_hz        = mp.Array(ctypes.c_double, 1)
    shm_target_height  = mp.Array(ctypes.c_double, 1)          # live target height (m)

    # Initialise quaternion to identity so ghost renders sane before IMU data
    _np(shm_imu_quat)[:] = [1.0, 0.0, 0.0, 0.0]
    _np(shm_target_height)[0] = args.target

    connect_mp = mp.Event()
    stop_mp    = mp.Event()

    # ── Spawn real process ─────────────────────────────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(
            args.checkpoint,
            shm_target_height,
            shm_q_real,
            shm_q_target,
            shm_imu_quat,
            shm_real_hz,
            connect_mp,
            stop_mp,
            args.dry_run,
        ),
        daemon=True,
    )
    real_proc.start()

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    print("Loading MuJoCo model …")
    model       = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data_sim    = mujoco.MjData(model)
    data_target = mujoco.MjData(model)
    data_real   = mujoco.MjData(model)

    # Start from the sitting keyframe: all joints 0 rad, matches real sit-pose zero
    key_id = model.key("sitting").id
    mujoco.mj_resetDataKeyframe(model, data_sim,    key_id)
    mujoco.mj_resetDataKeyframe(model, data_target, key_id)
    mujoco.mj_resetDataKeyframe(model, data_real,   key_id)
    mujoco.mj_forward(model, data_sim)
    mujoco.mj_forward(model, data_target)
    mujoco.mj_forward(model, data_real)

    # Initialise shared arrays to keyframe joint positions
    q0 = data_sim.qpos[7:19].copy()
    _np(shm_q_sim)[:]    = q0
    _np(shm_q_target)[:] = q0
    _np(shm_q_real)[:]   = q0

    # ── Load policy (main process, for sim thread) ─────────────────────────────
    print("Loading policy …")
    from policy import GetupPolicy
    policy = GetupPolicy(args.checkpoint)
    print()

    # ── Sim thread ─────────────────────────────────────────────────────────────
    th_stop = threading.Event()
    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(model, data_sim, policy, shm_target_height,
              shm_q_sim, shm_sim_hz, th_stop),
    ).start()

    # ── Viser ─────────────────────────────────────────────────────────────────
    print("Starting viser …")
    server      = viser.ViserServer(label="Rumi Getup Validation")
    scene       = ViserMujocoScene.create(server, model)
    sim_view    = scene.add_robot("sim",    color=(0.75, 0.75, 0.75, 1.00))
    target_view = scene.add_robot("target", color=(0.20, 0.80, 0.20, 0.80))
    real_view   = scene.add_robot("real",   color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(
        camera_distance=1.2, camera_azimuth=135.0, camera_elevation=25.0
    )

    # ── GUI panels ─────────────────────────────────────────────────────────────
    with server.gui.add_folder("Validation"):
        chk_connect  = server.gui.add_checkbox("Connect robot", initial_value=False)
        txt_status   = server.gui.add_text("Robot status", initial_value="Disconnected")
        txt_sim_hz   = server.gui.add_text("Sim rate",     initial_value="— Hz")
        txt_real_hz  = server.gui.add_text("Real rate",    initial_value="— Hz")
        txt_imu_grav = server.gui.add_text("IMU gravity",  initial_value="—")

        sl_target_height = server.gui.add_slider(
            "Target height (m)", min=0.18, max=0.31, step=0.01,
            initial_value=args.target,
        )

    @sl_target_height.on_update
    def _on_target_height(_):
        _np(shm_target_height)[0] = sl_target_height.value

    @chk_connect.on_update
    def _on_connect_toggle(_):
        if chk_connect.value and not connect_mp.is_set():
            connect_mp.set()
            txt_status.value = "Connecting …"
        elif not chk_connect.value and connect_mp.is_set():
            stop_mp.set()
            txt_status.value = "Disconnected"

    # ── Viz thread ─────────────────────────────────────────────────────────────
    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(sim_view, target_view, real_view,
              model, data_sim, data_target, data_real,
              shm_q_sim, shm_q_target, shm_q_real, th_stop),
    ).start()

    # ── Main loop (60 Hz) ─────────────────────────────────────────────────────
    print("Running. Open http://localhost:8080 in a browser. Ctrl+C to stop.\n")
    start_time = time.time()

    try:
        while True:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[INFO] Duration {args.duration:.1f} s reached, stopping.")
                break

            # Update GUI status
            sim_hz  = _np(shm_sim_hz)[0]
            real_hz = _np(shm_real_hz)[0]
            txt_sim_hz.value  = f"{sim_hz:.0f} Hz"
            txt_real_hz.value = f"{real_hz:.0f} Hz"

            if connect_mp.is_set() and real_hz > 0:
                txt_status.value = f"Connected ({real_hz:.0f} Hz)"

            # Show projected gravity from IMU in GUI (sanity check)
            q = _np(shm_imu_quat).copy()
            w, x, y, z = q
            xyz = np.array([x, y, z])
            v   = np.array([0.0, 0.0, -1.0])
            t_  = 2.0 * np.cross(xyz, v)
            g   = v + w * t_ + np.cross(xyz, t_)
            txt_imu_grav.value = f"[{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]"

            time.sleep(1.0 / 60)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received.")

    finally:
        th_stop.set()
        stop_mp.set()
        real_proc.join(timeout=5.0)
        server.stop()
        print("[DONE]")


if __name__ == "__main__":
    main()
