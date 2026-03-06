"""Rumi quadruped sysid — joint group, MuJoCo sim + real motors in parallel.

All 12 joints are spawned in the sim and visualised. The selected joint defines
its part group (hip / thigh / calf). The test signal is sent to ALL joints in
that group with the appropriate sign convention (left=-1, right=+1). All other
joints (different part) are held at zero in both sim and real.

Sign convention (from basic_sitstand.py):
  Left  side (FL, BL): sign = -1
  Right side (FR, BR): sign = +1
  The selected joint's signal is always applied with sign +1; peers are scaled
  by their_sign / selected_sign so the motion is symmetric.

GUI limits and amplitude are based on the selected joint's MuJoCo limits.
Recording captures only the selected joint's state (q_real, dq_real, q_sim).

Signal modes (selectable via GUI):
  hold  — manual target via slider
  sine  — A·sin(2π·f·t) around centre
  step  — ±A square wave at period T
  chirp — linearly swept sine f0→f1 over sweep_t seconds

Recording:
  Press "Record" in the GUI to start/stop recording.
  Data is saved as data/<timestamp>_<joint>_<mode>_recording.npz with keys:
    t, target, q_real, dq_real, q_sim, dq_sim, tau_meas, control_hz, joint

Processes / threads:
  Main process:
    - Sim  thread  (control_hz): write ctrl for all group joints → mj_step
    - Viz  thread  (30 Hz):      copy qpos → mj_kinematics → update views
    - Main thread  (60 Hz):      signal gen → shared memory, plot refresh
  Real process (control_hz):
    - GroupSyncRead all 12 motors (position only) → shm_q_real[12]
    - read selected joint state (pos+vel+cur) → shm_dq_real, shm_tau_meas
    - GroupSyncWrite all 12 motors

Run:
  python rumi_sync.py --joint FL_calf_joint
  python rumi_sync.py --joint BL_calf_joint --control-hz 100
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import viser
import viser.uplot as uplot

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from viewer import ViserMujocoScene, ViserRobotView  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCENE_XML = _HERE / "scene.xml"

# ── Constants ──────────────────────────────────────────────────────────────────
PHYSICS_DT    = 0.004   # s
CONTROL_HZ    = 50
VIZ_HZ        = 30
PLOT_HZ       = 15
PLOT_WINDOW_S = 15.0

SIM_COLOR  = None
REAL_COLOR = (0.20, 0.55, 0.90, 0.60)

DEFAULT_JOINT = "BL_calf_joint"

# ── Joint info ─────────────────────────────────────────────────────────────────

def build_joint_info(mj_model: mujoco.MjModel) -> dict:
    """Return dict: joint_name → (qpos_id, qvel_id, ctrl_id, jnt_min, jnt_max).

    Only includes joints that have a corresponding actuator.
    """
    # build actuator name → ctrl_id  and  actuator joint name → ctrl_id
    act_joint_to_ctrl: dict[str, int] = {}
    for i in range(mj_model.nu):
        trnid = mj_model.actuator_trnid[i, 0]   # joint id this actuator drives
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        if jname:
            act_joint_to_ctrl[jname] = i

    info: dict[str, tuple[int, int, int, float, float]] = {}
    for jname, ctrl_id in act_joint_to_ctrl.items():
        jid     = mj_model.joint(jname).id
        qpos_id = int(mj_model.jnt_qposadr[jid])
        qvel_id = int(mj_model.jnt_dofadr[jid])
        jnt_min = float(mj_model.jnt_range[jid, 0])
        jnt_max = float(mj_model.jnt_range[jid, 1])
        info[jname] = (qpos_id, qvel_id, ctrl_id, jnt_min, jnt_max)

    return info


# ── Joint sign convention & group helpers ─────────────────────────────────────

# Left side = -1, Right side = +1  (from basic_sitstand.py convention)
_LOCATION_SIGN: dict[str, int] = {
    "FL": -1,
    "BL": -1,
    "BR": +1,
    "FR": +1,
}


def joint_sign(joint_name: str) -> int:
    """Return +1 or -1 for a joint based on its location prefix."""
    loc = joint_name.split("_")[0]   # "FL", "BL", "BR", "FR"
    return _LOCATION_SIGN[loc]


def joint_part(joint_name: str) -> str:
    """Return part string: 'hip', 'thigh', or 'calf'."""
    # e.g. "BL_calf_joint" → "calf"
    return joint_name.split("_")[1]


def build_group_sign_ratios(
    selected: str,
    joint_info: dict,
) -> dict[str, float]:
    """Return {joint_name: sign_ratio} for all joints in the same part group.

    sign_ratio = their_sign / selected_sign so the selected joint always moves
    exactly as commanded and peers mirror with correct relative sign.
    Non-group joints are not included (caller sends them zero).
    """
    part   = joint_part(selected)
    sel_sign = joint_sign(selected)
    return {
        jname: joint_sign(jname) / sel_sign
        for jname in joint_info
        if joint_part(jname) == part
    }


# ── Shared-memory helper ───────────────────────────────────────────────────────

def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── Real process ───────────────────────────────────────────────────────────────

def real_process_fn(
    selected_motor_id: int,
    group_motor_ids:   list[int],   # all motors in the part group
    group_sign_ratios: list[float], # sign_ratio per group motor (same order)
    all_motor_ids:     list[int],   # all 12 motor IDs (defines shm_q_real index order)
    control_hz:        int,
    shm_q_real:        mp.Array,    # 12-element: abs position per motor in all_motor_ids order
    shm_dq_real:       mp.Array,    # scalar: selected motor velocity (rad/s)
    shm_tau_meas:      mp.Array,    # scalar: selected motor torque (N·m)
    shm_target:        mp.Array,    # scalar: offset from initial position (rad)
    shm_init_pos:      mp.Array,    # 12-element: abs initial position per motor in all_motor_ids order
    shm_real_hz:       mp.Array,
    stop_event:        mp.Event,
) -> None:
    """Hardware control loop — drives all group motors, holds others at init.

    Each loop:
      1. GroupSyncRead  → read all 12 motor positions in one USB packet → shm_q_real
      2. Individual read of selected motor (vel + current) → shm_dq_real, shm_tau_meas
      3. GroupSyncWrite → send goal positions to all 12 motors in one USB packet
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from hardware_rumi import (  # noqa: PLC0415
        MX64Hardware, DEFAULT_DEVICE, DEFAULT_BAUDRATE, _RAD_PER_TICK,
        _ADDR_PRESENT_POSITION, _to_signed32,
    )
    from dynamixel_sdk import (  # noqa: PLC0415
        PacketHandler, PortHandler, GroupSyncWrite, GroupSyncRead, COMM_SUCCESS,
    )

    ADDR_GOAL_POSITION  = 116
    LEN_GOAL_POSITION   = 4
    ADDR_PRESENT_POS    = _ADDR_PRESENT_POSITION   # 132
    LEN_PRESENT_POS     = 4

    dt = 1.0 / control_hz

    # Open a single port for all motors
    port = PortHandler(DEFAULT_DEVICE)
    ph   = PacketHandler(2.0)
    if not port.openPort():
        print(f"[real] Cannot open port {DEFAULT_DEVICE}")
        return
    if not port.setBaudRate(DEFAULT_BAUDRATE):
        print(f"[real] Cannot set baudrate {DEFAULT_BAUDRATE}")
        port.closePort()
        return

    # Enable torque on all motors using individual MX64Hardware instances
    # (mode set + PID gains written per motor)
    hws: dict[int, MX64Hardware] = {}
    for mid in all_motor_ids:
        hw = MX64Hardware(motor_id=mid)
        hw._port = port
        hw._ph   = ph
        hw.enable()
        hws[mid] = hw

    # Read initial positions for all motors (hold positions for non-group)
    print("[real] Reading initial positions…")
    init_pos: dict[int, float] = {}
    for mid, hw in hws.items():
        state = hw.read_state()
        if state is not None:
            init_pos[mid] = state.pos_rad
        else:
            init_pos[mid] = 0.0
            print(f"[real] Warning: could not read motor {mid}, defaulting to 0.0")

    # Populate shm_init_pos and shm_q_real (both indexed by all_motor_ids order).
    # shm_target is an OFFSET from sel_init (0.0 at startup).
    q_real_np    = _np(shm_q_real)
    init_pos_np  = _np(shm_init_pos)
    for i, mid in enumerate(all_motor_ids):
        q_real_np[i]   = init_pos[mid]
        init_pos_np[i] = init_pos[mid]

    _np(shm_dq_real)[0]  = 0.0
    _np(shm_target)[0]   = 0.0   # offset = 0 at startup

    sel_init = init_pos[selected_motor_id]
    sel_idx  = all_motor_ids.index(selected_motor_id)
    print(f"[real] Selected motor {selected_motor_id} (idx {sel_idx}) initial pos: "
          f"{np.rad2deg(sel_init):.1f} deg — target offset seeded to 0.")
    print(f"[real] Group motors: {group_motor_ids}  "
          f"sign_ratios: {group_sign_ratios}")

    # GroupSyncWrite for all 12 motors (goal position)
    gsw = GroupSyncWrite(port, ph, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def _write_all(target_signal: float) -> None:
        if not np.isfinite(target_signal):
            target_signal = 0.0
        gsw.clearParam()
        for mid in all_motor_ids:
            if mid in group_motor_ids:
                idx = group_motor_ids.index(mid)
                pos_rad = init_pos[mid] + target_signal * group_sign_ratios[idx]
            else:
                pos_rad = init_pos[mid]   # hold at initial position
            raw = int(round(pos_rad / _RAD_PER_TICK))
            raw = int(np.clip(raw, -(2**31 - 1), 2**31 - 1))
            if raw < 0:
                raw += 2**32
            param = [
                (raw >> 0)  & 0xFF,
                (raw >> 8)  & 0xFF,
                (raw >> 16) & 0xFF,
                (raw >> 24) & 0xFF,
            ]
            gsw.addParam(mid, param)
        gsw.txPacket()

    # GroupSyncRead for all 12 motors (present position only — 4 bytes @ addr 132)
    gsr = GroupSyncRead(port, ph, ADDR_PRESENT_POS, LEN_PRESENT_POS)
    for mid in all_motor_ids:
        gsr.addParam(mid)

    def _read_all_positions() -> None:
        """GroupSyncRead all motor positions → shm_q_real (one USB round-trip)."""
        result = gsr.txRxPacket()
        if result != COMM_SUCCESS:
            return  # keep stale values on comm error
        for i, mid in enumerate(all_motor_ids):
            if gsr.isAvailable(mid, ADDR_PRESENT_POS, LEN_PRESENT_POS):
                raw = gsr.getData(mid, ADDR_PRESENT_POS, LEN_PRESENT_POS)
                q_real_np[i] = _to_signed32(raw) * _RAD_PER_TICK

    print("[real] Starting control loop.")
    iters  = 0
    t_rate = time.time()

    try:
        while not stop_event.is_set():
            t = time.time()

            # 1. Read all motor positions in one packet
            _read_all_positions()

            # 2. Read selected motor velocity + current (individual read for scalar shm)
            state = hws[selected_motor_id].read_state()
            if state is not None:
                _np(shm_dq_real)[0]  = state.vel_rads
                _np(shm_tau_meas)[0] = state.torque_nm

            # 3. Write all motors (shm_target is offset from sel_init)
            _write_all(_np(shm_target)[0])

            iters += 1
            dt_rate = time.time() - t_rate
            if dt_rate >= 1.0:
                _np(shm_real_hz)[0] = iters / dt_rate
                iters  = 0
                t_rate = time.time()

            elapsed = time.time() - t
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        # Disable torque on all motors
        for hw in hws.values():
            try:
                hw._torque_enable(False)
            except Exception:
                pass
        port.closePort()
        print("[real] Done.")


# ── Sim thread ─────────────────────────────────────────────────────────────────

def sim_thread_fn(
    mj_model:          mujoco.MjModel,
    mj_data:           mujoco.MjData,
    qpos_id:           int,
    qvel_id:           int,
    group_ctrl_ids:    list[int],    # ctrl indices for all group joints
    group_sign_ratios: list[float],  # sign_ratio per group joint (same order)
    shm_target:        mp.Array,
    shm_sim_hz:        mp.Array,
    shm_q_sim:         mp.Array,
    shm_dq_sim:        mp.Array,
    data_lock:         threading.Lock,
    stop_event:        threading.Event,
    sim_substeps:      int,
    control_hz:        int,
) -> None:
    dt     = 1.0 / control_hz
    iters  = 0
    t_rate = time.time()

    while not stop_event.is_set():
        t = time.time()

        target = _np(shm_target)[0]

        with data_lock:
            _np(shm_q_sim)[0]  = float(mj_data.qpos[qpos_id])
            _np(shm_dq_sim)[0] = float(mj_data.qvel[qvel_id])
            # Write all group joints with their sign ratios; non-group ctrl stays 0
            for ctrl_id, ratio in zip(group_ctrl_ids, group_sign_ratios):
                mj_data.ctrl[ctrl_id] = target * ratio
            for _ in range(sim_substeps):
                mujoco.mj_step(mj_model, mj_data)

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Viz thread ─────────────────────────────────────────────────────────────────

def viz_thread_fn(
    sim_view:        ViserRobotView,
    real_view:       ViserRobotView,
    mj_model:        mujoco.MjModel,
    mj_data:         mujoco.MjData,
    joint_info:      dict,        # joint_name → (qpos_id, qvel_id, ctrl_id, min, max)
    all_motor_ids:   list[int],   # motor IDs in shm_q_real index order
    motor_id_to_joint: dict[int, str],  # motor_id → joint_name
    shm_q_real:      mp.Array,    # 12-element: abs position per motor
    shm_init_pos:    mp.Array,    # 12-element: abs initial position per motor
    data_lock:       threading.Lock,
    stop_event:      threading.Event,
) -> None:
    """Update sim and real ghost views at VIZ_HZ.

    Real ghost: for each motor, set qpos[qpos_id] = q_real[i] - init_pos[i]
    so the ghost shows joint angles relative to the model's zero position.
    All 12 joints are synced from hardware in one go.
    """
    sim_viz  = mujoco.MjData(mj_model)
    real_viz = mujoco.MjData(mj_model)
    period   = 1.0 / VIZ_HZ

    # Build lookup: motor index (in all_motor_ids) → qpos_id  (only actuated joints)
    motor_idx_to_qpos: list[int | None] = []
    for mid in all_motor_ids:
        jname = motor_id_to_joint.get(mid)
        if jname is not None and jname in joint_info:
            motor_idx_to_qpos.append(joint_info[jname][0])  # qpos_id
        else:
            motor_idx_to_qpos.append(None)

    while not stop_event.is_set():
        t = time.time()

        with data_lock:
            sim_viz.qpos[:] = mj_data.qpos
        mujoco.mj_kinematics(mj_model, sim_viz)
        sim_view.update(sim_viz)

        # Real ghost: start from sim qpos (free-joint orientation etc.), then
        # overwrite each actuated joint from hardware reading (offset from init).
        q_real_np   = _np(shm_q_real)
        init_pos_np = _np(shm_init_pos)
        real_viz.qpos[:] = sim_viz.qpos
        for i, qpos_id in enumerate(motor_idx_to_qpos):
            if qpos_id is not None:
                real_viz.qpos[qpos_id] = q_real_np[i] - init_pos_np[i]
        mujoco.mj_kinematics(mj_model, real_viz)
        real_view.update(real_viz)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Test-signal generator ──────────────────────────────────────────────────────

class SignalGen:
    """Generates position targets in radians, clamped to joint limits."""

    def __init__(self, jnt_min: float, jnt_max: float) -> None:
        self.jnt_min   = jnt_min
        self.jnt_max   = jnt_max
        self.mode      = "hold"
        self.amplitude = min(0.3, (jnt_max - jnt_min) / 4.0)
        self.frequency = 0.5
        self.chirp_f1  = 5.0
        self.chirp_t   = 10.0
        self.step_T    = 2.0
        self.centre    = float(np.clip(0.0, jnt_min, jnt_max))
        self._t0: float | None = None

    def reset(self) -> None:
        self._t0 = None

    def __call__(self, t_wall: float) -> float:
        if self._t0 is None:
            self._t0 = t_wall
        t = t_wall - self._t0
        A = self.amplitude
        c = self.centre

        if self.mode == "sine":
            raw = c + A * np.sin(2.0 * np.pi * self.frequency * t)
        elif self.mode == "step":
            phase = (t % self.step_T) / self.step_T
            raw = c + (A if phase < 0.5 else -A)
        elif self.mode == "chirp":
            f0, f1, T = self.frequency, self.chirp_f1, self.chirp_t
            t_mod  = t % T
            inst_f = f0 + (f1 - f0) * t_mod / T
            raw = c + A * np.sin(2.0 * np.pi * inst_f * t_mod)
        else:
            raw = c   # hold

        return float(np.clip(raw, self.jnt_min, self.jnt_max))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rumi sysid: joint group, sim + real in parallel."
    )
    parser.add_argument(
        "--joint", type=str, default=DEFAULT_JOINT,
        help=f"Joint name to actuate (default: {DEFAULT_JOINT}).",
    )
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    args = parser.parse_args()

    # ── MuJoCo model (needed before real process for joint info) ───────────────
    mj_model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    mj_model.opt.timestep = PHYSICS_DT
    mj_data  = mujoco.MjData(mj_model)

    joint_info = build_joint_info(mj_model)
    if args.joint not in joint_info:
        print(f"[error] Unknown joint '{args.joint}'. Available: {list(joint_info)}")
        sys.exit(1)

    qpos_id, qvel_id, _, jnt_min, jnt_max = joint_info[args.joint]

    from hardware_rumi import JOINT_TO_MOTOR_ID  # noqa: PLC0415
    motor_id = JOINT_TO_MOTOR_ID[args.joint]

    # ── Build group: all joints of the same part (hip/thigh/calf) ─────────────
    group_sign_map = build_group_sign_ratios(args.joint, joint_info)
    # ordered lists for passing to threads/process
    group_joints      = list(group_sign_map.keys())
    group_sign_ratios = [group_sign_map[j] for j in group_joints]
    group_ctrl_ids    = [joint_info[j][2] for j in group_joints]
    group_motor_ids   = [JOINT_TO_MOTOR_ID[j] for j in group_joints]
    all_motor_ids     = list(JOINT_TO_MOTOR_ID.values())

    sim_substeps = max(1, round(1.0 / (args.control_hz * PHYSICS_DT)))
    print(f"Joint: {args.joint}  motor_id={motor_id}  "
          f"limits=[{np.rad2deg(jnt_min):.1f}, {np.rad2deg(jnt_max):.1f}] deg  "
          f"control_hz={args.control_hz}  sim_substeps={sim_substeps}")
    print(f"Group ({joint_part(args.joint)}): "
          + "  ".join(f"{j}(×{r:+.0f})" for j, r in zip(group_joints, group_sign_ratios)))

    # Apply "zero" keyframe so the freejoint starts in the inverted orientation
    # that matches the physical robot on the desk.
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "zero")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mj_data.qpos[:] = 0.0
        mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # ── Shared memory ──────────────────────────────────────────────────────────
    n_motors     = len(all_motor_ids)
    shm_target   = mp.Array(ctypes.c_double, 1)
    shm_q_real   = mp.Array(ctypes.c_double, n_motors)  # abs pos per motor
    shm_dq_real  = mp.Array(ctypes.c_double, 1)         # selected motor velocity
    shm_tau_meas = mp.Array(ctypes.c_double, 1)         # selected motor torque
    shm_real_hz  = mp.Array(ctypes.c_double, 1)
    shm_sim_hz   = mp.Array(ctypes.c_double, 1)
    shm_q_sim    = mp.Array(ctypes.c_double, 1)
    shm_dq_sim   = mp.Array(ctypes.c_double, 1)
    shm_init_pos = mp.Array(ctypes.c_double, n_motors)  # abs initial pos per motor

    _np(shm_target)[0]  = 0.0
    _np(shm_q_real)[:] = 0.0
    _np(shm_q_sim)[0]   = 0.0
    _np(shm_init_pos)[:] = 0.0

    # Index of selected motor in all_motor_ids (for scalar reads in main loop)
    sel_idx = all_motor_ids.index(motor_id)

    stop_event = mp.Event()

    # ── Spawn real process BEFORE GPU / CUDA init ──────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(
            motor_id, group_motor_ids, group_sign_ratios, all_motor_ids,
            args.control_hz,
            shm_q_real, shm_dq_real, shm_tau_meas,
            shm_target, shm_init_pos, shm_real_hz,
            stop_event,
        ),
        daemon=True,
    )
    real_proc.start()

    # ── Viser ──────────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label=f"Rumi Sysid — {args.joint}")
    scene     = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=SIM_COLOR)
    real_view = scene.add_robot("real", color=REAL_COLOR)
    scene.create_visualization_gui(
        camera_distance=1.5,
        camera_azimuth=45.0,
        camera_elevation=-20.0,  # negative: look from below to match inverted robot
    )

    # ── Status ─────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Status"):
        txt_joint   = server.gui.add_text("Active joint", initial_value=args.joint)
        txt_motor   = server.gui.add_text("Motor ID",     initial_value=str(motor_id))
        txt_group   = server.gui.add_text(
            "Group",
            initial_value="  ".join(
                f"{j}(×{r:+.0f})" for j, r in zip(group_joints, group_sign_ratios)
            ),
        )
        txt_limits  = server.gui.add_text(
            "Limits (deg)",
            initial_value=f"[{np.rad2deg(jnt_min):.1f}, {np.rad2deg(jnt_max):.1f}]"
        )
        txt_sim_hz  = server.gui.add_text("Sim Hz",  initial_value="— Hz")
        txt_real_hz = server.gui.add_text("Real Hz", initial_value="— Hz")

    # ── Signal GUI ─────────────────────────────────────────────────────────────
    sig = SignalGen(jnt_min, jnt_max)

    _amp_max_deg  = float(np.rad2deg((jnt_max - jnt_min) / 2.0))
    _cen_min_deg  = float(np.rad2deg(jnt_min))
    _cen_max_deg  = float(np.rad2deg(jnt_max))

    with server.gui.add_folder("Test Signal"):
        dd_mode = server.gui.add_dropdown(
            "Mode",
            options=["hold", "sine", "step", "chirp"],
            initial_value="hold",
        )
        sl_centre = server.gui.add_slider(
            "Centre (deg)",
            min=_cen_min_deg, max=_cen_max_deg, step=0.5,
            initial_value=float(np.rad2deg(sig.centre)),
        )
        sl_amp = server.gui.add_slider(
            "Amplitude (deg)",
            min=0.0, max=_amp_max_deg, step=0.5,
            initial_value=float(np.rad2deg(sig.amplitude)),
        )
        sl_freq = server.gui.add_slider(
            "Frequency (Hz)", min=0.05, max=10.0, step=0.05, initial_value=0.5
        )
        sl_step_T = server.gui.add_slider(
            "Step period (s)", min=0.2, max=10.0, step=0.1, initial_value=2.0
        )
        sl_chirp_f1 = server.gui.add_slider(
            "Chirp f1 (Hz)", min=0.1, max=20.0, step=0.1, initial_value=5.0
        )
        sl_chirp_t = server.gui.add_slider(
            "Chirp sweep (s)", min=1.0, max=30.0, step=0.5, initial_value=10.0
        )
        btn_reset = server.gui.add_button("Reset signal phase")

        @btn_reset.on_click
        def _(_) -> None:
            sig.reset()

    with server.gui.add_folder("Manual (hold mode)"):
        sl_target = server.gui.add_slider(
            "Target (deg)",
            min=_cen_min_deg, max=_cen_max_deg, step=0.5,
            initial_value=0.0,
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    _buf = int(PLOT_WINDOW_S * 60)
    _t_buf        = deque(maxlen=_buf)
    _q_sim_buf    = deque(maxlen=_buf)
    _q_real_buf   = deque(maxlen=_buf)
    _q_des_buf    = deque(maxlen=_buf)
    _dq_sim_buf   = deque(maxlen=_buf)
    _dq_real_buf  = deque(maxlen=_buf)
    _tau_meas_buf = deque(maxlen=_buf)

    def _s(label: str, color: str, dash: bool = False) -> uplot.Series:
        return uplot.Series(label=label, stroke=color, width=1.5,
                            **({"dash": (6.0, 3.0)} if dash else {}))

    with server.gui.add_folder("Plots", expand_by_default=True):
        plt_q = server.gui.add_uplot(
            tuple(np.zeros(2) for _ in range(4)),
            (uplot.Series(label="t"),
             _s("des",  "#fa0"),
             _s("sim",  "#4af"),
             _s("real", "#f55", dash=True)),
            title="Position (deg)", aspect=2.0,
            axes=(uplot.Axis(), uplot.Axis(label="deg")),
        )
        plt_dq = server.gui.add_uplot(
            tuple(np.zeros(2) for _ in range(3)),
            (uplot.Series(label="t"),
             _s("sim",  "#4af"),
             _s("real", "#f55", dash=True)),
            title="Velocity (deg/s)", aspect=2.0,
            axes=(uplot.Axis(), uplot.Axis(label="deg/s")),
        )
        plt_tau_meas = server.gui.add_uplot(
            (np.zeros(2), np.zeros(2)),
            (uplot.Series(label="t"), _s("tau_meas", "#4d4")),
            title="Measured Torque (N·m)", aspect=2.0,
            axes=(uplot.Axis(), uplot.Axis(label="N·m")),
        )

    # ── Recording GUI ──────────────────────────────────────────────────────────
    _rec_lock     = threading.Lock()
    _rec_active   = False
    _rec_t:       list[float] = []
    _rec_target:  list[float] = []
    _rec_q_real:  list[float] = []
    _rec_dq_real: list[float] = []
    _rec_q_sim:   list[float] = []
    _rec_dq_sim:  list[float] = []
    _rec_tau:     list[float] = []

    _data_dir = _HERE / "data"
    _data_dir.mkdir(exist_ok=True)

    with server.gui.add_folder("Recording"):
        txt_rec_status = server.gui.add_text("Status", initial_value="Idle")
        btn_rec = server.gui.add_button("Start Recording", color="green")

        @btn_rec.on_click
        def _(_) -> None:
            nonlocal _rec_active
            with _rec_lock:
                if not _rec_active:
                    _rec_active = True
                    _rec_t.clear(); _rec_target.clear()
                    _rec_q_real.clear(); _rec_dq_real.clear()
                    _rec_q_sim.clear(); _rec_dq_sim.clear()
                    _rec_tau.clear()
                    btn_rec.label = "Stop Recording"
                    btn_rec.color = "red"
                    txt_rec_status.value = "Recording…"
                    print("[rec] Started recording.")
                else:
                    _rec_active = False
                    btn_rec.label = "Start Recording"
                    btn_rec.color = "green"
                    n = len(_rec_t)
                    if n > 0:
                        # name: stamp_joint_mode_recording.npz
                        safe_joint = args.joint.replace("_joint", "")
                        import datetime as _dt
                        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = (_data_dir /
                                 f"{_ts}_{safe_joint}_{sig.mode}_recording.npz")
                        from hardware_rumi import DEFAULT_KP, DEFAULT_KD  # noqa: PLC0415
                        _kp_sim = float(mj_model.actuator_gainprm[0, 0])
                        _kd_sim = float(-mj_model.actuator_biasprm[0, 2])
                        np.savez(
                            fname,
                            t          = np.array(_rec_t,       dtype=np.float64),
                            target     = np.array(_rec_target,   dtype=np.float64),
                            q_real     = np.array(_rec_q_real,   dtype=np.float64),
                            dq_real    = np.array(_rec_dq_real,  dtype=np.float64),
                            q_sim      = np.array(_rec_q_sim,    dtype=np.float64),
                            dq_sim     = np.array(_rec_dq_sim,   dtype=np.float64),
                            tau_meas   = np.array(_rec_tau,      dtype=np.float64),
                            control_hz = np.array([args.control_hz], dtype=np.float64),
                            joint      = np.array([args.joint]),
                            kp_sim     = np.array([_kp_sim]),
                            kd_sim     = np.array([_kd_sim]),
                            kp_real    = np.array([DEFAULT_KP]),
                            kd_real    = np.array([DEFAULT_KD]),
                        )
                        print(f"[rec] Saved {n} samples → {fname}")
                        txt_rec_status.value = f"Saved {n} pts → {fname.name}"
                    else:
                        txt_rec_status.value = "Idle (no data)"

    # ── Threads ────────────────────────────────────────────────────────────────
    th_stop   = threading.Event()
    data_lock = threading.Lock()

    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(
            mj_model, mj_data,
            qpos_id, qvel_id,
            group_ctrl_ids, group_sign_ratios,
            shm_target, shm_sim_hz,
            shm_q_sim, shm_dq_sim,
            data_lock, th_stop, sim_substeps, args.control_hz,
        ),
    ).start()

    from hardware_rumi import MOTOR_ID_TO_JOINT as _MOTOR_ID_TO_JOINT  # noqa: PLC0415
    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(
            sim_view, real_view,
            mj_model, mj_data,
            joint_info, all_motor_ids, _MOTOR_ID_TO_JOINT,
            shm_q_real, shm_init_pos,
            data_lock, th_stop,
        ),
    ).start()

    # ── Wait for real process to connect ───────────────────────────────────────
    # shm_target is an offset from the motor's initial position.
    # GUI sliders and SignalGen all operate in offset-space (0 = initial position).
    print("Waiting for real process to connect…")
    time.sleep(2.0)
    print("[main] Real process connected. offset-space: 0 = initial motor position.")
    print("Running. Ctrl+C to stop.")

    _plot_skip = max(1, round(60 / PLOT_HZ))
    _loop_iter = 0
    _t0        = time.time()

    try:
        while True:
            t_wall = time.time()

            # update signal params from GUI
            sig.mode      = dd_mode.value
            sig.amplitude = float(np.clip(
                np.deg2rad(sl_amp.value), 0.0, (jnt_max - jnt_min) / 2.0
            ))
            sig.frequency = float(sl_freq.value)
            sig.centre    = float(np.clip(np.deg2rad(sl_centre.value), jnt_min, jnt_max))
            sig.step_T    = float(sl_step_T.value)
            sig.chirp_f1  = float(sl_chirp_f1.value)
            sig.chirp_t   = float(sl_chirp_t.value)

            # generate target
            if sig.mode == "hold":
                target_rad = float(np.clip(np.deg2rad(sl_target.value), jnt_min, jnt_max))
            else:
                target_rad = sig(t_wall)

            _np(shm_target)[0] = target_rad

            # status
            txt_sim_hz.value  = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            txt_real_hz.value = f"{_np(shm_real_hz)[0]:.0f} Hz"

            # read shared values once
            now       = t_wall - _t0
            q_sim_r   = float(_np(shm_q_sim)[0])
            q_real_r  = float(_np(shm_q_real)[sel_idx]) - float(_np(shm_init_pos)[sel_idx])
            dq_sim_r  = float(_np(shm_dq_sim)[0])
            dq_real_r = float(_np(shm_dq_real)[0])
            tau_r     = float(_np(shm_tau_meas)[0])

            # plot buffers (degrees)
            _t_buf.append(now)
            _q_des_buf.append(float(np.rad2deg(target_rad)))
            _q_sim_buf.append(float(np.rad2deg(q_sim_r)))
            _q_real_buf.append(float(np.rad2deg(q_real_r)))
            _dq_sim_buf.append(float(np.rad2deg(dq_sim_r)))
            _dq_real_buf.append(float(np.rad2deg(dq_real_r)))
            _tau_meas_buf.append(tau_r)

            # recording (raw radians)
            if _rec_active:
                with _rec_lock:
                    if _rec_active:
                        _rec_t.append(now)
                        _rec_target.append(target_rad)
                        _rec_q_real.append(q_real_r)
                        _rec_dq_real.append(dq_real_r)
                        _rec_q_sim.append(q_sim_r)
                        _rec_dq_sim.append(dq_sim_r)
                        _rec_tau.append(tau_r)
                        txt_rec_status.value = f"Recording… {len(_rec_t)} pts"

            # plot refresh
            _loop_iter += 1
            if _loop_iter % _plot_skip == 0 and len(_t_buf) > 1:
                t_arr = np.array(_t_buf, dtype=np.float64)
                plt_q.data        = (t_arr,
                                     np.array(_q_des_buf),
                                     np.array(_q_sim_buf),
                                     np.array(_q_real_buf))
                plt_dq.data       = (t_arr,
                                     np.array(_dq_sim_buf),
                                     np.array(_dq_real_buf))
                plt_tau_meas.data = (t_arr, np.array(_tau_meas_buf))

            time.sleep(1.0 / 60)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        th_stop.set()
        stop_event.set()
        real_proc.join(timeout=10.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
