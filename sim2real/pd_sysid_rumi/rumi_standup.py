"""Rumi quadruped standup — full-robot coordinated sim + real in parallel.

Unlike rumi_sync.py (single joint group, scalar target), this script drives
ALL 12 joints simultaneously with per-joint targets, enabling coordinated
standup/sitdown motions that require thigh + calf to move together.

shm_target is a 12-element array (one entry per motor in all_motor_ids order).
The signal generator writes all 12 targets each frame; sim thread and real
process both consume the full array.

Sign convention (from basic_sitstand.py):
  Left  side (FL, BL): sign = -1
  Right side (FR, BR): sign = +1

Motion groups:
  hip   — motors 1,4,7,10  (FL/BL/BR/FR hip)
  thigh — motors 2,5,8,11  (FL/BL/BR/FR thigh)
  calf  — motors 3,6,9,12  (FL/BL/BR/FR calf)

Each group has an independent amplitude. The signal is applied as:
  target[joint] = amplitude_for_group * sign(joint)

Signal modes:
  hold     — manual per-group sliders, robot stays wherever set
  standup  — execute one coordinated ramp: 0 → amp → 0  (sit-stand-sit)
  sine     — all groups oscillate sinusoidally (good for sysid)
  step     — all groups do a square wave

Recording:
  Press "Record" to start/stop. Saves data/<timestamp>_standup_<mode>.npz:
    t          : (N,)    wall-clock time (s)
    target     : (N, 12) commanded joint positions (rad, offset-space)
    q_real     : (N, 12) measured joint positions  (rad, offset-space)
    dq_real    : (N, 12) measured joint velocities (rad/s)
    tau_meas   : (N, 12) measured joint torques    (N·m)
    q_sim      : (N, 12) simulated joint positions (rad)
    dq_sim     : (N, 12) simulated joint velocities(rad/s)
    control_hz : (1,)
    joint_names: (12,)   string names in all_motor_ids order

Processes / threads:
  Main process:
    - Sim  thread  (control_hz): write ctrl[12] → mj_step
    - Viz  thread  (30 Hz):      copy qpos → mj_kinematics → update views
    - Main thread  (60 Hz):      signal gen → shm_target[12], plot refresh
  Real process (control_hz):
    - GroupSyncRead  all 12 (pos+vel+cur, 10-byte block) → shm arrays
    - GroupSyncWrite all 12 goal positions

Run:
  python rumi_standup.py
  python rumi_standup.py --control-hz 100
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

# ── Paths ───────────────────────────────────────────────────────────────────────
_SCENE_XML = _HERE / "scene.xml"

# ── Constants ───────────────────────────────────────────────────────────────────
PHYSICS_DT    = 0.004   # s
CONTROL_HZ    = 50
VIZ_HZ        = 30
PLOT_HZ       = 15
PLOT_WINDOW_S = 15.0

SIM_COLOR  = None
REAL_COLOR = (0.20, 0.55, 0.90, 0.60)

# ── Joint sign convention ───────────────────────────────────────────────────────
_LOCATION_SIGN: dict[str, int] = {
    "FL": -1, "BL": -1,
    "BR": +1, "FR": +1,
}


def joint_sign(joint_name: str) -> int:
    return _LOCATION_SIGN[joint_name.split("_")[0]]


def joint_part(joint_name: str) -> str:
    return joint_name.split("_")[1]   # "hip" | "thigh" | "calf"


# ── Joint info ──────────────────────────────────────────────────────────────────

def build_joint_info(mj_model: mujoco.MjModel) -> dict:
    """joint_name → (qpos_id, qvel_id, ctrl_id, jnt_min, jnt_max)."""
    act_joint_to_ctrl: dict[str, int] = {}
    for i in range(mj_model.nu):
        trnid = mj_model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        if jname:
            act_joint_to_ctrl[jname] = i

    info: dict[str, tuple] = {}
    for jname, ctrl_id in act_joint_to_ctrl.items():
        jid     = mj_model.joint(jname).id
        qpos_id = int(mj_model.jnt_qposadr[jid])
        qvel_id = int(mj_model.jnt_dofadr[jid])
        jnt_min = float(mj_model.jnt_range[jid, 0])
        jnt_max = float(mj_model.jnt_range[jid, 1])
        info[jname] = (qpos_id, qvel_id, ctrl_id, jnt_min, jnt_max)
    return info


# ── Shared-memory helper ────────────────────────────────────────────────────────

def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── Real process ────────────────────────────────────────────────────────────────

def real_process_fn(
    all_motor_ids:  list[int],   # 12 motor IDs — defines index order in all shm arrays
    control_hz:     int,
    shm_target:     mp.Array,    # (12,) offset from init per motor (rad)
    shm_q_real:     mp.Array,    # (12,) abs present position per motor (rad)
    shm_dq_real:    mp.Array,    # (12,) present velocity per motor (rad/s)
    shm_tau_meas:   mp.Array,    # (12,) present torque per motor (N·m)
    shm_init_pos:   mp.Array,    # (12,) abs initial position per motor (rad)
    shm_real_hz:    mp.Array,
    stop_event:     mp.Event,
) -> None:
    """Hardware control loop.

    Each iteration:
      1. GroupSyncRead  all 12 — 10-byte block (cur+vel+pos) in one USB packet
      2. GroupSyncWrite all 12 — goal positions from shm_target + init_pos
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from hardware_rumi import (  # noqa: PLC0415
        MX64Hardware, DEFAULT_DEVICE, DEFAULT_BAUDRATE,
        _RAD_PER_TICK, _RADS_PER_LSB, _KT_PER_LSB,
        _READ_BASE, _READ_LEN,
        _ADDR_PRESENT_POSITION,
        _to_signed32, _to_signed16,
    )
    from dynamixel_sdk import (  # noqa: PLC0415
        PacketHandler, PortHandler,
        GroupSyncWrite, GroupSyncRead,
        COMM_SUCCESS,
    )

    ADDR_GOAL   = 116
    LEN_GOAL    = 4
    ADDR_READ   = _READ_BASE          # 126 — cur(2) + vel(4) + pos(4) = 10 bytes
    LEN_READ    = _READ_LEN           # 10
    ADDR_POS    = _ADDR_PRESENT_POSITION  # 132

    dt = 1.0 / control_hz

    port = PortHandler(DEFAULT_DEVICE)
    ph   = PacketHandler(2.0)
    if not port.openPort():
        print(f"[real] Cannot open port {DEFAULT_DEVICE}")
        return
    if not port.setBaudRate(DEFAULT_BAUDRATE):
        print(f"[real] Cannot set baudrate {DEFAULT_BAUDRATE}")
        port.closePort()
        return

    # Enable all motors
    hws: dict[int, MX64Hardware] = {}
    for mid in all_motor_ids:
        hw = MX64Hardware(motor_id=mid)
        hw._port = port
        hw._ph   = ph
        hw.enable()
        hws[mid] = hw

    # Read initial positions
    print("[real] Reading initial positions…")
    init_pos: dict[int, float] = {}
    for mid, hw in hws.items():
        state = hw.read_state()
        if state is not None:
            init_pos[mid] = state.pos_rad
        else:
            init_pos[mid] = 0.0
            print(f"[real] Warning: motor {mid} read failed, defaulting to 0.0")

    # Write shm_init_pos and seed shm_q_real
    q_np    = _np(shm_q_real)
    ip_np   = _np(shm_init_pos)
    tgt_np  = _np(shm_target)
    dq_np   = _np(shm_dq_real)
    tau_np  = _np(shm_tau_meas)
    for i, mid in enumerate(all_motor_ids):
        ip_np[i]  = init_pos[mid]
        q_np[i]   = init_pos[mid]
        tgt_np[i] = 0.0   # target offset = 0 at startup
        dq_np[i]  = 0.0
        tau_np[i] = 0.0

    for mid in all_motor_ids:
        print(f"[real]   motor {mid:2d}: init {np.rad2deg(init_pos[mid]):+.1f} deg")

    # GroupSyncRead — full 10-byte block per motor (cur + vel + pos)
    gsr = GroupSyncRead(port, ph, ADDR_READ, LEN_READ)
    for mid in all_motor_ids:
        gsr.addParam(mid)

    def _read_all() -> None:
        result = gsr.txRxPacket()
        if result != COMM_SUCCESS:
            return
        for i, mid in enumerate(all_motor_ids):
            if not gsr.isAvailable(mid, ADDR_READ, LEN_READ):
                continue
            # current (2 bytes at offset 0 from ADDR_READ=126)
            cur_raw = gsr.getData(mid, ADDR_READ, 2)
            # velocity (4 bytes at offset 2)
            vel_raw = gsr.getData(mid, ADDR_READ + 2, 4)
            # position (4 bytes at offset 6)
            pos_raw = gsr.getData(mid, ADDR_READ + 6, 4)
            q_np[i]   = _to_signed32(pos_raw) * _RAD_PER_TICK
            dq_np[i]  = _to_signed32(vel_raw) * _RADS_PER_LSB
            tau_np[i] = _to_signed16(cur_raw) * _KT_PER_LSB

    # GroupSyncWrite — goal positions
    gsw = GroupSyncWrite(port, ph, ADDR_GOAL, LEN_GOAL)

    def _write_all() -> None:
        gsw.clearParam()
        for i, mid in enumerate(all_motor_ids):
            tgt = tgt_np[i]
            if not np.isfinite(tgt):
                tgt = 0.0
            pos_rad = init_pos[mid] + tgt
            raw = int(round(pos_rad / _RAD_PER_TICK))
            raw = int(np.clip(raw, -(2**31 - 1), 2**31 - 1))
            if raw < 0:
                raw += 2**32
            gsw.addParam(mid, [
                (raw >> 0)  & 0xFF,
                (raw >> 8)  & 0xFF,
                (raw >> 16) & 0xFF,
                (raw >> 24) & 0xFF,
            ])
        gsw.txPacket()

    print("[real] Starting control loop.")
    iters  = 0
    t_rate = time.time()

    try:
        while not stop_event.is_set():
            t0 = time.time()
            _read_all()
            _write_all()

            iters += 1
            dt_rate = time.time() - t_rate
            if dt_rate >= 1.0:
                _np(shm_real_hz)[0] = iters / dt_rate
                iters  = 0
                t_rate = time.time()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        for hw in hws.values():
            try:
                hw._torque_enable(False)
            except Exception:
                pass
        port.closePort()
        print("[real] Done.")


# ── Sim thread ──────────────────────────────────────────────────────────────────

def sim_thread_fn(
    mj_model:       mujoco.MjModel,
    mj_data:        mujoco.MjData,
    all_motor_ids:  list[int],
    motor_to_ctrl:  dict[int, int],    # motor_id → ctrl index
    shm_target:     mp.Array,          # (12,) rad, offset-space
    shm_q_sim:      mp.Array,          # (12,) rad
    shm_dq_sim:     mp.Array,          # (12,) rad/s
    shm_sim_hz:     mp.Array,
    data_lock:      threading.Lock,
    stop_event:     threading.Event,
    sim_substeps:   int,
    control_hz:     int,
) -> None:
    dt     = 1.0 / control_hz
    iters  = 0
    t_rate = time.time()
    tgt_np = _np(shm_target)
    q_np   = _np(shm_q_sim)
    dq_np  = _np(shm_dq_sim)

    while not stop_event.is_set():
        t0 = time.time()

        with data_lock:
            # Read sim state for all actuated joints
            for i, mid in enumerate(all_motor_ids):
                ctrl_id = motor_to_ctrl.get(mid)
                if ctrl_id is None:
                    continue
                jid     = mj_model.actuator_trnid[ctrl_id, 0]
                qpos_id = int(mj_model.jnt_qposadr[jid])
                qvel_id = int(mj_model.jnt_dofadr[jid])
                q_np[i]  = float(mj_data.qpos[qpos_id])
                dq_np[i] = float(mj_data.qvel[qvel_id])
                mj_data.ctrl[ctrl_id] = tgt_np[i]
            for _ in range(sim_substeps):
                mujoco.mj_step(mj_model, mj_data)

        iters += 1
        dt_rate = time.time() - t_rate
        if dt_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / dt_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Viz thread ──────────────────────────────────────────────────────────────────

def viz_thread_fn(
    sim_view:          ViserRobotView,
    real_view:         ViserRobotView,
    mj_model:          mujoco.MjModel,
    mj_data:           mujoco.MjData,
    joint_info:        dict,             # joint_name → (qpos_id, ...)
    all_motor_ids:     list[int],
    motor_id_to_joint: dict[int, str],
    shm_q_real:        mp.Array,         # (12,) abs position
    shm_init_pos:      mp.Array,         # (12,) abs initial position
    data_lock:         threading.Lock,
    stop_event:        threading.Event,
) -> None:
    sim_viz  = mujoco.MjData(mj_model)
    real_viz = mujoco.MjData(mj_model)
    period   = 1.0 / VIZ_HZ

    # motor index → qpos_id lookup
    motor_idx_to_qpos: list[int | None] = []
    for mid in all_motor_ids:
        jname = motor_id_to_joint.get(mid)
        if jname and jname in joint_info:
            motor_idx_to_qpos.append(joint_info[jname][0])
        else:
            motor_idx_to_qpos.append(None)

    q_real_np = _np(shm_q_real)
    ip_np     = _np(shm_init_pos)

    while not stop_event.is_set():
        t0 = time.time()

        # Sim ghost — copy live mj_data under lock
        with data_lock:
            sim_viz.qpos[:] = mj_data.qpos
        mujoco.mj_kinematics(mj_model, sim_viz)
        sim_view.update(sim_viz)

        # Real ghost — start from sim qpos (freejoint orientation),
        # overwrite each actuated joint from hardware (offset from init)
        real_viz.qpos[:] = sim_viz.qpos
        for i, qpos_id in enumerate(motor_idx_to_qpos):
            if qpos_id is not None:
                real_viz.qpos[qpos_id] = q_real_np[i] - ip_np[i]
        mujoco.mj_kinematics(mj_model, real_viz)
        real_view.update(real_viz)

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Signal generator ────────────────────────────────────────────────────────────

class StandupSignalGen:
    """Per-group signal generator for coordinated full-robot motion.

    Generates a (12,) target array (rad, offset-space) each call.
    Each joint's target = group_amplitude * joint_sign.
    """

    def __init__(self) -> None:
        self.mode = "hold"

        # Per-group centre offsets (rad) — hold target and oscillation centre
        self.amp: dict[str, float] = {"hip": 0.0, "thigh": 0.0, "calf": 0.0}

        # Per-group swing (rad) — oscillation amplitude around centre
        self.swing: dict[str, float] = {"hip": 0.0, "thigh": 0.0, "calf": 0.0}

        # Sine / step / chirp params
        self.frequency = 0.5
        self.step_T    = 2.0
        self.chirp_f1  = 5.0
        self.chirp_t   = 10.0

        # Standup ramp state
        self._ramp_active = False
        self._ramp_t0:  float = 0.0
        self._ramp_dur: float = 3.0   # s
        self._ramp_phase = 0   # 0=up, 1=hold, 2=down, done

        # Phase origin for periodic signals
        self._t0: float | None = None

        # Joint order (set from main before use)
        self.all_motor_ids: list[int] = []

    def reset(self) -> None:
        self._t0 = None

    def trigger_standup(self, duration: float) -> None:
        self._ramp_active = True
        self._ramp_t0     = time.time()
        self._ramp_dur    = duration
        self._ramp_phase  = 0

    def _ramp_envelope(self, t_wall: float) -> float:
        """Returns 0→1→0 envelope for standup ramp. 0 when inactive."""
        if not self._ramp_active:
            return 0.0
        t = t_wall - self._ramp_t0
        half = self._ramp_dur / 2.0
        if t < half:
            return t / half
        elif t < self._ramp_dur:
            return 1.0 - (t - half) / half
        else:
            self._ramp_active = False
            return 0.0

    def __call__(
        self,
        t_wall:       float,
        joint_names:  list[str],   # 12 names in all_motor_ids order
    ) -> np.ndarray:
        """Return (12,) target array in rad (offset-space)."""
        if self._t0 is None:
            self._t0 = t_wall
        t = t_wall - self._t0

        targets = np.zeros(len(joint_names))

        for i, jname in enumerate(joint_names):
            part   = joint_part(jname)
            sign   = joint_sign(jname)
            centre = self.amp.get(part, 0.0)   * sign   # hold position
            swing  = self.swing.get(part, 0.0) * sign   # oscillation amplitude

            if self.mode == "hold":
                val = centre
            elif self.mode == "standup":
                val = centre * self._ramp_envelope(t_wall)
            elif self.mode == "sine":
                val = centre + swing * np.sin(2.0 * np.pi * self.frequency * t)
            elif self.mode == "step":
                phase = (t % self.step_T) / self.step_T
                val   = centre + (swing if phase < 0.5 else -swing)
            elif self.mode == "chirp":
                f0, f1, T = self.frequency, self.chirp_f1, self.chirp_t
                t_mod     = t % T
                inst_f    = f0 + (f1 - f0) * t_mod / T
                val       = centre + swing * np.sin(2.0 * np.pi * inst_f * t_mod)
            else:
                val = 0.0

            targets[i] = val

        return targets


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rumi full-robot standup: coordinated sim + real."
    )
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    args = parser.parse_args()

    # ── MuJoCo model ───────────────────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    mj_model.opt.timestep = PHYSICS_DT
    mj_data  = mujoco.MjData(mj_model)

    joint_info = build_joint_info(mj_model)

    from hardware_rumi import JOINT_TO_MOTOR_ID, MOTOR_ID_TO_JOINT  # noqa: PLC0415

    all_motor_ids  = list(JOINT_TO_MOTOR_ID.values())   # canonical order: 1..12
    n_motors       = len(all_motor_ids)
    joint_names    = [MOTOR_ID_TO_JOINT[mid] for mid in all_motor_ids]

    # motor_id → ctrl index (for sim thread)
    motor_to_ctrl: dict[int, int] = {}
    for jname, (_, _, ctrl_id, _, _) in joint_info.items():
        mid = JOINT_TO_MOTOR_ID.get(jname)
        if mid is not None:
            motor_to_ctrl[mid] = ctrl_id

    sim_substeps = max(1, round(1.0 / (args.control_hz * PHYSICS_DT)))
    print(f"control_hz={args.control_hz}  sim_substeps={sim_substeps}  n_motors={n_motors}")
    print(f"Joint order: {joint_names}")

    # Apply "zero" keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "zero")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mj_data.qpos[:] = 0.0
        mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # ── Shared memory ───────────────────────────────────────────────────────────
    shm_target   = mp.Array(ctypes.c_double, n_motors)
    shm_q_real   = mp.Array(ctypes.c_double, n_motors)
    shm_dq_real  = mp.Array(ctypes.c_double, n_motors)
    shm_tau_meas = mp.Array(ctypes.c_double, n_motors)
    shm_init_pos = mp.Array(ctypes.c_double, n_motors)
    shm_q_sim    = mp.Array(ctypes.c_double, n_motors)
    shm_dq_sim   = mp.Array(ctypes.c_double, n_motors)
    shm_real_hz  = mp.Array(ctypes.c_double, 1)
    shm_sim_hz   = mp.Array(ctypes.c_double, 1)

    _np(shm_target)[:] = 0.0
    _np(shm_q_real)[:] = 0.0
    _np(shm_q_sim)[:] = 0.0

    stop_event = mp.Event()

    # ── Spawn real process BEFORE GPU / CUDA init ───────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(
            all_motor_ids, args.control_hz,
            shm_target, shm_q_real, shm_dq_real, shm_tau_meas,
            shm_init_pos, shm_real_hz, stop_event,
        ),
        daemon=True,
    )
    real_proc.start()

    # ── Viser ───────────────────────────────────────────────────────────────────
    server   = viser.ViserServer(label="Rumi Standup")
    scene    = ViserMujocoScene.create(server, mj_model)
    sim_view = scene.add_robot("sim",  color=SIM_COLOR)
    real_view= scene.add_robot("real", color=REAL_COLOR)
    scene.create_visualization_gui(
        camera_distance=1.5,
        camera_azimuth=45.0,
        camera_elevation=-20.0,
    )

    # ── Status ──────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Status"):
        txt_sim_hz  = server.gui.add_text("Sim Hz",  initial_value="— Hz")
        txt_real_hz = server.gui.add_text("Real Hz", initial_value="— Hz")

    # ── Signal GUI ───────────────────────────────────────────────────────────────
    sig = StandupSignalGen()
    sig.all_motor_ids = all_motor_ids

    with server.gui.add_folder("Signal"):
        dd_mode = server.gui.add_dropdown(
            "Mode",
            options=["hold", "standup", "sine", "step", "chirp"],
            initial_value="hold",
        )
        sl_hip_amp = server.gui.add_slider(
            "Hip centre (deg)", min=0.0, max=30.0, step=0.5, initial_value=0.0,
        )
        sl_thigh_amp = server.gui.add_slider(
            "Thigh centre (deg)", min=-20.0, max=40.0, step=0.5, initial_value=0.0,
        )
        sl_calf_amp = server.gui.add_slider(
            "Calf centre (deg)", min=0.0, max=70.0, step=0.5, initial_value=0.0,
        )
        sl_hip_swing = server.gui.add_slider(
            "Hip swing (deg)", min=0.0, max=30.0, step=0.5, initial_value=0.0,
        )
        sl_thigh_swing = server.gui.add_slider(
            "Thigh swing (deg)", min=0.0, max=40.0, step=0.5, initial_value=0.0,
        )
        sl_calf_swing = server.gui.add_slider(
            "Calf swing (deg)", min=0.0, max=70.0, step=0.5, initial_value=0.0,
        )
        sl_freq = server.gui.add_slider(
            "Frequency (Hz)", min=0.05, max=10.0, step=0.05, initial_value=0.5,
        )
        sl_step_T = server.gui.add_slider(
            "Step period (s)", min=0.2, max=10.0, step=0.1, initial_value=2.0,
        )
        sl_chirp_f1 = server.gui.add_slider(
            "Chirp f1 (Hz)", min=0.1, max=20.0, step=0.1, initial_value=5.0,
        )
        sl_chirp_t = server.gui.add_slider(
            "Chirp sweep (s)", min=1.0, max=30.0, step=0.5, initial_value=10.0,
        )
        sl_standup_dur = server.gui.add_slider(
            "Standup duration (s)", min=0.5, max=10.0, step=0.25, initial_value=3.0,
        )
        btn_execute = server.gui.add_button("Execute standup", color="green")
        btn_reset   = server.gui.add_button("Reset signal phase")

        @btn_execute.on_click
        def _(_) -> None:
            sig.trigger_standup(float(sl_standup_dur.value))

        @btn_reset.on_click
        def _(_) -> None:
            sig.reset()

    # ── Plots ───────────────────────────────────────────────────────────────────
    _buf    = int(PLOT_WINDOW_S * 60)
    _t_buf  = deque(maxlen=_buf)
    _groups = ["hip", "thigh", "calf"]

    # Map group → ordered list of motor indices (consistent order)
    _group_indices: dict[str, list[int]] = {g: [] for g in _groups}
    for i, jname in enumerate(joint_names):
        _group_indices[joint_part(jname)].append(i)

    # Per-group, per-joint buffers: _q_real_buf[g][j_idx] is a deque
    # j_idx is position within the group's index list (0..3)
    _q_real_buf: dict[str, list[deque]] = {
        g: [deque(maxlen=_buf) for _ in _group_indices[g]] for g in _groups
    }
    _q_sim_buf: dict[str, list[deque]] = {
        g: [deque(maxlen=_buf) for _ in _group_indices[g]] for g in _groups
    }
    _q_des_buf: dict[str, deque] = {g: deque(maxlen=_buf) for g in _groups}
    _tau_buf:   dict[str, list[deque]] = {
        g: [deque(maxlen=_buf) for _ in _group_indices[g]] for g in _groups
    }

    # Colours for the 4 joints in each group: FL, BL, BR, FR
    # group indices always follow all_motor_ids order:
    # hip:   1(FL), 4(BL), 7(BR), 10(FR)
    # thigh: 2(FL), 5(BL), 8(BR), 11(FR)
    # calf:  3(FL), 6(BL), 9(BR), 12(FR)
    _JOINT_COLORS = ["#e06c75", "#98c379", "#61afef", "#c678dd"]  # FL BL BR FR

    def _s(label: str, color: str, dash: bool = False, width: float = 1.5) -> uplot.Series:
        return uplot.Series(label=label, stroke=color, width=width,
                            **({"dash": (6.0, 3.0)} if dash else {}))

    with server.gui.add_folder("Plots", expand_by_default=True):
        _plt_pos: dict[str, object] = {}
        _plt_tau: dict[str, object] = {}
        for g in _groups:
            idxs   = _group_indices[g]
            labels = [joint_names[i].replace("_joint", "").replace(f"_{g}", "") for i in idxs]
            # Series: t, target, sim_FL, sim_BL, sim_BR, sim_FR, real_FL, real_BL, real_BR, real_FR
            n_series = 1 + 1 + len(idxs) + len(idxs)   # t + des + 4 sim + 4 real
            _plt_pos[g] = server.gui.add_uplot(
                tuple(np.zeros(2) for _ in range(n_series)),
                (
                    uplot.Series(label="t"),
                    _s("target", "#fa0", width=2.0),
                    *[_s(f"sim {lb}", c, width=1.2)
                      for lb, c in zip(labels, _JOINT_COLORS)],
                    *[_s(f"real {lb}", c, dash=True, width=1.5)
                      for lb, c in zip(labels, _JOINT_COLORS)],
                ),
                title=f"{g.capitalize()} position (deg)", aspect=2.5,
                axes=(uplot.Axis(), uplot.Axis(label="deg")),
            )
            # Torque: t + 4 joints
            _plt_tau[g] = server.gui.add_uplot(
                tuple(np.zeros(2) for _ in range(1 + len(idxs))),
                (
                    uplot.Series(label="t"),
                    *[_s(f"τ {lb}", c, width=1.2)
                      for lb, c in zip(labels, _JOINT_COLORS)],
                ),
                title=f"{g.capitalize()} torque (N·m)", aspect=1.5,
                axes=(uplot.Axis(), uplot.Axis(label="N·m")),
            )

    # ── Recording ───────────────────────────────────────────────────────────────
    _rec_lock   = threading.Lock()
    _rec_active = False
    _rec_t:      list[float]         = []
    _rec_target: list[np.ndarray]    = []
    _rec_q_real: list[np.ndarray]    = []
    _rec_dq_real:list[np.ndarray]    = []
    _rec_tau:    list[np.ndarray]    = []
    _rec_q_sim:  list[np.ndarray]    = []
    _rec_dq_sim: list[np.ndarray]    = []

    _data_dir = _HERE / "data"
    _data_dir.mkdir(exist_ok=True)

    with server.gui.add_folder("Recording"):
        txt_rec = server.gui.add_text("Status", initial_value="Idle")
        btn_rec = server.gui.add_button("Start Recording", color="green")

        @btn_rec.on_click
        def _(_) -> None:
            nonlocal _rec_active
            with _rec_lock:
                if not _rec_active:
                    _rec_active = True
                    _rec_t.clear(); _rec_target.clear()
                    _rec_q_real.clear(); _rec_dq_real.clear()
                    _rec_tau.clear(); _rec_q_sim.clear(); _rec_dq_sim.clear()
                    btn_rec.label = "Stop Recording"
                    btn_rec.color = "red"
                    txt_rec.value = "Recording…"
                    print("[rec] Started.")
                else:
                    _rec_active = False
                    btn_rec.label = "Start Recording"
                    btn_rec.color = "green"
                    n = len(_rec_t)
                    if n > 0:
                        import datetime as _dt
                        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = (_data_dir /
                                 f"{_ts}_standup_{sig.mode}.npz")
                        from hardware_rumi import DEFAULT_KP, DEFAULT_KD  # noqa: PLC0415
                        _kp_sim = float(mj_model.actuator_gainprm[0, 0])
                        _kd_sim = float(-mj_model.actuator_biasprm[0, 2])
                        np.savez(
                            fname,
                            t          = np.array(_rec_t),
                            target     = np.array(_rec_target),
                            q_real     = np.array(_rec_q_real),
                            dq_real    = np.array(_rec_dq_real),
                            tau_meas   = np.array(_rec_tau),
                            q_sim      = np.array(_rec_q_sim),
                            dq_sim     = np.array(_rec_dq_sim),
                            control_hz = np.array([args.control_hz]),
                            joint_names= np.array(joint_names),
                            kp_sim     = np.array([_kp_sim]),
                            kd_sim     = np.array([_kd_sim]),
                            kp_real    = np.array([DEFAULT_KP]),
                            kd_real    = np.array([DEFAULT_KD]),
                        )
                        print(f"[rec] Saved {n} samples → {fname.name}")
                        txt_rec.value = f"Saved {n} pts → {fname.name}"
                    else:
                        txt_rec.value = "Idle (no data)"

    # ── Threads ──────────────────────────────────────────────────────────────────
    th_stop   = threading.Event()
    data_lock = threading.Lock()

    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(
            mj_model, mj_data,
            all_motor_ids, motor_to_ctrl,
            shm_target, shm_q_sim, shm_dq_sim, shm_sim_hz,
            data_lock, th_stop, sim_substeps, args.control_hz,
        ),
    ).start()

    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(
            sim_view, real_view,
            mj_model, mj_data,
            joint_info, all_motor_ids, MOTOR_ID_TO_JOINT,
            shm_q_real, shm_init_pos,
            data_lock, th_stop,
        ),
    ).start()

    # ── Wait for real process ────────────────────────────────────────────────────
    print("Waiting for real process…")
    time.sleep(2.0)
    print("[main] Ready. offset-space: 0 = initial motor position.")
    print("Running. Ctrl+C to stop.")

    _plot_skip = max(1, round(60 / PLOT_HZ))
    _loop_iter = 0
    _t0        = time.time()
    tgt_np     = _np(shm_target)
    q_real_np  = _np(shm_q_real)
    dq_real_np = _np(shm_dq_real)
    tau_np     = _np(shm_tau_meas)
    ip_np      = _np(shm_init_pos)
    q_sim_np   = _np(shm_q_sim)
    dq_sim_np  = _np(shm_dq_sim)

    try:
        while True:
            t_wall = time.time()

            # Update signal params from GUI
            sig.mode = dd_mode.value
            sig.amp["hip"]     = float(np.deg2rad(sl_hip_amp.value))
            sig.amp["thigh"]   = float(np.deg2rad(sl_thigh_amp.value))
            sig.amp["calf"]    = float(np.deg2rad(sl_calf_amp.value))
            sig.swing["hip"]   = float(np.deg2rad(sl_hip_swing.value))
            sig.swing["thigh"] = float(np.deg2rad(sl_thigh_swing.value))
            sig.swing["calf"]  = float(np.deg2rad(sl_calf_swing.value))
            sig.frequency      = float(sl_freq.value)
            sig.step_T       = float(sl_step_T.value)
            sig.chirp_f1     = float(sl_chirp_f1.value)
            sig.chirp_t      = float(sl_chirp_t.value)

            # Generate targets for all 12 joints and write to shm
            targets = sig(t_wall, joint_names)
            tgt_np[:] = targets

            # Status
            txt_sim_hz.value  = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            txt_real_hz.value = f"{_np(shm_real_hz)[0]:.0f} Hz"

            # Snapshot shared arrays (offset-space for real)
            now        = t_wall - _t0
            q_real_off = q_real_np.copy() - ip_np   # offset from init
            q_sim_now  = q_sim_np.copy()
            dq_real_now= dq_real_np.copy()
            tau_now    = tau_np.copy()

            # Accumulate per-group, per-joint plot buffers
            _t_buf.append(now)
            for g in _groups:
                idxs = _group_indices[g]
                # target: use first right-side joint as representative scalar
                right = [i for i in idxs if joint_sign(joint_names[i]) > 0]
                rep   = right[0] if right else idxs[0]
                _q_des_buf[g].append(float(np.rad2deg(targets[rep])))
                for j, mi in enumerate(idxs):
                    _q_sim_buf[g][j].append(float(np.rad2deg(q_sim_now[mi])))
                    _q_real_buf[g][j].append(float(np.rad2deg(q_real_off[mi])))
                    _tau_buf[g][j].append(float(tau_now[mi]))

            # Recording
            if _rec_active:
                with _rec_lock:
                    if _rec_active:
                        _rec_t.append(now)
                        _rec_target.append(targets.copy())
                        _rec_q_real.append(q_real_off.copy())
                        _rec_dq_real.append(dq_real_now.copy())
                        _rec_tau.append(tau_now.copy())
                        _rec_q_sim.append(q_sim_now.copy())
                        _rec_dq_sim.append(dq_sim_np.copy())
                        txt_rec.value = f"Recording… {len(_rec_t)} pts"

            # Plot refresh
            _loop_iter += 1
            if _loop_iter % _plot_skip == 0 and len(_t_buf) > 1:
                t_arr = np.array(_t_buf)
                for g in _groups:
                    n = len(_group_indices[g])
                    _plt_pos[g].data = (
                        t_arr,
                        np.array(_q_des_buf[g]),
                        *[np.array(_q_sim_buf[g][j])  for j in range(n)],
                        *[np.array(_q_real_buf[g][j]) for j in range(n)],
                    )
                    _plt_tau[g].data = (
                        t_arr,
                        *[np.array(_tau_buf[g][j]) for j in range(n)],
                    )

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
