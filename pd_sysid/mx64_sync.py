"""MX-64 single-joint sysid — MuJoCo sim + real motor in parallel.

Both sim and real receive the same position target at the same control rate.
The goal is to observe and compare their responses for system identification.

Signal modes (selectable via GUI):
  hold  — manual target via slider
  sine  — A·sin(2π·f·t) around centre
  step  — ±A square wave at period T
  chirp — linearly swept sine f0→f1 over sweep_t seconds

Recording:
  Press "Record" in the GUI to start/stop recording.
  Data is saved as data/<timestamp>_recording.npz with keys:
    t         — time (s)
    target    — commanded position (rad)
    q_real    — real motor position (rad)
    dq_real   — real motor velocity (rad/s)
    q_sim     — sim position (rad)
    dq_sim    — sim velocity (rad/s)
    tau_meas  — estimated torque from current (N·m)
    control_hz — scalar, stored as metadata

Processes / threads:
  Main process:
    - Sim  thread  (control_hz): write ctrl → mj_step
    - Viz  thread  (30 Hz):      copy qpos → mj_kinematics → update views
    - Main thread  (60 Hz):      signal gen → shared memory, plot refresh
  Real process (control_hz):
    - read state → write goal position

Run:
  python mx64_sync.py
  python mx64_sync.py --motor-id 17 --control-hz 50
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
_SCENE_XML = _HERE / "motor_assembly.xml"

# ── Constants ──────────────────────────────────────────────────────────────────
JOINT_NAME = "disk_joint"
MOTOR_NAME = "disk_motor"

HOME_RAD  = 0.0
JOINT_MIN = -np.pi
JOINT_MAX =  np.pi

PHYSICS_DT = 0.004   # s  (matches XML timestep)
CONTROL_HZ = 50
VIZ_HZ     = 30
PLOT_HZ    = 15

PLOT_WINDOW_S = 15.0

SIM_COLOR  = None
REAL_COLOR = (0.20, 0.55, 0.90, 0.60)  # blue ghost


# ── Shared-memory helper ───────────────────────────────────────────────────────

def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── MuJoCo index helpers ───────────────────────────────────────────────────────

def get_joint_indices(mj_model: mujoco.MjModel) -> tuple[int, int, int]:
    qpos_id = mj_model.jnt_qposadr[mj_model.joint(JOINT_NAME).id]
    qvel_id = mj_model.jnt_dofadr [mj_model.joint(JOINT_NAME).id]
    ctrl_id = mj_model.actuator(MOTOR_NAME).id
    return int(qpos_id), int(qvel_id), int(ctrl_id)


# ── Real process ───────────────────────────────────────────────────────────────

def real_process_fn(
    motor_id:    int,
    control_hz:  int,
    shm_q_real:  mp.Array,
    shm_dq_real: mp.Array,
    shm_tau_meas: mp.Array,
    shm_target:  mp.Array,
    shm_real_hz: mp.Array,
    stop_event:  mp.Event,
) -> None:
    """Hardware control loop — runs in a separate process."""
    sys.path.insert(0, str(Path(__file__).parent))
    from hardware_mx import MX64Hardware  # noqa: PLC0415

    dt = 1.0 / control_hz
    hw = MX64Hardware(motor_id=motor_id)

    try:
        print("[real] Connecting…")
        hw.connect()
        hw.enable()

        # seed target to current position so motor holds still at startup
        state = hw.read_state()
        if state is not None:
            _np(shm_q_real)[0]  = state.pos_rad
            _np(shm_dq_real)[0] = state.vel_rads
            _np(shm_target)[0]  = state.pos_rad
            print(f"[real] Initial pos: {np.rad2deg(state.pos_rad):.1f} deg — "
                  f"target seeded to current position.")

        print("[real] Starting control loop.")
        iters  = 0
        t_rate = time.time()

        while not stop_event.is_set():
            t = time.time()

            state = hw.read_state()
            if state is not None:
                _np(shm_q_real)[0]   = state.pos_rad
                _np(shm_dq_real)[0]  = state.vel_rads
                _np(shm_tau_meas)[0] = state.torque_nm

            hw.set_position_rad(_np(shm_target)[0])

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
        hw.shutdown()
        print("[real] Done.")


# ── Sim thread ─────────────────────────────────────────────────────────────────

def sim_thread_fn(
    mj_model:     mujoco.MjModel,
    mj_data:      mujoco.MjData,
    qpos_id:      int,
    qvel_id:      int,
    ctrl_id:      int,
    shm_target:   mp.Array,
    shm_sim_hz:   mp.Array,
    shm_q_sim:    mp.Array,
    shm_dq_sim:   mp.Array,
    data_lock:    threading.Lock,
    stop_event:   threading.Event,
    sim_substeps: int,
    control_hz:   int,
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
            mj_data.ctrl[ctrl_id] = target
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
    sim_view:   ViserRobotView,
    real_view:  ViserRobotView,
    mj_model:   mujoco.MjModel,
    mj_data:    mujoco.MjData,
    qpos_id:    int,
    shm_q_real: mp.Array,
    data_lock:  threading.Lock,
    stop_event: threading.Event,
) -> None:
    sim_viz  = mujoco.MjData(mj_model)
    real_viz = mujoco.MjData(mj_model)
    period   = 1.0 / VIZ_HZ

    while not stop_event.is_set():
        t = time.time()

        with data_lock:
            sim_viz.qpos[:] = mj_data.qpos
        mujoco.mj_kinematics(mj_model, sim_viz)
        sim_view.update(sim_viz)

        real_viz.qpos[qpos_id] = _np(shm_q_real)[0]
        mujoco.mj_kinematics(mj_model, real_viz)
        real_view.update(real_viz)

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Test-signal generator ──────────────────────────────────────────────────────

class SignalGen:
    """Generates position targets in radians."""

    def __init__(self) -> None:
        self.mode      = "hold"
        self.amplitude = 0.5      # rad
        self.frequency = 0.5      # Hz
        self.chirp_f1  = 5.0      # Hz  (chirp end frequency)
        self.chirp_t   = 10.0     # s   (chirp sweep duration)
        self.step_T    = 2.0      # s   (square wave full period)
        self.centre    = 0.0      # rad
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
            return c + A * np.sin(2.0 * np.pi * self.frequency * t)

        if self.mode == "step":
            phase = (t % self.step_T) / self.step_T
            return c + (A if phase < 0.5 else -A)

        if self.mode == "chirp":
            f0, f1, T = self.frequency, self.chirp_f1, self.chirp_t
            t_mod = t % T
            inst_f = f0 + (f1 - f0) * t_mod / T
            return c + A * np.sin(2.0 * np.pi * inst_f * t_mod)

        return c   # hold


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MX-64 sysid: sim + real in parallel.")
    parser.add_argument("--motor-id",   type=int, default=17)
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    args = parser.parse_args()

    sim_substeps = max(1, round(1.0 / (args.control_hz * PHYSICS_DT)))
    print(f"Motor ID: {args.motor_id}  control_hz={args.control_hz}  "
          f"sim_substeps={sim_substeps}")

    # ── Shared memory ──────────────────────────────────────────────────────────
    shm_target   = mp.Array(ctypes.c_double, 1)
    shm_q_real   = mp.Array(ctypes.c_double, 1)
    shm_dq_real  = mp.Array(ctypes.c_double, 1)
    shm_tau_meas = mp.Array(ctypes.c_double, 1)
    shm_real_hz  = mp.Array(ctypes.c_double, 1)
    shm_sim_hz   = mp.Array(ctypes.c_double, 1)
    shm_q_sim    = mp.Array(ctypes.c_double, 1)
    shm_dq_sim   = mp.Array(ctypes.c_double, 1)

    _np(shm_target)[0] = HOME_RAD
    _np(shm_q_real)[0] = HOME_RAD
    _np(shm_q_sim)[0]  = HOME_RAD

    stop_event = mp.Event()

    # ── Spawn real process BEFORE GPU / CUDA init ──────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(
            args.motor_id, args.control_hz,
            shm_q_real, shm_dq_real, shm_tau_meas,
            shm_target, shm_real_hz,
            stop_event,
        ),
        daemon=True,
    )
    real_proc.start()

    # ── MuJoCo model ───────────────────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    mj_model.opt.timestep = PHYSICS_DT
    mj_data  = mujoco.MjData(mj_model)
    qpos_id, qvel_id, ctrl_id = get_joint_indices(mj_model)

    mj_data.qpos[qpos_id] = HOME_RAD
    mj_data.qvel[:]       = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # ── Viser ──────────────────────────────────────────────────────────────────
    server    = viser.ViserServer(label="MX-64 Sysid")
    scene     = ViserMujocoScene.create(server, mj_model)
    sim_view  = scene.add_robot("sim",  color=SIM_COLOR)
    real_view = scene.add_robot("real", color=REAL_COLOR)
    scene.create_visualization_gui(
        camera_distance=0.5,
        camera_azimuth=135.0,
        camera_elevation=30.0,
    )

    # ── Status ─────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Status"):
        txt_sim_hz  = server.gui.add_text("Sim Hz",  initial_value="— Hz")
        txt_real_hz = server.gui.add_text("Real Hz", initial_value="— Hz")

    # ── Signal GUI ─────────────────────────────────────────────────────────────
    sig = SignalGen()

    with server.gui.add_folder("Test Signal"):
        dd_mode = server.gui.add_dropdown(
            "Mode",
            options=["hold", "sine", "step", "chirp"],
            initial_value="hold",
        )
        sl_centre = server.gui.add_slider(
            "Centre (deg)", min=-150.0, max=150.0, step=1.0, initial_value=0.0
        )
        sl_amp = server.gui.add_slider(
            "Amplitude (deg)", min=0.0, max=90.0, step=0.5, initial_value=30.0
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
            min=float(np.rad2deg(JOINT_MIN)),
            max=float(np.rad2deg(JOINT_MAX)),
            step=0.5,
            initial_value=0.0,
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    _buf = int(PLOT_WINDOW_S * 60)
    _t_buf       = deque(maxlen=_buf)
    _q_sim_buf   = deque(maxlen=_buf)
    _q_real_buf  = deque(maxlen=_buf)
    _q_des_buf   = deque(maxlen=_buf)
    _dq_sim_buf  = deque(maxlen=_buf)
    _dq_real_buf = deque(maxlen=_buf)
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
    _rec_lock    = threading.Lock()
    _rec_active  = False
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
                    _rec_t.clear()
                    _rec_target.clear()
                    _rec_q_real.clear()
                    _rec_dq_real.clear()
                    _rec_q_sim.clear()
                    _rec_dq_sim.clear()
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
                        fname = _data_dir / f"{int(time.time())}_recording.npz"
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
                        )
                        print(f"[rec] Saved {n} samples → {fname}")
                        txt_rec_status.value = f"Saved {n} pts → {fname.name}"
                    else:
                        txt_rec_status.value = "Idle (no data)"
                        print("[rec] Stopped — no data recorded.")

    # ── Threads ────────────────────────────────────────────────────────────────
    th_stop   = threading.Event()
    data_lock = threading.Lock()

    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(
            mj_model, mj_data,
            qpos_id, qvel_id, ctrl_id,
            shm_target, shm_sim_hz,
            shm_q_sim, shm_dq_sim,
            data_lock, th_stop, sim_substeps, args.control_hz,
        ),
    ).start()

    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(
            sim_view, real_view,
            mj_model, mj_data,
            qpos_id, shm_q_real,
            data_lock, th_stop,
        ),
    ).start()

    # ── Wait for real process to seed position ─────────────────────────────────
    print("Waiting for real process to connect…")
    time.sleep(2.0)
    seeded_deg = float(np.clip(
        np.rad2deg(_np(shm_target)[0]),
        np.rad2deg(JOINT_MIN), np.rad2deg(JOINT_MAX)
    ))
    sl_target.value = seeded_deg
    sl_centre.value = seeded_deg
    sig.centre      = float(np.deg2rad(seeded_deg))
    print(f"[main] Seeded centre to motor position: {seeded_deg:.1f} deg")
    print("Running. Ctrl+C to stop.")

    _plot_skip = max(1, round(60 / PLOT_HZ))
    _loop_iter = 0
    _t0        = time.time()

    try:
        while True:
            t_wall = time.time()

            # update signal params from GUI
            sig.mode      = dd_mode.value
            sig.amplitude = float(np.deg2rad(sl_amp.value))
            sig.frequency = float(sl_freq.value)
            sig.centre    = float(np.deg2rad(sl_centre.value))
            sig.step_T    = float(sl_step_T.value)
            sig.chirp_f1  = float(sl_chirp_f1.value)
            sig.chirp_t   = float(sl_chirp_t.value)

            # generate target
            if sig.mode == "hold":
                target_rad = float(np.deg2rad(sl_target.value))
            else:
                target_rad = float(np.clip(sig(t_wall), JOINT_MIN, JOINT_MAX))

            _np(shm_target)[0] = target_rad

            # status
            txt_sim_hz.value  = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            txt_real_hz.value = f"{_np(shm_real_hz)[0]:.0f} Hz"

            # buffer (plot, in degrees for display)
            now       = t_wall - _t0
            q_sim_r   = float(_np(shm_q_sim)[0])
            q_real_r  = float(_np(shm_q_real)[0])
            dq_sim_r  = float(_np(shm_dq_sim)[0])
            dq_real_r = float(_np(shm_dq_real)[0])
            tau_r     = float(_np(shm_tau_meas)[0])

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
                    if _rec_active:   # re-check inside lock
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
