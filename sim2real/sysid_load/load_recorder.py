"""Motor load sysid recorder — single joint with configurable excitation signals.

Drives one MX-64 motor (with a pendulum load) with a chosen waveform while
running the MuJoCo sim in parallel.  Real and sim state are recorded together
and saved to an .npz file on demand.

Signal modes (all configured via GUI sliders):
  hold     — hold at a fixed offset angle
  sine     — offset + amplitude * sin(2π f t)
  step     — alternating ±amplitude around offset, period = 1/freq
  triangle — linear ramp between offset±amplitude, period = 1/freq
  chirp    — linear frequency sweep from freq_start → freq_end over chirp_sweep_s,
             then repeats

Zero convention:
  On Connect the motor's raw position is read and stored as q0_hw.
  All subsequent real positions are reported as (raw - q0_hw).
  The sim also starts at qpos = 0.

Processes / threads:
  Real process  (control_hz): hardware read/write loop
  Main process:
    Sim    thread (control_hz): MuJoCo step + live physics params
    Viz    thread (control_hz): viser 3D update + live plots
    Rec    thread (control_hz): data recording (starts/stops on GUI button)
    Main   thread (60 Hz):      signal gen → shm_ctrl, GUI poll

Run:
  python load_recorder.py
  python load_recorder.py --control-hz 100 --device /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
import viser.uplot as uplot

_HERE = Path(__file__).parent
_XML  = _HERE / "motor_assembly.xml"
_DATA = _HERE / "data"

sys.path.insert(0, str(_HERE))
from viewer import ViserMujocoScene  # noqa: E402

# ── Timing ──────────────────────────────────────────────────────────────────────
CONTROL_HZ  = 50
PLOT_WINDOW = 15.0   # seconds of history shown in live plots

# ── Default physics / control params ────────────────────────────────────────────
DEFAULT_KP           = 20.0
DEFAULT_KD           = 0.0
DEFAULT_DAMPING      = 0.2
DEFAULT_ARMATURE     = 0.05
DEFAULT_FRICTIONLOSS = 0.01

# ── Shared memory helpers ────────────────────────────────────────────────────────
def _np(shm: mp.Array) -> np.ndarray:
    return np.frombuffer(shm.get_obj(), dtype=np.float64)


# ── Thread-shared physics params (main → sim thread + recorder) ─────────────────
_phys_lock   = threading.Lock()
_phys_params = {
    "kp":           DEFAULT_KP,
    "kd":           DEFAULT_KD,
    "damping":      DEFAULT_DAMPING,
    "armature":     DEFAULT_ARMATURE,
    "frictionloss": DEFAULT_FRICTIONLOSS,
}

# ── Thread-shared signal params (main → signal gen thread) ──────────────────────
_sig_lock   = threading.Lock()
_sig_params = {
    "mode":         "hold",   # hold | sine | step | triangle | chirp
    "offset":       0.0,      # rad — centre of oscillation
    "amplitude":    0.5,      # rad — half-swing
    "frequency":    0.5,      # Hz — for sine/step/triangle and chirp start
    "chirp_f_end":  5.0,      # Hz — chirp end frequency
    "chirp_sweep":  10.0,     # s  — chirp sweep duration before repeating
}


# ── Signal generator ─────────────────────────────────────────────────────────────

class LoadSignalGen:
    """Single-joint waveform generator."""

    def __init__(self) -> None:
        self._t0: float | None = None

    def reset(self) -> None:
        self._t0 = None

    def __call__(self, t_wall: float) -> float:
        """Return target angle (rad, offset-space) for the current wall time."""
        if self._t0 is None:
            self._t0 = t_wall
        t = t_wall - self._t0

        with _sig_lock:
            p = _sig_params.copy()

        mode      = p["mode"]
        offset    = p["offset"]
        amp       = p["amplitude"]
        freq      = p["frequency"]
        f_end     = p["chirp_f_end"]
        sweep_s   = p["chirp_sweep"]

        if mode == "hold":
            return offset

        elif mode == "sine":
            return offset + amp * np.sin(2.0 * np.pi * freq * t)

        elif mode == "step":
            # period = 1/freq; alternate ±amp around offset
            period = 1.0 / max(freq, 1e-6)
            phase  = (t % period) / period
            return offset + (amp if phase < 0.5 else -amp)

        elif mode == "triangle":
            # period = 1/freq; linear ramp between offset-amp and offset+amp
            period = 1.0 / max(freq, 1e-6)
            phase  = (t % period) / period
            env    = phase / 0.5 if phase < 0.5 else 1.0 - (phase - 0.5) / 0.5
            return (offset - amp) + env * 2.0 * amp

        elif mode == "chirp":
            # linear instantaneous frequency sweep f0→f_end over sweep_s, then repeats
            f0    = freq
            T     = max(sweep_s, 0.1)
            t_mod = t % T
            inst_f = f0 + (f_end - f0) * t_mod / T
            return offset + amp * np.sin(2.0 * np.pi * inst_f * t_mod)

        return offset


# ── Real process ─────────────────────────────────────────────────────────────────

def real_process_fn(
    control_hz:    int,
    device:        str,
    shm_q_real:    mp.Array,
    shm_dq_real:   mp.Array,
    shm_tau_real:  mp.Array,
    shm_ctrl:      mp.Array,
    shm_real_hz:   mp.Array,
    shm_kp_real:   mp.Array,
    shm_kd_real:   mp.Array,
    connect_event: mp.Event,
    stop_event:    mp.Event,
) -> None:
    import sys as _sys
    _sys.path.insert(0, str(_HERE))
    from hardware_mx import MX64Hardware

    dt = 1.0 / control_hz

    print("[real] Waiting for Connect…")
    connect_event.wait()
    if stop_event.is_set():
        return

    hw = MX64Hardware(device=device)
    try:
        hw.connect()

        with _phys_lock:
            kp = _phys_params["kp"]
            kd = _phys_params["kd"]
        hw.enable(kp=kp, kd=kd)
        # Record the gains actually written to hardware
        _np(shm_kp_real)[0] = kp
        _np(shm_kd_real)[0] = kd

        # Zero offset: retry until we get a valid read
        q0_hw = None
        while q0_hw is None and not stop_event.is_set():
            state = hw.read_state()
            if state is not None:
                q0_hw = state.pos_rad
            else:
                print("[real] Warning: could not read initial position, retrying…")
                time.sleep(0.05)
        if q0_hw is None:
            return
        print(f"[real] Hardware zero offset: {q0_hw:.4f} rad")

        print("[real] Starting control loop.")
        iters  = 0
        t_rate = time.time()

        while not stop_event.is_set():
            t = time.time()

            state = hw.read_state()
            if state is not None:
                _np(shm_q_real)[0]   = state.pos_rad - q0_hw
                _np(shm_dq_real)[0]  = state.vel_rads
                _np(shm_tau_real)[0] = state.torque_nm

            hw.set_position_rad(q0_hw + _np(shm_ctrl)[0])

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
        hw.shutdown()
        print("[real] Done.")


# ── Sim thread ───────────────────────────────────────────────────────────────────

def sim_thread_fn(
    model:       mujoco.MjModel,
    data:        mujoco.MjData,
    shm_ctrl:    mp.Array,
    shm_q_sim:   mp.Array,
    shm_dq_sim:  mp.Array,
    shm_tau_sim: mp.Array,
    shm_sim_hz:  mp.Array,
    stop_event:  threading.Event,
) -> None:
    dt         = 1.0 / CONTROL_HZ
    n_substeps = max(1, round(dt / model.opt.timestep))
    iters      = 0
    t_rate     = time.time()

    while not stop_event.is_set():
        t = time.time()

        with _phys_lock:
            p = _phys_params.copy()

        model.dof_damping[:]         = p["damping"]
        model.dof_armature[:]        = p["armature"]
        model.dof_frictionloss[:]    = p["frictionloss"]
        model.actuator_gainprm[0, 0] = p["kp"]
        model.actuator_biasprm[0, 1] = -p["kp"]
        model.actuator_biasprm[0, 2] = -p["kd"]

        data.ctrl[0] = _np(shm_ctrl)[0]
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        _np(shm_q_sim)[0]   = data.qpos[0]
        _np(shm_dq_sim)[0]  = data.qvel[0]
        _np(shm_tau_sim)[0] = data.actuator_force[0]

        iters += 1
        elapsed_rate = time.time() - t_rate
        if elapsed_rate >= 1.0:
            _np(shm_sim_hz)[0] = iters / elapsed_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ── Recorder thread ──────────────────────────────────────────────────────────────

def recorder_thread_fn(
    control_hz:   int,
    shm_ctrl:     mp.Array,
    shm_q_sim:    mp.Array,
    shm_dq_sim:   mp.Array,
    shm_tau_sim:  mp.Array,
    shm_q_real:   mp.Array,
    shm_dq_real:  mp.Array,
    shm_tau_real: mp.Array,
    shm_kp_real:  mp.Array,
    shm_kd_real:  mp.Array,
    start_event:  threading.Event,   # set → start recording
    stop_event:   threading.Event,   # set → stop & save
) -> None:
    dt = 1.0 / control_hz

    while not stop_event.is_set():
        print("[rec] Waiting for Record to start…")
        # Wait for start, but wake up periodically to check stop
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.05)
        if stop_event.is_set():
            return

        print("[rec] Recording started.")
        t_start = time.time()

        buf_t            = []
        buf_target       = []
        buf_q_sim        = []
        buf_dq_sim       = []
        buf_tau_sim      = []
        buf_q_real       = []
        buf_dq_real      = []
        buf_tau_real     = []
        buf_kp_sim       = []
        buf_kd_sim       = []
        buf_damping      = []
        buf_armature     = []
        buf_frictionloss = []
        buf_sig_mode     = []

        while start_event.is_set() and not stop_event.is_set():
            t = time.time()

            with _phys_lock:
                p = _phys_params.copy()
            with _sig_lock:
                sp = _sig_params.copy()

            buf_t.append(t - t_start)
            buf_target.append(_np(shm_ctrl)[0])
            buf_q_sim.append(_np(shm_q_sim)[0])
            buf_dq_sim.append(_np(shm_dq_sim)[0])
            buf_tau_sim.append(_np(shm_tau_sim)[0])
            buf_q_real.append(_np(shm_q_real)[0])
            buf_dq_real.append(_np(shm_dq_real)[0])
            buf_tau_real.append(_np(shm_tau_real)[0])
            buf_kp_sim.append(p["kp"])
            buf_kd_sim.append(p["kd"])
            buf_damping.append(p["damping"])
            buf_armature.append(p["armature"])
            buf_frictionloss.append(p["frictionloss"])
            buf_sig_mode.append(sp["mode"])

            elapsed = time.time() - t
            if elapsed < dt:
                time.sleep(dt - elapsed)

        # Save
        if not buf_t:
            print("[rec] No data collected, skipping save.")
            start_event.clear()
            continue

        _DATA.mkdir(exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sig_mode = buf_sig_mode[0] if buf_sig_mode else "unknown"
        out_path = _DATA / f"{ts}_load_{sig_mode}.npz"

        with _sig_lock:
            sp_final = _sig_params.copy()

        np.savez(
            str(out_path),
            t            = np.array(buf_t),
            target       = np.array(buf_target),
            q_sim        = np.array(buf_q_sim),
            dq_sim       = np.array(buf_dq_sim),
            tau_sim      = np.array(buf_tau_sim),
            q_real       = np.array(buf_q_real),
            dq_real      = np.array(buf_dq_real),
            tau_real     = np.array(buf_tau_real),
            kp_sim       = np.array(buf_kp_sim),
            kd_sim       = np.array(buf_kd_sim),
            kp_real      = np.array([_np(shm_kp_real)[0]]),
            kd_real      = np.array([_np(shm_kd_real)[0]]),
            damping      = np.array(buf_damping),
            armature     = np.array(buf_armature),
            frictionloss = np.array(buf_frictionloss),
            control_hz   = np.array([control_hz]),
            signal_mode  = np.array([sig_mode]),
            sig_offset_rad    = np.array([sp_final["offset"]]),
            sig_amplitude_rad = np.array([sp_final["amplitude"]]),
            sig_frequency_hz  = np.array([sp_final["frequency"]]),
            sig_chirp_f_end   = np.array([sp_final["chirp_f_end"]]),
            sig_chirp_sweep_s = np.array([sp_final["chirp_sweep"]]),
        )
        print(f"[rec] Saved {len(buf_t)} samples → {out_path}")
        start_event.clear()


# ── Viz thread ───────────────────────────────────────────────────────────────────

def viz_thread_fn(
    sim_view:     object,
    real_view:    object,
    model:        mujoco.MjModel,
    data_sim:     mujoco.MjData,
    data_real:    mujoco.MjData,
    shm_q_sim:    mp.Array,
    shm_dq_sim:   mp.Array,
    shm_tau_sim:  mp.Array,
    shm_q_real:   mp.Array,
    shm_dq_real:  mp.Array,
    shm_tau_real: mp.Array,
    shm_viz_hz:   mp.Array,
    plot_qpos:    object,
    plot_qvel:    object,
    plot_tau:     object,
    stop_event:   threading.Event,
) -> None:
    period  = 1.0 / CONTROL_HZ
    iters   = 0
    t_rate  = time.time()
    t_start = time.time()

    max_pts = int(PLOT_WINDOW * CONTROL_HZ)

    buf_t        = np.zeros(max_pts)
    buf_q_sim    = np.zeros(max_pts)
    buf_q_real   = np.zeros(max_pts)
    buf_dq_sim   = np.zeros(max_pts)
    buf_dq_real  = np.zeros(max_pts)
    buf_tau_sim  = np.zeros(max_pts)
    buf_tau_real = np.zeros(max_pts)
    idx    = 0
    filled = 0

    while not stop_event.is_set():
        t = time.time()

        data_sim.qpos[0] = _np(shm_q_sim)[0]
        mujoco.mj_kinematics(model, data_sim)
        sim_view.update(data_sim)

        data_real.qpos[0] = _np(shm_q_real)[0]
        mujoco.mj_kinematics(model, data_real)
        real_view.update(data_real)

        now             = t - t_start
        buf_t[idx]        = now
        buf_q_sim[idx]    = _np(shm_q_sim)[0]
        buf_q_real[idx]   = _np(shm_q_real)[0]
        buf_dq_sim[idx]   = _np(shm_dq_sim)[0]
        buf_dq_real[idx]  = _np(shm_dq_real)[0]
        buf_tau_sim[idx]  = _np(shm_tau_sim)[0]
        buf_tau_real[idx] = _np(shm_tau_real)[0]
        idx    = (idx + 1) % max_pts
        filled = min(filled + 1, max_pts)

        if filled < max_pts:
            sl          = slice(0, filled)
            t_plot      = buf_t[sl].copy()
            q_sim_p     = buf_q_sim[sl].copy()
            q_real_p    = buf_q_real[sl].copy()
            dq_sim_p    = buf_dq_sim[sl].copy()
            dq_real_p   = buf_dq_real[sl].copy()
            tau_sim_p   = buf_tau_sim[sl].copy()
            tau_real_p  = buf_tau_real[sl].copy()
        else:
            order       = np.roll(np.arange(max_pts), -idx)
            t_plot      = buf_t[order]
            q_sim_p     = buf_q_sim[order]
            q_real_p    = buf_q_real[order]
            dq_sim_p    = buf_dq_sim[order]
            dq_real_p   = buf_dq_real[order]
            tau_sim_p   = buf_tau_sim[order]
            tau_real_p  = buf_tau_real[order]

        plot_qpos.data = (t_plot, q_sim_p,  q_real_p)
        plot_qvel.data = (t_plot, dq_sim_p, dq_real_p)
        plot_tau.data  = (t_plot, tau_sim_p, tau_real_p)

        iters += 1
        elapsed_rate = time.time() - t_rate
        if elapsed_rate >= 1.0:
            _np(shm_viz_hz)[0] = iters / elapsed_rate
            iters  = 0
            t_rate = time.time()

        elapsed = time.time() - t
        if elapsed < period:
            time.sleep(period - elapsed)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Motor load sysid recorder")
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    parser.add_argument("--device", type=str, default="/dev/ttyUSB0")
    args = parser.parse_args()

    # ── Shared memory ────────────────────────────────────────────────────────────
    shm_q_real   = mp.Array(ctypes.c_double, 1)
    shm_dq_real  = mp.Array(ctypes.c_double, 1)
    shm_tau_real = mp.Array(ctypes.c_double, 1)
    shm_q_sim    = mp.Array(ctypes.c_double, 1)
    shm_dq_sim   = mp.Array(ctypes.c_double, 1)
    shm_tau_sim  = mp.Array(ctypes.c_double, 1)
    shm_ctrl     = mp.Array(ctypes.c_double, 1)
    shm_real_hz  = mp.Array(ctypes.c_double, 1)
    shm_sim_hz   = mp.Array(ctypes.c_double, 1)
    shm_viz_hz   = mp.Array(ctypes.c_double, 1)
    shm_kp_real  = mp.Array(ctypes.c_double, 1)
    shm_kd_real  = mp.Array(ctypes.c_double, 1)

    connect_mp = mp.Event()
    stop_mp    = mp.Event()

    # ── Spawn real process ───────────────────────────────────────────────────────
    real_proc = mp.Process(
        target=real_process_fn,
        args=(args.control_hz, args.device,
              shm_q_real, shm_dq_real, shm_tau_real,
              shm_ctrl, shm_real_hz, shm_kp_real, shm_kd_real,
              connect_mp, stop_mp),
        daemon=True,
    )
    real_proc.start()

    # ── Load MuJoCo model ────────────────────────────────────────────────────────
    print("Loading MuJoCo model…")
    model     = mujoco.MjModel.from_xml_path(str(_XML))
    data_sim  = mujoco.MjData(model)
    data_real = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data_sim,  model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data_real, model.key("home").id)
    mujoco.mj_forward(model, data_sim)
    mujoco.mj_forward(model, data_real)

    # ── Viser ────────────────────────────────────────────────────────────────────
    print("Starting viser…")
    server   = viser.ViserServer(label="Load Sysid Recorder")
    scene    = ViserMujocoScene.create(server, model)
    sim_view  = scene.add_robot("sim",  color=(0.75, 0.75, 0.75, 1.00))
    real_view = scene.add_robot("real", color=(0.20, 0.55, 0.90, 0.65))
    scene.create_visualization_gui(
        camera_distance=0.5, camera_azimuth=135.0, camera_elevation=25.0
    )

    _zeros      = np.zeros(1)
    _t0         = np.zeros(1)
    _sim_series  = uplot.Series(label="sim",  stroke="#222222", width=1.5)
    _real_series = uplot.Series(label="real", stroke="#3388eb", width=1.5)

    # ── GUI ──────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Sim2Real"):
        chk_connect  = server.gui.add_checkbox("Connect robot", initial_value=False)
        txt_status   = server.gui.add_text("Robot status", initial_value="Disconnected")
        txt_real_hz  = server.gui.add_text("Real rate",  initial_value="— Hz")
        txt_sim_hz   = server.gui.add_text("Sim rate",   initial_value="— Hz")
        txt_viz_hz   = server.gui.add_text("Viz rate",   initial_value="— Hz")

    @chk_connect.on_update
    def _on_connect(_):
        if chk_connect.value and not connect_mp.is_set():
            connect_mp.set()
            txt_status.value = "Connecting…"
        elif not chk_connect.value and connect_mp.is_set():
            stop_mp.set()
            txt_status.value = "Disconnected"

    with server.gui.add_folder("Signal"):
        dd_mode = server.gui.add_dropdown(
            "Mode",
            options=["hold", "sine", "step", "triangle", "chirp"],
            initial_value="hold",
        )
        sl_offset = server.gui.add_slider(
            "Offset (rad)", min=-3.14, max=3.14, step=0.01, initial_value=0.0,
        )
        sl_amplitude = server.gui.add_slider(
            "Amplitude (rad)", min=0.0, max=3.14, step=0.01, initial_value=0.5,
        )
        sl_freq = server.gui.add_slider(
            "Frequency (Hz)", min=0.05, max=10.0, step=0.05, initial_value=0.5,
        )
        sl_chirp_f_end = server.gui.add_slider(
            "Chirp f_end (Hz)", min=0.1, max=20.0, step=0.1, initial_value=5.0,
        )
        sl_chirp_sweep = server.gui.add_slider(
            "Chirp sweep (s)", min=1.0, max=60.0, step=0.5, initial_value=10.0,
        )
        btn_reset_phase = server.gui.add_button("Reset signal phase")

    sig = LoadSignalGen()

    @btn_reset_phase.on_click
    def _on_reset(_):
        sig.reset()

    with server.gui.add_folder("Physics params"):
        sl_kp           = server.gui.add_slider("kp",            min=0.0,  max=120.0, step=0.5,   initial_value=DEFAULT_KP)
        sl_kd           = server.gui.add_slider("kd",            min=0.0,  max=5.0,   step=0.05,  initial_value=DEFAULT_KD)
        sl_damping      = server.gui.add_slider("joint_damping", min=0.0,  max=5.0,   step=0.01,  initial_value=DEFAULT_DAMPING)
        sl_armature     = server.gui.add_slider("armature",      min=0.0,  max=1.0,   step=0.001, initial_value=DEFAULT_ARMATURE)
        sl_frictionloss = server.gui.add_slider("frictionloss",  min=0.0,  max=1.0,   step=0.005, initial_value=DEFAULT_FRICTIONLOSS)

    with server.gui.add_folder("Recording"):
        btn_record  = server.gui.add_button("Start recording")
        btn_stop    = server.gui.add_button("Stop & save")
        txt_rec     = server.gui.add_text("Rec status", initial_value="Idle")

    # ── Plots ────────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Plots"):
        plot_qpos = server.gui.add_uplot(
            data=(_t0, _zeros, _zeros),
            series=(uplot.Series(label="t"), _sim_series, _real_series),
            title="Joint position (rad)",
            aspect=2.5,
        )
        plot_qvel = server.gui.add_uplot(
            data=(_t0, _zeros, _zeros),
            series=(uplot.Series(label="t"), _sim_series, _real_series),
            title="Joint velocity (rad/s)",
            aspect=2.5,
        )
        plot_tau = server.gui.add_uplot(
            data=(_t0, _zeros, _zeros),
            series=(uplot.Series(label="t"), _sim_series, _real_series),
            title="Torque (Nm)",
            aspect=2.5,
        )

    # ── Recording thread events ──────────────────────────────────────────────────
    th_stop      = threading.Event()
    rec_start    = threading.Event()
    rec_stop     = threading.Event()

    @btn_record.on_click
    def _on_record(_):
        if not rec_start.is_set():
            rec_start.set()
            txt_rec.value = "Recording…"
        else:
            txt_rec.value = "Already recording — press Stop to save."

    @btn_stop.on_click
    def _on_stop(_):
        if rec_start.is_set():
            rec_start.clear()
            txt_rec.value = "Saving…"
        else:
            txt_rec.value = "Not recording."

    # ── Start threads ────────────────────────────────────────────────────────────
    threading.Thread(
        target=sim_thread_fn, daemon=True,
        args=(model, data_sim, shm_ctrl,
              shm_q_sim, shm_dq_sim, shm_tau_sim,
              shm_sim_hz, th_stop),
    ).start()

    threading.Thread(
        target=viz_thread_fn, daemon=True,
        args=(sim_view, real_view, model, data_sim, data_real,
              shm_q_sim, shm_dq_sim, shm_tau_sim,
              shm_q_real, shm_dq_real, shm_tau_real,
              shm_viz_hz, plot_qpos, plot_qvel, plot_tau, th_stop),
    ).start()

    rec_thread = threading.Thread(
        target=recorder_thread_fn, daemon=False,
        args=(args.control_hz,
              shm_ctrl,
              shm_q_sim, shm_dq_sim, shm_tau_sim,
              shm_q_real, shm_dq_real, shm_tau_real,
              shm_kp_real, shm_kd_real,
              rec_start, rec_stop),
    )
    rec_thread.start()

    # ── Main loop (60 Hz) ────────────────────────────────────────────────────────
    print("Running. Ctrl+C to stop.")
    try:
        while True:
            # Update signal params from GUI
            with _sig_lock:
                _sig_params["mode"]        = dd_mode.value
                _sig_params["offset"]      = float(sl_offset.value)
                _sig_params["amplitude"]   = float(sl_amplitude.value)
                _sig_params["frequency"]   = float(sl_freq.value)
                _sig_params["chirp_f_end"] = float(sl_chirp_f_end.value)
                _sig_params["chirp_sweep"] = float(sl_chirp_sweep.value)

            # Update physics params from GUI
            with _phys_lock:
                _phys_params["kp"]           = float(sl_kp.value)
                _phys_params["kd"]           = float(sl_kd.value)
                _phys_params["damping"]      = float(sl_damping.value)
                _phys_params["armature"]     = float(sl_armature.value)
                _phys_params["frictionloss"] = float(sl_frictionloss.value)

            # Generate target signal and push to shared memory
            _np(shm_ctrl)[0] = sig(time.time())

            # Update status text
            real_hz = _np(shm_real_hz)[0]
            txt_real_hz.value = f"{real_hz:.0f} Hz"
            txt_sim_hz.value  = f"{_np(shm_sim_hz)[0]:.0f} Hz"
            txt_viz_hz.value  = f"{_np(shm_viz_hz)[0]:.0f} Hz"

            if not connect_mp.is_set() or stop_mp.is_set():
                txt_status.value = "Disconnected"
            elif real_hz > 0:
                txt_status.value = f"Connected ({real_hz:.0f} Hz)"

            if rec_start.is_set():
                txt_rec.value = "Recording…"
            elif not rec_start.is_set() and txt_rec.value == "Saving…":
                pass  # recorder thread will update after save

            time.sleep(1.0 / 60)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        th_stop.set()
        stop_mp.set()
        rec_start.clear()   # stop recording if active
        rec_stop.set()
        rec_thread.join()   # wait for recorder to finish saving
        real_proc.join(timeout=5.0)
        server.stop()
        print("Done.")


if __name__ == "__main__":
    main()
