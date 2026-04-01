"""Rumi getup validation — npz replay with independent sim re-inference.

Workflow
────────
  1. Record on real hardware:
         python getup/run_getup.py --record --duration 5
     → saves getup/recordings/getup_<timestamp>.npz

  2. Replay on laptop (no hardware needed):
         python validation/run_validation.py getup/recordings/getup_<timestamp>.npz

Three robots in viser
─────────────────────
  target (green) : recorded raw_action * action_scale — the exact position command
                   the real robot received each step.
  real   (blue)  : recorded joint_pos — what the real robot's encoders measured.
  sim    (grey)  : MuJoCo re-simulation using Option B:
                     each step the sim builds its own observations from its own state,
                     calls the policy independently, and steps forward.
                   This shows how sim obs → policy diverges from the real obs → policy
                   loop over time — the full sim2real gap.

The target-height slider lets you override the goal mid-replay to see how each
robot responds.  The replay pauses and resets when it reaches the end, or loops
if --loop is passed.

Run
───
  python run_validation.py <recording.npz>
  python run_validation.py <recording.npz> --loop
  python run_validation.py <recording.npz> --speed 2.0   # 2× faster
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
_DEPLOY        = _HERE.parent
_SIM2REAL      = _DEPLOY.parent
_RUMI_MJLAB_SRC = _SIM2REAL / "rumi-mjlab" / "src"
_GETUP_DIR     = _DEPLOY / "getup"
_SCENE_XML     = _HERE / "scene.xml"
_DEFAULT_REC_DIR = _GETUP_DIR / "recordings"

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_DEPLOY))
sys.path.insert(0, str(_GETUP_DIR))
sys.path.insert(0, str(_RUMI_MJLAB_SRC))

from viewer import ViserMujocoScene    # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
VIZ_HZ = 50

# Joint order: FL FR BL BR  (matches observations.py JOINT_ORDER and MuJoCo XML)
JOINT_NAMES = [
    "FL_hip_joint",  "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint",  "FR_calf_joint",
    "BL_hip_joint",  "BL_thigh_joint",  "BL_calf_joint",
    "BR_hip_joint",  "BR_thigh_joint",  "BR_calf_joint",
]
N_JOINTS = len(JOINT_NAMES)


# ── Obs building from MuJoCo state ────────────────────────────────────────────

def build_obs_sim(
    data:            mujoco.MjData,
    model:           mujoco.MjModel,
    last_action:     np.ndarray,
    target_height_m: float,
) -> np.ndarray:
    """41-dim observation from MuJoCo sim state (mirrors observations.py layout)."""
    _H_MIN, _H_MAX = 0.10, 0.25

    body_height_obs = float(np.clip(
        (float(data.qpos[2]) - _H_MIN) / (_H_MAX - _H_MIN), -0.5, 1.5
    ))
    target_height_obs = float((target_height_m - _H_MIN) / (_H_MAX - _H_MIN))

    body_id   = model.body("body").id
    R_w2b     = data.xmat[body_id].reshape(3, 3).T
    proj_grav = (R_w2b @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

    obs = np.concatenate([
        [body_height_obs],
        [target_height_obs],
        proj_grav,
        data.qpos[7:19].astype(np.float32),
        data.qvel[6:18].astype(np.float32),
        last_action.astype(np.float32),
    ])
    return obs.astype(np.float32)


# ── Ghost foot-centroid placement ─────────────────────────────────────────────

def _place_ghost(
    model:             mujoco.MjModel,
    data:              mujoco.MjData,
    q_joints:          np.ndarray,
    sim_foot_centroid: np.ndarray,
    foot_ids:          list,
) -> None:
    data.qpos[7:]  = q_joints
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [0.0, 0.0, 0.2]
    mujoco.mj_kinematics(model, data)
    ghost_centroid = np.mean([data.site(sid).xpos for sid in foot_ids], axis=0)
    data.qpos[0:3] += sim_foot_centroid - ghost_centroid
    mujoco.mj_kinematics(model, data)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rumi getup validation (npz replay)")
    parser.add_argument("recording", type=str,
                        help="Path to .npz file from run_getup.py --record")
    parser.add_argument("--loop",    action="store_true",
                        help="Loop replay continuously instead of stopping at end")
    parser.add_argument("--speed",   type=float, default=1.0,
                        help="Replay speed multiplier (default: 1.0 = real-time)")
    args = parser.parse_args()

    # ── Load recording ────────────────────────────────────────────────────────
    rec_path = Path(args.recording)
    if not rec_path.exists():
        print(f"[ERROR] Recording not found: {rec_path}")
        sys.exit(1)

    rec          = np.load(rec_path)
    rec_quat      = rec["quat"]        # (T, 4)  float32  IMU quaternion (w,x,y,z)
    rec_joint_pos = rec["joint_pos"]   # (T, 12) float32  encoder pos rel. sit-pose zero
    rec_joint_vel = rec["joint_vel"]   # (T, 12) float32  encoder vel rad/s
    rec_action    = rec["raw_action"]  # (T, 12) float32  unscaled policy output
    action_scale  = float(rec["action_scale"])
    control_hz    = float(rec["control_hz"])
    target_height_default = float(rec["target_height"])
    T             = rec_quat.shape[0]
    DT            = 1.0 / control_hz

    print("\n" + "=" * 60)
    print("   Rumi Getup — Validation (npz replay)")
    print("=" * 60)
    print(f"  Recording  : {rec_path.name}")
    print(f"  Steps      : {T}  ({T/control_hz:.1f} s @ {control_hz:.0f} Hz)")
    print(f"  Target ht  : {target_height_default:.3f} m")
    print(f"  Act scale  : {action_scale}")
    print(f"  Speed      : {args.speed}×")
    print(f"  Loop       : {args.loop}")
    print()

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    print("Loading MuJoCo model …")
    model       = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data_sim    = mujoco.MjData(model)
    data_target = mujoco.MjData(model)
    data_real   = mujoco.MjData(model)

    key_id = model.key("sitting").id
    for d in (data_sim, data_target, data_real):
        mujoco.mj_resetDataKeyframe(model, d, key_id)
        mujoco.mj_forward(model, d)

    foot_ids     = [model.site(f).id for f in ["FL", "FR", "BL", "BR"]]
    n_substeps   = max(1, round(DT / model.opt.timestep))

    # ── Load policy ───────────────────────────────────────────────────────────
    print("Loading policy …")
    from policy import GetupPolicy
    policy = GetupPolicy(str(_GETUP_DIR / "checkpoint" / "latest_getup.pt"))

    # Warmup
    policy(build_obs_sim(data_sim, model, np.zeros(12, np.float32), target_height_default))
    print()

    # ── Shared state (main thread ↔ viz thread) ───────────────────────────────
    _lock          = threading.Lock()
    _state         = {
        "q_sim":          data_sim.qpos[7:19].copy(),    # sim joint angles
        "q_target":       np.zeros(N_JOINTS),             # recorded action * scale
        "q_real":         np.zeros(N_JOINTS),             # recorded encoder pos
        "step":           0,
        "total_steps":    T,
        "target_height":  target_height_default,
        "paused":         False,
        "reset":          False,
    }

    # ── Replay + sim thread ───────────────────────────────────────────────────
    stop_event = threading.Event()

    def replay_thread_fn():
        # Reset sim to sitting pose
        mujoco.mj_resetDataKeyframe(model, data_sim, key_id)
        mujoco.mj_forward(model, data_sim)

        sim_last_action = np.zeros(12, np.float32)
        step = 0

        while not stop_event.is_set():
            with _lock:
                paused        = _state["paused"]
                do_reset      = _state["reset"]
                target_height = _state["target_height"]

            if do_reset:
                mujoco.mj_resetDataKeyframe(model, data_sim, key_id)
                mujoco.mj_forward(model, data_sim)
                sim_last_action = np.zeros(12, np.float32)
                step = 0
                with _lock:
                    _state["reset"] = False
                    _state["step"]  = 0
                continue

            if paused:
                time.sleep(0.02)
                continue

            t_loop = time.time()

            if step >= T:
                if args.loop:
                    # Reset sim and restart
                    mujoco.mj_resetDataKeyframe(model, data_sim, key_id)
                    mujoco.mj_forward(model, data_sim)
                    sim_last_action = np.zeros(12, np.float32)
                    step = 0
                    with _lock:
                        _state["step"] = 0
                    continue
                else:
                    with _lock:
                        _state["paused"] = True
                    continue

            # ── Option B: sim builds its own obs and calls policy ─────────────
            mujoco.mj_forward(model, data_sim)
            obs_sim    = build_obs_sim(data_sim, model, sim_last_action, target_height)
            raw_sim    = policy(obs_sim)
            data_sim.ctrl[:] = raw_sim * action_scale
            for _ in range(n_substeps):
                mujoco.mj_step(model, data_sim)
            sim_last_action = raw_sim

            # ── Recorded data for target and real ghosts ──────────────────────
            q_target = rec_action[step] * action_scale   # what was commanded
            q_real   = rec_joint_pos[step]               # what encoders measured

            with _lock:
                _state["q_sim"]    = data_sim.qpos[7:19].copy()
                _state["q_target"] = q_target.copy()
                _state["q_real"]   = q_real.copy()
                _state["step"]     = step + 1

            step += 1

            elapsed = time.time() - t_loop
            sleep   = DT / args.speed - elapsed
            if sleep > 0:
                time.sleep(sleep)

    threading.Thread(target=replay_thread_fn, daemon=True).start()

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

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Validation"):
        txt_step   = server.gui.add_text("Step",          initial_value=f"0 / {T}")
        txt_time   = server.gui.add_text("Time",          initial_value="0.00 s")

        sl_target  = server.gui.add_slider(
            "Target height (m)", min=0.18, max=0.31, step=0.01,
            initial_value=target_height_default,
        )
        sl_speed   = server.gui.add_slider(
            "Speed", min=0.1, max=4.0, step=0.1, initial_value=args.speed,
        )
        btn_pause  = server.gui.add_button("Pause / Resume")
        btn_reset  = server.gui.add_button("Reset")

    @sl_target.on_update
    def _(_):
        with _lock:
            _state["target_height"] = sl_target.value

    @btn_pause.on_click
    def _(_):
        with _lock:
            _state["paused"] = not _state["paused"]

    @btn_reset.on_click
    def _(_):
        with _lock:
            _state["reset"]  = True
            _state["paused"] = False

    # ── Viz loop (main thread) ────────────────────────────────────────────────
    print("Running. Open http://localhost:8080 in a browser. Ctrl+C to stop.\n")
    period = 1.0 / VIZ_HZ

    try:
        while True:
            t = time.time()

            with _lock:
                q_sim    = _state["q_sim"].copy()
                q_target = _state["q_target"].copy()
                q_real   = _state["q_real"].copy()
                step     = _state["step"]
                speed    = sl_speed.value

            args.speed = speed   # feed slider back to replay thread

            # ── Sim (grey) ────────────────────────────────────────────────────
            data_sim.qpos[7:] = q_sim
            mujoco.mj_kinematics(model, data_sim)
            sim_view.update(data_sim)

            sim_foot_centroid = np.mean(
                [data_sim.site(sid).xpos for sid in foot_ids], axis=0
            )

            # ── Target (green) ────────────────────────────────────────────────
            _place_ghost(model, data_target, q_target, sim_foot_centroid, foot_ids)
            target_view.update(data_target)

            # ── Real (blue) ───────────────────────────────────────────────────
            _place_ghost(model, data_real, q_real, sim_foot_centroid, foot_ids)
            real_view.update(data_real)

            # ── GUI status ────────────────────────────────────────────────────
            txt_step.value = f"{step} / {T}"
            txt_time.value = f"{step / control_hz:.2f} s"

            elapsed = time.time() - t
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received.")
    finally:
        stop_event.set()
        server.stop()
        print("[DONE]")


if __name__ == "__main__":
    main()
