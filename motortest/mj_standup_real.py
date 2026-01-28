#!/usr/bin/env python3
# rumi-custom-quadruped — Reinforcement-learning-based quadruped-robot control framework for custom quadruped
# Copyright (C) 2025  Vishwanath R
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Real Robot Stand-Up Controller with Parallel MuJoCo Simulation

Architecture:
    Process 1 (Simulation): Runs MuJoCo + IK at 50 Hz
        - Computes target joint positions using inverse kinematics
        - Updates shared memory with latest targets
        - Provides visual feedback via MuJoCo viewer

    Process 2 (Control): Runs motor control at 100 Hz
        - Reads target positions from shared memory
        - Commands real motors via mx64_controller
        - Non-blocking, real-time priority

Key Features:
    - Parallel processing prevents simulation from blocking motor control
    - Shared memory for low-latency IPC (no queues, no locks)
    - Automatic motor initialization with PID configuration
    - Safe shutdown on Ctrl+C
"""

import sys
import os
import time
import signal
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Process, Array, Value
from ctypes import c_double, c_bool

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "policy"))

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

from mx64_controller import MX64Controller


class QuadController:
    """State machine for sit-stand motion control."""

    def __init__(self):
        # Height definitions
        self.sit_height = 0.05        # Height when sitting (5cm)
        self.stand_height = 0.2       # Height when standing (20cm)
        self.transition_speed = 0.003 # Speed of height transition (3mm per step)

        # State machine
        self.state = "sitting"        # "sitting", "standing_up", "standing", "sitting_down"
        self.target_height = self.sit_height
        self.hold_duration = 2.0      # Hold each position for 2 seconds
        self.last_state_change = time.time()

    def update_height(self, data, base_mid):
        """
        Update the mocap height based on current state.

        Args:
            data: MuJoCo data object
            base_mid: Mocap ID for base target

        Returns:
            tuple: (state, current_height)
        """
        current_time = time.time()
        current_height = data.mocap_pos[base_mid][2]

        # State transitions based on time
        if current_time - self.last_state_change > self.hold_duration:
            if self.state == "sitting":
                self.state = "standing_up"
                self.target_height = self.stand_height
                self.last_state_change = current_time
            elif self.state == "standing":
                self.state = "sitting_down"
                self.target_height = self.sit_height
                self.last_state_change = current_time

        # Check if we've reached target height
        if abs(current_height - self.target_height) < 0.01:
            if self.state == "standing_up":
                self.state = "standing"
                self.last_state_change = current_time
            elif self.state == "sitting_down":
                self.state = "sitting"
                self.last_state_change = current_time

        # Move towards target height
        if current_height < self.target_height:
            data.mocap_pos[base_mid][2] += self.transition_speed
        elif current_height > self.target_height:
            data.mocap_pos[base_mid][2] -= self.transition_speed

        return self.state, current_height


def simulation_process(joint_positions, running_flag, xml_path):
    """
    Simulation process: Runs MuJoCo + IK at 50 Hz.

    Computes target joint positions and updates shared memory.
    Provides visual feedback via MuJoCo viewer.

    Args:
        joint_positions: Shared memory array for joint positions (12 floats)
        running_flag: Shared flag to signal shutdown
        xml_path: Path to MuJoCo XML model
    """
    print("[SIM] Simulation process started")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize quad controller
    quad_controller = QuadController()

    # Setup IK
    configuration = mink.Configuration(model)
    feet = ["FL", "FR", "BR", "BL"]

    base_task = mink.FrameTask(
        frame_name="body_mid",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    base_mid = model.body("body_target").mocapid[0]
    feet_end = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    # IK settings
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # Configure visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset to home position
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, "body_target", "body_mid", "site")

        rate = RateLimiter(frequency=50.0, warn=False)
        print("[SIM] Starting simulation loop at 50 Hz")

        while viewer.is_running() and running_flag.value:
            # Update quad height using the controller
            state, current_height = quad_controller.update_height(data, base_mid)

            # Set IK targets
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_end[i]))

            # Solve IK
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)

                # Check convergence
                pos_achieved = True
                ori_achieved = True
                for task in [base_task, *feet_tasks]:
                    err = task.compute_error(configuration)
                    pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
                    ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
                if pos_achieved and ori_achieved:
                    break

            # Update simulation
            data.ctrl = configuration.q[7:]  # Skip base pose (7 DOF: xyz + quat)

            # Write joint positions to shared memory
            # configuration.q[7:] gives us the 12 joint positions
            for i, pos in enumerate(configuration.q[7:]):
                joint_positions[i] = pos

            # Step simulation
            for _ in range(10):
                mujoco.mj_step(model, data)

            # Visualize
            viewer.sync()
            rate.sleep()

    print("[SIM] Simulation process ended")
    running_flag.value = False


def control_process(joint_positions, running_flag, control_freq, mujoco_to_motor_map, radians_to_raw, back_motors):
    """
    Control process: Commands real motors at high frequency.

    Reads target joint positions from shared memory and commands motors.
    Non-blocking, real-time control loop.

    Args:
        joint_positions: Shared memory array for joint positions (12 floats, in radians from MuJoCo)
        running_flag: Shared flag to signal shutdown
        control_freq: Control loop frequency in Hz
        mujoco_to_motor_map: Mapping from MuJoCo joint indices to motor IDs
        radians_to_raw: Conversion factor from radians to raw motor units
        back_motors: List of motor IDs that need sign flip (back motors)
    """
    print("[CTRL] Control process started")

    # Initialize motor controller
    controller = MX64Controller()

    # Initialize motors (connect, discover, configure, enable)
    motor_ids = controller.initialize(expected_motors=12)
    if motor_ids is None:
        print("[CTRL] Failed to initialize motors")
        running_flag.value = False
        return

    print(f"[CTRL] Motors initialized: {motor_ids}")

    # Get reference positions (initial standing position)
    reference_positions = controller.initial_positions.copy()
    print(f"[CTRL] Reference positions: {reference_positions}")

    # Setup control loop
    rate = RateLimiter(frequency=control_freq, warn=False)
    print(f"[CTRL] Starting control loop at {control_freq} Hz")

    iteration = 0
    last_print = time.time()

    try:
        while running_flag.value:
            # Read target positions from shared memory (in radians from MuJoCo)
            target_positions_radians = [joint_positions[i] for i in range(12)]

            # Convert to motor positions
            # MuJoCo gives joint angles in radians
            # Motors expect raw units (0-4095)
            # mapping: mujoco_idx -> motor_id
            target_positions_raw = {}
            for mujoco_idx, motor_id in enumerate(mujoco_to_motor_map):
                # Get offset from MuJoCo (in radians)
                offset_radians = target_positions_radians[mujoco_idx]

                # Apply sign flip for back motors (BL and BR)
                if motor_id in back_motors:
                    offset_radians = -offset_radians

                # Convert radians to raw units, add reference position
                offset_raw = int(offset_radians * radians_to_raw)
                target_positions_raw[motor_id] = reference_positions[motor_id] + offset_raw

            # Command motors (all at once via GroupSync)
            success = controller.sync_write_positions(target_positions_raw)

            if not success and iteration % 100 == 0:
                print(f"[CTRL] Warning: Failed to write positions at iteration {iteration}")

            # Status update (every second)
            iteration += 1
            if time.time() - last_print > 1.0:
                print(f"[CTRL] Running at {iteration / (time.time() - last_print + 1e-6):.1f} Hz")
                last_print = time.time()
                iteration = 0

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[CTRL] Interrupted by user")
    finally:
        print("[CTRL] Returning to reference position...")
        controller.sync_write_positions(reference_positions)
        time.sleep(0.5)

        print("[CTRL] Disconnecting motors...")
        controller.disconnect()
        print("[CTRL] Control process ended")
        running_flag.value = False


def main():
    parser = argparse.ArgumentParser(
        description="Real robot stand-up controller with parallel MuJoCo simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 100 Hz control, 50 Hz simulation
  python mj_standup_real.py

  # Custom control frequency
  python mj_standup_real.py -f 200

  # Custom XML path
  python mj_standup_real.py --xml /path/to/model.xml

Architecture:
  Process 1 (Simulation): MuJoCo + IK at 50 Hz
  Process 2 (Control):    Motor control at 100 Hz (or specified)

Press Ctrl+C to stop gracefully.
        """
    )

    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=100.0,
        help="Motor control frequency in Hz (default: 100)"
    )

    parser.add_argument(
        "--xml",
        type=str,
        default="/home/robot/vishwa/rumi/policy/xml/rumi.xml",
        help="Path to MuJoCo XML model (default: policy/xml/rumi.xml)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.freq <= 0 or args.freq > 1000:
        print("[ERROR] Control frequency must be between 0 and 1000 Hz")
        sys.exit(1)

    if not os.path.exists(args.xml):
        print(f"[ERROR] XML file not found: {args.xml}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("   Real Robot Stand-Up Controller")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Control frequency: {args.freq} Hz")
    print(f"  Simulation frequency: 50 Hz")
    print(f"  MuJoCo model: {args.xml}")
    print("=" * 60)

    # Define motor mapping
    # MuJoCo joint order (from configuration.q[7:]): FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, BR_hip, BR_thigh, BR_calf, BL_hip, BL_thigh, BL_calf
    # Real motor IDs: FL=[1,2,3], RL=[4,5,6], RR=[7,8,9], FR=[10,11,12]
    # Need to verify this mapping from your URDF/XML
    mujoco_to_motor_map = [
        1, 2, 3,    # FL: hip, thigh, calf
        10, 11, 12, # FR: hip, thigh, calf
        7, 8, 9,    # BR (Rear Right): hip, thigh, calf
        4, 5, 6,    # BL (Rear Left): hip, thigh, calf
    ]

    # Conversion factor: radians to raw motor units
    # MX-64: 4096 units per revolution = 4096 / (2*pi) units per radian
    radians_to_raw = 4096.0 / (2.0 * np.pi)

    # Sign flip for back motors (both BL and BR)
    # Back motors need opposite sign due to mechanical mounting or URDF convention
    BACK_MOTORS = [4, 5, 6, 7, 8, 9]  # BL (4,5,6) + BR (7,8,9)

    # Shared memory for joint positions (12 joints)
    joint_positions = Array(c_double, 12)
    for i in range(12):
        joint_positions[i] = 0.0

    # Shared flag for graceful shutdown
    running_flag = Value(c_bool, True)

    # Signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\n[MAIN] Ctrl+C detected, shutting down...")
        running_flag.value = False

    signal.signal(signal.SIGINT, signal_handler)

    # Create processes
    sim_proc = Process(
        target=simulation_process,
        args=(joint_positions, running_flag, args.xml),
        name="SimulationProcess"
    )

    ctrl_proc = Process(
        target=control_process,
        args=(joint_positions, running_flag, args.freq, mujoco_to_motor_map, radians_to_raw, BACK_MOTORS),
        name="ControlProcess"
    )

    print("\n[MAIN] Starting processes...")

    # Start simulation first (so viewer opens)
    sim_proc.start()
    time.sleep(2.0)  # Give simulation time to initialize

    # Start control process
    ctrl_proc.start()

    print("[MAIN] Both processes running. Press Ctrl+C to stop.\n")

    # Wait for processes to finish
    try:
        sim_proc.join()
        ctrl_proc.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted, waiting for processes to finish...")
        running_flag.value = False
        sim_proc.join(timeout=5.0)
        ctrl_proc.join(timeout=5.0)

        if sim_proc.is_alive():
            print("[MAIN] Terminating simulation process...")
            sim_proc.terminate()
        if ctrl_proc.is_alive():
            print("[MAIN] Terminating control process...")
            ctrl_proc.terminate()

    print("\n[MAIN] All processes stopped. Goodbye!\n")


if __name__ == "__main__":
    main()
