#!/usr/bin/env python3

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to perform desk cleanup task with tactile sensor visualization.

This script demonstrates a desk cleanup task where the robot:
- Identifies objects on a desk
- Grasps individual objects
- Moves them to target locations
- Real-time visualization of GelSight sensor data

Usage:
    python scripts/desk_cleanup_demo.py --num_objects 6 --target_object 0
    python scripts/desk_cleanup_demo.py --headless --num_objects 4
"""
 # NOTE： with USD files failed, try URDF files /home/pi-zero/isaac-sim/TacEx/scripts/desk_cleanup_urdf_demo.py
"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Desk cleanup with tactile visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_objects", type=int, default=6, help="Number of objects on desk (default: 6, max: 6).")
parser.add_argument("--target_object", type=int, default=0, help="Target object index to grasp.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Force enable cameras for visualization
args_cli.enable_cameras = True

# Check if running in headless mode
HEADLESS = args_cli.headless

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

# Import configuration
from tacex_tasks.desk_cleanup.desk_cleanup_env_cfg import DeskCleanupSceneCfg, create_desk_objects
from tacex import GelSightSensor


class CleanupState:
    """State machine for desk cleanup sequence."""
    IDLE = 0
    MOVE_TO_PREGRASP = 1
    MOVE_TO_GRASP = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    MOVE_TO_PLACE = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8


def visualize_tactile_sensors(
    gelsight_left: GelSightSensor,
    gelsight_right: GelSightSensor,
    window_name_left: str = "GelSight Left",
    window_name_right: str = "GelSight Right",
    headless: bool = False,
    debug: bool = False
):
    """Visualize tactile sensor data in OpenCV windows."""
    if headless:
        return

    # Get tactile RGB data from output dictionary
    if hasattr(gelsight_left.data, 'output') and gelsight_left.data.output:
        tactile_rgb = gelsight_left.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            left_rgb = tactile_rgb[0].cpu().numpy()

            if debug:
                print(f"[DEBUG] Left sensor RGB shape: {left_rgb.shape}, dtype: {left_rgb.dtype}")
                print(f"[DEBUG] Left sensor RGB range: [{left_rgb.min():.3f}, {left_rgb.max():.3f}]")

            left_rgb_scaled = np.clip(left_rgb, 0.0, 1.0)
            left_bgr = cv2.cvtColor((left_rgb_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name_left, left_bgr)

    if hasattr(gelsight_right.data, 'output') and gelsight_right.data.output:
        tactile_rgb = gelsight_right.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            right_rgb = tactile_rgb[0].cpu().numpy()

            if debug:
                print(f"[DEBUG] Right sensor RGB shape: {right_rgb.shape}, dtype: {right_rgb.dtype}")

            right_rgb_scaled = np.clip(right_rgb, 0.0, 1.0)
            right_bgr = cv2.cvtColor((right_rgb_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name_right, right_bgr)

    cv2.waitKey(1)


def run_cleanup_demo(
    scene: InteractiveScene,
    sim: SimulationContext,
    target_object_idx: int = 0
):
    """
    Run desk cleanup demonstration with tactile visualization.

    Args:
        scene: Interactive scene
        sim: Simulation context
        target_object_idx: Index of target object to grasp
    """
    # Extract scene entities
    robot: Articulation = scene["robot"]
    object_collection: RigidObjectCollection = scene["object_collection"]
    gelsight_left: GelSightSensor = scene.sensors["gelsight_left"]
    gelsight_right: GelSightSensor = scene.sensors["gelsight_right"]

    print("\n" + "=" * 80)
    print("DESK CLEANUP DEMO WITH TACTILE VISUALIZATION")
    print("=" * 80)

    # Get object names
    object_names = object_collection.object_names
    num_objects = len(object_names)

    target_name = object_names[target_object_idx] if target_object_idx < num_objects else "unknown"

    print(f"\n[INFO] Target object: {target_name} (index {target_object_idx})")
    print(f"[INFO] Number of environments: {scene.num_envs}")
    print(f"[INFO] Robot joints: {robot.num_joints}")
    print(f"[INFO] Objects on desk: {num_objects} ({', '.join(object_names)})")

    print("\n[CONTROLS]")
    print("  - Press 'q' in tactile windows to quit")
    print("  - Press 'r' to reset")
    print("  - Press Ctrl+C to exit")

    print("\n[CLEANUP SEQUENCE]")
    print("  1. Move to pre-grasp position above object")
    print("  2. Move down to grasp position")
    print("  3. Close gripper")
    print("  4. Lift object")
    print("  5. Move to placement location")
    print("  6. Release object")
    print("  7. Retreat")
    print("  8. Done")

    print("\n" + "=" * 80)

    # Print initial object positions
    print("\n[INFO] Initial object positions:")
    for i, name in enumerate(object_names):
        pos = object_collection.data.object_pos_w[0, i].cpu().numpy()
        print(f"  {name:15s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

    # Create OpenCV windows
    if not HEADLESS:
        cv2.namedWindow("GelSight Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("GelSight Right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GelSight Left", 400, 300)
        cv2.resizeWindow("GelSight Right", 400, 300)

    # State machine
    state = CleanupState.IDLE
    state_timer = 0

    # Use predefined joint positions
    home_joints = robot.data.default_joint_pos[0, :7].clone()

    print(f"\n[DEBUG] Home joint positions: {home_joints.cpu().numpy()}")

    # Define placement location (to the side of the desk)
    placement_location = torch.tensor([0.3, 0.3, 0.15], device=robot.device)

    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Get current object position
        object_positions = object_collection.data.object_pos_w[0]
        target_pos = object_positions[target_object_idx].clone()

        # Get current end-effector position
        ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]

        # Get current gripper state
        gripper_pos = robot.data.joint_pos[:, -2:].mean(dim=1, keepdim=True)

        # State machine
        if state == CleanupState.IDLE:
            if count > 50:
                state = CleanupState.MOVE_TO_PREGRASP
                state_timer = 0
                print(f"\n[STATE] Moving to pre-grasp position above {target_name}")
                print(f"        Target position: {target_pos.cpu().numpy()}")

        elif state == CleanupState.MOVE_TO_PREGRASP:
            # Move above the object
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0.3
            target_joints[2] = 0.0
            target_joints[3] = -2.2
            target_joints[4] = 0.0
            target_joints[5] = 2.0
            target_joints[6] = 0.785

            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Keep gripper open
            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            joint_error = torch.norm(robot.data.joint_pos[0, :7] - target_joints).item()

            if joint_error < 1.0 and state_timer > 300:
                state = CleanupState.MOVE_TO_GRASP
                state_timer = 0
                print(f"[STATE] Moving to grasp position")
            state_timer += 1

        elif state == CleanupState.MOVE_TO_GRASP:
            # Move down to grasp
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = 0.0
            target_joints[2] = 0.0
            target_joints[3] = -2.6
            target_joints[4] = 0.0
            target_joints[5] = 2.5
            target_joints[6] = 0.785

            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            joint_error = torch.norm(robot.data.joint_pos[0, :7] - target_joints).item()

            if joint_error < 1.0 and state_timer > 300:
                state = CleanupState.CLOSE_GRIPPER
                state_timer = 0
                print(f"[STATE] Closing gripper to grasp {target_name}")
            state_timer += 1

        elif state == CleanupState.CLOSE_GRIPPER:
            # Hold arm position
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = 0.0
            target_joints[2] = 0.0
            target_joints[3] = -2.6
            target_joints[4] = 0.0
            target_joints[5] = 2.5
            target_joints[6] = 0.785
            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Close gripper
            gripper_target = torch.zeros(scene.num_envs, 2, device=robot.device)
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            state_timer += 1
            if state_timer > 100:
                state = CleanupState.LIFT
                state_timer = 0
                print(f"[STATE] Lifting {target_name}")

        elif state == CleanupState.LIFT:
            # Lift up
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0.5
            target_joints[2] = 0.0
            target_joints[3] = -2.0
            target_joints[4] = 0.0
            target_joints[5] = 1.8
            target_joints[6] = 0.785

            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Keep gripper closed
            gripper_target = torch.zeros(scene.num_envs, 2, device=robot.device)
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            state_timer += 1
            if state_timer > 200:
                state = CleanupState.MOVE_TO_PLACE
                state_timer = 0
                print(f"[STATE] Moving to placement location")

        elif state == CleanupState.MOVE_TO_PLACE:
            # Move to placement location (left side)
            target_joints = home_joints.clone()
            target_joints[0] = 0.8  # Rotate to left
            target_joints[1] = -0.3
            target_joints[2] = 0.0
            target_joints[3] = -2.0
            target_joints[4] = 0.0
            target_joints[5] = 1.8
            target_joints[6] = 0.785

            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            gripper_target = torch.zeros(scene.num_envs, 2, device=robot.device)
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            state_timer += 1
            if state_timer > 300:
                state = CleanupState.RELEASE
                state_timer = 0
                print(f"[STATE] Releasing {target_name}")

        elif state == CleanupState.RELEASE:
            # Hold position and open gripper
            target_joints = home_joints.clone()
            target_joints[0] = 0.8
            target_joints[1] = -0.3
            target_joints[2] = 0.0
            target_joints[3] = -2.0
            target_joints[4] = 0.0
            target_joints[5] = 1.8
            target_joints[6] = 0.785
            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Open gripper
            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            state_timer += 1
            if state_timer > 100:
                state = CleanupState.RETREAT
                state_timer = 0
                print(f"[STATE] Retreating to home position")

        elif state == CleanupState.RETREAT:
            # Return to home
            robot.set_joint_position_target(home_joints.unsqueeze(0), joint_ids=list(range(7)))

            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            state_timer += 1
            if state_timer > 300:
                state = CleanupState.DONE
                state_timer = 0
                print(f"[STATE] Cleanup complete!")

        elif state == CleanupState.DONE:
            # Hold home position
            robot.set_joint_position_target(home_joints.unsqueeze(0), joint_ids=list(range(7)))
            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            # Check for reset or quit
            if not HEADLESS:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    print("\n[INFO] Resetting scene...")
                    state = CleanupState.IDLE
                    state_timer = 0
                    count = 0

                    # Reset robot
                    robot.write_joint_state_to_sim(
                        robot.data.default_joint_pos,
                        robot.data.default_joint_vel
                    )
                    robot.reset()

                    # Reset objects
                    root_state = object_collection.data.default_object_state.clone()
                    root_state[:, :, :3] += scene.env_origins.unsqueeze(1)
                    object_collection.write_object_pose_to_sim(root_state[:, :, :7])
            else:
                # In headless mode, break after completion
                if state_timer > 50:
                    print("\n[INFO] Cleanup sequence completed (headless mode)")
                    break
                state_timer += 1

        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update buffers
        scene.update(sim_dt)

        # Visualize tactile sensors
        if count % 5 == 0:
            show_debug = (count < 15)
            visualize_tactile_sensors(gelsight_left, gelsight_right, headless=HEADLESS, debug=show_debug)

        # Check for quit
        if not HEADLESS:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break

        # Print status
        if count % 100 == 0:
            print(f"[INFO] Step {count}, State: {state}, EE pos: {ee_pos[0].cpu().numpy()}, Gripper: {gripper_pos[0].item():.3f}")

        count += 1

    # Cleanup
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("\n[INFO] Closing simulation...")


def main():
    """Main function."""

    # Clamp num_objects to max of 6
    num_objects = min(args_cli.num_objects, 6)

    # Create scene configuration
    scene_cfg = DeskCleanupSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.replicate_physics = False

    # Override object_collection with custom number of objects
    scene_cfg.object_collection = create_desk_objects(num_objects=num_objects)

    print(f"\n[CONFIG] Creating desk cleanup scene with {num_objects} objects")

    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.5, 0.0, 0.3])

    # Create scene
    print("\n[INFO] Creating desk cleanup scene...")
    scene = InteractiveScene(scene_cfg)
    print("[INFO] Scene created successfully!")

    # Play the simulator
    sim.reset()
    print("\n[INFO] Starting simulation...")

    # Run the cleanup demo
    run_cleanup_demo(scene, sim, target_object_idx=args_cli.target_object)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
