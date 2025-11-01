#!/usr/bin/env python3

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to perform grasping with real-time tactile sensor visualization.
Written by xinpan meng. 
Test Done

This script allows the robot to:
- Select a target object in the bin
- Move to grasp position
- Close gripper to grasp
- Lift the object
- Real-time visualization of left and right GelSight sensor images

Usage:
    /home/pi-zero/anaconda3/envs/env45_isaacsim/bin/python scripts/grasp_with_visualization.py --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Grasp objects with tactile visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--target_object", type=int, default=4, help="Target object index (0-4).")

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
from tacex_tasks.grasping_bin.grasping_bin_env_cfg import GraspingBinSceneCfg
from tacex import GelSightSensor


class GraspingState:
    """State machine for grasping sequence."""
    IDLE = 0
    MOVE_TO_PREGRASP = 1
    MOVE_TO_GRASP = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    DONE = 5


def compute_ik_target(target_pos: torch.Tensor, offset: tuple = (0.0, 0.0, 0.1)) -> torch.Tensor:
    """
    Compute IK target position with offset.

    Args:
        target_pos: Target object position [x, y, z]
        offset: Offset from target [dx, dy, dz]

    Returns:
        IK target position
    """
    target = target_pos.clone()
    target[0] += offset[0]
    target[1] += offset[1]
    target[2] += offset[2]
    return target


def visualize_tactile_sensors(
    gelsight_left: GelSightSensor,
    gelsight_right: GelSightSensor,
    window_name_left: str = "GelSight Left",
    window_name_right: str = "GelSight Right",
    headless: bool = False,
    debug: bool = False
):
    """
    Visualize tactile sensor data in OpenCV windows.

    Args:
        gelsight_left: Left GelSight sensor
        gelsight_right: Right GelSight sensor
        window_name_left: Window name for left sensor
        window_name_right: Window name for right sensor
        headless: Skip visualization if running in headless mode
        debug: Print debug information
    """
    if headless:
        return

    # Get tactile RGB data from output dictionary (correct data access pattern)
    if hasattr(gelsight_left.data, 'output') and gelsight_left.data.output:
        tactile_rgb = gelsight_left.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            left_rgb = tactile_rgb[0].cpu().numpy()  # [H, W, 3]

            if debug:
                print(f"[DEBUG] Left sensor RGB shape: {left_rgb.shape}, dtype: {left_rgb.dtype}")
                print(f"[DEBUG] Left sensor RGB range: [{left_rgb.min():.3f}, {left_rgb.max():.3f}]")
                print(f"[DEBUG] Left sensor RGB mean: {left_rgb.mean(axis=(0,1))}")

            # Convert from RGB float [0,1] to BGR uint8 [0,255] for OpenCV
            # Make sure data is in correct range
            left_rgb_scaled = np.clip(left_rgb, 0.0, 1.0)
            left_bgr = cv2.cvtColor((left_rgb_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name_left, left_bgr)

    if hasattr(gelsight_right.data, 'output') and gelsight_right.data.output:
        tactile_rgb = gelsight_right.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            right_rgb = tactile_rgb[0].cpu().numpy()  # [H, W, 3]

            if debug:
                print(f"[DEBUG] Right sensor RGB shape: {right_rgb.shape}, dtype: {right_rgb.dtype}")
                print(f"[DEBUG] Right sensor RGB range: [{right_rgb.min():.3f}, {right_rgb.max():.3f}]")

            right_rgb_scaled = np.clip(right_rgb, 0.0, 1.0)
            right_bgr = cv2.cvtColor((right_rgb_scaled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name_right, right_bgr)

    cv2.waitKey(1)


def run_grasping_demo(
    scene: InteractiveScene,
    sim: SimulationContext,
    target_object_idx: int = 0
):
    """
    Run grasping demonstration with tactile visualization.

    Args:
        scene: Interactive scene
        sim: Simulation context
        target_object_idx: Index of target object to grasp (0-4)
    """
    # Extract scene entities
    robot: Articulation = scene["robot"]
    object_collection: RigidObjectCollection = scene["object_collection"]
    gelsight_left: GelSightSensor = scene.sensors["gelsight_left"]
    gelsight_right: GelSightSensor = scene.sensors["gelsight_right"]

    print("\n" + "=" * 80)
    print("GRASPING DEMO WITH TACTILE VISUALIZATION")
    print("=" * 80)

    print(f"\n[INFO] Target object: object_{target_object_idx}")
    print(f"[INFO] Number of environments: {scene.num_envs}")
    print(f"[INFO] Robot joints: {robot.num_joints}")
    print(f"[INFO] Objects in bin: 5")

    print("\n[CONTROLS]")
    print("  - Press 'q' in tactile windows to quit")
    print("  - Press 'r' to reset and retry")
    print("  - Press Ctrl+C to exit")

    print("\n[GRASPING SEQUENCE]")
    print("  1. Move to pre-grasp position (10cm above object)")
    print("  2. Move down to grasp position")
    print("  3. Close gripper")
    print("  4. Lift object")
    print("  5. Done")

    print("\n" + "=" * 80)

    # Create OpenCV windows only if not in headless mode
    if not HEADLESS:
        cv2.namedWindow("GelSight Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("GelSight Right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GelSight Left", 400, 300)
        cv2.resizeWindow("GelSight Right", 400, 300)

    # State machine
    state = GraspingState.IDLE
    state_timer = 0

    # Grasping parameters
    pregrasp_height = 0.15  # 15cm above object
    grasp_height = 0.01     # 2cm above object (adjusted for gripper)
    lift_height = 0.20      # 20cm lift

    # Use simple position control instead of velocity control
    # Define target joint positions for different poses
    # These are approximate values - ideally you'd use IK
    home_joints = robot.data.default_joint_pos[0, :7].clone()

    print(f"\n[DEBUG] Home joint positions: {home_joints.cpu().numpy()}")
    print(f"[DEBUG] Initial EE position: {robot.data.body_pos_w[0, robot.body_names.index('panda_hand'), :].cpu().numpy()}")

    # Define the simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Get current object position (RigidObjectCollection uses object_pos_w)
        # Shape: [num_envs, num_objects, 3]
        object_positions = object_collection.data.object_pos_w[0]  # [num_objects, 3]
        target_pos = object_positions[target_object_idx].clone()

        # Get current end-effector position
        ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]

        # Get current gripper state
        gripper_pos = robot.data.joint_pos[:, -2:].mean(dim=1, keepdim=True)  # Average of two finger joints

        # State machine
        if state == GraspingState.IDLE:
            if count > 50:  # Wait for scene to stabilize
                state = GraspingState.MOVE_TO_PREGRASP
                state_timer = 0
                print(f"\n[STATE] Moving to pre-grasp position above object_{target_object_idx}")
                print(f"        Target position: {target_pos.cpu().numpy()}")

        elif state == GraspingState.MOVE_TO_PREGRASP:
            # Use simple predefined joint positions to move over the object
            # This is a simplified approach - real implementation should use IK
            target_joints = home_joints.clone()
            # Move to a position roughly over the bin
            target_joints[0] = 0.0  # Joint 1
            target_joints[1] = -0.3  # Joint 2
            target_joints[2] = 0.0  # Joint 3
            target_joints[3] = -2.2  # Joint 4
            target_joints[4] = 0.0  # Joint 5
            target_joints[5] = 2.0  # Joint 6
            target_joints[6] = 0.785  # Joint 7

            # Set joint position targets
            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Keep gripper open
            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            # Check if reached (simple joint position error)
            joint_error = torch.norm(robot.data.joint_pos[0, :7] - target_joints).item()

            # Print debug info every 200 steps
            if state_timer % 200 == 0:
                print(f"[DEBUG PREGR ASP] joint_error={joint_error:.4f}, timer={state_timer}")
                print(f"        Current joints: {robot.data.joint_pos[0, :7].cpu().numpy()}")
                print(f"        Target joints:  {target_joints.cpu().numpy()}")

            # More lenient threshold - just check if we're moving at all
            if joint_error < 1.0 and state_timer > 300:  # Increased threshold and wait time
                state = GraspingState.MOVE_TO_GRASP
                state_timer = 0
                print(f"[STATE] Moving to grasp position")
            state_timer += 1

        elif state == GraspingState.MOVE_TO_GRASP:
            # Move down closer to grasp
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0  # Lower position
            target_joints[2] = 0.0
            target_joints[3] = -2.6  # Reach further down
            target_joints[4] = 0.0
            target_joints[5] = 2.5  # Adjust wrist
            target_joints[6] = 0.785

            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            # Keep gripper open
            gripper_target = torch.ones(scene.num_envs, 2, device=robot.device) * 0.04
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            # Check if reached
            joint_error = torch.norm(robot.data.joint_pos[0, :7] - target_joints).item()

            # Print debug info every 200 steps
            if state_timer % 200 == 0:
                print(f"[DEBUG GRASP] joint_error={joint_error:.4f}, timer={state_timer}")
                print(f"        Current joints: {robot.data.joint_pos[0, :7].cpu().numpy()}")
                print(f"        Target joints:  {target_joints.cpu().numpy()}")

            if joint_error < 1.0 and state_timer > 300:
                state = GraspingState.CLOSE_GRIPPER
                state_timer = 0
                print(f"[STATE] Closing gripper to grasp object")
            state_timer += 1

        elif state == GraspingState.CLOSE_GRIPPER:
            # Hold arm position
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0
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
            if state_timer > 100:  # Wait for gripper to close
                state = GraspingState.LIFT
                state_timer = 0
                print(f"[STATE] Lifting object")

        elif state == GraspingState.LIFT:
            # Lift up
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0.5  # Lift higher
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
            if state_timer > 200:  # Wait for lift
                state = GraspingState.DONE
                state_timer = 0
                print(f"[STATE] Grasping complete!")
                print(f"        Final EE position: {ee_pos[0].cpu().numpy()}")
                print(f"        Object position: {target_pos.cpu().numpy()}")

        elif state == GraspingState.DONE:
            # Hold lifted position
            target_joints = home_joints.clone()
            target_joints[0] = 0.0
            target_joints[1] = -0.5
            target_joints[2] = 0.0
            target_joints[3] = -2.0
            target_joints[4] = 0.0
            target_joints[5] = 1.8
            target_joints[6] = 0.785
            robot.set_joint_position_target(target_joints.unsqueeze(0), joint_ids=list(range(7)))

            gripper_target = torch.zeros(scene.num_envs, 2, device=robot.device)
            robot.set_joint_position_target(gripper_target, joint_ids=[-2, -1])

            # Check for reset command (only in non-headless mode)
            if not HEADLESS:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    print("\n[INFO] Resetting scene...")
                    state = GraspingState.IDLE
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
                    print("\n[INFO] Grasping sequence completed (headless mode)")
                    break
                state_timer += 1

        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update buffers
        scene.update(sim_dt)

        # Visualize tactile sensors (update every 5 steps for performance)
        if count % 5 == 0:
            # Show debug info for first 3 visualizations
            show_debug = (count < 15)
            visualize_tactile_sensors(gelsight_left, gelsight_right, headless=HEADLESS, debug=show_debug)

        # Check for quit only if not in headless mode
        if not HEADLESS:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break

        # Print detailed sensor diagnostics during contact states
        if state in [GraspingState.CLOSE_GRIPPER, GraspingState.LIFT, GraspingState.DONE] and count % 20 == 0:
            print(f"\n[SENSOR DIAGNOSTIC] Step {count}, State: {state}")

            # Check camera data
            if gelsight_left.camera:
                camera_depth = gelsight_left.camera.data.output.get("depth")
                if camera_depth is not None:
                    depth_stats = camera_depth[0].cpu().numpy()
                    print(f"  Camera depth: shape={depth_stats.shape}, min={np.min(depth_stats):.4f}, max={np.max(depth_stats):.4f}, mean={np.mean(depth_stats):.4f}")

            # Check height map
            if hasattr(gelsight_left.data, 'output') and gelsight_left.data.output:
                height_map = gelsight_left.data.output.get("height_map")
                if height_map is not None:
                    hm_stats = height_map[0].cpu().numpy()
                    print(f"  Height map: shape={hm_stats.shape}, min={np.min(hm_stats):.4f}, max={np.max(hm_stats):.4f}, mean={np.mean(hm_stats):.4f}")

                # Check tactile RGB
                tactile_rgb = gelsight_left.data.output.get("tactile_rgb")
                if tactile_rgb is not None:
                    tac_stats = tactile_rgb[0].cpu().numpy()
                    print(f"  Tactile RGB: shape={tac_stats.shape}, min={np.min(tac_stats):.4f}, max={np.max(tac_stats):.4f}")
                    # Check color variance
                    r_var = np.var(tac_stats[:, :, 0])
                    g_var = np.var(tac_stats[:, :, 1])
                    b_var = np.var(tac_stats[:, :, 2])
                    print(f"  RGB variance: R={r_var:.6f}, G={g_var:.6f}, B={b_var:.6f}")

            # Check indentation depth
            print(f"  Indentation depth: {gelsight_left.indentation_depth[0].item():.4f}")

        # Print status every 100 steps
        if count % 100 == 0:
            print(f"[INFO] Step {count}, State: {state}, EE pos: {ee_pos[0].cpu().numpy()}, Gripper: {gripper_pos[0].item():.3f}")
            if count < 500:  # Print joint positions for first few updates
                print(f"       Joint pos: {robot.data.joint_pos[0, :7].cpu().numpy()}")

        # Increment counter
        count += 1
        sim_time += sim_dt

    # Cleanup
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("\n[INFO] Closing simulation...")


def main():
    """Main function."""

    # Create scene configuration
    scene_cfg = GraspingBinSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.replicate_physics = False

    # Initialize simulation context first (required before InteractiveScene)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.5, 0.0, 0.5])

    # Create scene
    print("[INFO] Creating scene...")
    scene = InteractiveScene(scene_cfg)
    print("[INFO] Scene created successfully!")

    # Play the simulator
    sim.reset()
    print("[INFO] Starting simulation...")

    # Run the grasping demo
    run_grasping_demo(scene, sim, target_object_idx=args_cli.target_object)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
