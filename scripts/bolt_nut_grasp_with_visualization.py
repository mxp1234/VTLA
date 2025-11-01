#!/usr/bin/env python3

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to perform bolt/nut grasping with real-time tactile sensor visualization.

This script allows the robot to:
- Select a target object (bolt or nut) in the bin
- Move to grasp position
- Close gripper to grasp
- Lift the object
- Real-time visualization of left and right GelSight sensor images

Usage:
    python scripts/bolt_nut_grasp_with_visualization.py  --enable_cameras --num_bolts 3 --num_nuts 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Grasp bolts/nuts with tactile visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_bolts", type=int, default=5, help="Number of bolts to spawn (default: 5).")
parser.add_argument("--num_nuts", type=int, default=5, help="Number of nuts to spawn (default: 5).")
parser.add_argument("--target_object", type=int, default=0, help="Target object index.")

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
from tacex_tasks.grasping_bin.bolt_nut_bin_env_cfg_v2 import create_bolt_nut_scene_with_counts
from tacex import GelSightSensor


class GraspingState:
    """State machine for grasping sequence."""
    IDLE = 0
    MOVE_TO_PREGRASP = 1
    MOVE_TO_GRASP = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    DONE = 5


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

    # Get tactile RGB data from output dictionary
    if hasattr(gelsight_left.data, 'output') and gelsight_left.data.output:
        tactile_rgb = gelsight_left.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            left_rgb = tactile_rgb[0].cpu().numpy()  # [H, W, 3]

            if debug:
                print(f"[DEBUG] Left sensor RGB shape: {left_rgb.shape}, dtype: {left_rgb.dtype}")
                print(f"[DEBUG] Left sensor RGB range: [{left_rgb.min():.3f}, {left_rgb.max():.3f}]")
                print(f"[DEBUG] Left sensor RGB mean: {left_rgb.mean(axis=(0,1))}")

            # Convert from RGB float [0,1] to BGR uint8 [0,255] for OpenCV
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


def reset_articulation_positions(scene: InteractiveScene, num_bolts: int, num_nuts: int):
    """
    Manually reset articulation positions to their configured locations (V2 pattern).

    Args:
        scene: The interactive scene
        num_bolts: Number of bolt objects
        num_nuts: Number of nut objects
    """
    import numpy as np

    # Two bin centers (matching the V2 configuration)
    bin_left_center = np.array([0.4, 0.2, 0.01])   # Left bin for bolts
    bin_right_center = np.array([0.4, -0.2, 0.01])  # Right bin for nuts

    print(f"\n[INFO] Resetting positions for {num_bolts} bolts and {num_nuts} nuts")

    device = scene.device

    # Reset each bolt articulation - place in left bin
    for i in range(num_bolts):
        bolt_name = f"bolt_{i}"
        if bolt_name not in scene.articulations:
            print(f"[WARNING] {bolt_name} not found in scene articulations")
            continue

        bolt = scene.articulations[bolt_name]

        # Calculate bolt position (same logic as in cfg)
        angle = 2 * np.pi * i / max(num_bolts, 1)
        radius = 0.08  # Smaller radius to fit in bin
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)

        pos = torch.tensor([
            bin_left_center[0] + x_offset,
            bin_left_center[1] + y_offset,
            bin_left_center[2]
        ], device=device, dtype=torch.float32)

        # Add environment origin offset
        pos_with_origin = pos + scene.env_origins[0]

        # Set root state: [num_envs, 13] where 13 = [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((scene.num_envs, 13), device=device)
        root_state[:, 0:3] = pos_with_origin
        root_state[:, 3] = 1.0  # qw (identity quaternion)
        root_state[:, 4:7] = 0.0  # qx, qy, qz
        root_state[:, 7:13] = 0.0  # velocities

        bolt.write_root_state_to_sim(root_state)

    # Reset each nut articulation - place in right bin
    for i in range(num_nuts):
        nut_name = f"nut_{i}"
        if nut_name not in scene.articulations:
            print(f"[WARNING] {nut_name} not found in scene articulations")
            continue

        nut = scene.articulations[nut_name]

        # Calculate nut position (same logic as in cfg)
        angle = 2 * np.pi * i / max(num_nuts, 1) + np.pi / max(num_nuts, 1)
        radius = 0.06  # Even smaller radius for nuts
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)

        pos = torch.tensor([
            bin_right_center[0] + x_offset,
            bin_right_center[1] + y_offset,
            bin_right_center[2]
        ], device=device, dtype=torch.float32)

        # Add environment origin offset
        pos_with_origin = pos + scene.env_origins[0]

        # Set root state
        root_state = torch.zeros((scene.num_envs, 13), device=device)
        root_state[:, 0:3] = pos_with_origin
        root_state[:, 3] = 1.0  # qw
        root_state[:, 4:7] = 0.0  # qx, qy, qz
        root_state[:, 7:13] = 0.0  # velocities

        nut.write_root_state_to_sim(root_state)

    print(f"[INFO] Object positions reset complete")


def reset_object_positions(object_collection: RigidObjectCollection, scene: InteractiveScene):
    """
    Manually reset object positions to their configured locations.

    Args:
        object_collection: The rigid object collection containing bolts and nuts
        scene: The interactive scene
    """
    import numpy as np

    # Bin center and size (matching the configuration)
    bin_center = np.array([0.5, 0.0, 0.15])

    # Get object names
    object_names = object_collection.object_names
    num_bolts = sum(1 for name in object_names if 'bolt' in name)
    num_nuts = sum(1 for name in object_names if 'nut' in name)

    print(f"\n[INFO] Resetting positions for {num_bolts} bolts and {num_nuts} nuts")

    # Create position tensors
    device = object_collection.device
    num_objects = len(object_names)

    # Reset each object to its configured position
    bolt_idx = 0
    nut_idx = 0

    for obj_idx, obj_name in enumerate(object_names):
        if 'bolt' in obj_name:
            # Calculate bolt position (same logic as in cfg)
            angle = 2 * np.pi * bolt_idx / max(num_bolts, 1)
            radius = 0.1
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)

            pos = torch.tensor([
                bin_center[0] + x_offset,
                bin_center[1] + y_offset,
                bin_center[2]
            ], device=device, dtype=torch.float32)

            bolt_idx += 1
        else:  # nut
            # Calculate nut position (same logic as in cfg)
            angle = 2 * np.pi * nut_idx / max(num_nuts, 1) + np.pi / max(num_nuts, 1)
            radius = 0.06
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)

            pos = torch.tensor([
                bin_center[0] + x_offset,
                bin_center[1] + y_offset,
                bin_center[2]
            ], device=device, dtype=torch.float32)

            nut_idx += 1

        # Set object pose: [num_envs, num_objects, 7] where 7 = [x, y, z, qw, qx, qy, qz]
        object_pose = torch.zeros((scene.num_envs, num_objects, 7), device=device)

        # Set position for this object
        object_pose[:, obj_idx, 0:3] = pos
        # Set rotation (identity quaternion: w=1, x=0, y=0, z=0)
        object_pose[:, obj_idx, 3] = 1.0  # qw
        object_pose[:, obj_idx, 4:7] = 0.0  # qx, qy, qz

    # Write all object poses to simulation
    # Note: We need to set all objects at once, not one by one
    all_poses = torch.zeros((scene.num_envs, num_objects, 7), device=device)

    bolt_idx = 0
    nut_idx = 0
    for obj_idx, obj_name in enumerate(object_names):
        if 'bolt' in obj_name:
            angle = 2 * np.pi * bolt_idx / max(num_bolts, 1)
            radius = 0.1
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            pos = [bin_center[0] + x_offset, bin_center[1] + y_offset, bin_center[2]]
            bolt_idx += 1
        else:
            angle = 2 * np.pi * nut_idx / max(num_nuts, 1) + np.pi / max(num_nuts, 1)
            radius = 0.06
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            pos = [bin_center[0] + x_offset, bin_center[1] + y_offset, bin_center[2]]
            nut_idx += 1

        # Add environment origin offset
        all_poses[:, obj_idx, 0:3] = torch.tensor(pos, device=device) + scene.env_origins
        all_poses[:, obj_idx, 3] = 1.0  # qw
        all_poses[:, obj_idx, 4:7] = 0.0  # qx, qy, qz

    # Write poses to simulation
    object_collection.write_object_pose_to_sim(all_poses)

    # Zero out velocities
    zero_vel = torch.zeros((scene.num_envs, num_objects, 6), device=device)
    object_collection.write_object_velocity_to_sim(zero_vel)

    print(f"[INFO] Object positions reset complete")


def run_grasping_demo(
    scene: InteractiveScene,
    sim: SimulationContext,
    target_object_idx: int = 0,
    num_bolts: int = 5,
    num_nuts: int = 5
):
    """
    Run grasping demonstration with tactile visualization (V2 pattern).

    Args:
        scene: Interactive scene
        sim: Simulation context
        target_object_idx: Index of target object to grasp
        num_bolts: Number of bolts in the scene
        num_nuts: Number of nuts in the scene
    """
    # Extract scene entities
    robot: Articulation = scene["robot"]
    gelsight_left: GelSightSensor = scene.sensors["gelsight_left"]
    gelsight_right: GelSightSensor = scene.sensors["gelsight_right"]

    # Collect all bolt and nut articulations
    all_objects = {}
    for i in range(num_bolts):
        bolt_name = f"bolt_{i}"
        if bolt_name in scene.articulations:
            all_objects[bolt_name] = scene.articulations[bolt_name]

    for i in range(num_nuts):
        nut_name = f"nut_{i}"
        if nut_name in scene.articulations:
            all_objects[nut_name] = scene.articulations[nut_name]

    # Create sorted list of object names
    object_names = sorted(all_objects.keys())
    num_objects = len(object_names)

    print("\n" + "=" * 80)
    print("BOLT/NUT GRASPING DEMO WITH TACTILE VISUALIZATION")
    print("=" * 80)

    # Determine if target is bolt or nut
    target_name = object_names[target_object_idx] if target_object_idx < num_objects else "unknown"

    print(f"\n[INFO] Target object: {target_name} (index {target_object_idx})")
    print(f"[INFO] Number of environments: {scene.num_envs}")
    print(f"[INFO] Robot joints: {robot.num_joints}")
    print(f"[INFO] Objects in bin: {num_objects} ({', '.join(object_names)})")

    print("\n[CONTROLS]")
    print("  - Press 'q' in tactile windows to quit")
    print("  - Press 'r' to reset and retry")
    print("  - Press Ctrl+C to exit")

    print("\n[GRASPING SEQUENCE]")
    print("  1. Move to pre-grasp position (15cm above object)")
    print("  2. Move down to grasp position")
    print("  3. Close gripper")
    print("  4. Lift object")
    print("  5. Done")

    print("\n" + "=" * 80)

    # Reset object positions at the start
    print("\n[INFO] Resetting object positions to configured locations...")
    reset_articulation_positions(scene, num_bolts, num_nuts)
    print("reset done !!!")
    for name in object_names:
        obj = all_objects[name]
        pos = obj.data.root_pos_w[0].cpu().numpy()
        print(f"  {name:10s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
    # Create OpenCV windows only if not in headless mode
    if not HEADLESS:
        cv2.namedWindow("GelSight Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("GelSight Right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GelSight Left", 400, 300)
        cv2.resizeWindow("GelSight Right", 400, 300)

    # State machine
    state = GraspingState.IDLE
    state_timer = 0

    # Use predefined joint positions
    home_joints = robot.data.default_joint_pos[0, :7].clone()

    print(f"\n[DEBUG] Home joint positions: {home_joints.cpu().numpy()}")
    print(f"[DEBUG] Initial EE position: {robot.data.body_pos_w[0, robot.body_names.index('panda_hand'), :].cpu().numpy()}")

    # Define the simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Get current target object position (V2 pattern - access individual articulation)
        target_obj = all_objects[target_name]
        target_pos = target_obj.data.root_pos_w[0].clone()

        # Get current end-effector position
        ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_hand"), :]

        # Get current gripper state
        gripper_pos = robot.data.joint_pos[:, -2:].mean(dim=1, keepdim=True)

        # State machine
        if state == GraspingState.IDLE:
            if count > 50:  # Wait for scene to stabilize
                state = GraspingState.MOVE_TO_PREGRASP
                state_timer = 0
                print(f"\n[STATE] Moving to pre-grasp position above {target_name}")
                print(f"        Target position: {target_pos.cpu().numpy()}")

        elif state == GraspingState.MOVE_TO_PREGRASP:
            # Move to a position roughly over the bin
            target_joints = home_joints.clone()
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

            # Check if reached
            joint_error = torch.norm(robot.data.joint_pos[0, :7] - target_joints).item()

            # Print debug info every 200 steps
            if state_timer % 200 == 0:
                print(f"[DEBUG PREGRASP] joint_error={joint_error:.4f}, timer={state_timer}")
                print(f"        Current joints: {robot.data.joint_pos[0, :7].cpu().numpy()}")
                print(f"        Target joints:  {target_joints.cpu().numpy()}")

            if joint_error < 1.0 and state_timer > 300:
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
                print(f"[STATE] Closing gripper to grasp {target_name}")
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
                print(f"[STATE] Lifting {target_name}")

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

                    # Reset objects to configured positions (V2 pattern)
                    reset_articulation_positions(scene, num_bolts, num_nuts)
                    
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
            if count < 500:
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

    # Dynamically create scene class with specified number of bolts and nuts
    print(f"\n[CONFIG] Creating scene with {args_cli.num_bolts} bolts and {args_cli.num_nuts} nuts")
    SceneCfg = create_bolt_nut_scene_with_counts(
        num_bolts=args_cli.num_bolts,
        num_nuts=args_cli.num_nuts
    )

    # Create scene configuration instance
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.replicate_physics = False

    print(f"[CONFIG] Scene class generated with {args_cli.num_bolts + args_cli.num_nuts} articulated objects")

    # Initialize simulation context first (required before InteractiveScene)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.5, 0.0, 0.5])

    # Create scene
    print("\n[INFO] Creating bolt-nut bin scene...")
    scene = InteractiveScene(scene_cfg)
    print("[INFO] Scene created successfully!")

    # Play the simulator
    sim.reset()
    print("\n[INFO] Starting simulation...")

    # Run the grasping demo
    run_grasping_demo(scene, sim, target_object_idx=args_cli.target_object,
                      num_bolts=args_cli.num_bolts, num_nuts=args_cli.num_nuts)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
