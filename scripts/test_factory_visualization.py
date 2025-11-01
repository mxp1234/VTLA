#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to test and visualize the Factory Version 1 environment with tactile sensors.

This script creates a visualization of the factory_env_cfg.py environment to debug
the black screen issue and verify proper rendering of:
- Robot (Franka with GelSight Mini sensors)
- Fixed asset (peg hole, gear base, or bolt)
- Held asset (peg, gear, or nut)
- Tactile sensor data
- Lighting and materials

Usage:
    /home/pi-zero/anaconda3/envs/env45_isaacsim/bin/python scripts/test_factory_visualization.py --num_envs 1 --task peg_insert

Tasks available: peg_insert, gear_mesh, nut_thread
"""

import argparse
import numpy as np
import torch
import cv2
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test Factory Environment Visualization")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
parser.add_argument("--task", type=str, default="peg_insert",
                    choices=["peg_insert", "gear_mesh", "nut_thread"],
                    help="Factory task to visualize")
# parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--save_images", action="store_true", help="Save tactile images to disk")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force enable cameras for tactile sensors
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import factory environment configuration
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../source/tacex_tasks"))
from tacex_tasks.factory_version1.factory_env_cfg import (
    FactoryTaskPegInsertCfg,
    FactoryTaskGearMeshCfg,
    FactoryTaskNutThreadCfg,
)
from tacex import GelSightSensor


def setup_lighting(prim_path="/World"):
    """Setup proper lighting for the scene to avoid black rendering."""
    print("[INFO] Setting up scene lighting...")

    # Add dome light for ambient lighting
    dome_light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
        texture_file=None,  # Use uniform white light
    )
    dome_light_cfg.func(f"{prim_path}/DomeLight", dome_light_cfg)

    # Add distant light for directional lighting (simulates sun)
    distant_light_cfg = sim_utils.DistantLightCfg(
        intensity=2000.0,
        color=(1.0, 1.0, 0.95),
        angle=0.53,  # Sun-like angle
    )
    distant_light_cfg.func(f"{prim_path}/DistantLight", distant_light_cfg,
                          translation=(2.0, 2.0, 3.0))

    # Add sphere light for close-up illumination of the workspace
    sphere_light_cfg = sim_utils.SphereLightCfg(
        intensity=10000.0,
        color=(1.0, 1.0, 1.0),
        radius=0.1,
    )
    sphere_light_cfg.func(f"{prim_path}/SphereLight", sphere_light_cfg,
                         translation=(0.6, 0.0, 0.5))

    print("[INFO] Lighting setup complete")


def visualize_tactile_sensors(
    gsmini_left: GelSightSensor,
    gsmini_right: GelSightSensor,
    save_dir: str | None = None,
    step_count: int = 0,
):
    """
    Visualize and optionally save tactile sensor data.

    Args:
        gsmini_left: Left GelSight sensor
        gsmini_right: Right GelSight sensor
        save_dir: Directory to save images (None = display only)
        step_count: Current simulation step
    """
    if not hasattr(gsmini_left.data, 'output') or not gsmini_left.data.output:
        return

    # Get tactile RGB data
    tactile_left = gsmini_left.data.output.get("tactile_rgb")
    tactile_right = gsmini_right.data.output.get("tactile_rgb")

    if tactile_left is None or tactile_right is None:
        return

    # Get first environment data [H, W, 3]
    left_rgb = tactile_left[0].cpu().numpy()
    right_rgb = tactile_right[0].cpu().numpy()

    # Ensure data is in correct range [0, 1]
    left_rgb = np.clip(left_rgb, 0.0, 1.0)
    right_rgb = np.clip(right_rgb, 0.0, 1.0)

    # Convert to uint8 and BGR for OpenCV
    left_bgr = cv2.cvtColor((left_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor((right_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # TODO
    # Resize for better visibility (32x32 -> 256x256)
    # left_large = cv2.resize(left_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    # right_large = cv2.resize(right_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)

    left_large = left_bgr
    right_large = right_bgr
    # Create combined image
    combined = np.hstack([left_large, right_large])

    # Add labels
    cv2.putText(combined, "Left Sensor", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Right Sensor", (266, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"Step: {step_count}", (10, 246),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display
    if not args_cli.headless:
        cv2.imshow("Tactile Sensors", combined)
        cv2.waitKey(1)

    # Save if requested
    if save_dir is not None and step_count % 50 == 0:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"tactile_step_{step_count:06d}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, combined)
        if step_count % 200 == 0:
            print(f"[INFO] Saved tactile image: {filepath}")


def print_sensor_diagnostics(gsmini_left, gsmini_right, step_count):
    """Print detailed sensor diagnostics."""
    if step_count % 100 != 0:
        return

    print(f"\n[SENSOR DIAGNOSTIC] Step {step_count}")

    # Left sensor
    if hasattr(gsmini_left.data, 'output') and gsmini_left.data.output:
        tactile_rgb = gsmini_left.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            rgb_data = tactile_rgb[0].cpu().numpy()
            print(f"  Left sensor:")
            print(f"    Shape: {rgb_data.shape}")
            print(f"    Range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
            print(f"    Mean RGB: {rgb_data.mean(axis=(0,1))}")
            print(f"    Std RGB: {rgb_data.std(axis=(0,1))}")

    # Right sensor
    if hasattr(gsmini_right.data, 'output') and gsmini_right.data.output:
        tactile_rgb = gsmini_right.data.output.get("tactile_rgb")
        if tactile_rgb is not None:
            rgb_data = tactile_rgb[0].cpu().numpy()
            print(f"  Right sensor:")
            print(f"    Shape: {rgb_data.shape}")
            print(f"    Range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
            print(f"    Mean RGB: {rgb_data.mean(axis=(0,1))}")


def run_visualization(scene_cfg, task_name):
    """
    Run the visualization test for the factory environment.

    Args:
        scene_cfg: Scene configuration for the task
        task_name: Name of the task being visualized
    """
    print("\n" + "=" * 80)
    print(f"FACTORY ENVIRONMENT VISUALIZATION TEST - {task_name.upper()}")
    print("=" * 80)

    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=1/120,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physx=scene_cfg.sim.physx,
        physics_material=scene_cfg.sim.physics_material,
    )
    sim = SimulationContext(sim_cfg)

    # Set camera view for good visualization
    sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.6, 0.0, 0.3])

    # Create ground plane
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                       translation=(0.0, 0.0, -1.05))

    # Spawn table
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    table_cfg.func(
        "/World/envs/env_0/Table", table_cfg,
        translation=(0.55, 0.0, 0.0),
        orientation=(0.70711, 0.0, 0.0, 0.70711)
    )

    # Setup lighting
    setup_lighting("/World")

    # Create InteractiveScene
    print(f"[INFO] Creating scene for {task_name}...")
    scene = InteractiveScene(scene_cfg.scene)

    # Add articulations to scene
    scene.articulations["robot"] = Articulation(scene_cfg.robot)
    scene.articulations["fixed_asset"] = Articulation(scene_cfg.task.fixed_asset)
    scene.articulations["held_asset"] = Articulation(scene_cfg.task.held_asset)

    # Add gear assets if needed
    if task_name == "gear_mesh":
        scene.articulations["small_gear"] = Articulation(scene_cfg.task.small_gear_cfg)
        scene.articulations["large_gear"] = Articulation(scene_cfg.task.large_gear_cfg)

    # Create tactile sensors
    print("[INFO] Creating tactile sensors...")
    scene.sensors["gsmini_left"] = GelSightSensor(scene_cfg.gsmini_left)
    scene.sensors["gsmini_right"] = GelSightSensor(scene_cfg.gsmini_right)

    # Get references to scene objects
    robot = scene.articulations["robot"]
    fixed_asset = scene.articulations["fixed_asset"]
    held_asset = scene.articulations["held_asset"]
    gsmini_left = scene.sensors["gsmini_left"]
    gsmini_right = scene.sensors["gsmini_right"]

    # Play the simulator
    sim.reset()
    print("[INFO] Simulation started")

    # Create OpenCV window
    if not args_cli.headless:
        cv2.namedWindow("Tactile Sensors", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tactile Sensors", 800, 400)

    # Setup save directory
    save_dir = None
    if args_cli.save_images:
        save_dir = os.path.join(os.getcwd(), f"tactile_images_{task_name}")
        print(f"[INFO] Saving tactile images to: {save_dir}")

    print("\n[INFO] Starting visualization loop...")
    print("[CONTROLS]")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset robot pose")
    print("  - Press 's' to save current tactile image")

    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0

    # Define some test joint positions for the robot
    test_joint_positions = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        # Apply simple joint position targets to keep robot stable
        robot.set_joint_position_target(test_joint_positions)

        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update scene (this updates all articulations and sensors)
        scene.update(sim_dt)

        # Visualize tactile sensors every 5 steps
        if count % 5 == 0:
            visualize_tactile_sensors(gsmini_left, gsmini_right, save_dir, count)

        # Print diagnostics
        print_sensor_diagnostics(gsmini_left, gsmini_right, count)

        # Print status every 100 steps
        if count % 100 == 0:
            ee_pos = robot.data.body_pos_w[0, robot.body_names.index("panda_hand"), :]
            fixed_pos = fixed_asset.data.root_pos_w[0]
            held_pos = held_asset.data.root_pos_w[0]
            print(f"\n[STATUS] Step {count}")
            print(f"  EE position: {ee_pos.cpu().numpy()}")
            print(f"  Fixed asset position: {fixed_pos.cpu().numpy()}")
            print(f"  Held asset position: {held_pos.cpu().numpy()}")

        # Handle keyboard input
        if not args_cli.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break
            elif key == ord('r'):
                print("\n[INFO] Resetting robot pose...")
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel
                )
            elif key == ord('s') and save_dir is not None:
                print(f"\n[INFO] Saving tactile image at step {count}")
                visualize_tactile_sensors(gsmini_left, gsmini_right, save_dir, count)

        count += 1

        # Auto-quit after 1000 steps in headless mode
        if args_cli.headless and count > 1000:
            print("\n[INFO] Headless mode: completed 1000 steps")
            break

    # Cleanup
    if not args_cli.headless:
        cv2.destroyAllWindows()
    print("\n[INFO] Visualization test complete")


def main():
    """Main function."""

    # Select task configuration
    task_configs = {
        "peg_insert": FactoryTaskPegInsertCfg,
        "gear_mesh": FactoryTaskGearMeshCfg,
        "nut_thread": FactoryTaskNutThreadCfg,
    }

    if args_cli.task not in task_configs:
        raise ValueError(f"Unknown task: {args_cli.task}. Choose from {list(task_configs.keys())}")

    # Create scene configuration
    scene_cfg_class = task_configs[args_cli.task]
    scene_cfg = scene_cfg_class()
    scene_cfg.scene.num_envs = args_cli.num_envs
    scene_cfg.scene.env_spacing = 2.0

    print(f"\n[INFO] Task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Device: {args_cli.device}")
    print(f"[INFO] Headless: {args_cli.headless}")

    # Run visualization
    run_visualization(scene_cfg, args_cli.task)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close sim app
        simulation_app.close()
