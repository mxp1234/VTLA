#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for Factory Peg Insert task with GelSight tactile sensors.

This script verifies that the new task can be loaded and run successfully.
"""

from isaaclab.app import AppLauncher
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description="Test Factory Tactile environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
# parser.add_argument("--headless", action="store_true", help="Run in headless mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Import task
import tacex_tasks  # noqa: F401

def main():
    """Test the Factory Tactile environment."""

    print("=" * 80)
    print("Testing Factory Peg Insert with GelSight Sensors")
    print("=" * 80)

    # Create environment
    task_name = "Isaac-Factory-PegInsert-Tactile-Direct-v0"

    try:
        # Load the environment configuration
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        env_cfg.scene.num_envs = args_cli.num_envs

        # Create environment with config
        env = gym.make(task_name, cfg=env_cfg)
        print(f"\n✓ Successfully created environment: {task_name}")
        print(f"  - Number of environments: {env.unwrapped.num_envs}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space shape: {env.observation_space.shape}")

        # Reset environment
        print("\n" + "-" * 80)
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs['policy'].shape}")
        print(f"  - State shape: {obs['critic'].shape}")

        # Run a few steps
        print("\n" + "-" * 80)
        print("Running 10 test steps...")

        for step in range(10):
            # Sample random actions
            actions = torch.randn(env.unwrapped.num_envs, env.action_space.shape[0], device=env.unwrapped.device)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            if step % 5 == 0:
                print(f"  Step {step}: reward mean = {reward.mean().item():.4f}")

        print(f"\n✓ Environment stepping successful")

        # Close environment
        env.close()
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    simulation_app.close()
    exit(0 if success else 1)
