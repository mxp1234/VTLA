# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg, FactoryTestPegInsertCfg, FactoryTestPegInsertNoRandomCfg

# ---
# Register Gym environments with TACTILE suffix to indicate these include tactile observations
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Tactile-v1 --num_envs 100 --enable_cameras
gym.register(
    id="TacEx-Factory-PegInsert-Tactile-v1",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTestPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Tactile-v1 --num_envs 100 --enable_cameras
gym.register(
    id="TacEx-Factory-GearMesh-Tactile-v1",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Tactile-v1 --num_envs 20 --enable_cameras
gym.register(
    id="TacEx-Factory-NutThread-Tactile-v1",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# ---
# 测试环境注册 - 用于检测本地资产的初始化效果
# Test Environment Registration - For checking initialization effects of local assets
# ---

# 测试环境 (有随机化): 用于验证本地peg_10mm.usd和hole_10mm.usd的初始化
# Test environment (with randomization): For validating initialization of local peg_10mm.usd and hole_10mm.usd
# python scripts/play.py --task TacEx-Factory-PegInsert-Test-v1 --num_envs 1
gym.register(
    id="TacEx-Factory-PegInsert-Test-v1",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTestPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# 测试环境 (无随机化): 用于验证固定初始状态下的物理仿真
# Test environment (without randomization): For validating physics simulation with fixed initial state
# python scripts/play.py --task TacEx-Factory-PegInsert-Test-NoRand-v1 --num_envs 1
gym.register(
    id="TacEx-Factory-PegInsert-Test-NoRand-v1",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTestPegInsertNoRandomCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

