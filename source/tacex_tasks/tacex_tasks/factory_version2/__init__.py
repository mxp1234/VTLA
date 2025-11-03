import gymnasium as gym

from . import agents
from .peg_in_hole_env_cfg import PegInHoleCircleHole_I_Cfg, PegInHoleCircleHole_test_Cfg, PegInHoleSquareHole_II_Cfg, PegInHoleLHole_III_Cfg

##
# Register Gym environments.
##

gym.register(
    id="Peg-In-Hole-Cricle-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole with tolerance level I (2mm)

gym.register(
    id="Peg-In-Hole-Cricle-test-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_test_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole test with official assets

gym.register(
    id="Peg-In-Hole-Square-II-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleSquareHole_II_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_square_cfg.yaml",
    },
) # Square hole with tolerance level II (0.5mm)

gym.register(
    id="Peg-In-Hole-LHole-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleLHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_LHole_cfg.yaml",
    },
) # L-shaped hole with tolerance level III (0.2mm)