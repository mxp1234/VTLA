import gymnasium as gym

from . import agents
from .peg_in_hole_env_cfg import PegInHoleCircleHole_I_Cfg, PegInHoleCircleHole_II_Cfg, PegInHoleCircleHole_III_Cfg, PegInHoleCircleHole_IV_Cfg, PegInHoleCircleHole_test_Cfg, PegInHoleSquareHole_I_Cfg, PegInHoleSquareHole_II_Cfg, PegInHoleSquareHole_III_Cfg, PegInHoleSquareHole_IV_Cfg, PegInHoleLHole_I_Cfg, PegInHoleLHole_II_Cfg, PegInHoleLHole_III_Cfg, PegInHoleLHole_IV_Cfg, PegInHoleTriangleHole_I_Cfg, PegInHoleTriangleHole_II_Cfg, PegInHoleTriangleHole_III_Cfg, PegInHoleTriangleHole_IV_Cfg, PegInHoleHexagonHole_I_Cfg, PegInHoleHexagonHole_II_Cfg, PegInHoleHexagonHole_III_Cfg, PegInHoleHexagonHole_IV_Cfg

##
# Register Gym environments.
##

gym.register(
    id="Peg-In-Hole-Circle-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole with tolerance level I (2mm)

gym.register(
    id="Peg-In-Hole-Circle-II-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_II_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole with tolerance level II (0.5mm)

gym.register(
    id="Peg-In-Hole-Circle-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole with tolerance level III (0.1mm)

gym.register(
    id="Peg-In-Hole-Circle-IV-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_IV_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole with tolerance level IV (0.02mm)

gym.register(
    id="Peg-In-Hole-Circle-test-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleCircleHole_test_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_circle_cfg.yaml",
    },
) # Circle hole test with official assets

gym.register(
    id="Peg-In-Hole-Square-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleSquareHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_square_cfg.yaml",
    },
) # Square hole with tolerance level I (2mm)

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
    id="Peg-In-Hole-Square-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleSquareHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_square_cfg.yaml",
    },
) # Square hole with tolerance level III (0.1mm)

gym.register(
    id="Peg-In-Hole-Square-IV-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleSquareHole_IV_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_square_cfg.yaml",
    },
) # Square hole with tolerance level IV (0.02mm)

gym.register(
    id="Peg-In-Hole-LHole-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleLHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_L_cfg.yaml",
    },
) # L-shaped hole with tolerance level I (2mm)

gym.register(
    id="Peg-In-Hole-LHole-II-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleLHole_II_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_L_cfg.yaml",
    },
) # L-shaped hole with tolerance level II (0.5mm)

gym.register(
    id="Peg-In-Hole-LHole-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleLHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_L_cfg.yaml",
    },
) # L-shaped hole with tolerance level III (0.1mm)

gym.register(
    id="Peg-In-Hole-LHole-IV-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleLHole_IV_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_L_cfg.yaml",
    },
) # L-shaped hole with tolerance level IV (0.02mm)

gym.register(
    id="Peg-In-Hole-Triangle-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleTriangleHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_triangle_cfg.yaml",
    },
) # Triangle hole with tolerance level I (2mm)

gym.register(
    id="Peg-In-Hole-Triangle-II-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleTriangleHole_II_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_triangle_cfg.yaml",
    },
) # Triangle hole with tolerance level II (0.5mm)

gym.register(
    id="Peg-In-Hole-Triangle-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleTriangleHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_triangle_cfg.yaml",
    },
) # Triangle hole with tolerance level III (0.1mm)

gym.register(
    id="Peg-In-Hole-Triangle-IV-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleTriangleHole_IV_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_triangle_cfg.yaml",
    },
) # Triangle hole with tolerance level IV (0.02mm)

gym.register(
    id="Peg-In-Hole-Hexagon-I-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleHexagonHole_I_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_hexagon_cfg.yaml",
    },
) # Hexagon hole with tolerance level I (2mm)

gym.register(
    id="Peg-In-Hole-Hexagon-II-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleHexagonHole_II_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_hexagon_cfg.yaml",
    },
) # Hexagon hole with tolerance level II (0.5mm)

gym.register(
    id="Peg-In-Hole-Hexagon-III-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleHexagonHole_III_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_hexagon_cfg.yaml",
    },
) # Hexagon hole with tolerance level III (0.1mm)

gym.register(
    id="Peg-In-Hole-Hexagon-IV-Tactile-v2",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegInHoleHexagonHole_IV_Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_hexagon_cfg.yaml",
    },
) # Hexagon hole with tolerance level IV (0.02mm)