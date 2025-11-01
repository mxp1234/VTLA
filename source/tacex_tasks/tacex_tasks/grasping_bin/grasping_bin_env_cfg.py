# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for grasping bin environment with 5 geometric shapes.

This environment uses standard geometric primitives for grasping demonstrations:
- Sphere (red)
- Cube (green)
- Cylinder (blue)
- Cone (yellow)
- Capsule (magenta)
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

# Import Franka with gripper and GelSight sensors
from tacex_assets.robots.franka import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini import GelSightMiniCfg


@configclass
class GraspingBinSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the grasping bin scene with 5 geometric shapes.

    This scene contains 5 standard geometric primitives arranged in a bin:
    - Sphere (red, center)
    - Cube (green, right)
    - Cylinder (blue, left)
    - Cone (yellow, top)
    - Capsule (magenta, bottom)

    Each shape has unique visual properties (color, metallic, roughness)
    for easy identification during grasping tasks.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Container/Bin for objects
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.12), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
            scale=(1.8, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # Robot - Franka with Gripper and GelSight sensors
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Object collection - 5 geometric shapes arranged in bin
    # Standard geometric primitives: Sphere, Cube, Cylinder, Cone, Torus
    object_collection = RigidObjectCollectionCfg(
        rigid_objects={
            # Object 0: Sphere (Center)
            "sphere": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sphere",
                spawn=sim_utils.SphereCfg(
                    radius=0.025,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=100.0,
                        max_linear_velocity=100.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),  # Red
                        metallic=0.5,
                        roughness=0.3,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.15), rot=(1, 0, 0, 0)),
            ),
            # Object 1: Cube (Right)
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.04),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=100.0,
                        max_linear_velocity=100.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),  # Green
                        metallic=0.4,
                        roughness=0.4,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.57, 0.0, 0.15), rot=(1, 0, 0, 0)),
            ),
            # Object 2: Cylinder (Left)
            "cylinder": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=0.02,
                    height=0.05,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=100.0,
                        max_linear_velocity=100.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 1.0),  # Blue
                        metallic=0.6,
                        roughness=0.3,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.43, 0.0, 0.15), rot=(1, 0, 0, 0)),
            ),
            # Object 3: Cone (Top)
            "cone": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cone",
                spawn=sim_utils.ConeCfg(
                    radius=0.025,
                    height=0.05,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=100.0,
                        max_linear_velocity=100.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0),  # Yellow
                        metallic=0.5,
                        roughness=0.4,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.07, 0.15), rot=(1, 0, 0, 0)),
            ),
            # Object 4: Capsule (Bottom) - replacing torus with capsule for better stability
            "capsule": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Capsule",
                spawn=sim_utils.CapsuleCfg(
                    radius=0.015,
                    height=0.05,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=100.0,
                        max_linear_velocity=100.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 1.0),  # Magenta
                        metallic=0.6,
                        roughness=0.3,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.07, 0.15), rot=(1, 0, 0, 0)),
            ),
        }
    )

    # End-effector frame for task-space control
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/EndEffectorFrameTransformer",
            markers={
                "end_effector": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            },
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
        ],
    )

    # GelSight sensors on gripper fingers
    gelsight_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb", "camera_depth"],
    )
    # Configure optical simulation settings
    gelsight_left.optical_sim_cfg = gelsight_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    # Right sensor - copy configuration from left
    gelsight_right = gelsight_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )
