# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for desk cleanup environment.

This environment simulates a desk cleanup task where a robot needs to pick up
various desktop objects and organize them. Objects from USD files include:
- Bowl
- Cup
- Clamp
- Sponge
- Sashuihu (watering can)

The task is to grasp these objects and place them in designated areas.
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


def create_desk_objects(num_objects: int = 5) -> RigidObjectCollectionCfg:
    """
    Create desktop objects for cleanup task using USD files.

    Args:
        num_objects: Number of objects to spawn (default: 5, max: 5)

    Returns:
        RigidObjectCollectionCfg with desktop objects from USD files
    """
    rigid_objects = {}

    # Desk center and work area
    desk_center = [0.5, 0.0, 0.05]

    # Object definitions: (name, spawn_cfg, position_offset, color)
    # Using USD files from /home/pi-zero/Documents/USD-file
    usd_base_path = "/home/pi-zero/Documents/USD-file"

    object_defs = [
        # Bowl - center
        ("bowl", UsdFileCfg(
            usd_path=f"{usd_base_path}/bowl.usd",
            scale=(1.0, 1.0, 1.0),
        ), (0.0, -0.15, 0.0), (1.0, 1.0, 1.0)),

        # Cup - front right
        ("cup", UsdFileCfg(
            usd_path=f"{usd_base_path}/cup.usd",
            scale=(1.0, 1.0, 1.0),
        ), (0.15, -0.10, 0.0), (1.0, 1.0, 1.0)),

        # Clamp - front left
        ("clamp", UsdFileCfg(
            usd_path=f"{usd_base_path}/clamp.usd",
            scale=(1.0, 1.0, 1.0),
        ), (-0.15, -0.10, 0.0), (1.0, 1.0, 1.0)),

        # Sponge - back right
        ("sponge", UsdFileCfg(
            usd_path=f"{usd_base_path}/sponge.usd",
            scale=(1.0, 1.0, 1.0),
        ), (0.10, 0.10, 0.0), (1.0, 1.0, 1.0)),

        # Sashuihu (watering can) - back left
        ("sashuihu", UsdFileCfg(
            usd_path=f"{usd_base_path}/sashuihu.usd",
            scale=(1.0, 1.0, 1.0),
        ), (-0.10, 0.10, 0.0), (1.0, 1.0, 1.0)),
    ]

    # Create objects up to num_objects
    for i in range(min(num_objects, len(object_defs))):
        name, spawn_cfg, offset, color = object_defs[i]

        # Calculate position
        pos = (
            desk_center[0] + offset[0],
            desk_center[1] + offset[1],
            desk_center[2] + offset[2] + 0.25  # Higher elevation to prevent initial overlap
        )

        # Common rigid body properties
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=100.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0
        )

        mass_props = sim_utils.MassPropertiesCfg(mass=0.05)

        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            metallic=0.5,
            roughness=0.3,
        )

        # All objects are now USD files, set their properties
        if isinstance(spawn_cfg, UsdFileCfg):
            spawn_cfg.rigid_props = rigid_props
            spawn_cfg.collision_props = collision_props
            spawn_cfg.mass_props = mass_props
            spawn_cfg.activate_contact_sensors = True  # Enable collision detection
            spawn_cfg.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )

        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1, 0, 0, 0)),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class DeskCleanupSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the desk cleanup scene.

    This scene simulates a desk environment with various real-world objects
    that need to be organized. The robot (Franka) with GelSight tactile sensors
    must grasp and manipulate these objects.

    Objects: Bowl, Cup, Clamp, Sponge, and Sashuihu (watering can)
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

    # Desk/Table - using Seattle Lab table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Robot - Franka with Gripper and GelSight sensors
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Desktop objects collection
    object_collection: RigidObjectCollectionCfg = create_desk_objects(num_objects=5)

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
            resolution=(64, 64),
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
        tactile_img_res=(64, 64),
    )

    # Right sensor - copy configuration from left
    gelsight_right = gelsight_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )
