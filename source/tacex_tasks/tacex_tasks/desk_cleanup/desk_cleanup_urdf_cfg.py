#!/usr/bin/env python3
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for desk cleanup environment using URDF imports.

This version imports objects from URDF files instead of USD files.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg, UrdfFileCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

# Import Franka with gripper and GelSight sensors
from tacex_assets.robots.franka import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini import GelSightMiniCfg


def create_desk_objects_from_urdf(num_objects: int = 5) -> RigidObjectCollectionCfg:
    """
    Create desktop objects for cleanup task using URDF files and USD files.

    Args:
        num_objects: Number of objects to spawn (default: 5, max: 8)

    Returns:
        RigidObjectCollectionCfg with desktop objects from URDF/USD files
    """
    rigid_objects = {}

    # Desk center and work area
    desk_center = [0.5, 0.0, 0.05]

    # Base path for YCB URDF files
    urdf_base_path = "/home/pi-zero/Downloads/ycb_urdfs-main/ycb_assets"

    # Object definitions: (name, file_type, filename, position_offset, scale, color)
    # file_type: 'urdf' or 'usd'
    object_defs = [
        # Bowl - 原始大小
        ("bowl", "urdf", "024_bowl.urdf", (0.0, -0.15, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Cup - 放大1.2倍
        ("cup", "urdf", "065-g_cups.urdf", (0.15, -0.10, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Clamp 1 - 原始大小
        ("clamp_0", "urdf", "050_medium_clamp.urdf", (-0.15, -0.10, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Clamp 2 - 新增
        ("clamp_1", "urdf", "050_medium_clamp.urdf", (-0.20, 0.0, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Clamp 3 - 新增
        ("clamp_2", "urdf", "050_medium_clamp.urdf", (-0.15, 0.10, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Sponge - 缩小到0.8倍
        ("sponge", "urdf", "026_sponge.urdf", (0.10, 0.10, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Watering can (Sashuihu) - 原始大小
        # ("sashuihu", "urdf", "022_windex_bottle.urdf", (-0.10, 0.10, 0.0), 0.1, (0.1, 0.1, 0.1)),

        # Small KLT container - USD file from Isaac Sim assets
        ("small_KLT", "usd", f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd", (0.0, 0.0, 0.0), 1.0, (0.8, 0.6, 0.4)),
    ]

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

    # Create objects up to num_objects
    for i in range(min(num_objects, len(object_defs))):
        name, file_type, file_path, offset, scale, color = object_defs[i]

        # Calculate position
        pos = (
            desk_center[0] + offset[0],
            desk_center[1] + offset[1],
            desk_center[2] + offset[2] + 0.25  # Higher elevation to prevent initial overlap
        )

        # Create spawn configuration based on file type
        if file_type == "urdf":
            # Create URDF import configuration
            spawn_cfg = UrdfFileCfg(
                asset_path=f"{urdf_base_path}/{file_path}",
                rigid_props=rigid_props,
                collision_props=collision_props,
                mass_props=mass_props,
                activate_contact_sensors=True,
                # URDF import specific settings
                fix_base=False,  # Objects should be free to move
                merge_fixed_joints=True,
                force_usd_conversion=True,  # Force regeneration of USD from URDF
                # Joint drive configuration (None for URDFs without joints - rigid objects)
                joint_drive=None,
            )
        else:  # file_type == "usd"
            # Create USD import configuration with detailed rigid body properties
            # Following the pattern from factory_tasks_cfg.py for USD objects
            spawn_cfg = UsdFileCfg(
                usd_path=file_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=mass_props,
                collision_props=collision_props,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,  # This is a rigid object, not an articulation
                ),
                # Set visual material for USD objects
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    metallic=0.2,
                    roughness=0.5,
                ),
            )

        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1, 0, 0, 0)),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class DeskCleanupUrdfSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the desk cleanup scene using URDF imports.

    This scene simulates a desk environment with YCB objects imported from URDF.
    The robot (Franka) with GelSight tactile sensors must grasp and manipulate these objects.
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

    # Desktop objects collection from URDF and USD (8 objects: bowl, cup, 3x clamp, sponge, sashuihu, small_KLT)
    object_collection: RigidObjectCollectionCfg = create_desk_objects_from_urdf(num_objects=8)

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
