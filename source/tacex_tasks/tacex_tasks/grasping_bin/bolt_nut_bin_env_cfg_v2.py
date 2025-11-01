# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Bolt-Nut Bin Grasping Environment Configuration (V2 - Articulation-based).

This configuration follows the official Factory task pattern (NutThread)
from factory_tasks_cfg.py, using ArticulationCfg for all objects instead
of RigidObjectCollection.

Key differences from V1:
- Uses ArticulationCfg instead of RigidObjectCfg/RigidObjectCollection
- Fixed number of objects (3 bolts + 2 nuts) as scene attributes
- Physics APIs are automatically applied by Isaac Sim
- Proper visual rendering guaranteed
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg,RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

# Import Franka with gripper and GelSight sensors
from tacex_assets.robots.franka import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini import GelSightMiniCfg

# Import factory task assets for USD paths
from tacex_tasks.factory.factory_tasks_cfg import BoltM16, NutM16

##
# Helper functions for dynamic scene creation
##

def create_bolt_nut_scene_with_counts(num_bolts: int = 3, num_nuts: int = 2) -> type:
    """
    Dynamically create a BoltNutBinSceneCfg class with specified number of bolts and nuts.

    This function generates ArticulationCfg attributes for each bolt and nut,
    following the NutThread pattern from factory_tasks_cfg.py.

    Args:
        num_bolts: Number of bolts to create (default: 3)
        num_nuts: Number of nuts to create (default: 2)

    Returns:
        A BoltNutBinSceneCfg class with the specified objects
    """
    import numpy as np
    from isaaclab.utils import configclass

    # Base positions for objects - two bins
    bin_left_center = [0.4, 0.2, 0.05]   # Left bin for bolts
    bin_right_center = [0.4, -0.2, 0.05]  # Right bin for nuts

    # Create class attributes dictionary
    class_attrs = {}

    # Add standard scene components
    class_attrs['ground'] = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    class_attrs['light'] = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    class_attrs['table'] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Bin 1 - Left bin for bolts
    class_attrs['bin_left'] = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.2, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
            scale=(1.8, 1.5, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # Bin 2 - Right bin for nuts
    class_attrs['bin_right'] = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BinRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
            scale=(1.8, 1.5, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    class_attrs['robot'] = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Create bolt ArticulationCfg objects - place in left bin
    for i in range(num_bolts):
        angle = 2 * np.pi * i / max(num_bolts, 1)
        radius = 0.04  # Smaller radius to fit in bin
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        pos = (bin_left_center[0] + x_offset, bin_left_center[1] + y_offset, bin_left_center[2])

        class_attrs[f'bolt_{i}'] = ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Bolt_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=BoltM16().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

    # Create nut ArticulationCfg objects - place in right bin
    for i in range(num_nuts):
        angle = 2 * np.pi * i / max(num_nuts, 1) + np.pi / max(num_nuts, 1)
        radius = 0.03  # Even smaller radius for nuts
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        pos = (bin_right_center[0] + x_offset, bin_right_center[1] + y_offset, bin_right_center[2])

        class_attrs[f'nut_{i}'] = ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Nut_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=NutM16().usd_path,
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

    # Add sensors
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
    gelsight_left.optical_sim_cfg = gelsight_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(64, 64),
    )

    class_attrs['gelsight_left'] = gelsight_left
    class_attrs['gelsight_right'] = gelsight_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    # Add end-effector frame
    class_attrs['ee_frame'] = FrameTransformerCfg(
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

    # Create the class dynamically
    DynamicSceneCfg = type(
        'BoltNutBinSceneCfg',
        (InteractiveSceneCfg,),
        class_attrs
    )

    # Apply @configclass decorator
    return configclass(DynamicSceneCfg)


##
# Scene definition (default with 3 bolts + 2 nuts)
##

@configclass
class BoltNutBinSceneCfg(InteractiveSceneCfg):
    """
    Scene configuration for bolt-nut bin grasping environment.

    Following the NutThread pattern: each object is a separate ArticulationCfg attribute.
    This ensures proper physics API application and visual rendering.
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
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
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

    # Bolt objects - Following NutThread/Factory pattern
    # Each bolt is a separate ArticulationCfg (not part of a collection)
    bolt_0: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bolt_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BoltM16().usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # BoltM16 mass
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.15), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    bolt_1: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bolt_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BoltM16().usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.05, 0.01), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    bolt_2: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bolt_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BoltM16().usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, -0.05, 0.01), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    # # Nut objects - Following NutThread/Factory pattern
    # nut_0: ArticulationCfg = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Nut_0",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=NutM16().usd_path,
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=192,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.03),  # NutM16 mass
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.45, 0.03, 0.15), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    #     ),
    #     actuators={},
    # )

    # nut_1: ArticulationCfg = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Nut_1",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=NutM16().usd_path,
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=3666.0,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=192,
    #             solver_velocity_iteration_count=1,
    #             max_contact_impulse=1e32,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.55, -0.03, 0.15), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    #     ),
    #     actuators={},
    # )

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
    gelsight_left.optical_sim_cfg = gelsight_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(64, 64),
    )

    gelsight_right = gelsight_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot joint positions
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # Robot joint velocities
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # Gripper state
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
        )

        # Bolt positions (each as separate observation term)
        bolt_0_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bolt_0")})
        bolt_1_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bolt_1")})
        bolt_2_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bolt_2")})

        # Nut positions
        nut_0_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("nut_0")})
        nut_1_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("nut_1")})

        # Tactile observations from left sensor
        tactile_left_rgb = ObsTerm(
            func=lambda env: env.scene.sensors["gelsight_left"].data.output.get("tactile_rgb",
                torch.zeros(env.num_envs, 32, 32, 3, device=env.device)).flatten(start_dim=1)
            if hasattr(env.scene.sensors["gelsight_left"].data, 'output') and env.scene.sensors["gelsight_left"].data.output
            else torch.zeros(env.num_envs, 32*32*3, device=env.device),
        )

        # Tactile observations from right sensor
        tactile_right_rgb = ObsTerm(
            func=lambda env: env.scene.sensors["gelsight_right"].data.output.get("tactile_rgb",
                torch.zeros(env.num_envs, 32, 32, 3, device=env.device)).flatten(start_dim=1)
            if hasattr(env.scene.sensors["gelsight_right"].data, 'output') and env.scene.sensors["gelsight_right"].data.output
            else torch.zeros(env.num_envs, 32*32*3, device=env.device),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot state
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    # Reset gripper state to open
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        },
    )

    # Reset bolt positions using mdp.reset_root_state_uniform (like NutThread)
    reset_bolt_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.55), "y": (-0.05, 0.05), "z": (0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bolt_0"),
        },
    )

    reset_bolt_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.65), "y": (0.0, 0.1), "z": (0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bolt_1"),
        },
    )

    reset_bolt_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.55), "y": (-0.1, 0.0), "z": (0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bolt_2"),
        },
    )

    # Reset nut positions
    reset_nut_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.40, 0.50), "y": (0.0, 0.06), "z": (0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("nut_0"),
        },
    )

    reset_nut_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.50, 0.60), "y": (-0.06, 0.0), "z": (0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("nut_1"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Penalize large actions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalize joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##

@configclass
class BoltNutBinEnvCfgV2(ManagerBasedRLEnvCfg):
    """
    Configuration for the bolt-nut bin grasping environment (V2 - Articulation-based).

    This configuration follows the official Factory task pattern from NutThread,
    ensuring proper physics API application and visual rendering.

    Fixed configuration:
    - 3 bolts (bolt_0, bolt_1, bolt_2)
    - 2 nuts (nut_0, nut_1)
    - Each object is a separate Articulation in the scene
    """

    # Scene settings
    scene: BoltNutBinSceneCfg = BoltNutBinSceneCfg(num_envs=4, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.dt = 1 / 60  # 60 Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Enable scene query support for sensors
        self.sim.enable_scene_query_support = True

        # viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.5, 0.0, 0.5)
