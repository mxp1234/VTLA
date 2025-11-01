# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
'''
old version, new version: /home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/grasping_bin/bolt_nut_bin_env_cfg_v2.py
'''
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
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

# Import factory task assets
# from tacex_tasks.factory.factory_tasks_cfg import BoltM16, NutM16  # Commented out - using local USD files instead

##
# Helper functions
##

def create_object_collection(num_bolts: int = 5, num_nuts: int = 5) -> RigidObjectCollectionCfg:
    """
    Create a dynamic object collection with specified number of bolts and nuts.

    Args:
        num_bolts: Number of bolt objects to spawn (default: 5)
        num_nuts: Number of nut objects to spawn (default: 5)

    Returns:
        RigidObjectCollectionCfg with the specified objects
    """
    import numpy as np

    rigid_objects = {}

    # Bin center and size
    bin_center = np.array([0.5, 0.0, 0.15])  # Bin is at (0.5, 0.0, 0.12), objects at 0.15
    bin_half_size = np.array([0.08, 0.12, 0.0])  # Bin dimensions

    # Create bolts
    for i in range(num_bolts):
        # Distribute objects in a circular pattern within bin bounds
        angle = 2 * np.pi * i / max(num_bolts, 1)
        radius = 0.1  # Smaller radius to keep objects inside bin (bin is ~0.08m wide)
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)

        pos = (
            bin_center[0] + x_offset,
            bin_center[1] + y_offset,
            bin_center[2]
        )

        rigid_objects[f"bolt_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Bolt_{i}",
            spawn=UsdFileCfg(
                usd_path="/home/pi-zero/Documents/USD-file/factory_bolt_m16_loose.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # BoltM16 default mass
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                # Set bolt color - red/orange
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.3, 0.1),  # RGB: orange-red color
                    metallic=0.6,
                    roughness=0.4,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1, 0, 0, 0)),
        )
    print("[rnm dengping] create bolts done---------")
    # Create nuts
    for i in range(num_nuts):
        angle = 2 * np.pi * i / max(num_nuts, 1) + np.pi / max(num_nuts, 1)  # Offset from bolts
        radius = 0.06  # Even smaller radius for nuts to avoid overlap
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)

        pos = (
            bin_center[0] + x_offset,
            bin_center[1] + y_offset,
            bin_center[2]
        )

        rigid_objects[f"nut_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Nut_{i}",
            spawn=UsdFileCfg(
                usd_path="/home/pi-zero/Documents/USD-file/factory_nut_m16.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),  # NutM16 mass
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                # Set nut color - blue/cyan
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.5, 0.9),  # RGB: blue color
                    metallic=0.7,
                    roughness=0.3,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1, 0, 0, 0)),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)

##
# Scene definition
##

@configclass
class BoltNutBinSceneCfg(InteractiveSceneCfg):
    """Configuration for the bolt-nut bin scene with a Franka robot and tactile sensors."""

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

    # Bolt and Nut objects using RigidObjectCollection
    # Default: 5 bolts + 5 nuts, can be overridden by calling create_object_collection()
    object_collection: RigidObjectCollectionCfg = create_object_collection(num_bolts=5, num_nuts=5)

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

    # GelSight sensors on gripper fingers - following factory_env_cfg pattern
    gelsight_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(64, 64),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb", "camera_depth"],
    )
    # Configure optical simulation settings (important for visualization)
    gelsight_left.optical_sim_cfg = gelsight_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(64,64),
    )

    # Right sensor - copy configuration from left for consistency
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

    # Arm action: differential IK
    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )

    # Gripper action: binary gripper action (close/open)
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

        # Object positions (from collection - includes all bolts and nuts)
        object_positions = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object_collection")},
        )

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

    # NOTE: reset_object_position event is REMOVED because mdp.reset_root_state_uniform
    # does not work with RigidObjectCollection (it only works with RigidObject and Articulation).
    # The RigidObjectCollection uses default_object_state instead of default_root_state.
    # Objects will spawn at their configured init_state positions without randomization.


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
class BoltNutBinEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bolt-nut bin grasping environment."""

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
