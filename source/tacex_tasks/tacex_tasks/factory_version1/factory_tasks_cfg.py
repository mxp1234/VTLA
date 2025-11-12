# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from dataclasses import field
import numpy as np
from isaaclab.assets import RigidObjectCfg  
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
# 本地资产目录
LOCAL_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    #----解耦奖励设置------
    use_decoupled_reward: bool = False
    requires_orientation_logic: bool = False
    symmetry_angles_deg: list[float] = field(default_factory=lambda: [180])  # 对称角（如方形 peg 为 [0,90,180,270]）
    # -----触觉相关设置-----
    tactile_enabled_in_obs = False
    # tactile_enabled_in_state = False
    '''TACTILE_ENCODE_METHODS = ["raw", "tactile_force_field", "others"]'''

    tactile_encode_method: str = "tactile_force_field"

    if tactile_enabled_in_obs == True and tactile_encode_method == "tactile_force_field":
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
        ##TODO  暂时简化逻辑为默认采用tactile_force_field作为obs与state编码触觉的方式， 后续完善
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        # "tactile_force_field",
    ]
    # --- 奖励参数 ---
    xy_dist_coef: tuple[float, float] = (50.0, 2.0)
    z_dist_coef: tuple[float, float] = (20.0, 4.0)
    xy_dist_reward_scale: float = 2.0
    z_dist_reward_scale: float = 2.0

    orientation_reward_scale: float = 3
    yaw_success_threshold: float = float(np.deg2rad(5.0)) # 成功判断用
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚权重。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚权重。鼓励动作的连续性，防止抖动
    
    success_threshold: float = 0.04
    success_threshold_scale: float = 1.0
    engage_threshold: float = 0.9
    engage_threshold_scale: float = 1.0
    engage_half_threshold: float = 0.55
    engage_half_threshold_scale: float = 1.0

    orientation_coef: list = [5, 4]

    fixed_asset: ArticulationCfg = field(default=None, init=False)
    held_asset: ArticulationCfg = field(default=None, init=False)
    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_thread task only.
    action_penalty_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]  # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]  # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    kp_baseline_scale: float = 1.0 # 基线奖励的权重
    kp_coarse_scale: float = 1.0 # 粗略奖励的权重
    kp_fine_scale : float = 1.0
    engage_threshold: float = 0.9

    def __post_init__(self):
        """
        在对象初始化完成后，动态地构建 ArticulationCfg。
        """
        # --- 为 fixed_asset (Hole) 构建配置 ---
        self.fixed_asset = ArticulationCfg(
            prim_path="/World/envs/env_.*/FixedAsset",
            spawn=sim_utils.UsdFileCfg(
                # 【核心】从 self.fixed_asset_cfg 中动态获取 usd_path
                usd_path=self.fixed_asset_cfg.usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                # 【核心】从 self.fixed_asset_cfg 中动态获取 mass
                mass_props=sim_utils.MassPropertiesCfg(mass=self.fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

        # --- 为 held_asset (Peg) 构建配置 ---
        self.held_asset = ArticulationCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                # 【核心】从 self.held_asset_cfg 中动态获取 usd_path
                usd_path=self.held_asset_cfg.usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                # 【核心】从 self.held_asset_cfg 中动态获取 mass
                mass_props=sim_utils.MassPropertiesCfg(mass=self.held_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


# 本地资产目录
LOCAL_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


# ================================================================================================
# 本地资产配置类
# Local Asset Configuration Classes
# ================================================================================================


@configclass
class Peg10mmLocal(HeldAssetCfg):
    """
    本地10mm插销配置 - 用于测试初始化
    Local 10mm peg configuration - for testing initialization

    尺寸参数:
    - 直径: 10mm
    - 高度: 50mm
    - 质量: 25g
    """
    usd_path = f"{LOCAL_ASSETS_DIR}/peg.usd"
    diameter = 0.010  # 10mm 直径
    height = 0.050    # 50mm 高度
    mass = 0.025      # 25g 质量
    friction = 0.75   # 摩擦系数


@configclass
class Hole10mmLocal(FixedAssetCfg):
    """
    本地10mm插孔配置 - 用于测试初始化
    Local 10mm hole configuration - for testing initialization

    尺寸参数:
    - 内径: 10mm
    - 深度: 25mm
    - 质量: 50g
    """
    usd_path = f"{LOCAL_ASSETS_DIR}/hole/my_project/hole.usd"
    diameter = 0.010  # 10mm 内径
    height = 0.025    # 25mm 深度
    base_height = 0.0 # 无底座
    mass = 0.05       # 50g 质量
    friction = 0.75   # 摩擦系数

@configclass
class LPeg_III(HeldAssetCfg):
    """ 定义一个公差III的15mm L形插销 """
    usd_path = f"{LOCAL_ASSETS_DIR}/L_hole/L_peg_III.usd"

    diameter: float = 0.0149
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.003

# ================================================================================================
# 测试任务配置
# Test Task Configuration
# ================================================================================================
@configclass
class LHole(FixedAssetCfg):
    """ 定义一个匹配15mm L形插销的L孔 """
    usd_path = f"{LOCAL_ASSETS_DIR}/L_hole/L_hole.usd"
    diameter: float = 0.015
    # 孔深和基座高度与圆形资产保持一致
    height: float = 0.025
    base_height: float = 0.0

@configclass
class PegInsert_L_III(FactoryTask):

    name = "peg_insert_test"
    # -----新增奖励相关设置------

    use_decoupled_reward = True
    requires_orientation_logic = True
    xy_dist_reward_scale = 2.0
    xy_dist_coef: list = [50, 2]
    z_dist_reward_scale = 2.0
    z_dist_coef: list = [20, 4]

    duration_s = 20
    # 唯一的区别: 使用本地资产而非Nucleus资产
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]  # 无位置随机
    hand_init_orn_noise: list = [0.0, 0.0, 0.2]  # 无姿态随机

    # 禁用固定资产(Hole)的随机化
    fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]  # Hole位置固定
    fixed_asset_init_orn_deg: float = 180.0  # Hole初始方向
    fixed_asset_init_orn_range_deg: float = 30.0  # Hole方向随机范围=0，即不随机

    # 禁用手持资产(Peg)在夹爪中的随机化
    held_asset_pos_noise: list = [0.003, 0, 0.003]  # Peg在夹爪中位置固定
    # 指定构成"状态"的条目及其顺序
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    
    # L形只有1个对称方向：0度
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励
    held_asset_cfg = LPeg_III()
    fixed_asset_cfg = LHole()
    asset_size = held_asset_cfg.diameter

#   | 旋转描述      | 四元数 (w, x, y, z)       |
#   |-----------|----------------------------|
#   | 无旋转       | (1.0, 0.0, 0.0, 0.0)       |
#   | 绕X轴旋转90°  | (0.7071, 0.7071, 0.0, 0.0) |
#   | 绕Y轴旋转90°  | (0.7071, 0.0, 0.7071, 0.0) |
#   | 绕Z轴旋转90°  | (0.7071, 0.0, 0.0, 0.7071) |
#   | 绕X轴旋转180° | (0.0, 1.0, 0.0, 0.0)       |
#   | 绕Y轴旋转180° | (0.0, 0.0, 1.0, 0.0)       |
#   | 绕Z轴旋转180° | (0.0, 0.0, 0.0, 1.0)       |

@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012


@configclass
class GearMesh(FactoryTask):
    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    target_gear = "gear_medium"
    duration_s = 20.0

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=small_gear_usd,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=large_gear_usd,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    # Gears Asset
    add_flanking_gears = True
    add_flanking_gears_prob = 1.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.035]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of gear peg height.
    success_threshold: float = 0.05
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class NutM16(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_nut_m16.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25


@configclass
class BoltM16(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002


@configclass
class NutThread(FactoryTask):
    name = "nut_thread"
    fixed_asset_cfg = BoltM16()
    held_asset_cfg = NutM16()
    asset_size = 16.0
    duration_s = 30.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 1.83]
    hand_init_orn_noise: list = [0.0, 0.0, 0.26]

    # Action
    unidirectional_rot: bool = True

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.003, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward.
    ee_success_yaw = 0.0
    keypoint_coef_baseline: list = [100, 2]
    keypoint_coef_coarse: list = [500, 2]  # 100, 2
    keypoint_coef_fine: list = [1500, 0]  # 500, 0
    # Fraction of thread-height.
    success_threshold: float = 0.375
    engage_threshold: float = 0.5
    keypoint_scale: float = 0.05

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
