# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils # Isaac Lab的仿真工具模块
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg # 驱动器配置类
from isaaclab.assets import ArticulationCfg # 关节链配置类
from isaaclab.envs import DirectRLEnvCfg # 强化学习环境配置基类
from isaaclab.scene import InteractiveSceneCfg # 交互场景配置类
from isaaclab.sim import PhysxCfg, SimulationCfg # 物理引擎和仿真配置类
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg # 物理材质配置类
from isaaclab.utils import configclass # 配置类装饰器

from tacex_assets import TACEX_ASSETS_DATA_DIR # TacEx资源数据目录
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg # GelSight Mini传感器配置类

from .factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert, PegInsert_L_III # 任务配置类

# -- 观测空间维度配置字典 --
# 定义了"观测(Observation)"中各个组成部分的维度
# 观测是提供给RL策略(Actor)的信息,不包含特权信息(如真实位置、增益等)
OBS_DIM_CFG = {
    "fingertip_pos": 3, # 夹爪指尖中心点的三维坐标(x,y,z)
    "fingertip_pos_rel_fixed": 3, # 夹爪指尖相对于固定工件的相对坐标
    "fingertip_quat": 4, # 夹爪指尖姿态的四元数(w,x,y,z)
    "ee_linvel": 3, # 末端执行器的线速度(vx,vy,vz)
    "ee_angvel": 3, # 末端执行器的角速度(wx,wy,wz)
    # "tactile_left": 224 * 224 * 3,  # 左触觉传感器224x224 RGB原始图像(如果使用)
    # "tactile_right": 224 * 224 * 3,  # 右触觉传感器224x224 RGB原始图像(如果使用)
    "tactile_force_field": 3,  # 触觉力场特征: normal_sum, shear_x_sum, shear_y_sum
    "prev_actions": 6, # 上一时刻的动作(用于提供动作连续性信息)
}

# -- 状态空间维度配置字典 --
# 定义了"状态(State)"中各个组成部分的维度
# 状态是提供给RL Critic的信息,包含Actor的观测以及额外的特权信息
STATE_DIM_CFG = {
    # --- 包含所有观测信息 ---
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    # --- 额外的特权信息(只有Critic能看到) ---
    "joint_pos": 7, # 机器人手臂7个关节的角度
    "held_pos": 3, # 被夹持工件的三维坐标
    "held_pos_rel_fixed": 3, # 被夹持工件相对于固定工件的相对位置
    "held_quat": 4, # 被夹持工件的姿态(四元数)
    "fixed_pos": 3, # 固定工件的真实三维坐标
    "fixed_quat": 4, # 固定工件的真实姿态(四元数)
    "task_prop_gains": 6, # 任务空间控制器的P-gain(比例增益),6维对应x,y,z平移和roll,pitch,yaw旋转
    # "ema_factor": 1, # 指数移动平均平滑因子
    "pos_threshold": 3, # 位置动作的缩放阈值
    "rot_threshold": 3, # 旋转动作的缩放阈值
    # "tactile_left": 224 * 224 * 3,  # 左触觉传感器图像
    # "tactile_right": 224 * 224 * 3,  # 右触觉传感器图像
    "tactile_force_field": 3, # 触觉力场特征
    "prev_actions": 6, # 上一时刻的动作

}


@configclass
class ObsRandCfg:
    """
    观测随机化配置类。
    用于定义在观测数据上添加的噪声,以模拟传感器误差或感知不确定性。
    这种随机化有助于提高策略的鲁棒性,使其在真实环境中表现更好。
    """
    # 固定工件位置观测的高斯噪声标准差(单位:米)
    # [0.001, 0.001, 0.001]表示在x,y,z三个轴向上分别添加标准差为1毫米的噪声
    # 注意: 这不会改变工件在仿真中的真实物理位置,只会改变策略"看到"的位置
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    """
    控制配置类。
    汇集了所有与控制相关的参数,包括动作处理、重置状态和零空间控制等。
    """
    # -- 动作平滑参数 --
    # 指数移动平均(EMA)因子,用于平滑策略输出的动作
    # new_action = ema_factor * current_policy_action + (1 - ema_factor) * previous_action
    # 较小的值(如0.2)意味着更平滑的动作,可以防止机器人运动过于剧烈、抖动
    ema_factor = 0.2

    # -- 动作边界参数 --
    # 位置和旋转动作的边界,用于限制单步内机器人末端目标点相对于固定工件的最大移动范围
    pos_action_bounds = [0.05, 0.05, 0.05] # 位置边界: 末端目标点最多偏离固定工件中心±5cm
    rot_action_bounds = [1.0, 1.0, 1.0] # 旋转边界: ±1弧度

    # -- 动作缩放参数 --
    # 动作阈值/缩放因子,用于将策略输出的归一化动作[-1,1]映射到实际的物理空间
    # 实际的目标位移 = 策略输出 * 阈值
    # 例如,如果策略在x轴上输出0.5,则实际目标位置增量为 0.5 * 0.02 = 0.01米(1cm)
    pos_action_threshold = [0.02, 0.02, 0.02] # 位置阈值: 每步最多2cm
    rot_action_threshold = [0.097, 0.097, 0.097] # 旋转阈值: 每步最多0.097弧度(约5.6度)

    # -- 重置与默认状态参数 --
    # 环境重置时,机器人手臂7个关节的默认角度(弧度)
    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    # 在重置阶段(例如,用IK将手臂移动到初始位置、抓取物体)使用的控制器P-gain
    # 通常会设置得比较大,以便快速、稳定地到达目标
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0 # 重置时使用的旋转D-gain(微分增益)的缩放因子
    default_task_prop_gains = [100, 100, 100, 30, 30, 30] # 在正常RL交互阶段使用的默认控制器P-gain

    # -- 零空间控制参数 --
    # Null space parameters.
    # 零空间控制的目标关节姿态: 即机械臂在不影响末端姿态的前提下,希望达到的"舒适"关节姿态
    # 它的作用是在机器人执行任务的过程中,作为一个"期望的休息姿态"或"舒适姿态"
    # 控制器在保证完成末端跟踪任务(主要目标)的前提下,会利用剩余的自由度,
    # 施加一个微小的力矩,引导关节趋向于default_dof_pos_tensor所定义的姿态
    # 这可以防止关节漂移到奇异或极限位置,提高机械臂的整体操控性
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0 # 零空间控制的P-gain(比例增益)
    kd_null = 6.3246 # 零空间控制的D-gain(微分增益)


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    """
    Factory任务环境的主配置类。
    定义了强化学习环境的所有参数,包括动作/观测/状态空间、仿真设置、机器人配置等。
    继承自DirectRLEnvCfg,是Isaac Lab强化学习环境框架的配置基类。
    """
    # -- 强化学习核心参数 --
    decimation = 8 # 动作抽取频率,意味着RL策略每做出一次决策,物理仿真会执行8个步长
    action_space = 6 # 动作空间的维度(末端执行器x,y,z平移 + roll,pitch,yaw旋转)
    # num_*: will be overwritten to correspond to obs_order, state_order.
    # 观测空间和状态空间的维度会在运行时根据下面的obs_order和state_order自动重新计算
    observation_space = 21
    state_space = 72
    # 指定构成"观测"的条目及其顺序

    # -- 任务与控制配置 --
    task_name: str = "peg_insert"  # 默认任务名称 (可选: peg_insert, gear_mesh, nut_thread)
    task: FactoryTask = FactoryTask() # 具体的任务配置对象(会被子类覆写)
    obs_rand: ObsRandCfg = ObsRandCfg() # 观测随机化配置
    ctrl: CtrlCfg = CtrlCfg() # 控制相关配置
    evaluation_mode: bool = False # 一个标志位，用于区分训练和评估模式
    debug: bool = True
    episode_length_s = 10.0  # 每回合的最大时长(秒),Probably need to override.
    # -- 仿真配置 --
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", # 仿真在GPU上运行
        dt=1 / 120, # 物理仿真时间步长,120Hz
        gravity=(0.0, 0.0, -9.81), # 重力加速度(m/s²)
        physx=PhysxCfg( # PhysX物理引擎的详细配置
            solver_type=1, # 0: PGS (Projected Gauss-Seidel), 1: TGS (Temporal Gauss-Seidel,更稳定)
            max_position_iteration_count=192,  # 位置求解器的迭代次数,对于需要精确接触的任务(如插孔)非常重要,越高越能避免穿模
            max_velocity_iteration_count=1, # 速度求解器的迭代次数
            bounce_threshold_velocity=0.2, # 产生反弹的最小速度阈值(m/s)
            friction_offset_threshold=0.01, # 摩擦偏移阈值
            friction_correlation_distance=0.00625, # 摩擦关联距离
            gpu_max_rigid_contact_count=2**23, # GPU上支持的最大刚体接触数量
            gpu_max_rigid_patch_count=2**23, # GPU上支持的最大刚体接触面片数量
            gpu_max_num_partitions=1,  # GPU分区数量,设为1对稳定仿真很重要
        ),
        physics_material=RigidBodyMaterialCfg( # 默认的物理材质属性
            static_friction=1.0, # 静摩擦系数
            dynamic_friction=1.0, # 动摩擦系数
        ),
    )

    # -- 场景配置 --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0) # 并行128个环境,环境中心间距2米

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd",
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(224, 224),
        ),
        device="cuda",
        debug_vis=False,  # Set to False for headless mode (no GUI)
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],  # marker_motion
    )
    # settings for optical sim
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(224, 224),
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

@configclass
class FactoryTestPegInsertCfg(FactoryEnvCfg):
    """
    Peg插入测试环境配置 公差0.05mm的L型peg

    """
    # 任务配置
    task_name = "peg_insert_test"
    task = PegInsert_L_III()

    # 回合长度 (秒)
    # 测试时可以设置更长,便于观察初始化过程
    episode_length_s = 15.0  # 15秒

    # 场景配置
    # 测试时建议只用1个环境,便于观察
    # scene.num_envs = 1  # 在实例化后设置,这里定义的是默认值
    # 注意: num_envs需要在环境创建时通过命令行参数或代码指定

    # 观测空间和状态空间保持默认 (从FactoryEnvCfg继承)
    # observation_space和state_space会根据obs_order和state_order自动计算


@configclass
class FactoryTestPegInsertNoRandomCfg(FactoryEnvCfg):
    """
    Peg插入测试环境配置 - 无随机化版本

    使用方法:
    ```bash
    python scripts/play.py --task Isaac-Velocity-Flat-Franka-Factory-Test-Peg-Insert-NoRand-v0 --num_envs 1
    ```
    """
    task_name = "peg_insert_test_norand"

    # 创建一个修改后的任务配置,禁用所有随机化
    task = PegInsert_L_III()

    # 禁用所有位置和方向的随机化
    # 注意: 由于task是在类定义时创建的,我们需要在初始化时修改
    # 这里展示如何覆盖配置 (实际使用时需要在__post_init__中处理)
    # task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    # task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    # task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    # task.fixed_asset_init_orn_range_deg = 0.0
    # task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    # 设置非常长的episode长度,这样就只会初始化一次,不会重复reset
    # 方便观察初始化后的位姿
    episode_length_s = 1000.0  # 1000秒,足够长时间观察

    def __post_init__(self):
        """
        初始化后处理,用于修改任务配置中的随机化参数
        """
        super().__post_init__()

        # 禁用机器人手部初始化的随机化
        self.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

        # 禁用固定工件的随机化
        self.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_orn_range_deg = 0.0

        # 禁用被夹持工件的随机化
        self.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

        print("[FactoryTestPegInsertNoRandomCfg] 已禁用所有随机化,用于测试固定初始状态")
        print(f"[FactoryTestPegInsertNoRandomCfg] Episode长度设置为 {self.episode_length_s} 秒,环境将只初始化一次")


# ================================================================================================
# 调试和验证辅助信息
# Debug and Validation Helper Information
# ================================================================================================

"""
使用测试配置的步骤:

运行测试环境 (有随机化):
   ```bash
   cd /home/pi-zero/isaac-sim/TacEx
   python scripts/play.py --task Isaac-Velocity-Flat-Franka-Factory-Test-Peg-Insert-v0 --num_envs 1
   ```

运行测试环境 (无随机化):
   ```bash
   python scripts/play.py --task Isaac-Velocity-Flat-Franka-Factory-Test-Peg-Insert-NoRand-v0 --num_envs 1

"""
@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0
