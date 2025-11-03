import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from dataclasses import field

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from .peg_in_hole_tasks_cfg import ASSET_DIR, PegInHoleTask, PegInHoleCircleHole_I, PegInHoleCircleHole_test, PegInHoleLHole_III, PegInHoleSquareHole_II

OBS_DIM_CFG = { #  这个字典定义了“观测（Observation）”中各个组成部分的维度
    "fingertip_pos": 3, # 夹爪指尖中心点的三维坐标 (x, y, z)，所以维度是3
    "fingertip_pos_rel_fixed": 3, # 夹爪指尖中心点相对于固定工件（例如插孔）的三维相对坐标，维度是3。相对坐标通常比绝对坐标更有利于学习
    "fingertip_quat": 4, # 夹爪指尖中心点的姿态，使用四元数 (w, x, y, z) 表示，所以维度是4
    "fingertip_quat_rel_fixed": 4, # 【新增】夹爪指尖中心点相对于固定工件的相对姿态四元数，维度是4。这有助于策略理解夹爪与插孔之间的相对方向关系
    "ee_linvel": 3, # 末端执行器（End-Effector）的线性速度 (vx, vy, vz)，维度是3
    "ee_angvel": 3, # 末端执行器的角速度 (wx, wy, wz)，维度是3
    "tactile_force_field": 3, # 力场特征: normal_sum, shear_x_sum, shear_y_sum，维度是3
}

STATE_DIM_CFG = { # 这个字典定义了“状态（State）”中各个组成部分的维度
    # --- 包含所有观测信息 ---
    # 下面是Actor能看到的部分
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_quat_rel_fixed": 4, # 【新增】
    "ee_linvel": 3,
    "ee_angvel": 3,
    "tactile_force_field": 3,
    # --- 额外的、只有Critic能看到的特权信息 ---
    "joint_pos": 7, # 机器人手臂7个关节的角度
    "held_pos": 3, # 被夹爪持有的工件（例如插销）的三维坐标
    "held_pos_rel_fixed": 3, # 被持有工件相对于固定工件的相对位置
    "held_quat": 4, # 被持有工件的姿态（四元数）
    "fixed_pos": 3, # 固定工件（例如插孔）的真实三维坐标
    "fixed_quat": 4, # 固定工件的真实姿态（四元数）
    "task_prop_gains": 6, # 任务空间控制器（OSC）的P-gain（比例增益），6个值分别对应x,y,z平移和roll,pitch,yaw旋转
    "ema_factor": 1, # 指数移动平均（EMA）平滑因子，用于平滑动作
    "pos_threshold": 3, # 位置动作的缩放阈值
    "rot_threshold": 3, # 旋转动作的缩放阈值
}

@configclass
class ObsRandCfg:
    """
    这个配置类专门用于定义“观测随机化”的参数。
    在提供给策略的“观测数据”上添加噪声，以模拟传感器误差或感知不确定性。
    """
    # 定义了在“固定工件位置”的观测值上所添加的高斯噪声的标准差。
    # 例如，[0.001, 0.001, 0.001] 意味着在x, y, z三个轴向上分别添加标准差为0.001米（1毫米）的噪声。
    # 注意：这不会改变工件在仿真中的真实物理位置，只会改变策略“看到”的位置。
    fixed_asset_pos = [0.001, 0.001, 0.001]
    
@configclass
class CtrlCfg:
    """
    这个配置类汇集了所有与“控制（Control）”相关的参数。
    它详细定义了如何解释策略输出的动作、如何控制机器人以及如何进行重置。
    """
    # --- 动作处理 ---
    # 指数移动平均（EMA）因子，用于平滑策略输出的动作。
    # new_action = ema_factor * current_policy_action + (1 - ema_factor) * previous_action
    # 较小的值意味着更平滑的动作，可以防止机器人运动过于剧烈、抖动。
    ema_factor = 0.2

    # 位置和旋转动作的边界。用于限制单步内机器人末端目标点相对于固定工件的最大移动范围。
    # 这可以防止策略在探索时发出过于激进的指令，从而提高训练稳定性。
    # 确保了不论策略输出什么，最终发送给控制器的目标位置指令都不会超出以固定工件为中心的一个 5cm x 5cm x 5cm 的立方体区域
    pos_action_bounds = [0.05, 0.05, 0.05] # 末端目标点最多偏离固定工件中心5厘米
    rot_action_bounds = [1.0, 1.0, 1.0] # 旋转边界（弧度）

    # 动作阈值/缩放因子。策略网络输出的动作值通常在[-1, 1]之间。
    # 实际的目标位移 = 策略输出 * 阈值。
    # 例如，如果策略在x轴上输出0.5，则实际目标位置增量为 0.5 * 0.02 = 0.01米。
    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    # --- 重置与默认状态 ---
    # 环境重置时，机器人手臂7个关节的默认角度（弧度）。
    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    # 在重置阶段（例如，用IK将手臂移动到初始位置、抓取物体）使用的控制器P-gain。
    # 通常会设置得比较大，以便快速、稳定地到达目标。
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0 # 重置时使用的旋转D-gain（微分增益）的缩放因子
    default_task_prop_gains = [100, 100, 100, 30, 30, 30] # 在正常RL交互阶段使用的默认控制器P-gain

    # --- 零空间（Null Space）控制参数 ---
    # 零空间控制的目标：即机械臂在不影响末端姿态的前提下，希望达到的“舒适”关节姿态
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754] # 它的作用是在机器人执行任务的过程中，作为一个“期望的休息姿态”或“舒适姿态”。控制器在保证完成末端跟踪任务（主要目标）的前提下，会利用剩余的自由度，施加一个微小的力矩，引导关节趋向于 default_dof_pos_tensor 所定义的姿态。这可以防止关节漂移到奇异或极限位置，提高机械臂的整体操控性
    kp_null = 10.0 # 零空间控制的P-gain（比例增益）
    kd_null = 6.3246 # 零空间控制的D-gain（微分增益）
    
@configclass
class PegInHoleEnvCfg(DirectRLEnvCfg):
    """
    这个配置类定义了Peg-in-Hole任务的强化学习环境。
    它继承自DirectRLEnvCfg，并根据任务需求进行了扩展和修改。
    """
    # --- 强化学习核心参数 ---
    decimation = int(8) # 动作抽取频率。意味着RL策略每做出一次决策，物理仿真会执行8个步长。这使得策略可以在一个更长的时间尺度上进行控制
    action_space = int(6) # 动作空间的维度（末端执行器x,y,z平移 + roll,pitch,yaw旋转）
    observation_space = int(21) # 观测空间的维度。这个值会在`peg_in_hole_v0_env.py`的`__init__`中根据下面的`obs_order`被自动重新计算
    state_space = int(72) # 状态空间的维度。同样会被自动重新计算
    # 指定构成“观测”的条目及其顺序。`peg_in_hole_v0_env.py`会按照这个顺序从OBS_DIM_CFG查找维度并拼接张量
    state_order: list = [ # 指定构成“状态”的条目及其顺序
        "fingertip_pos", # 夹爪指尖位置
        "fingertip_pos_rel_fixed", # 夹爪指尖（peg）相对于固定工件（hole）的位置
        "fingertip_quat", # 夹爪指尖姿态
        "fingertip_quat_rel_fixed", # 夹爪指尖（peg）相对于固定工件（hole）的姿态
        "ee_linvel", # 末端线速度
        "ee_angvel", # 末端角速度
        "tactile_force_field", # 力场特征
        "joint_pos", # 机器人关节位置，7dof
        "held_pos", # 被夹持工件位置
        "held_pos_rel_fixed", # 被夹持工件相对于固定工件的位置
        "held_quat", # 被夹持工件姿态
        "fixed_pos", # 固定工件位置
        "fixed_quat", # 固定工件姿态
    ]
    # --- 任务与控制配置 ---
    task_name: str = "peg_in_hole_circle_test"  # 默认任务名称，会被具体的任务配置类覆写
    task: PegInHoleTask = PegInHoleTask() # 具体的任务配置对象，包含了随机化、奖励函数等参数。会被覆写
    obs_rand: ObsRandCfg = ObsRandCfg() # 观测随机化配置
    ctrl: CtrlCfg = CtrlCfg() # 控制相关配置

    evaluation_mode: bool = False # 一个标志位，用于区分训练和评估模式
    episode_length_s = 10.0  # 每回合（episode）的最大时长（秒）
    # --- 仿真（Simulation）配置 ---
    sim: SimulationCfg = SimulationCfg( # SimulationCfg在前面import进来
        device="cuda:0", # 指定仿真在哪个GPU上运行
        dt=1 / 120, # 物理仿真的时间步长，这里是120Hz
        gravity=(0.0, 0.0, -9.81), # 重力加速度
        physx=PhysxCfg( # PhysX物理引擎的详细配置，同样在前面import
            solver_type=1, # 0: PGS, 1: TGS。TGS是默认的，通常更稳定
            max_position_iteration_count=192, # 位置求解器的迭代次数。对于需要精确接触的任务（如插孔），这个值非常重要，越高越能避免穿模
            max_velocity_iteration_count=1, # 速度求解器的迭代次数
            bounce_threshold_velocity=0.2, # 产生反弹的最小速度阈值
            friction_offset_threshold=0.01, # 摩擦偏移阈值
            friction_correlation_distance=0.00625, # 摩擦关联距离
            gpu_max_rigid_contact_count=2**23, # GPU上支持的最大刚体接触数量
            gpu_max_rigid_patch_count=2**23, # GPU上支持的最大刚体接触面片数量
            gpu_collision_stack_size=2**28, # GPU碰撞堆栈大小
            gpu_max_num_partitions=1,  # GPU分区数量。设为1对稳定仿真很重要
        ),
        physics_material=RigidBodyMaterialCfg( # 默认的物理材质属性，同样在前面import
            static_friction=1.0, # 默认静摩擦系数
            dynamic_friction=1.0, # 默认动摩擦系数
        ),
    )
    
    # --- 场景（Scene）配置 ---
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(
    #     num_envs=128, # 并行环境的数量，即同时训练128个机器人
    #     env_spacing=2.0, # 每个环境中心点之间的距离
    #     # clone_in_fabric=True # 是否在Fabric（Isaac Sim的数据后端）中克隆，可以提高效率
    #     ) 
    
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
            num_envs=128,
            env_spacing=2.0,
            # 注意：这里我们默认不设置 clone_in_fabric
        )
    )
    
    # --- 机器人（Robot）配置 ---
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot", # prim_path 定义了机器人在USD场景图中的路径
        spawn=sim_utils.UsdFileCfg( # spawn 定义了如何生成这个机器人。这里是从一个USD文件加载
            # usd_path=f"{ASSET_DIR}/franka_mimic.usd", # 机器人模型的USD文件路径
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd", # 使用带有GelSight Mini触觉传感器的夹爪模型
            activate_contact_sensors=True, # 激活接触传感器
            rigid_props=sim_utils.RigidBodyPropertiesCfg( # 机器人连杆的刚体属性
                disable_gravity=True, # 对机器人禁用重力，因为控制器会补偿重力，可以简化控制问题
                max_depenetration_velocity=5.0,  # 最大反穿透速度，当物理引擎检测到两个物体发生了穿透（即互相嵌入了一点），它会施加一个力将它们推开。这个参数限制了推开时物体的最大速度。设置一个合理的值可以防止物体因为求解器强大的修正力而“爆炸”式地弹开，让接触分离更柔和、更稳定
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32, # 冲量是力在时间上的积累。这个参数限制了在单次接触事件中，物理引擎可以施加的最大冲量。它也是一个为了防止仿真“爆炸”，提高稳定性的参数。在需要模拟非常巨大冲击力的场景外，通常保持一个较大的默认值即可
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg( # 机器人作为关节链的整体属性
                enabled_self_collisions=False, # 禁用自碰撞，可以提高性能和稳定性
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0), # 碰撞属性，物理引擎为了提高效率和稳定性，并不会等到物体表面真正几何接触时才开始计算碰撞。它会在物体表面外包围一层看不见的“外壳”，这个外壳的厚度就是 contact_offset (这里是5毫米)。当两个物体的“外壳”接触时，碰撞检测就会被触发。这给了求解器一个提前反应的缓冲期。定义了物体稳定接触（静置）时，它们几何表面之间的距离。一个负值可以让物体在静止时轻微互相嵌入，一个正值则会使它们保持微小间隙，0.0 表示理想情况下几何表面正好贴合
        ),
        # init_state 定义了机器人的初始状态
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={ # 各个关节的初始角度
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0), # 机器人基座的初始位置（相对于环境原点）
            rot=(1.0, 0.0, 0.0, 0.0), # 机器人基座的初始姿态（四元数，表示无旋转）
        ),
        # actuators 定义了机器人的驱动器
        actuators={
            # 将手臂关节分为两组，并为它们配置不同的驱动器参数
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"], # 正则表达式，匹配1到4号关节
                stiffness=0.0, # 刚度设为0，因为我们直接用力矩控制
                damping=0.0, # 阻尼设为0
                friction=0.0,
                armature=0.0, # 这是一个在机器人仿真中很常见的技巧性参数。它向关节的惯性矩阵的对角线元素上增加一个小的数值。从物理上讲，这有点像人为地增加了电机转子的转动惯量。主要作用是提高数值稳定性
                effort_limit_sim=87, # 力矩上限
                velocity_limit_sim=124.6, # 速度上限
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg( # 为手爪配置驱动器，这里使用了较大的刚度和阻尼，因为手爪是通过位置控制（PD控制）来闭合的，而不是直接力矩控制
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
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
        debug_vis=False,
        data_types=["tactile_rgb"],
        marker_motion_sim_cfg=None,
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
    
    def __post_init__(self):
        """
        在所有子类配置都完成后，进行最终的动态配置。
        """
        # 检查当前实例的task配置，动态决定是否启用 fabric 克隆
        if hasattr(self.task, 'tactile') and self.task.tactile["tactile_enabled"]:
            print(f"[INFO] Task '{self.task.name}' has tactile enabled. Disabling 'clone_in_fabric' for compatibility.")
            self.scene.clone_in_fabric = False
        else:
            print(f"[INFO] Task '{self.task.name}' has tactile disabled. Enabling 'clone_in_fabric' for performance.")
            self.scene.clone_in_fabric = True
    
@configclass
class PegInHoleCircleHole_test_Cfg(PegInHoleEnvCfg):
    """
    这个配置类专门用于“圆形插销-测试”任务环境。
    它继承自PegInHoleEnvCfg，并覆写了一些参数以适应具体任务需求。
    """
    task= PegInHoleCircleHole_test() # 使用圆形插销测试任务的具体配置对象
    task_name: str = task.name
    episode_length_s = task.duration_s
    
@configclass
class PegInHoleCircleHole_I_Cfg(PegInHoleEnvCfg):
    """
    这个配置类专门用于“圆形插销-任务I”环境。
    它继承自PegInHoleEnvCfg，并覆写了一些参数以适应具体任务需求。
    """
    task= PegInHoleCircleHole_I() # 使用圆形插销任务I的具体配置对象
    task_name: str = task.name
    episode_length_s = task.duration_s
    
@configclass
class PegInHoleSquareHole_II_Cfg(PegInHoleEnvCfg):
    """
    这个配置类专门用于“方形插销-任务II”环境。
    它继承自PegInHoleEnvCfg，并覆写了一些参数以适应具体任务需求。
    """
    task= PegInHoleSquareHole_II() # 使用方形插销任务II的具体配置对象
    task_name: str = task.name
    episode_length_s = task.duration_s
    
@configclass
class PegInHoleLHole_III_Cfg(PegInHoleEnvCfg):
    """
    这个配置类专门用于“L形插销-任务III”环境。
    它继承自PegInHoleEnvCfg，并覆写了一些参数以适应具体任务需求。
    """
    task= PegInHoleLHole_III() # 使用L形插销任务III的具体配置对象
    task_name: str = task.name
    episode_length_s = task.duration_s