import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg, RigidObject  # 若你用 omni 命名空间则改为 omni.isaac.lab.assets

from isaaclab.utils import configclass
from dataclasses import field
import os
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
PEG_AND_HOLE_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets") # 定义一个全局变量，指向本地存放Peg和Hole资产（USD文件）的目录

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory" # 定义一个全局变量，指向Isaac Lab在NVIDIA Nucleus服务器上存放Factory任务相关模型资产（USD文件）的目录

@configclass
class FixedAssetCfg:
    """
    用于配置“固定工件”（Fixed Asset）的通用模板。
    固定工件指的是任务中通常保持静止的目标物体，如Hole
    """
    usd_path: str = "" # 物体模型的USD文件路径
    diameter: float = 0.0 # 物体的特征直径
    height: float = 0.0 # 物体的特征高度
    base_height: float = 0.0  # 物体底座的高度
    friction: float = 0.75 # 物体表面的摩擦系数
    mass: float = 0.05 # 物体的质量（千克）
    
@configclass
class HeldAssetCfg:
    """
    用于配置“手持工件”（Held Asset）的通用模板。
    手持工件指的是被机器人夹爪抓取并移动的物体，如peg
    """
    usd_path: str = "" # 物体模型的USD文件路径
    diameter: float = 0.0  # 物体的特征直径
    height: float = 0.0 # 物体的特征高度
    friction: float = 0.75 # 物体表面的摩擦系数
    mass: float = 0.05 # 物体的质量（千克）
    
@configclass
class RobotCfg:
    """
    用于配置机器人特定参数的通用模板。
    """
    robot_usd: str = "" # 机器人模型的USD文件路径
    franka_fingerpad_length: float = 0.017608 # Franka机器人指垫的长度
    friction: float = 0.75 # 机器人夹爪表面的摩擦系数
    
@configclass
class PegInHoleTask:
    """
    用于配置Peg-in-Hole任务的通用模板。
    具体的任务配置类（例如PegInHoleCircleHoleCfg）将继承自这个模板，并根据具体任务需求进行扩展和修改。
    """
    # -- 基础配置 --
    robot_cfg: RobotCfg = RobotCfg() # 机器人相关配置
    name: str = "" # 任务名称，会被子类覆写
    duration_s = 5.0 # 任务默认时长（秒），会被子类覆写
    # -- 资产配置占位符 --
    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0 # 资产的特征尺寸（例如10mm的插销，就是10.0）
    
    # ================= 领域随机化 (Domain Randomization) 参数 =================
    # -- 机器人初始姿态随机化 --
    hand_init_pos: list = [0.0, 0.0, 0.047]  # 机械臂末端（手）在重置时的初始位置，这个位置是相对于“固定工件顶端”的
    hand_init_pos_noise: list = [0.02, 0.02, 0.01] # 在上述基础位置上添加的均匀随机噪声范围。[-0.02, 0.02] on x, [-0.02, 0.02] on y, [-0.01, 0.01] on z
    hand_init_orn: list = [3.1416, 0.0, 0.0] # 机械臂末端的初始姿态（欧拉角 roll, pitch, yaw），[pi, 0, 2.356] 是一个朝下的姿态
    hand_init_orn_noise: list = [0.0, 0.0, 0.0] # 在上述基础姿态上添加的均匀随机噪声范围（弧度）

    # -- 固定工件初始位姿随机化 --
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05] # 固定工件初始位置的随机噪声范围
    # fixed_asset_init_pos_noise: list = [0, 0, 0] # 固定工件初始位置的随机噪声范围
    fixed_asset_init_orn_deg: float = 0.0 # 固定工件初始yaw（绕Z轴旋转）的基础值（度）
    fixed_asset_init_orn_range_deg: float = 0.0 # 在基础yaw上添加的均匀随机噪声范围（度）

    # -- 手持工件初始位姿随机化 --
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # 在夹爪已经抓好工件后，给工件的相对位置添加的噪声
    held_asset_rot_init: float = 0.0 # 手持工件的初始旋转（度）
    
    # ================= 奖励函数 (Reward Function) 参数 =================
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚权重。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚权重。鼓励动作的连续性，防止抖动
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # 奖励函数使用了一个压缩函数 r(x) = 1 / (exp(ax) + b + exp(-ax))，其中x是关键点距离。
    # [a, b] 这两个参数控制了函数曲线的形状（a控制陡峭程度，b控制最大值）。
    
    # -- 关键点奖励参数 --
    num_keypoints: int = 4 # 在工件上用于计算距离的关键点数量
    keypoint_scale: float = 0.15 # 关键点沿Z轴分布的尺度。值越大，关键点分布越分散
    # "基线"奖励：a=5, b=4。曲线非常平缓，用于在距离很远时提供一个大致正确的方向引导
    keypoint_coef_baseline: list = [5, 4]
    kp_baseline_scale: float = 1.0 # 基线奖励的权重
    # "粗略"奖励：a=50, b=2。曲线变陡，用于在两个工件大致对齐时提供更强的引导信号
    keypoint_coef_coarse: list = [50, 2]
    kp_coarse_scale: float = 1.0 # 粗略奖励的权重
    # "精细"奖励：a=100, b=0。曲线最陡峭，用于在最后“临门一脚”的插入或拧紧阶段，对极小的位置误差提供非常强的奖励信号
    keypoint_coef_fine: list = [100, 0]
    kp_fine_scale: float = 1.0 # 精细奖励的权重
    
    # -- 成功/状态判断阈值 --
    # 这两个值都是“固定工件高度”的一个比例。
    # 判断任务是否成功的阈值。例如，0.04意味着插销插入插孔深度的4%即算成功
    success_threshold: float = 0.04
    success_threshold_scale: float = 1.0
    engage_threshold: float = 0.9
    engage_threshold_scale: float = 1.0
    engage_half_threshold: float = 0.55
    engage_half_threshold_scale: float = 1.0
    
    # -- 解耦奖励参数 --
    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0005 # 当x，y平面距离小于0.5mm时，才激活Z轴奖励
    
    # -- 为姿态对齐添加新的奖励和成功参数，只有插入方形孔等不具有完全旋转对称性任务会用到 --
    # 这个新的奖励项将专门用于鼓励Agent对齐yaw
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.05  # 弧度
    # 物体的旋转对称角度列表（度），例如正方形是 [0, 90, 180, 270] 度
    symmetry_angles_deg: list = []
    orientation_coef: list = [5, 4]
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool =  True
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": True, # 是否启用触觉传感器
        # "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": True, # 是否使用接触力作为触觉信息
        # "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "fingertip_quat_rel_fixed", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "fingertip_quat_rel_fixed", "ee_linvel", "ee_angvel"]
    
    # ================ 仿真对象生成配置 =================
    # -- 仿真对象生成配置 --
    fixed_asset: ArticulationCfg = field(default=None, init=False)
    # held_asset: ArticulationCfg = field(default=None, init=False)
    # held_asset: RigidObjectCfg = field(default=None, init=False)
    
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

        # # --- 为 held_asset (Peg) 构建配置 ---
        # self.held_asset = RigidObjectCfg(
        # prim_path="/World/envs/env_.*/HeldAsset/mesh",
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=self.held_asset_cfg.usd_path,
        #     activate_contact_sensors=True,
        #     # 质量与碰撞来自你的 held_asset_cfg
        #     mass_props=sim_utils.MassPropertiesCfg(mass=self.held_asset_cfg.mass),
        #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         max_depenetration_velocity=5.0,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=3666.0,
        #         enable_gyroscopic_forces=True,
        #         solver_position_iteration_count=192,
        #         solver_velocity_iteration_count=1,
        #         max_contact_impulse=1e32,
        #     ),
        # ),
        # init_state=RigidObjectCfg.InitialStateCfg(
        #     pos=(0.0, 0.4, 0.1),
        #     rot=(1.0, 0.0, 0.0, 0.0),
        #     lin_vel=(0.0, 0.0, 0.0),
        #     ang_vel=(0.0, 0.0, 0.0),
        # ),
        # )



    #     self.held_asset = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/HeldAsset",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=self.held_asset_cfg.usd_path,
    #         activate_contact_sensors=True,
    #         # 确保物理属性完整
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,  # 确保启用重力
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
    #         mass_props=sim_utils.MassPropertiesCfg(mass=self.held_asset_cfg.mass),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.005, 
    #             rest_offset=0.0,
    #             # 确保碰撞几何体被正确识别
    #             collision_enabled=True,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             enabled_self_collisions=False,
    #             solver_position_iteration_count=192,
    #             solver_velocity_iteration_count=1,
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.4, 0.1),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         joint_pos={},  # 无关节
    #         joint_vel={},
    #     ),
    #     actuators={},  # 无执行器
    # )
    
    
# ================= Peg in hole Circle Hole =================
@configclass
class CircleHole(FixedAssetCfg):
    """ 定义一个圆形hole，所有公差等级共用同一个hole """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_hole.usd" 
    diameter = 0.010 # hole的直径
    height = 0.025 # 孔的深度25mm
    base_height = 0.0

@configclass
class CirclePeg_I(HeldAssetCfg):
    """ 定义一个公差I的圆形peg """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_peg_I.usd"
    diameter = 0.008 # peg的直径
    height = 0.050 # 高度50mm
    mass = 0.019 # 质量19g
    
@configclass
class PegInHoleCircleHole_I(PegInHoleTask):
    """
    公差I的圆形插销插入任务配置
    """
    # -- 基本信息 --
    name = "peg_in_hole_circle_I" # 任务名称
    fixed_asset_cfg = CircleHole() # 指定固定工件是圆孔
    held_asset_cfg = CirclePeg_I() # 指定手持工件是公差I的圆形插销
    asset_size = held_asset_cfg.diameter
    duration_s = 10.0 # 任务时长10秒
    
    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动
    
    # Fixed Asset
    fixed_asset_init_orn_range_deg: float = 360.0
    
    # Held Asset
    
    # -- 奖励函数参数覆写 --
    z_reward_activation_threshold: float = 0.0005
    orientation_reward_scale: float = 0.0 # 圆孔任务不需要姿态对齐奖励
    yaw_success_threshold: float = 0.0  # 圆孔任务不需要姿态对齐成功判定
    symmetry_angles_deg: list = [] # 圆孔任务不需要姿态对齐对称性处理
    orientation_coef: list = [0, 0] # 圆孔任务不需要姿态对齐奖励系数
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool = False
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 注意：圆形孔任务中，不需要fingertip_quat_rel_fixed这个观测量
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
        
    
@configclass
class CirclePeg_II(HeldAssetCfg):
    """ 定义一个公差II的圆形peg """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_peg_II.usd"
    diameter = 0.0095 # peg的直径
    height = 0.050 # 高度50mm
    mass = 0.019 # 质量19g
    
@configclass
class PegInHoleCircleHole_II(PegInHoleTask):
    """
    公差II的圆形插销插入任务配置
    """
    # -- 基本信息 --
    name = "peg_in_hole_circle_II" # 任务名称
    fixed_asset_cfg = CircleHole() # 指定固定工件是圆孔
    held_asset_cfg = CirclePeg_II() # 指定手持工件是公差I的圆形插销
    asset_size = held_asset_cfg.diameter
    duration_s = 10.0 # 任务时长10秒
    
    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动
    
    # Fixed Asset
    fixed_asset_init_orn_range_deg: float = 360.0
    
    # Held Asset
    
    # -- 奖励函数参数覆写 --
    z_reward_activation_threshold: float = 0.0005
    orientation_reward_scale: float = 0.0 # 圆孔任务不需要姿态对齐奖励
    yaw_success_threshold: float = 0.0  # 圆孔任务不需要姿态对齐成功判定
    symmetry_angles_deg: list = [] # 圆孔任务不需要姿态对齐对称性处理
    orientation_coef: list = [0, 0] # 圆孔任务不需要姿态对齐奖励系数
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool = False
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 注意：圆形孔任务中，不需要fingertip_quat_rel_fixed这个观测量
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
        
@configclass
class CirclePeg_III(HeldAssetCfg):
    """ 定义一个公差III的圆形peg """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_peg_III.usd"
    diameter = 0.0099 # peg的直径
    height = 0.050 # 高度50mm
    mass = 0.019 # 质量19g
    
@configclass
class PegInHoleCircleHole_III(PegInHoleTask):
    """
    公差III的圆形插销插入任务配置
    """
    # -- 基本信息 --
    name = "peg_in_hole_circle_III" # 任务名称
    fixed_asset_cfg = CircleHole() # 指定固定工件是圆孔
    held_asset_cfg = CirclePeg_III() # 指定手持工件是公差I的圆形插销
    asset_size = held_asset_cfg.diameter
    duration_s = 10.0 # 任务时长10秒
    
    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动
    
    # Fixed Asset
    fixed_asset_init_orn_range_deg: float = 360.0
    
    # Held Asset
    
    # -- 奖励函数参数覆写 --
    z_reward_activation_threshold: float = 0.0005
    orientation_reward_scale: float = 0.0 # 圆孔任务不需要姿态对齐奖励
    yaw_success_threshold: float = 0.0  # 圆孔任务不需要姿态对齐成功判定
    symmetry_angles_deg: list = [] # 圆孔任务不需要姿态对齐对称性处理
    orientation_coef: list = [0, 0] # 圆孔任务不需要姿态对齐奖励系数
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool = False
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 注意：圆形孔任务中，不需要fingertip_quat_rel_fixed这个观测量
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    
@configclass
class CirclePeg_IV(HeldAssetCfg):
    """ 定义一个公差IV的圆形peg """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_peg_IV.usd"
    diameter = 0.00998 # peg的直径
    height = 0.050 # 高度50mm
    mass = 0.019 # 质量19g
    
@configclass
class PegInHoleCircleHole_IV(PegInHoleTask):
    """
    公差IV的圆形插销插入任务配置
    """
    # -- 基本信息 --
    name = "peg_in_hole_circle_IV" # 任务名称
    fixed_asset_cfg = CircleHole() # 指定固定工件是圆孔
    held_asset_cfg = CirclePeg_IV() # 指定手持工件是公差I的圆形插销
    asset_size = held_asset_cfg.diameter
    duration_s = 10.0 # 任务时长10秒
    
    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动
    
    # Fixed Asset
    fixed_asset_init_orn_range_deg: float = 360.0
    
    # Held Asset
    
    # -- 奖励函数参数覆写 --
    z_reward_activation_threshold: float = 0.0005
    orientation_reward_scale: float = 0.0 # 圆孔任务不需要姿态对齐奖励
    yaw_success_threshold: float = 0.0  # 圆孔任务不需要姿态对齐成功判定
    symmetry_angles_deg: list = [] # 圆孔任务不需要姿态对齐对称性处理
    orientation_coef: list = [0, 0] # 圆孔任务不需要姿态对齐奖励系数
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool = False
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 注意：圆形孔任务中，不需要fingertip_quat_rel_fixed这个观测量
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    
@configclass
class CirclePeg_test(HeldAssetCfg):
    """ 定义一个测试用的圆形peg，官方资产 """
    # usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd" # 使用官方资产，与下方本地资产等效
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_peg_test.usd"
    diameter = 0.007986 # peg的直径
    height = 0.050 # 高度50mm
    mass = 0.019 # 质量19g

@configclass
class CircleHole_test(FixedAssetCfg):
    """ 定义一个测试用的插孔，官方资产 """
    # usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd" # 使用官方资产，与下方本地资产等效
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/circle/circle_hole_test.usd"
    diameter = 0.0081 # 孔的直径略大于销的直径，留有余量
    height = 0.025 # 孔的深度25mm
    base_height = 0.0

@configclass
class PegInHoleCircleHole_test(PegInHoleTask):
    """
    插入官方圆形插销到官方圆孔的任务配置，测试用
    """
    # -- 基本信息 --
    name = "peg_in_hole_circle_test" # 任务名称
    fixed_asset_cfg = CircleHole_test() # 指定固定工件是圆孔
    held_asset_cfg = CirclePeg_test() # 指定手持工件是公差I的圆形插销
    asset_size = held_asset_cfg.diameter
    duration_s = 10.0 # 任务时长10秒
    
    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动
    
    # Fixed Asset
    fixed_asset_init_orn_range_deg: float = 360.0
    
    # Held Asset
    
    # -- 奖励函数参数覆写 --
    z_reward_activation_threshold: float = 0.0005
    orientation_reward_scale: float = 0.0 # 圆孔任务不需要姿态对齐奖励
    yaw_success_threshold: float = 0.0  # 圆孔任务不需要姿态对齐成功判定
    symmetry_angles_deg: list = [] # 圆孔任务不需要姿态对齐对称性处理
    orientation_coef: list = [0, 0] # 圆孔任务不需要姿态对齐奖励系数
    
    # ================= 奖励函数选择 =================
    # 这个任务是否需要精确的姿态对齐逻辑？除Circle Hole任务外，其他任务都需要
    requires_orientation_logic: bool = False
    
    # 这个任务是否使用解耦的 XY/Z 奖励？非解耦意味着采用原始的关键点奖励函数
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 注意：圆形孔任务中，不需要fingertip_quat_rel_fixed这个观测量
    if tactile["tactile_enabled"] and tactile["use_contact_forces_as_obs"]:
    # 定义带有触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "tactile_force_field"]
    else:
    # 定义不带触觉的观测顺序
        obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
        
    
# ================= Peg in hole Square Hole =================

@configclass
class SquareHole(FixedAssetCfg):
    """ 定义一个10mm方形插销的方孔（固定工件）"""
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/square/square_hole.usd"
    diameter: float = 0.010
    height: float = 0.025
    base_height: float = 0.0

@configclass
class SquarePeg_I(HeldAssetCfg):
    """ 定义一个公差I的方形插销（手持工件） """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/square/square_peg_I.usd"
    # 使用边长作为'diameter'，用于夹爪宽度的计算
    diameter: float = 0.008
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.005 # 方形插销稍微重一点

@configclass
class PegInHoleSquareHole_I(PegInHoleTask):
    """
    公差I的方形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_square_I" # 任务名称
    fixed_asset_cfg = SquareHole()
    held_asset_cfg = SquarePeg_I()
    asset_size = held_asset_cfg.diameter
    duration_s = 15.0  # 方形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动，增大任务难度
    fixed_asset_init_orn_range_deg: float = 90.0 # 方孔的偏航角随机化范围缩小到15度，避免过大旋转导致任务过难
    
    # -- 奖励函数参数覆写 --
    
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.05  # 弧度，约等于2.86度
    
    # 正方形有4个对称方向：0, 90, 180, 270 度
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]
    
    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0005 # 当x，y平面距离小于0.5mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动

@configclass
class SquarePeg_II(HeldAssetCfg):
    """ 定义一个公差II的方形插销（手持工件） """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/square/square_peg_II.usd"
    # 使用边长作为'diameter'，用于夹爪宽度的计算
    diameter: float = 0.0095
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.005 # 方形插销稍微重一点

@configclass
class PegInHoleSquareHole_II(PegInHoleTask):
    """
    公差II的方形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_square_II" # 任务名称
    fixed_asset_cfg = SquareHole()
    held_asset_cfg = SquarePeg_II()
    asset_size = held_asset_cfg.diameter
    duration_s = 15.0  # 方形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动，增大任务难度
    fixed_asset_init_orn_range_deg: float = 90.0 # 方孔的偏航角随机化范围缩小到15度，避免过大旋转导致任务过难
    
    # -- 奖励函数参数覆写 --
    
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.05  # 弧度，约等于2.86度
    
    # 正方形有4个对称方向：0, 90, 180, 270 度
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]
    
    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0005 # 当x，y平面距离小于0.5mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
    
    
@configclass
class SquarePeg_III(HeldAssetCfg):
    """ 定义一个公差III的方形插销（手持工件） """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/square/square_peg_III.usd"
    # 使用边长作为'diameter'，用于夹爪宽度的计算
    diameter: float = 0.0099
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.005 # 方形插销稍微重一点

@configclass
class PegInHoleSquareHole_III(PegInHoleTask):
    """
    公差II的方形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_square_III" # 任务名称
    fixed_asset_cfg = SquareHole()
    held_asset_cfg = SquarePeg_III()
    asset_size = held_asset_cfg.diameter
    duration_s = 15.0  # 方形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动，增大任务难度
    fixed_asset_init_orn_range_deg: float = 90.0 # 方孔的偏航角随机化范围缩小到15度，避免过大旋转导致任务过难
    
    # -- 奖励函数参数覆写 --
    
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.05  # 弧度，约等于2.86度
    
    # 正方形有4个对称方向：0, 90, 180, 270 度
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]
    
    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0005 # 当x，y平面距离小于0.5mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
@configclass
class SquarePeg_IV(HeldAssetCfg):
    """ 定义一个公差IV的方形插销（手持工件） """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/square/square_peg_IV.usd"
    # 使用边长作为'diameter'，用于夹爪宽度的计算
    diameter: float = 0.00998
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.005 # 方形插销稍微重一点

@configclass
class PegInHoleSquareHole_IV(PegInHoleTask):
    """
    公差II的方形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_square_IV" # 任务名称
    fixed_asset_cfg = SquareHole()
    held_asset_cfg = SquarePeg_IV()
    asset_size = held_asset_cfg.diameter
    duration_s = 15.0  # 方形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    hand_init_orn_noise: list = [0.0, 0.0, 0.785] # 允许在偏航角上有 +/- 45度的随机转动，增大任务难度
    fixed_asset_init_orn_range_deg: float = 90.0 # 方孔的偏航角随机化范围缩小到15度，避免过大旋转导致任务过难
    
    # -- 奖励函数参数覆写 --
    
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.05  # 弧度，约等于2.86度
    
    # 正方形有4个对称方向：0, 90, 180, 270 度
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]
    
    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0005 # 当x，y平面距离小于0.5mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 ================= 
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
# ================= Peg in hole L Hole =================

@configclass
class LHole(FixedAssetCfg):
    """ 定义一个匹配15mm L形插销的L孔 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/L_hole/L_hole.usd"
    diameter: float = 0.015
    # 孔深和基座高度与圆形资产保持一致
    height: float = 0.025
    base_height: float = 0.0

@configclass
class LPeg_I(HeldAssetCfg):
    """ 定义一个公差I的15mm L形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/L_hole/L_peg_I.usd"

    diameter: float = 0.013
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.003

@configclass
class PegInHoleLHole_I(PegInHoleTask):
    """
    公差I的L形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_L_I" # 任务名称
    fixed_asset_cfg = LHole()
    held_asset_cfg = LPeg_I()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # L形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0 # 非常重要：L孔初始朝向默认是180度！
    fixed_asset_init_orn_range_deg: float = 30.0
    
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续

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
    
    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动

@configclass
class LPeg_II(HeldAssetCfg):
    """ 定义一个公差II的15mm L形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/L_hole/L_peg_II.usd"

    diameter: float = 0.0145
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.003

@configclass
class PegInHoleLHole_II(PegInHoleTask):
    """
    公差II的L形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_L_II" # 任务名称
    fixed_asset_cfg = LHole()
    held_asset_cfg = LPeg_II()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # L形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0 # 非常重要：L孔初始朝向默认是180度！
    fixed_asset_init_orn_range_deg: float = 30.0
    
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续

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
    
    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动

@configclass
class LPeg_III(HeldAssetCfg):
    """ 定义一个公差III的15mm L形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/L_hole/L_peg_III_rigid.usd"

    diameter: float = 0.0149
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.003

@configclass
class PegInHoleLHole_III(PegInHoleTask):
    """
    公差III的L形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_L_III" # 任务名称
    fixed_asset_cfg = LHole()
    held_asset_cfg = LPeg_III()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # L形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0 # 非常重要：L孔初始朝向默认是180度！
    fixed_asset_init_orn_range_deg: float = 30.0
    
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续

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
    
    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": True, # 是否启用触觉传感器
        "use_contact_forces_as_obs": True, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
@configclass
class LPeg_IV(HeldAssetCfg):
    """ 定义一个公差IV的15mm L形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/L_hole/L_peg_IV.usd"

    diameter: float = 0.01498
    # 高度和质量与圆形资产保持一致
    height: float = 0.050
    mass: float = 0.019 + 0.003

@configclass
class PegInHoleLHole_IV(PegInHoleTask):
    """
    公差IV的L形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_L_IV" # 任务名称
    fixed_asset_cfg = LHole()
    held_asset_cfg = LPeg_IV()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # L形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0 # 非常重要：L孔初始朝向默认是180度！
    fixed_asset_init_orn_range_deg: float = 30.0
    
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续

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
    
    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    
    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
    
# ================= Peg in hole Triangle Hole =================
@configclass
class TriangleHole(FixedAssetCfg):
    """ 定义一个匹配10mm三角形插销的三角孔 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/triangle/triangle_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0
    
@configclass
class TrianglePeg_I(HeldAssetCfg):
    """ 定义一个公差I的三角形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/triangle/triangle_peg_I.usd"
    diameter: float = 0.008 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleTriangleHole_I(PegInHoleTask):
    """
    公差I的三角形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_triangle_I" # 任务名称
    fixed_asset_cfg = TriangleHole()
    held_asset_cfg = TrianglePeg_I()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 三角形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续

    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度

    # 三角形只有3个对称方向： 0, 120, 240 度
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动


@configclass
class TrianglePeg_II(HeldAssetCfg):
    """ 定义一个公差II的三角形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/triangle/triangle_peg_II.usd"
    diameter: float = 0.011 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleTriangleHole_II(PegInHoleTask):
    """
    公差II的三角形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_triangle_II" # 任务名称
    fixed_asset_cfg = TriangleHole()
    held_asset_cfg = TrianglePeg_II()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 三角形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 三角形只有3个对称方向： 0, 120, 240 度
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动


@configclass
class TrianglePeg_III(HeldAssetCfg):
    """ 定义一个公差III的三角形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/triangle/triangle_peg_III.usd"
    diameter: float = 0.0118 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleTriangleHole_III(PegInHoleTask):
    """
    公差III的三角形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_triangle_III" # 任务名称
    fixed_asset_cfg = TriangleHole()
    held_asset_cfg = TrianglePeg_III()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 三角形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 三角形只有3个对称方向： 0, 120, 240 度
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
@configclass
class TrianglePeg_IV(HeldAssetCfg):
    """ 定义一个公差IV的三角形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/triangle/triangle_peg_IV.usd"
    diameter: float = 0.01196 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleTriangleHole_IV(PegInHoleTask):
    """
    公差IV的三角形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_triangle_IV" # 任务名称
    fixed_asset_cfg = TriangleHole()
    held_asset_cfg = TrianglePeg_IV()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 三角形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 三角形只有3个对称方向： 0, 120, 240 度
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    # 采用默认的观测顺序，不需要改动
    
    
# ================= Peg in hole Hexagon Hole =================
@configclass
class HexagonHole(FixedAssetCfg):
    """ 定义一个匹配10mm六边形插销的六边孔 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/hexagon/hexagon_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0
    
@configclass
class HexagonPeg_I(HeldAssetCfg):
    """ 定义一个公差I的六边形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/hexagon/hexagon_peg_I.usd"
    diameter: float = 0.00969 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleHexagonHole_I(PegInHoleTask):
    """
    公差I的六边形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_hexagon_I" # 任务名称
    fixed_asset_cfg = HexagonHole()
    held_asset_cfg = HexagonPeg_I()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 六边形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 六边形有6个对称方向： 0, 60, 120, 180, 240, 300 度
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    
@configclass
class HexagonPeg_II(HeldAssetCfg):
    """ 定义一个公差II的六边形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/hexagon/hexagon_peg_II.usd"
    diameter: float = 0.011423 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleHexagonHole_II(PegInHoleTask):
    """
    公差II的六边形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_hexagon_II" # 任务名称
    fixed_asset_cfg = HexagonHole()
    held_asset_cfg = HexagonPeg_II()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 六边形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 六边形有6个对称方向： 0, 60, 120, 180, 240, 300 度
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    
@configclass
class HexagonPeg_III(HeldAssetCfg):
    """ 定义一个公差III的六边形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/hexagon/hexagon_peg_III.usd"
    diameter: float = 0.011884 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleHexagonHole_III(PegInHoleTask):
    """
    公差III的六边形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_hexagon_III" # 任务名称
    fixed_asset_cfg = HexagonHole()
    held_asset_cfg = HexagonPeg_III()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 六边形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 六边形有6个对称方向： 0, 60, 120, 180, 240, 300 度
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }
    
@configclass
class HexagonPeg_IV(HeldAssetCfg):
    """ 定义一个公差IV的六边形插销 """
    usd_path = f"{PEG_AND_HOLE_ASSETS_DIR}/hexagon/hexagon_peg_IV.usd"
    diameter: float = 0.011976 # 外接圆直径
    height: float = 0.050
    mass: float = 0.019
    
@configclass
class PegInHoleHexagonHole_IV(PegInHoleTask):
    """
    公差IV的六边形插销插入任务配置
    """
    # -- 1. 基本信息 --
    name = "peg_in_hole_hexagon_IV" # 任务名称
    fixed_asset_cfg = HexagonHole()
    held_asset_cfg = HexagonPeg_IV()
    asset_size = held_asset_cfg.diameter
    duration_s = 20.0  # 六边形任务更难，可以适当增加任务时长

    # -- 随机化参数覆写 --
    # Robot末端初始位置
    hand_init_orn_noise: list = [0.0, 0.0, 0.2] # 允许在偏航角上有 +/- 11.5度的随机转动
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    # -- 奖励函数参数覆写 --
    action_penalty_ee_scale: float = 0.03 # 对动作大小的惩罚系数。较大的动作会受到惩罚，鼓励更平滑的控制
    action_grad_penalty_scale: float = 0.1 # 对动作变化率（梯度）的惩罚系数。鼓励动作的连续
    orientation_reward_scale: float = 3.0  # 姿态对齐奖励的权重
    # 成功完成任务时，允许的最终偏航角误差（弧度）
    yaw_success_threshold: float = 0.01  # 弧度，约等于1.15度
    # 六边形有6个对称方向： 0, 60, 120, 180, 240, 300 度
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_activation_threshold: float = 0.0002 # 当x，y平面距离小于0.2mm时，才激活Z轴奖励

    # ================= 奖励函数选择 =================
    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True

    # ================ 触觉 =================
    tactile = {
        "tactile_enabled": False, # 是否启用触觉传感器
        "use_contact_forces_as_obs": False, # 是否使用接触力作为触觉信息
    }