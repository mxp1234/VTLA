import numpy as np
import torch

import carb # carb是NVIDIA Omniverse平台的核心底层库，提供对仿真世界的基本操作接口，例如设置物理属性
import isaacsim.core.utils.torch as torch_utils # Isaac Sim专门为PyTorch提供的工具集，包含大量用于坐标变换、四元数数学运算的函数

# 导入Isaac Lab的核心模块和我们之前定义好的配置及工具函数
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
# 用于触觉处理
import cv2
import os
from tacex import GelSightSensor
from .network.tactile_feature_extractor import create_tactile_encoder
# 导入我们为Peg-in-Hole任务定义的配置类和工具函数
from . import peg_in_hole_control, peg_in_hole_utils
from .peg_in_hole_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, PegInHoleEnvCfg
# 导入触觉数据记录器
from .tactile_datalogger import TactileDataLogger

class PegInHoleEnv(DirectRLEnv):
    """
    这个类实现了Peg-in-Hole任务的环境。
    它继承自DirectRLEnv，利用Isaac Lab的强化学习环境框架，
    并根据Peg-in-Hole任务的具体需求进行了定制和扩展。
    """
    cfg: PegInHoleEnvCfg # 这个环境的配置对象，类型是PegInHoleEnvCfg
    
    def _setup_scene(self):
            """
            搭建仿真场景。这个函数会在环境初始化时被框架自动调用。
            """
            # 生成一个地面
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

            # 从USD文件加载一张桌子
            cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
            cfg.func(
                "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
            )

            # -- 关键步骤：根据配置生成机器人和工件 --
            # 这里的 `self.cfg.robot`, `self.cfg_task.fixed_asset` 正是我们在 self.cfg 文件中定义的 ArticulationCfg 对象。
            # Isaac Lab框架会解析这些配置对象，并在仿真中创建出对应的物体。
            self._robot = Articulation(self.cfg.robot)
            self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
            self._held_asset = Articulation(self.cfg_task.held_asset)

            # -- 克隆环境并添加到场景中 --
            # 这一步会根据 scene.num_envs 的设置（例如128），将我们刚刚创建的单个环境（env_0）复制127份
            self.scene.clone_environments(copy_from_source=False) # scene是在PegInHoleEnvCfg类中定义的，本身是InteractiveSceneCfg类的实例
            if self.device == "cpu": # 在CPU模式下需要手动设置碰撞过滤规则
                self.scene.filter_collisions()

            # 将创建好的物体（Articulation对象）注册到场景管理器中，方便后续统一管理和数据读取
            self.scene.articulations["robot"] = self._robot
            self.scene.articulations["fixed_asset"] = self._fixed_asset
            self.scene.articulations["held_asset"] = self._held_asset

            # 只有在需要时才创建传感器
            if self.cfg_task.tactile["tactile_enabled"]:
                # sensors
                self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
                self.scene.sensors["gsmini_left"] = self.gsmini_left

                self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
                self.scene.sensors["gsmini_right"] = self.gsmini_right
            # 添加一个穹顶光，用于场景照明
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)
            
    def _set_default_dynamics_parameters(self):
            """设置定义动态交互的参数。"""
            # 设置在RL交互阶段使用的默认控制器P-gain。
            self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
                (self.num_envs, 1)
            ) # self.cfg是调用父类DirectRLEnv初始化函数在初始化时创建的

            # 设置动作的缩放阈值，用于将policy输出的归一化的动作映射到实际的物理空间中
            self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
                (self.num_envs, 1)
            )
            self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
                (self.num_envs, 1)
            )

            # -- 设置物理材质属性 --
            # 调用工具函数，根据配置文件为手持工件、固定工件和机器人设置摩擦系数
            peg_in_hole_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
            peg_in_hole_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
            peg_in_hole_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)
    
    def _init_tensors(self):
        """仅在初始化时调用一次，用于创建所有需要的张量。"""
        # Control targets.
        # 关节位置的控制目标张量
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor # 动作平滑（EMA）因子
        self.dead_zone_thresholds = None # 控制死区阈值（当前未使用）

        # Fixed asset.
        # 用于存储固定工件位置的张量，包括添加的观测噪声
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device) # 固定工件位置的观测值
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device) # 固定工件位置的观测噪声

        # Computer body indices.
        # -- 获取机器人上特定连杆的索引号 --
        # 这些索引号在仿真中是固定的，所以只需要获取一次。用于后续高效地从数据张量中提取特定连杆的状态
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        # -- 用于有限差分（Finite-Differencing）的张量 --
        # 通过比较当前帧和上一帧的位置/姿态来估算速度。这种方法有时比直接读取物理引擎的速度更稳定、噪声更小。
        self.last_update_timestamp = 0.0  # 上一次更新张量的时间戳
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device) # 上一帧指尖位置
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ) # 一个有效的四元数必须是“单位四元数”，即它的模长（sqrt(w^2 + x^2 + y^2 + z^2)）必须为1，代表着“无旋转”或“单位姿态”
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device) # 上一帧机械臂关节位置，7DoF

        # -- 用于记录回合统计数据的张量 --
        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device) # 记录每个环境在本回合是否已成功过
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device) # 记录首次成功的时间步

    def __init__(self, cfg: PegInHoleEnvCfg, render_mode: str | None = None, **kwargs):
        """
        构造函数，初始化环境。
        Args:
            cfg (PegInHoleEnvCfg): 任务环境的配置对象，包含了所有可调参数
        """
        self.cfg_task = cfg.task # 将任务相关的具体配置保存为一个独立的属性，方便后续访问，cfg.task=PegInHoleTask()，定义在peg_in_hole_tasks_cfg.py，具体派生出任务类
        # -- 动态计算观测/状态空间维度 --
        # 从配置文件中读取obs_order和state_order列表，然后从OBS_DIM_CFG字典中查找每个条目的维度并求和。
        self.obs_order = self.cfg_task.obs_order
        # 这样做的好处是，我们只需要在配置文件中增删列表项，这里的代码就能自动适应，无需修改。
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in self.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        # 加上动作空间的维度=6，因为"上一时刻的动作" (prev_actions) 也是观测和状态的一部分
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        
        super().__init__(cfg, render_mode, **kwargs) # 调用父类（DirectRLEnv）的构造函数，完成Isaac Lab环境框架的初始化，在其中会创建self.cfg = cfg
         # -- 初始化额外的物理属性和内部张量 --
        # 调用工具函数，为机器人连杆的惯性矩阵添加一个小的偏置（armature），以提高仿真稳定性
        peg_in_hole_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters() # 设置默认的控制器增益和物体摩擦力等
        
        # 如果是评估模式，则初始化用于统计的变量
        if self.cfg.evaluation_mode:
            self.eval_ep_count = 0
        
        # 检查是否需要加载触觉相关的模块
        if self.cfg_task.tactile["tactile_enabled"]:
            # Counter for tactile image saving
            self.tactile_save_counter = 0
            self.tactile_save_interval = 100  # Save every 100 steps
            print("[INFO] Tactile observation enabled. Initializing tactile system.")
            # Create directory for saving tactile images
            self.tactile_img_save_dir = os.path.join(os.getcwd(), "tactile_images")
            os.makedirs(self.tactile_img_save_dir, exist_ok=True)
            self.tactile_save_counter = 0

            # Initialize tactile force field extractor
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                "network/last.ckpt"
            )
            print(f"    > Loading tactile feature extractor from: {checkpoint_path}")
            self.tactile_extractor = create_tactile_encoder(
                encoder_type='force_field',
                checkpoint_path=checkpoint_path,
                device=self.device,
                freeze_model=True
            )
            print(f"    > Tactile feature extractor loaded. Output dim: {self.tactile_extractor.get_output_dim()}")
            
        # 初始化触觉数据记录器
        self.tactile_logger = None
        self._current_tactile_force = None # 一个临时变量，用于在函数间传递当前帧的触觉力
        # 检查所有条件是否满足
        if (self.cfg.evaluation_mode and
                self.cfg_task.tactile.get("tactile_enabled", False) and
                self.cfg_task.tactile.get("log_tac_force", False) and
                self.cfg.scene.num_envs == 1):
            
            log_directory = os.path.join(os.path.dirname(__file__), "tac_force")
            self.tactile_logger = TactileDataLogger(task_name=self.cfg_task.name, log_dir=log_directory)
    
    # ==================================================================
    #                      RL Step Logic (RL 步进逻辑)
    #               以下函数共同实现了环境与RL Agent的交互循环
    # ==================================================================
    def _compute_intermediate_values(self, dt):
        """
        从原始的仿真张量中获取值，并计算出衍生的中间值。
        这个函数是环境的“状态更新”核心，确保所有代码都基于最新的物理状态。
        参数 dt: 时间步长 (self.physics_dt)。
        """        
        # --- 1. 读取工件（Assets）相对环境的位姿 ---
        # 从仿真数据中读取固定工件的 "root" (根连杆) 的世界坐标位置和姿态，然后减去环境原点，得到相对于各自环境原点的局部坐标
        # _fixed_asset.data.root_pos_w 的shape是 [num_envs, 3]
        # 减去 self.scene.env_origins 是为了将世界坐标转换为相对于各个环境原点的局部坐标
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        # 以同样的方式读取手持工件的局部坐标位姿
        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        # --- 2. 读取机器人夹爪中心点（Fingertip）的状态 ---
        # `self._robot.data.body_pos_w` 是一个大的张量，shape为 [num_envs, num_bodies, 3]。
        # 我们使用在 __init__ 中获取的 `self.fingertip_body_idx` 索引，
        # 来精确地提取出“夹爪中心点”这个特定连杆的位置和姿态。
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins # 减去环境原点，得到相对于环境的局部位置
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        # 直接从物理引擎读取夹爪中心点的线速度和角速度。
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        # --- 3. 读取用于控制的物理学张量 ---
        # 获取机器人所有连杆的雅可比矩阵。这是一个非常大的张量。
        # 雅可比矩阵描述了关节速度和末端执行器速度之间的线性关系，是OSC控制器的核心。
        jacobians = self._robot.root_physx_view.get_jacobians()

        # 从大的雅可比矩阵中，分别提取左、右指尖的雅可比矩阵。
        # 索引 `..._idx - 1` 是雅可比矩阵的索引和body索引存在一个偏移，因为雅可比矩阵不包含根连杆
        # [0:6, 0:7] 表示我们关心的是6D末端速度 (x,y,z,wx,wy,wz) 和 7个手臂关节的关系。
        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        # 将左、右指尖的雅可比矩阵取平均，得到一个近似的“夹爪中心”雅可比矩阵
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        # 获取手臂的广义质量矩阵（也叫惯性矩阵）。它描述了关节力矩和关节加速度之间的关系。
        # 同样是OSC控制器计算所需的核心物理量
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        # 克隆当前的关节位置和速度，一共有9个关节（7个机械臂+2个夹爪）
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # --- 4. 通过有限差分（Finite-Differencing）计算速度 ---
        # 通过位置/姿态的变化来估算速度，比直接读取物理引擎的速度值更稳定、噪声更小。
        # 这对于需要稳定速度信号的观测和控制来说非常重要。
        # (a) 计算线性速度
        # 速度 = (当前位置 - 上一帧记录的位置) / 时间差
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt # 这是夹爪中心点的线速度
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone() # !! 关键：更新“上一帧位置”

        # (b) 计算角速度（四元数版本，更复杂）
        # 1. 计算从上一帧姿态到当前姿态的“差值四元数”。q_diff = q_current * conjugate(q_previous)
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        # 2. 确保取的是最短路径的旋转 (因为q和-q代表相同旋转，我们希望w分量为正)
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        # 3. 将这个“差值四元数”转换为轴-角(axis-angle)表示法。
        #    结果是一个3D向量，方向是旋转轴，模长是旋转角度（弧度）。
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        # 4. 角速度 = 旋转角度 / 时间差
        self.ee_angvel_fd = rot_diff_aa / dt # 这是夹爪中心点的角速度
        # !! 关键：更新“上一帧姿态”。
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # (c) 计算手臂关节的速度
        # 逻辑与线性速度相同，只是操作对象是手臂的7个关节角度。
        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos # 只取前7个关节（机械臂部分）
        self.joint_vel_fd = joint_diff / dt
        # !! 关键：更新“上一帧关节角度”
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # --- 5. 更新时间戳 ---
        # 记录当前已处理到的仿真时间戳，用于避免在同一个时间步内重复计算
        self.last_update_timestamp = self._robot._data._sim_timestamp
        
    # def _get_peginhole_obs_state_dict(self):
    #     """
    #     一个辅助函数，负责填充观测和状态字典。
    #     """
    #     # 1. 在每个step都重新采样一次高斯噪声，添加到固定工件的位置观测上
    #     noise = torch.randn((self.num_envs, 3), device=self.device)
    #     # 2. 从配置中获取噪声的标准差
    #     noise_std = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, device=self.device)
        
    #     # 3. 缩放噪声
    #     scaled_noise = noise @ torch.diag(noise_std)
        
    #     # 4. 将实时噪声应用到观测上
    #     noisy_fixed_pos = self.fixed_pos_obs_frame + scaled_noise
    #     # -------------------
    #     # 计算相对姿态
    #     current_quat_inv = torch_utils.quat_conjugate(self.fingertip_midpoint_quat)
    #     fingertip_quat_rel_fixed = torch_utils.quat_mul(current_quat_inv, self.fixed_quat)
        
        
    #     # 如果是需要方向对齐任务，则计算相对于“最近”对称目标的相对姿态
    #     if self.cfg_task.requires_orientation_logic:
    #         symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
    #         # 调用新的核心函数
    #         _closest_target_quat, relative_quat_to_closest, _min_yaw_error = \
    #             peg_in_hole_utils.get_closest_symmetry_transform(
    #                 self.fingertip_midpoint_quat, self.fixed_quat, symmetry_rad
    #             )
    #         # 【关键】用计算出的、指向最近目标的相对姿态，覆盖掉默认的相对姿态
    #         fingertip_quat_rel_fixed = relative_quat_to_closest

    #     prev_actions = self.actions.clone()
            
    #     # 填充观测字典（给Actor的信息）
    #     obs_dict = {
    #         # 关键信息：指尖相对于“有噪声的目标”的相对位置。这是最重要的观测
    #         "fingertip_pos": self.fingertip_midpoint_pos,
    #         "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
    #         "fingertip_quat": self.fingertip_midpoint_quat,
    #         "fingertip_quat_rel_fixed": fingertip_quat_rel_fixed,
    #         "ee_linvel": self.ee_linvel_fd, # 使用有限差分计算的速度
    #         "ee_angvel": self.ee_angvel_fd,
    #         "prev_actions": prev_actions,
    #     }

    #     # 填充状态字典（给Critic的更完整的信息）
    #     state_dict = {
    #         # Critic可以看到无噪声的相对位置
    #         "fingertip_pos": self.fingertip_midpoint_pos,
    #         "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
    #         "fingertip_quat": self.fingertip_midpoint_quat,
    #         "fingertip_quat_rel_fixed": fingertip_quat_rel_fixed,
    #         # Critic可以看到真实的、从物理引擎读取的速度
    #         "ee_linvel": self.fingertip_midpoint_linvel,
    #         "ee_angvel": self.fingertip_midpoint_angvel,
    #         # Critic可以看到所有“上帝视角”的真实信息
    #         "joint_pos": self.joint_pos[:, 0:7],
    #         "held_pos": self.held_pos,
    #         "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
    #         "held_quat": self.held_quat,
    #         "fixed_pos": self.fixed_pos,
    #         "fixed_quat": self.fixed_quat,
    #         # ... 其他信息 ...
    #         "task_prop_gains": self.task_prop_gains,
    #         "pos_threshold": self.pos_threshold,
    #         "rot_threshold": self.rot_threshold,
    #         "prev_actions": prev_actions,
    #     }
        
    #     tactile_force_field = torch.zeros((self.num_envs, 3), device=self.device)
        
    #     if self.cfg_task.tactile.get("tactile_enabled", False):
    #         # 1. Get tactile sensor data
    #         tactile_left = self.gsmini_left.data.output.get("tactile_rgb")
    #         tactile_right = self.gsmini_right.data.output.get("tactile_rgb")
            
    #         # 2. Extract force field features for observation
    #         #    tactile_extractor expects (B, H, W, 3) in range [0, 1]
    #         if tactile_left is not None and tactile_right is not None:
    #             tactile_force_field = self.tactile_extractor(tactile_left.float() / 255.0, tactile_right.float() / 255.0)
            
    #     self._current_tactile_force = tactile_force_field
                
    #     obs_dict["tactile_force_field"] = tactile_force_field
    #     state_dict["tactile_force_field"] = tactile_force_field
                
    #     return obs_dict, state_dict
    
    # (在 PegInHoleEnv 类中)

    def _get_peginhole_obs_state_dict(self):
        """
        填充观测和状态字典，并根据配置应用实时动态噪声。
        """
        # --- 1. 获取所有“干净”的原始值 ---
        clean_fingertip_pos = self.fingertip_midpoint_pos
        clean_fingertip_quat = self.fingertip_midpoint_quat
        clean_ee_linvel = self.ee_linvel_fd
        clean_ee_angvel = self.ee_angvel_fd
        
        clean_tactile_force_field = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.tactile.get("tactile_enabled", False):
            tactile_left = self.gsmini_left.data.output.get("tactile_rgb")
            tactile_right = self.gsmini_right.data.output.get("tactile_rgb")
            if tactile_left is not None and tactile_right is not None:
                clean_tactile_force_field = self.tactile_extractor(tactile_left.float() / 255.0, tactile_right.float() / 255.0)

        # --- 2. 根据 use_all_noise 标志决定如何生成“有噪声”的值 ---
        if self.cfg.obs_rand.use_all_noise:
            # --- 模式一：为所有观测添加实时动态噪声 ---
            
            # a. 固定工件位置噪声
            noise = torch.randn_like(self.fixed_pos_obs_frame)
            noise_std = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, device=self.device)
            noisy_fixed_pos = self.fixed_pos_obs_frame + (noise @ torch.diag(noise_std))

            # b. 机器人末端位置噪声
            noise = torch.randn_like(clean_fingertip_pos)
            noise_std = torch.tensor(self.cfg.obs_rand.fingertip_pos, device=self.device)
            noisy_fingertip_pos = clean_fingertip_pos + (noise @ torch.diag(noise_std))

            # c. 机器人末端姿态噪声
            noise = torch.randn((self.num_envs, 3), device=self.device) # 保证是 (N,3)
            noise_std = torch.tensor(self.cfg.obs_rand.fingertip_quat, device=self.device)
            aa_noise = noise @ torch.diag(noise_std)
            angle = torch.norm(aa_noise, p=2, dim=-1)
            axis = aa_noise / (angle.unsqueeze(-1) + 1e-6) # 加上eps防止除以0
            quat_noise = torch_utils.quat_from_angle_axis(angle, axis)
            noisy_fingertip_quat = torch_utils.quat_mul(quat_noise, clean_fingertip_quat)
            
            # d. 速度噪声
            noise = torch.randn_like(clean_ee_linvel)
            noise_std = torch.tensor(self.cfg.obs_rand.ee_linvel, device=self.device)
            noisy_ee_linvel = clean_ee_linvel + (noise @ torch.diag(noise_std))

            noise = torch.randn_like(clean_ee_angvel)
            noise_std = torch.tensor(self.cfg.obs_rand.ee_angvel, device=self.device)
            noisy_ee_angvel = clean_ee_angvel + (noise @ torch.diag(noise_std))

            # e. 触觉噪声
            noise = torch.randn_like(clean_tactile_force_field)
            noise_std = torch.tensor(self.cfg.obs_rand.tactile_force_field, device=self.device)
            noisy_tactile_force_field = clean_tactile_force_field + (noise @ torch.diag(noise_std))

        else:
            # --- 模式二：保持原始行为（只对Hole位置有初始偏移） ---
            noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            noisy_fingertip_pos = clean_fingertip_pos
            noisy_fingertip_quat = clean_fingertip_quat
            noisy_ee_linvel = clean_ee_linvel
            noisy_ee_angvel = clean_ee_angvel
            noisy_tactile_force_field = clean_tactile_force_field

        # --- 3. 使用“有噪声”或“干净”的值来计算衍生的观测值 ---
        
        # 相对姿态的计算需要根据情况使用有噪声或干净的quat
        # 为了简化，我们统一在外部计算
        
        # --- 4. 填充字典 ---
        prev_actions = self.actions.clone()
        
        # -- 状态字典 (state_dict) 总是使用“干净”的真实值 --
        
        # a. 先计算干净的相对姿态
        clean_current_quat_inv = torch_utils.quat_conjugate(clean_fingertip_quat)
        clean_fingertip_quat_rel_fixed = torch_utils.quat_mul(clean_current_quat_inv, self.fixed_quat)
        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
            _, clean_relative_quat_to_closest, _ = peg_in_hole_utils.get_closest_symmetry_transform(
                    clean_fingertip_quat, self.fixed_quat, symmetry_rad
                )
            clean_fingertip_quat_rel_fixed = clean_relative_quat_to_closest
        
        state_dict = {
            "fingertip_pos": clean_fingertip_pos,
            "fingertip_pos_rel_fixed": clean_fingertip_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": clean_fingertip_quat,
            "fingertip_quat_rel_fixed": clean_fingertip_quat_rel_fixed,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "tactile_force_field": clean_tactile_force_field,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "prev_actions": prev_actions,
        }
        
        # -- 观测字典 (obs_dict) 使用所有“有噪声”的值 --
        
        # a. 用有噪声的值计算相对姿态
        noisy_current_quat_inv = torch_utils.quat_conjugate(noisy_fingertip_quat)
        noisy_fingertip_quat_rel_fixed = torch_utils.quat_mul(noisy_current_quat_inv, self.fixed_quat)
        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
            _, noisy_relative_quat_to_closest, _ = peg_in_hole_utils.get_closest_symmetry_transform(
                    noisy_fingertip_quat, self.fixed_quat, symmetry_rad
                )
            noisy_fingertip_quat_rel_fixed = noisy_relative_quat_to_closest
            
        obs_dict = {
            "fingertip_pos": noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": noisy_fingertip_quat,
            "fingertip_quat_rel_fixed": noisy_fingertip_quat_rel_fixed,
            "ee_linvel": noisy_ee_linvel,
            "ee_angvel": noisy_ee_angvel,
            "tactile_force_field": noisy_tactile_force_field,
            "prev_actions": prev_actions,
        }
                
        return obs_dict, state_dict
    
    def _get_observations(self):
        """
        收集所有需要的信息，并打包成给Actor和Critic的观测/状态字典。
        """
        obs_dict, state_dict = self._get_peginhole_obs_state_dict() # 调用一个辅助函数，从各类状态张量中提取信息，填充成两个字典

        # 使用工具函数，按照配置文件中 `obs_order` 和 `state_order` 的顺序，
        # 将字典中的张量拼接成一个扁平的张量。
        obs_tensors = peg_in_hole_utils.collapse_obs_dict(obs_dict, self.obs_order + ["prev_actions"])
        state_tensors = peg_in_hole_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        # 返回一个字典，包含给策略（policy）的观测和给评论家（critic）的状态。
        return {"policy": obs_tensors, "critic": state_tensors}
    
    def _reset_buffers(self, env_ids):
        """重置与回合统计相关的缓冲区。"""
        self.ep_succeeded[env_ids] = 0 # 将成功标记置为0
        self.ep_success_times[env_ids] = 0 # 将首次成功时间步置为0
        
    def _pre_physics_step(self, action):
        """
        在物理仿真步进之前被框架调用。主要用于处理和应用RL Agent传入的动作。
        """
        # 检查是否有环境需要重置。`self.reset_buf`是一个标记缓冲，为1的环境ID表示需要重置。
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            # 如果有环境需要重置，则清空这些环境的成功记录。
            self._reset_buffers(env_ids)

        # -- 动作平滑 (Action Smoothing) --
        # 使用指数移动平均（EMA）来平滑Agent的原始动作。
        # self.actions 存储的是上一时刻平滑后的动作。
        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions
        
    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """
        根据目标末端位姿，计算出每个关节应施加的力矩。
        """
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        # 调用在 factory_control.py 中定义的操作空间控制器（OSC）。
        # 这是整个项目的控制核心。
        # 得到关节力矩，对7个机械臂关节使用力矩控制
        self.joint_torque, self.applied_wrench = peg_in_hole_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            # ... 传入目标状态 ...
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            # ... 传入控制器增益 ...
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # -- 应用力矩和目标位置 --
        # 对于夹爪（关节7和8），使用位置控制。
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        # 将计算出的夹爪力矩清零，因为它们由内置PD控制器处理。
        self.joint_torque[:, 7:9] = 0.0

        # 将夹爪的目标位置写入仿真器。
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        # 将手臂（关节0-6）的计算力矩写入仿真器。
        self._robot.set_joint_effort_target(self.joint_torque)
        
    def close_gripper_in_place(self):
        """一个特殊的控制函数，在闭合夹爪的同时，生成保持手臂当前世界坐标位姿不变的控制信号。"""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # 得到位置控制目标
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # 得到姿态控制目标
        rot_actions = actions[:, 3:6]

        # 从rot_actions计算出轴-角表示法
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        # 将轴-角表示法转换为四元数
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        # 处理零旋转的特殊情况，避免数值不稳定
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat) # 得到姿态控制目标

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159 # 将roll固定为180度，也即夹爪“倒置”
        target_euler_xyz[:, 1] = 0.0 # 将pitch固定为0度，也即夹爪“不侧倾”

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        ) # 得到新的姿态控制目标四元数

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )
        
    def _apply_action(self):
        """
        将平滑后的动作 self.actions 解释为控制信号，并应用到机器人上。
        这是连接RL Agent和机器人控制器的核心桥梁。
        """
        # 首先，更新所有从仿真中读取的中间状态值（如位置、速度、雅可比矩阵等）。
        # 这一步确保我们基于最新的状态来计算控制目标。
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # -- 1. 解释动作，计算目标末端位姿 --
        # (a) 位置动作：将[-1, 1]范围的动作乘以阈值，得到一个位置增量。
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        # (b) 旋转动作：同样进行缩放。
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        # (c) 计算初步的目标位置：当前位置 + 动作引起的位置增量
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # (d) 位置约束：为了加快学习速度，限制目标位置不能离初始位置太远（5cm以内）。
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        # 最终的位置目标 = 参考点 + 被裁剪后的偏移量
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # (e) 计算目标姿态：将旋转动作（轴-角形式）转换为四元数，然后乘以当前姿态四元数。
        #     q_target = q_delta * q_current
        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # (f) 姿态约束：将目标姿态转换为欧拉角，然后将roll和pitch角强制设为固定值（垂直朝下），只允许yaw角变化，也即夹爪只能水平旋转
        #     这极大地简化了学习问题，因为策略不需要学习如何保持夹爪的水平。
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # 将roll固定为180度，也即夹爪“倒置”
        target_euler_xyz[:, 1] = 0.0   # 将pitch固定为0度，也即夹爪“不侧倾”

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        ) # 得到新的姿态控制目标四元数

        # -- 2. 生成并应用控制信号 --
        # 调用核心的控制信号生成函数，传入最终计算出的目标位姿。
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )
        
    def _get_dones(self):
        """ 判断回合是否结束

        回合结束的唯一条件是达到最大步数（超时）。
        
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        # 在这个环境中，唯一的结束条件就是超时。
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 返回两个值，第一个是 "done" (回合结束，需要重置)，第二个是 "termination" (任务失败或成功)
        return time_out, time_out # 在这里两者是相同的
    
    def _get_curr_successes(self, success_threshold):
        """
        根据工件的几何关系，判断当前时间步任务是否成功。
        """
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 1. 获取手持工件基准点位姿和和手持工件目标的基准点位姿，在世界坐标系下，一般是工件的底部中心点
        held_base_pos, held_base_quat = peg_in_hole_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = peg_in_hole_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task, self.num_envs, self.device,
        ) # 对于解耦奖励和关键点奖励，返回值有所不同，后者可能涉及课程学习

        # 2. 计算XY平面上的距离
        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        # 3. 计算Z轴上的误差，当完全插入时，z_disp应该等于0
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        # 4. 判断条件
        #    (a) 是否在XY平面上对齐了 (is_centered)
        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        #    (b) Z轴方向上是否插入得足够深 (is_close_or_below)
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        
        height_threshold = fixed_cfg.height * success_threshold # 插入深度阈值
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        ) # z_disp < height_threshold表示插入足够深，is_close_or_below代表已经插入成功的环境
        # 5. 最终成功条件是 (a) 和 (b) 必须同时满足
        curr_successes = torch.logical_and(is_centered, is_close_or_below)
        
         # --- 6. 如果任务需要方向对齐逻辑，则进一步检查方向对齐条件 ---
        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]

            # 1. 直接调用我们的核心工具函数来获取最小的偏航角误差
            _, _, min_yaw_error = \
                peg_in_hole_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
            
            # 2. 使用返回的最小误差进行判断
            is_oriented = min_yaw_error < self.cfg_task.yaw_success_threshold
            
            # 3. 组合最终的成功条件
            curr_successes = torch.logical_and(curr_successes, is_oriented)

        return curr_successes
    
    def _log_factory_metrics(self, rew_dict, curr_successes):
        """记录和日志相关的统计数据，用于在训练过程中监控性能。"""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()
            
    def _get_factory_rew_dict(self):
        """
        计算奖励项的字典。
        """
        rew_dict, rew_scales = {}, {}
        # 1. 获取所有任务都需要的基准位姿
        held_base_pos, held_base_quat = peg_in_hole_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = peg_in_hole_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task, self.num_envs, self.device,
        )
     
        # 使用解耦奖励
        if self.cfg_task.use_decoupled_reward:

            # a. 计算解耦的误差
            xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
            z_dist = torch.abs(target_held_base_pos[:, 2] - held_base_pos[:, 2])

            # b. 计算解耦的XY和Z奖励
            xy_coef_a, xy_coef_b = self.cfg_task.xy_dist_coef
            rew_xy_align = peg_in_hole_utils.squashing_fn(xy_dist, xy_coef_a, xy_coef_b)
            
            z_coef_a, z_coef_b = self.cfg_task.z_dist_coef
            rew_z_insert = peg_in_hole_utils.squashing_fn(z_dist, z_coef_a, z_coef_b)
            
            # c. 创建XY位置门控
            gate_sharpness = 100.0 
            z_reward_mask = torch.exp(-gate_sharpness * xy_dist)
            rew_z_insert_gated = rew_z_insert * z_reward_mask

            # d. 计算姿态奖励，对于需要方向对齐的任务
            if self.cfg_task.requires_orientation_logic:
                symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
                _, _, min_yaw_error = peg_in_hole_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = peg_in_hole_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )
            
                # e. 创建姿态门控
                orientation_threshold = np.deg2rad(1.0)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
            else: # 对于不需要方向对齐的任务（圆形），姿态门控始终为1
                orientation_mask = torch.ones((self.num_envs,), device=self.device)
            
            # f. 计算其他共享的奖励项
            action_penalty_ee = torch.norm(self.actions, p=2)
            action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
            curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold)
            curr_engaged_half = self._get_curr_successes(success_threshold=self.cfg_task.engage_half_threshold)
            curr_successes = self._get_curr_successes(success_threshold=self.cfg_task.success_threshold)

            # g. 填充奖励字典
            rew_dict = {
                "xy_align": rew_xy_align,
                "z_insert": rew_z_insert_gated * orientation_mask, # 应用双重门控
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_engaged_half": curr_engaged_half.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "xy_align": self.cfg_task.xy_dist_reward_scale,
                "z_insert": self.cfg_task.z_dist_reward_scale,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": self.cfg_task.engage_threshold_scale,
                "curr_engaged_half": self.cfg_task.engage_half_threshold_scale,
                "curr_success": self.cfg_task.success_threshold_scale,
            }
            
            if self.cfg_task.requires_orientation_logic: # 为需要对齐方向的任务添加姿态奖励
                rew_dict["orientation"] = orientation_rew
                rew_scales["orientation"] = self.cfg_task.orientation_reward_scale
            
        else:
            # 采用官方的关键点奖励
            
            # a. 计算耦合的keypoint距离
            keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            offsets = peg_in_hole_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
            keypoint_offsets = offsets * self.cfg_task.keypoint_scale
            for idx, keypoint_offset in enumerate(keypoint_offsets):
                keypoints_held[:, idx] = torch_utils.tf_combine(
                    held_base_quat, held_base_pos,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    keypoint_offset.repeat(self.num_envs, 1),
                )[1]
                keypoints_fixed[:, idx] = torch_utils.tf_combine(
                    target_held_base_quat, target_held_base_pos,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    keypoint_offset.repeat(self.num_envs, 1),
                )[1]
            keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

            # b. 计算kp_*奖励
            a0, b0 = self.cfg_task.keypoint_coef_baseline
            a1, b1 = self.cfg_task.keypoint_coef_coarse
            a2, b2 = self.cfg_task.keypoint_coef_fine

            # c. 计算其他共享的奖励项
            action_penalty_ee = torch.norm(self.actions, p=2)
            action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
            curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold)
            curr_engaged_half = self._get_curr_successes(success_threshold=self.cfg_task.engage_half_threshold)
            curr_successes = self._get_curr_successes(success_threshold=self.cfg_task.success_threshold)

            # d. 填充奖励字典
            rew_dict = {
                "kp_baseline": peg_in_hole_utils.squashing_fn(keypoint_dist, a0, b0),
                "kp_coarse": peg_in_hole_utils.squashing_fn(keypoint_dist, a1, b1),
                "kp_fine": peg_in_hole_utils.squashing_fn(keypoint_dist, a2, b2),
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_engaged_half": curr_engaged_half.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "kp_baseline": self.cfg_task.kp_baseline_scale, 
                "kp_coarse": self.cfg_task.kp_coarse_scale, 
                "kp_fine": self.cfg_task.kp_fine_scale,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": self.cfg_task.engage_threshold_scale,
                "curr_engaged_half": self.cfg_task.engage_half_threshold_scale,
                "curr_success": self.cfg_task.success_threshold_scale,
            }

            # e. 为方形任务添加姿态奖励和门控
            if self.cfg_task.requires_orientation_logic:
                symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
                _, _, min_yaw_error = peg_in_hole_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = peg_in_hole_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )

                rew_dict["orientation"] = orientation_rew
                rew_scales["orientation"] = self.cfg_task.orientation_reward_scale
                
                orientation_threshold = np.deg2rad(10.0)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
                # 门控应用于 kp_fine
                rew_dict["kp_fine"] *= orientation_mask
        
        if self.tactile_logger is not None:
            self.tactile_logger.log_step(
                step=self.episode_length_buf[0].item(),
                is_engaged=rew_dict["curr_engaged"][0].item(), # 从字典中获取
                is_engaged_half=rew_dict["curr_engaged_half"][0].item(), # 从字典中获取
                is_success=rew_dict["curr_success"][0].item(), # 从字典中获取
                tactile_force=self._current_tactile_force[0]
            )        
        return rew_dict, rew_scales, curr_successes
    
    def _get_rewards(self):
        """
        计算当前时间步的奖励值。
        在评估模式下，此函数还负责统计和准备打印信息。
        """
        if self.cfg.evaluation_mode:
            self.extras.clear() # 在每次计算奖励的开始，清空 extras 字典，防止旧信息被重复使用
        # 1. 获取所有奖励项和用于训练日志的“最终成功”状态
        rew_dict, rew_scales, curr_successes_for_training_log = self._get_factory_rew_dict()
        
        # 2. 加权求和总奖励 (不变)
        rew_buf = torch.zeros((self.num_envs,), device=self.device)
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        # 3. 更新 prev_actions (不变)
        self.prev_actions = self.actions.clone()

        # 根据当前是训练模式还是评估模式，执行不同的日志/统计逻辑
        if self.cfg.evaluation_mode:
            # a. 定义我们用于评估的成功标准
            eval_success_mask = rew_dict["curr_success"].bool()
            
            # b. 复用 _log_factory_metrics 来更新基于我们标准的回合状态
            self._log_factory_metrics(rew_dict, eval_success_mask)
            
            # c. 只在回合结束时，准备打印字符串
            if torch.any(self.reset_buf):
                self.eval_ep_count += 1
                
                num_successes = torch.sum(self.ep_succeeded)
                success_rate = (num_successes / self.num_envs) * 100
                
                print_str = f"\n--- Episode {self.eval_ep_count} Finished ---\n"
                print_str += f"    Success Rate (insert_success): {success_rate:.2f}% ({num_successes.item()}/{self.num_envs})\n"

                if num_successes > 0:
                    successful_times_steps = self.ep_success_times[self.ep_succeeded.bool()]
                    avg_success_time_sec = torch.mean(successful_times_steps.float()) * self.step_dt
                    print_str += f"    Average Success Time: {avg_success_time_sec:.2f} seconds\n"
                
                # d. 将准备好的字符串放入 extras 字典，供 play.py 使用
                self.extras["eval_printout"] = print_str
        else:
            # 在训练模式下，正常调用日志函数
            self._log_factory_metrics(rew_dict, curr_successes_for_training_log)
            # Save tactile images periodically when there is actual contact
            if self.cfg_task.tactile["tactile_enabled"]:
                self.tactile_save_counter += 1
                if self.tactile_save_counter % self.tactile_save_interval == 0:
                    self._save_tactile_images_during_episode()
            
        return rew_buf 
    
    # ==================================================================
    #                       Reset Logic (重置逻辑)
    #                以下函数共同实现了环境的重置和领域随机化
    # ==================================================================
    def _reset_idx(self, env_ids):
        """
        重置指定ID的环境。在我们的例子中，所有环境总是同时重置，所以env_ids通常是所有环境的ID。
        这是重置流程的入口，该函数由仿真器在需要重置时调自动调用
        """
        super()._reset_idx(env_ids) # 调用父类的重置方法，处理一些通用的重置事务（如重置回合计数器）

        self._set_assets_to_default_pose(env_ids) # 1. 将机器人和工件移动到它们在USD文件中定义的默认姿态
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids) # 2. 将Franka机器人手臂的关节设置到配置文件中指定的 `reset_joints` 姿态
        self.step_sim_no_action() # 3. 在不施加任何动作的情况下，让物理仿真运行一小步，以确保所有物体都“安顿”下来

        self.randomize_initial_state(env_ids) # 4. 执行核心的随机化流程
        
        if self.tactile_logger is not None:
            self.tactile_logger.start_new_episode() # 重置触觉日志记录器
        
        # Don't save at reset - sensors have no contact data yet
        # if self.cfg_task.tactile["tactile_enabled"]:
        #     self._save_tactile_images_on_reset(env_ids)
        
    def _set_assets_to_default_pose(self, env_ids):
        """将工件移动到默认姿态。"""
        # 手持工件
        held_state = self._held_asset.data.default_root_state.clone()[env_ids] # 获取默认状态
        held_state[:, 0:3] += self.scene.env_origins[env_ids] # 加上每个环境的原点偏移
        held_state[:, 7:] = 0.0 # 速度清零
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids) # 写入仿真
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        # 固定工件，逻辑同上
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()
        
    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """
        使用DLS（Damped Least Squares，阻尼最小二乘法）IK算法，
        在一个短暂的循环中迭代求解，将机器人末端移动到目标位姿。
        """
        ik_time = 0.0 # 在一个短暂的时间循环内（0.25秒）持续进行IK求解，而不是只解一次。
        while ik_time < 0.25:
            # 1. 计算误差：
            #    调用 peg_in_hole_control.get_pose_error 计算当前末端位姿和目标位姿之间的差距。
            #    返回一个位置误差向量 (3D) 和一个姿态误差的轴-角向量 (3D)。
            pos_error, axis_angle_error = peg_in_hole_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            # 2. 拼接误差：将位置和姿态误差合并成一个6D的“位姿误差”向量
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # 3. 求解IK核心步骤：
            #    调用 factory_control.get_delta_dof_pos，这是IK算法的核心。
            #    输入是6D的末端位姿误差，输出是7D的关节角度增量。
            #    它内部使用了 'dls' (Damped Least Squares) 方法，通过雅可比矩阵求解 `delta_q = J_pinv * delta_x`。
            #    DLS是一种鲁棒的IK算法，即使在奇异点附近也不容易出错。
            delta_dof_pos = peg_in_hole_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            # 4. 更新关节角度：将计算出的关节角度增量加到当前的关节角度上。
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            # 5. 应用到仿真：
        	#    将新的目标关节角度写入仿真器，让机器人“瞬移”到新姿态。
            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # 6. 步进仿真并更新状态：
            #    让物理引擎走一步，并更新所有的内部状态张量（比如雅可比矩阵、末端位姿等），
            #    为下一次循环的误差计算做准备。
            self.step_sim_no_action()
            ik_time += self.physics_dt
            
        return pos_error, axis_angle_error
    
    
    def get_handheld_asset_relative_pose(self):
        """
        根据不同的任务，计算出手持工件相对于夹爪指尖中心的“标准”相对位姿。
        这是一个纯几何计算，确保工件被以一个合理的方式“拿住”。
        """
        
        held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device) # 定义一个局部坐标系下的位置向量
        held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height # 插销的高度
        # 减去指垫的长度，这意味着夹爪中心点 (fingertip) 在Z轴上比插销的顶端还要高一点，从而让指垫正好夹在插销的上半部分，这是一个非常稳定和真实的抓取方式
        held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length

        # --- 计算相对姿态 (四元数) ---
    	# 默认情况下，工件的姿态与夹爪的姿态保持一致（无相对旋转）
        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        return held_asset_relative_pos, held_asset_relative_quat
    
    def _set_franka_to_default_pose(self, joints, env_ids):
        """将Franka机器人设置到默认姿态。"""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25 # 计算一个合适的夹爪初始宽度，与固定夹持物品的直径有关
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # 设置夹爪关节宽度
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :] # 设置手臂7个关节的角度
        joint_vel = torch.zeros_like(joint_pos) # 速度清零
        joint_effort = torch.zeros_like(joint_pos) # 力矩清零
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids) # 写入仿真
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()
        
    def step_sim_no_action(self):
        """
        在不施加任何动作的情况下推进仿真一步。仅在重置时使用
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt) # 更新内部状态张量
        
    def randomize_initial_state(self, env_ids):
        """
        执行每一回合开始时的领域随机化。这是整个重置流程的核心和最复杂的部分。
        """
        # 临时禁用重力，这样在移动机器人和工件到初始位置时，它们不会因为重力而掉落
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # ---步骤1：将插孔（Hole）随机放置在桌面上一个小的区域内，并随机旋转---
        # (1.0) 从内存中克隆固定工件的默认状态。
        # - self._fixed_asset.data.default_root_state: [128, 13]，包含了128个环境的默认 (pos, quat, lin_vel, ang_vel)
        # - fixed_state: [128, 13]
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a) 位置随机化
    	# - rand_sample: [128, 3]，每个元素在 [0, 1] 之间
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # - fixed_pos_init_rand: [128, 3]，每个元素在 [-1, 1] 之间
        # - self.cfg_task.fixed_asset_init_pos_noise: [0.05, 0.05, 0.05] (Python list)
    	# - fixed_asset_init_pos_rand: [3] (Tensor)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        # - 核心数学操作: 矩阵乘法，将 [-1, 1] 的噪声缩放到 [-0.05, 0.05] 的范围内
        #   @ torch.diag(...) 相当于对每一行乘以对应的缩放因子
        # - fixed_pos_init_rand: [128, 3]
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        # - 将计算出的随机偏移量，加到默认位置上。
        #   fixed_state[:, 0:3]: [128, 3] (位置部分)
        #   self.scene.env_origins[env_ids]: [128, 3] (每个环境在世界中的原点偏移)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        
        # (1.b) 姿态随机化 (仅绕Z轴旋转)
    	# - fixed_orn_init_yaw: 0.0 (标量)
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        # - fixed_orn_yaw_range: 6.28 (2*pi) (标量)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        # - rand_sample: [128, 3]，新的随机数
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        # - fixed_orn_euler: [128, 3]，计算出每个环境的随机欧拉角 (roll, pitch, yaw)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        # - fixed_orn_euler[:, 0:2]: [128, 2]，将roll和pitch角强制设为0
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        # - fixed_orn_quat: [128, 4]，将欧拉角转换为四元数
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        # - fixed_state[:, 3:7]: [128, 4] (姿态部分)，用新的随机姿态覆盖默认姿态
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c) 将速度清零
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d) 将最终计算出的位姿写入仿真器
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.f) 步进仿真一帧，确保所有状态更新
        self.step_sim_no_action()

        # (1.g) 计算一个用于观测的“参考框架” (Observation Frame)
        #     目标是获取插孔顶面的中心点，作为后续所有相对位置计算的基准。
        # - fixed_tip_pos_local: [128, 3]，在插孔的局部坐标系下，顶面中心的位置 (通常是 [0, 0, height])
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        

        # - 使用坐标变换，将这个局部点转换到世界坐标系
        #   输入: self.fixed_quat ([128, 4]), self.fixed_pos ([128, 3])
        #   输出: fixed_tip_pos ([128, 3])
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        # - 保存这个世界坐标系下的点，作为本回合的观测参考点
    	# - self.fixed_pos_obs_frame: [128, 3]
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # ---步骤2：使用IK将机器人移动到工件上方一个随机位置---
        # (2.a) 准备IK循环
    	# - bad_envs: [128] (初始时，所有环境都需要IK)
        bad_envs = env_ids.clone()
        ik_attempt = 0

        # - hand_down_quat: [128, 4]，用于存储所有环境的目标姿态
        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            # - n_bad: 128 (第一次循环)
            n_bad = bad_envs.shape[0]

            # (2.b) 计算目标位置
        	# - above_fixed_pos: [128, 3]，从 fixed_tip_pos 开始
            above_fixed_pos = fixed_tip_pos.clone()
            # - 在Z轴上增加一个偏移量
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2] # 0.047

            # - rand_sample: [n_bad, 3]
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
             # - ... 与步骤1.a类似的逻辑，生成位置噪声 ...
        	# - above_fixed_pos_rand: [n_bad, 3]
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            # - 只对IK失败的环境(bad_envs)添加位置噪声
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (2.c) 计算目标姿态
        	# - hand_down_euler: [n_bad, 3]，从配置读取基础姿态 [3.1416, 0.0, 0.0]
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            # - ... 与步骤1.b类似的逻辑，生成姿态噪声 ...
        	# - above_fixed_orn_noise: [n_bad, 3]
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            # - 将欧拉角转为四元数，并只更新bad_envs对应的目标姿态
        	# - hand_down_quat: [128, 4]
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (2.d) 调用IK求解器
        	#   输入: above_fixed_pos ([128, 3]), hand_down_quat ([128, 4]), bad_envs ([n_bad])
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            # (2.e) 检查IK成功与否
        	# - pos_error: [n_bad]，每个元素是bool值
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            # - angle_error: [n_bad]，每个元素是bool值
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            # - any_error: [n_bad]，只要有一个误差超标，就为True
            any_error = torch.logical_or(pos_error, angle_error)
            # - 更新 bad_envs 列表，只保留那些 any_error 为 True 的环境
        	#   例如，如果第一次循环有10个环境IK失败，bad_envs 的 shape 会从 [128] 变为 [10]
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break # 所有环境都成功了，跳出循环

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            ) # 如果有失败的，将这些失败环境的机器人重置到默认姿态，再次尝试

            ik_attempt += 1

        # (2.f) 步进仿真，让机器人状态更新
        self.step_sim_no_action()

        # ---步骤3：放置手持工件（Held Asset）到夹爪中，计算出插销（Peg）应该在世界中的哪个位置，才能正好被已经就位的夹爪“完美”抓住---
        # (3.a) 准备坐标变换
    	# - flip_z_quat: [128, 4]，一个表示绕Z轴旋转180度的四元数 [0,0,1,0]
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        # - self.fingertip_midpoint_quat: [128, 4]，当前夹爪中心的世界姿态
        # - self.fingertip_midpoint_pos: [128, 3]，当前夹爪中心的世界位置
        # - 这一步的物理含义不明确，可能是为了校正USD中定义的坐标系
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # (3.b) 获取工件相对于夹爪的“标准抓取”相对位姿
        # - held_asset_relative_pos: [128, 3]
        # - held_asset_relative_quat: [128, 4]
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        # (3.c) 计算从夹爪中心到工件原点的变换
        #   tf_inverse 的作用是：如果 B 相对于 A 的位姿是 T_AB，那么 A 相对于 B 的位姿就是 inv(T_AB)
        # - asset_in_hand_quat: [128, 4]
        # - asset_in_hand_pos: [128, 3]
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        # (3.d) 核心变换：计算手持工件在世界坐标系下的位姿
        #   世界->工件 = (世界->夹爪) * (夹爪->工件)
        # - translated_held_asset_quat: [128, 4]
        # - translated_held_asset_pos: [128, 3]
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # (3.e) 在计算出的“完美”位姿上，添加抓取噪声
        # - ... 类似的噪声生成逻辑 ...
        # - held_asset_pos_noise: [128, 3]
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        # - 再次使用 tf_combine 将噪声（纯位移）叠加到工件位姿上
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        # (3.f) 将最终位姿写入仿真
    	# - held_state: [128, 13]
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0 # 速度清零
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # ---步骤4：模拟夹爪闭合的动作，并重置所有内部状态，为RL循环的 step(action) 做好准备
        # (4.a) 临时切换到高增益的PD控制器，用于快速抓取
    	# - reset_task_prop_gains: [128, 6]
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = peg_in_hole_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        # (4.b) 在一个0.25秒的循环中闭合夹爪
        grasp_time = 0.0
        while grasp_time < 0.25:
            # - self.ctrl_target_joint_pos: [128, 9] (7手臂+2夹爪)
        	#   将最后两个关节（夹爪）的目标位置设为0，表示完全闭合
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            #   调用一个特殊的控制函数，在闭合夹爪的同时保持手臂不动
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        # (5.a) 重置所有用于有限差分计算的 "上一帧" 状态
    	#   确保下一帧的速度计算是从当前状态开始的
        # - self.joint_pos: [128, 9] -> self.prev_joint_pos: [128, 7]
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        # - self.fingertip_midpoint_pos: [128, 3] -> self.prev_fingertip_pos: [128, 3]
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        # - self.fingertip_midpoint_quat: [128, 4] -> self.prev_fingertip_quat: [128, 4]
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # (5.b) 重置动作相关的状态
    	# - self.actions: [128, 6]
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # (5.c) 重置计算出的速度，避免第一帧出现很大的速度值
        # - self.ee_angvel_fd: [128, 3]
        self.ee_angvel_fd[:, :] = 0.0
        # - self.ee_linvel_fd: [128, 3]
        self.ee_linvel_fd[:, :] = 0.0

        # (5.d) 恢复为正常RL交互时使用的默认控制器增益
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = peg_in_hole_utils.get_deriv_gains(self.default_gains)

        # (收尾) 恢复重力
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        
    def _save_tactile_images_during_episode(self):
        """Save tactile sensor images during episode when there is actual contact data."""
        # Only save for environment 0 to avoid too many files
        env_id = 0

        try:
            # Safe access to tactile sensor data
            if not hasattr(self.gsmini_left.data, 'output') or not self.gsmini_left.data.output:
                return

            tactile_left_data = self.gsmini_left.data.output.get("tactile_rgb")
            tactile_right_data = self.gsmini_right.data.output.get("tactile_rgb")

            if tactile_left_data is None or tactile_right_data is None:
                return

            # Get tactile sensor data for env 0
            tactile_left = tactile_left_data[env_id]  # Shape: (H, W, 3)
            tactile_right = tactile_right_data[env_id]  # Shape: (H, W, 3)

            # Convert from torch tensor to numpy array
            # Tactile data is in range [0, 1], need to multiply by 255 before converting to uint8
            tactile_left_np = (tactile_left.cpu().numpy() * 255).astype(np.uint8)
            tactile_right_np = (tactile_right.cpu().numpy() * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            tactile_left_bgr = cv2.cvtColor(tactile_left_np, cv2.COLOR_RGB2BGR)
            tactile_right_bgr = cv2.cvtColor(tactile_right_np, cv2.COLOR_RGB2BGR)

            # Create a combined image (side by side)
            combined_img = np.hstack([tactile_left_bgr, tactile_right_bgr])

            # Resize for better visibility (upscale from 32x32 to 256x256)
            scale_factor = 8
            h, w = combined_img.shape[:2]
            combined_img_large = cv2.resize(combined_img, (w * scale_factor, h * scale_factor),
                                           interpolation=cv2.INTER_NEAREST)

            # Add text labels
            cv2.putText(combined_img_large, "Left Sensor", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_img_large, "Right Sensor", (w * scale_factor // 2 + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add episode and step info
            episode_step = int(self.episode_length_buf[env_id].item())
            cv2.putText(combined_img_large, f"Episode Step: {episode_step}", (10, h * scale_factor - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save image with global counter
            filename = f"tactile_contact_env{env_id}_counter{self.tactile_save_counter:06d}_step{episode_step}.png"
            filepath = os.path.join(self.tactile_img_save_dir, filename)
            cv2.imwrite(filepath, combined_img_large)

            # Print statistics
            mean_left = tactile_left_np.mean()
            mean_right = tactile_right_np.mean()
            print(f"[Tactile] Saved: {filename} | Left mean: {mean_left:.1f}, Right mean: {mean_right:.1f}")

        except Exception as e:
            print(f"[Tactile Visualization] Error saving tactile images: {e}")

    def _save_tactile_images_on_reset(self, env_ids):
        """Save tactile sensor images to disk for visualization using OpenCV."""
        # Only save images for the first environment to avoid too many files
        if len(env_ids) == 0:
            return

        # Get the first environment id to save
        env_id = env_ids[0].item() if torch.is_tensor(env_ids[0]) else env_ids[0]

        try:
            # Get tactile sensor data
            tactile_left = self.gsmini_left.data.output.get("tactile_rgb")[env_id]  # Shape: (H, W, 3)
            tactile_right = self.gsmini_right.data.output.get("tactile_rgb")[env_id]  # Shape: (H, W, 3)

            # Convert from torch tensor to numpy array
            # Tactile data is in range [0, 1], need to multiply by 255 before converting to uint8
            tactile_left_np = (tactile_left.cpu().numpy() * 255).astype(np.uint8)
            tactile_right_np = (tactile_right.cpu().numpy() * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            tactile_left_bgr = cv2.cvtColor(tactile_left_np, cv2.COLOR_RGB2BGR)
            tactile_right_bgr = cv2.cvtColor(tactile_right_np, cv2.COLOR_RGB2BGR)

            # Create a combined image (side by side)
            combined_img = np.hstack([tactile_left_bgr, tactile_right_bgr])

            # Resize for better visibility (upscale from 32x32 to 256x256)
            scale_factor = 8
            h, w = combined_img.shape[:2]
            combined_img_large = cv2.resize(combined_img, (w * scale_factor, h * scale_factor),
                                           interpolation=cv2.INTER_NEAREST)

            # Add text labels
            cv2.putText(combined_img_large, "Left Sensor", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_img_large, "Right Sensor", (w * scale_factor // 2 + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Save image with timestamp
            timestamp = int(self.episode_length_buf[env_id].item())
            filename = f"tactile_reset_env{env_id}_step{timestamp}.png"
            filepath = os.path.join(self.tactile_img_save_dir, filename)
            cv2.imwrite(filepath, combined_img_large)

            print(f"[Tactile Visualization] Saved tactile images to: {filepath}")

        except Exception as e:
            print(f"[Tactile Visualization] Error saving tactile images: {e}")
    
    def close(self):
        """关闭环境并释放资源。"""
        if self.tactile_logger is not None:
            self.tactile_logger.close()
        # -------------------
        # 调用父类的 close 方法
        super().close()