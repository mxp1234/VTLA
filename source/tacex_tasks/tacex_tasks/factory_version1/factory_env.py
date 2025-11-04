# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import cv2
import os

import carb # carb是NVIDIA Omniverse平台的核心底层库,提供对仿真世界的基本操作接口,例如设置物理属性
import isaacsim.core.utils.torch as torch_utils # Isaac Sim专门为PyTorch提供的工具集,包含大量用于坐标变换、四元数数学运算的函数

import isaaclab.sim as sim_utils # Isaac Lab的仿真工具模块
from isaaclab.assets import Articulation # Articulation类用于表示一个关节链(如机器人)
from isaaclab.envs import DirectRLEnv # DirectRLEnv是强化学习环境的基类
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane # 用于生成地面的工具
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # Isaac Sim资源库的路径
from isaaclab.utils.math import axis_angle_from_quat # 从四元数计算轴-角表示

from tacex import GelSightSensor # GelSight触觉传感器

from . import factory_control as fc # 导入控制模块
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg # 导入配置
from .network.tactile_feature_extractor import create_tactile_encoder # 导入触觉特征提取器
from . import factory_utils

class FactoryEnv(DirectRLEnv):
    """
    Factory任务环境类,支持多种精密装配任务(插销、齿轮、螺母等)。
    这个类实现了基于Isaac Lab的强化学习环境框架,并集成了GelSight触觉传感器。
    继承自DirectRLEnv,利用Isaac Lab的强化学习环境框架,根据具体任务需求进行扩展。
    """
    cfg: FactoryEnvCfg # 环境的配置对象,类型是FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        """
        构造函数,初始化Factory任务环境。

        Args:
            cfg (FactoryEnvCfg): 任务环境的配置对象,包含了所有可调参数
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        # -- 动态计算观测/状态空间维度 --
        # env配置文件包含task实例，task cfg定义了观测与状态空间
        # 从env配置文件中读取obs_order和state_order列表,然后从OBS_DIM_CFG字典中查找每个条目的维度并求和
        
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.task.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.task.state_order])
        # 加上动作空间的维度=6,因为"上一时刻的动作"(prev_actions)也是观测和状态的一部分
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        # 将任务相关的具体配置保存为一个独立的属性,方便后续访问
        self.cfg_task = cfg.task

        # 调用父类(DirectRLEnv)的构造函数,完成Isaac Lab环境框架的初始化
        super().__init__(cfg, render_mode, **kwargs)

        # -- 初始化额外的物理属性和内部张量 --
        # 调用工具函数,为机器人连杆的惯性矩阵添加一个小的偏置(armature),以提高仿真稳定性
        self._set_body_inertias()
        # 初始化所有需要的张量(用于存储状态、控制目标等)
        self._init_tensors()
        # 设置默认的控制器增益、摩擦力等动力学参数
        self._set_default_dynamics_parameters()
        # 计算中间值(如雅可比矩阵、速度等),确保所有状态都是最新的
        self._compute_intermediate_values(dt=self.physics_dt)
        if self.cfg_task.tactile_enabled_in_obs == True and self.cfg_task.tactile_encode_method == "tactile_force_field":
            # -- 创建触觉图像保存目录 --
            # 用于可视化和调试GelSight传感器的输出
            self.tactile_img_save_dir = os.path.join(os.getcwd(), "tactile_images")
            os.makedirs(self.tactile_img_save_dir, exist_ok=True)

            # 触觉图像保存计数器和保存间隔
            self.tactile_save_counter = 0
            self.tactile_save_interval = 100  # 每100步保存一次触觉图像

            # -- 初始化触觉力场特征提取器 --
            # 加载预训练的神经网络模型,用于从触觉图像提取力场特征
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                "network/last.ckpt"
            )
            print(f"[FactoryEnv] Loading tactile feature extractor from: {checkpoint_path}")
            self.tactile_extractor = create_tactile_encoder(
                encoder_type='force_field', # 使用力场特征提取器
                checkpoint_path=checkpoint_path, # 预训练模型路径
                device=self.device, # 使用与环境相同的设备(CPU/GPU)
                freeze_model=True # 冻结模型参数,不进行训练
            )
            print(f"[FactoryEnv] Tactile feature extractor loaded. Output dim: {self.tactile_extractor.get_output_dim()}")

        # TODO 完善后续逻辑部分

    def _set_body_inertias(self):
        """
        设置机器人连杆的惯性属性。
        注意: 这是为了补偿IGE(Isaac Gym Envs)中asset_options.armature参数的影响。
        通过在惯性矩阵的对角线上添加小的偏置来提高数值稳定性。
        """
        # 获取机器人所有连杆的当前惯性矩阵
        inertias = self._robot.root_physx_view.get_inertias()
        # 创建一个偏置张量,形状与惯性矩阵相同
        offset = torch.zeros_like(inertias)
        # 在惯性矩阵的对角线元素上添加0.01的偏置 (索引0,4,8对应3x3矩阵的对角线)
        offset[:, :, [0, 4, 8]] += 0.01
        # 计算新的惯性矩阵: 原惯性 + 偏置
        new_inertias = inertias + offset
        # 将新的惯性矩阵应用到所有环境的机器人上
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """
        设置定义动态交互的参数。
        包括控制器增益、动作阈值和物理材质属性(如摩擦力)。
        """
        # -- 设置默认控制器增益 --
        # 从配置中读取默认的任务空间PD控制器的P-gain,并复制到所有环境
        # shape: [num_envs, 6], 分别对应x,y,z平移和roll,pitch,yaw旋转
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # -- 设置动作缩放阈值 --
        # 这些阈值用于将策略输出的归一化动作[-1,1]映射到实际的物理空间
        # 位置动作阈值: 控制每步最大位置变化量 (米)
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        # 旋转动作阈值: 控制每步最大旋转变化量 (弧度)
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # -- 设置物理材质属性 --
        # 根据配置为手持工件、固定工件和机器人设置摩擦系数
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """
        更新给定资产的材质属性,主要是摩擦系数。

        Args:
            asset: 要设置摩擦力的资产(Articulation对象)
            value: 摩擦系数值(同时应用于静摩擦和动摩擦)
        """
        # 获取资产当前的材质属性
        materials = asset.root_physx_view.get_material_properties()
        # 设置静摩擦系数 (materials的第0维)
        materials[..., 0] = value  # Static friction.
        # 设置动摩擦系数 (materials的第1维)
        materials[..., 1] = value  # Dynamic friction.
        # 获取所有环境的ID
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        # 将新的材质属性应用到所有环境
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """
        初始化所有需要的张量。
        这个函数只在初始化时调用一次,用于创建所有用于存储状态、控制目标等的张量。
        """
        # -- 初始化单位四元数 --
        # 创建一个"无旋转"的四元数[1,0,0,0],并复制到所有环境
        # 这个四元数经常被用作默认姿态或姿态计算的中性元素
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # -- 控制目标张量 --
        # 存储机器人关节位置的控制目标 (包括7个手臂关节和2个夹爪关节)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        # 存储夹爪中心点位置的控制目标
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # 存储夹爪中心点姿态的控制目标(四元数)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # -- 固定工件相关张量 --
        # 固定工件在动作框架中的位置(用于计算动作边界)
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        # 固定工件在观测框架中的位置(用于提供给策略的观测)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        # 固定工件位置观测的初始噪声(在每个回合开始时采样一次)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # -- 计算手持工件的基准点偏移 --
        # 根据不同任务,手持工件的基准点(用于成功判断)相对于其重心有不同的偏移
        held_base_x_offset = 0.0
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "peg_insert_test":
            # 插销任务: Z轴偏移为0(基准点在底部)
            held_base_z_offset = 0.0
        else:
            raise NotImplementedError(f"Task '{self.cfg_task.name}' not implemented")

        # -- 手持工件基准点位姿(在手持工件的局部坐标系中) --
        # 局部位置
        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        # 局部姿态(默认为单位四元数,表示与工件坐标系对齐)
        self.held_base_quat_local = self.identity_quat.clone().detach()

        # 手持工件基准点在世界坐标系中的位姿(会在每步更新)
        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # -- 获取机器人特定连杆的索引 --
        # 这些索引用于从大的状态张量中快速提取特定连杆的数据
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # -- 用于有限差分的张量 --
        # 通过比较当前帧和上一帧的位置/姿态来估算速度,比直接读取更稳定
        self.last_update_timestamp = 0.0  # 上一次更新张量的时间戳
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # -- 目标位姿张量(用于成功判断) --
        # 手持工件的目标位置(手持工件基准点应该到达的位置)
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # 手持工件的目标姿态
        self.target_held_base_quat = self.identity_quat.clone().detach()

        # -- 关键点张量 --
        # 关键点是沿着工件轴线均匀分布的点,用于计算更细粒度的对齐奖励
        # 获取关键点偏移(在工件局部坐标系中)
        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        # 缩放关键点偏移
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        # 手持工件上的关键点在世界坐标系中的位置
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        # 固定工件上对应关键点的目标位置
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # -- 计算固定工件的成功位置(在固定工件的局部坐标系中) --
        # 这是手持工件基准点相对于固定工件原点应该到达的位置
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "peg_insert_test":
            # 插销任务: 插入后顶端应该与孔的顶部齐平(Z=0)
            self.fixed_success_pos_local[:, 2] = 0.0
        else:
            raise NotImplementedError(f"Task '{self.cfg_task.name}' not implemented")

        # -- 回合统计张量 --
        # 记录每个环境在本回合是否已成功过
        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # 记录首次成功的时间步
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _get_keypoint_offsets(self, num_keypoints):
        """
        获取均匀分布在单位长度线段上的关键点偏移。
        这些关键点以0为中心,沿着Z轴分布,用于计算精细的对齐奖励。

        Args:
            num_keypoints: 关键点的数量

        Returns:
            关键点偏移张量,shape: [num_keypoints, 3]
        """
        # 初始化关键点偏移张量(全零)
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        # 在Z轴上生成均匀分布的点,范围从0到1,然后减0.5使其以0为中心
        # 例如,如果num_keypoints=5,则点位于[-0.5, -0.25, 0, 0.25, 0.5]
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """
        初始化仿真场景。
        这个函数会在环境初始化时被框架自动调用,负责创建地面、桌子、机器人、工件和传感器。
        """
        # -- 生成地面 --
        # 在世界坐标系的/World/ground路径下创建一个地面,位置在Z=-1.05米
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # -- 加载桌子模型 --
        # 从Isaac Nucleus资源库加载一个USD格式的桌子模型
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        # 将桌子生成到场景中,使用正则表达式路径以支持多环境克隆
        # translation: 桌子的位置; orientation: 桌子的姿态(四元数,表示绕Y轴旋转90度)
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        # -- 创建机器人和工件 --
        # 这里的self.cfg.robot, self.cfg_task.fixed_asset等是在配置文件中定义的ArticulationCfg对象
        # Isaac Lab框架会解析这些配置对象,并在仿真中创建出对应的物体
        self._robot = Articulation(self.cfg.robot) # Franka机器人
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset) # 固定工件(如插孔、螺栓等)
        # print(self._fixed_asset.init_state)
        # print("="*80)
        self._held_asset = Articulation(self.cfg_task.held_asset) # 手持工件(如插销、螺母等)



        # -- 克隆环境并添加到场景中 --
        # 这一步会根据scene.num_envs的设置(如128),将刚刚创建的单个环境(env_0)复制127份
        self.scene.clone_environments(copy_from_source=False)
        # 在CPU模式下需要手动设置碰撞过滤规则
        if self.device == "cpu":
            self.scene.filter_collisions()

        # -- 将创建好的物体注册到场景管理器中 --
        # 方便后续统一管理和数据读取
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # -- 创建并注册触觉传感器 --
        # 左右两个GelSight Mini传感器分别安装在Franka夹爪的两个指尖上
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # -- 添加光源 --
        # 创建一个穹顶光(Dome Light)用于场景照明
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """
        从原始张量计算中间值,包括添加噪声。
        这个函数在每个仿真步都会被调用,用于更新所有状态相关的张量。

        Args:
            dt: 时间步长
        """
        # TODO: A lot of these can probably only be set once?
        # -- 获取工件的位姿 --
        # 固定工件的位置(相对于环境原点)和姿态
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        # 手持工件的位置(相对于环境原点)和姿态
        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        # -- 获取机器人末端的状态 --
        # 指尖中心点的位置、姿态、线速度和角速度
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        # -- 获取雅可比矩阵和质量矩阵 --
        jacobians = self._robot.root_physx_view.get_jacobians()

        # 左右手指的雅可比矩阵(6x7,6维力/力矩对应7个关节)
        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        # 指尖中心点的雅可比矩阵取左右手指的平均
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        # 手臂的质量矩阵(7x7,只包含手臂关节,不包含夹爪)
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        # 关节位置和速度
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # -- 使用有限差分法计算速度 --
        # 有限差分法的结果比直接读取的速度更可靠,可以减少噪声
        # Finite-differencing results in more reliable velocity estimates.
        # 末端线速度 = (当前位置 - 上一时刻位置) / 时间步长
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        # 计算末端角速度(使用四元数差分)
        # 差值四元数 = 当前四元数 * 上一时刻四元数的共轭
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        # 确保四元数在同一个半球(避免走远路)
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        # 将差值四元数转换为轴-角表示
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        # 角速度 = 轴-角 / 时间步长
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # 使用有限差分计算关节速度
        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # -- 计算关键点位置 --
        # Keypoint tensors.
        # 手持工件基准点的世界坐标 = 手持工件坐标系 + 局部偏移
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        # 目标位置(成功位置)的世界坐标 = 固定工件坐标系 + 成功位置的局部偏移
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        # 计算手持工件和固定工件上关键点的世界坐标
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            # 手持工件上的第idx个关键点
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            # 固定工件上对应的第idx个关键点
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        # 计算关键点之间的平均距离(用于奖励计算)
        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        # 更新时间戳
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """
        获取Actor/Critic的输入(使用非对称Critic)。
        返回包含policy(Actor观测)和critic(Critic状态)的字典。
        """
        # 为固定工件位置添加噪声(模拟传感器误差)
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # 获取上一时刻的动作
        prev_actions = self.actions.clone()
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos, # 使用带噪声的相对位置
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd, # 使用有限差分计算的速度
            "ee_angvel": self.ee_angvel_fd,
            
            "prev_actions": prev_actions,
        }
        # -- 获取触觉传感器数据 --
        # Get tactile sensor data
        if self.cfg_task.tactile_enabled_in_obs == True:
            tactile_left = self.gsmini_left.data.output.get("tactile_rgb")  # Shape: (num_envs, H, W, 3)
            tactile_right = self.gsmini_right.data.output.get("tactile_rgb")  # Shape: (num_envs, H, W, 3)
            # Flatten tactile images for state (critic): (num_envs, H*W*3)
            # 展平触觉图像用于状态(Critic),并归一化到[0,1]
            # tactile_left_flat = tactile_left.reshape(self.num_envs, -1).float() / 255.0  # Normalize to [0, 1]
            # tactile_right_flat = tactile_right.reshape(self.num_envs, -1).float() / 255.0  # Normalize to [0, 1]

            # Extract force field features for observation (actor): (num_envs, 3)
            # 使用预训练的神经网络提取力场特征用于观测(Actor)
            # tactile_extractor expects (B, H, W, 3) in range [0, 1]
            tactile_force_field = self.tactile_extractor(tactile_left.float() / 255.0, tactile_right.float() / 255.0)
            obs_dict["tactile_force_field"] = tactile_force_field
        # -- 构建观测字典(Actor使用) --

        # -- 构建状态字典(Critic使用,包含特权信息) --
        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame, # 无噪声的相对位置
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel, # 直接读取的速度
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7], # 关节角度(特权信息)
            "held_pos": self.held_pos, # 手持工件真实位置(特权信息)
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat, # 手持工件真实姿态(特权信息)
            "fixed_pos": self.fixed_pos, # 固定工件真实位置(特权信息)
            "fixed_quat": self.fixed_quat, # 固定工件真实姿态(特权信息)
            # "tactile_force_field": tactile_force_field,
            "task_prop_gains": self.task_prop_gains, # 当前控制器增益(特权信息)
            "pos_threshold": self.pos_threshold, # 动作阈值(特权信息)
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }

        # -- 按照配置的顺序拼接观测和状态张量 --
        # 根据cfg.obs_order的顺序从obs_dict中提取张量并拼接
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg_task.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        # 根据cfg.state_order的顺序从state_dict中提取张量并拼接
        state_tensors = [state_dict[state_name] for state_name in self.cfg_task.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)

        # 返回包含policy和critic输入的字典
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """
        重置缓冲区。
        在环境重置时清空成功标志。

        Args:
            env_ids: 需要重置的环境ID
        """
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """
        在物理步之前应用策略动作并进行平滑处理。
        使用指数移动平均(EMA)对动作进行平滑,避免剧烈抖动。

        Args:
            action: 策略输出的原始动作,shape: [num_envs, 6]
        """
        # 获取需要重置的环境ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # 使用EMA平滑动作: new = α * current + (1-α) * previous
        # 这可以防止策略输出的动作变化过于剧烈,使机器人运动更平稳
        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def close_gripper_in_place(self):
        """
        保持夹爪在当前位置不动,同时闭合夹爪。
        主要用于重置阶段抓取物体时使用。
        """
        # 创建零动作(不移动末端)
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0  # 0.0 = 闭合夹爪

        # Interpret actions as target pos displacements and set pos target
        # 位置动作: 保持当前位置不变
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        # 旋转动作: 保持当前姿态不变
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        # 将轴-角表示转换为四元数
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        # 如果旋转角度很小(<1e-6),则保持单位四元数(无旋转)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # 强制末端保持竖直向下的姿态(roll=180度, pitch=0)
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # roll = π (竖直向下)
        target_euler_xyz[:, 1] = 0.0  # pitch = 0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        # 生成控制信号(计算关节力矩)
        self.generate_ctrl_signals()

    def _apply_action(self):
        """
        将策略动作应用为相对于当前位置的增量目标。
        这是RL环境的核心函数,将策略的归一化动作转换为实际的控制目标。
        """
        # Get current yaw for success checking.
        # 获取当前的yaw角(绕Z轴的旋转),用于成功判断
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        # 将yaw角归一化到合理范围(处理角度的周期性)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        # 注意: 我们在控制和观测中使用有限差分速度
        # 检查是否需要在decimation循环中重新计算速度
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # -- 处理位置动作 --
        # Interpret actions as target pos displacements and set pos target
        # 策略输出的动作 * 阈值 = 实际位移(米)
        # 例如: action=0.5, threshold=0.02 => 实际位移=0.01米(1cm)
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # -- 处理旋转动作 --
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        # 如果任务要求单向旋转(如螺母拧紧),将Z轴旋转限制在[-1, 0]范围
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        # 旋转动作 * 阈值 = 实际旋转(弧度)
        rot_actions = rot_actions * self.rot_threshold

        # 设置新的目标位置 = 当前位置 + 位置增量
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        # 为了加速学习,永远不允许策略将末端移动到距离固定工件中心超过5cm的位置
        # 这个硬约束可以防止策略在探索时偏离太远,提高训练效率
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        # 将轴-角表示的旋转动作转换为四元数
        angle = torch.norm(rot_actions, p=2, dim=-1)  # 旋转角度(模长)
        axis = rot_actions / angle.unsqueeze(-1)  # 旋转轴(单位向量)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        # 如果旋转角度很小,则保持单位四元数
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        # 目标姿态 = 旋转增量 * 当前姿态
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # 强制末端保持竖直向下的姿态(roll=180度, pitch=0)
        # Restrict actions to be upright.
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # roll = π
        target_euler_xyz[:, 1] = 0.0  # pitch = 0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        # 设置夹爪目标位置(0.0 = 闭合)
        self.ctrl_target_gripper_dof_pos = 0.0
        # 生成控制信号(计算关节力矩)
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """
        使用临界阻尼设置机器人控制器增益。
        临界阻尼(critical damping): D = 2*sqrt(K),可以实现快速无振荡的响应。

        Args:
            prop_gains: 比例增益(P-gain),shape: [num_envs, 6]
            rot_deriv_scale: 旋转部分D-gain的额外缩放因子
        """
        # 设置P-gain
        self.task_prop_gains = prop_gains
        # 根据临界阻尼公式计算D-gain: D = 2*sqrt(K)
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        # 对旋转部分的D-gain进行额外缩放(通常旋转需要更小的阻尼)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """
        生成控制信号:设置Franka的关节位置目标(手指)或关节力矩(手臂)。
        调用操作空间控制器计算手臂关节的力矩,并设置夹爪的位置目标。
        """
        # 调用控制模块的compute_dof_torque函数,计算关节力矩
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,  # 使用有限差分计算的速度
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        # 设置夹爪关节的位置目标(使用PhysX的PD控制器)
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        # 夹爪关节不使用力矩控制,设为0
        self.joint_torque[:, 7:9] = 0.0

        # 将位置目标和力矩目标发送给机器人
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """
        更新用于奖励和观测的中间值,并判断回合是否结束。

        Returns:
            truncated: 超时结束的环境掩码
            terminated: 提前终止的环境掩码(这里与truncated相同)
        """
        # 更新中间值(位姿、速度、关键点等)
        self._compute_intermediate_values(dt=self.physics_dt)
        # 判断是否超时(达到最大步数)
        # 所以环境步进（step）是并行的，所有 env 会在同一时刻达到 max_episode_length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 这个环境没有提前终止的情况,所以truncated和terminated相同
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """
        获取当前时间步的成功掩码。
        判断每个环境是否达成了任务目标。

        Args:
            success_threshold: 成功阈值(相对于工件高度或螺距)
            check_rot: 是否检查旋转角度(用于螺母拧紧任务)

        Returns:
            curr_successes: 成功掩码,shape: [num_envs],dtype: bool
        """
        # 初始化成功掩码(全为False)
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 计算手持工件与目标位置在XY平面上的距离
        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        # 计算手持工件在Z方向上的位移(相对于目标位置)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        # 判断是否对中: XY距离 < 2.5mm
        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))

        # Height threshold to target
        # 根据任务类型设置不同的高度阈值
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "peg_insert_test" or self.cfg_task.name == "gear_mesh":
            # 插销/齿轮任务: 高度阈值 = 工件高度 * 成功阈值
            height_threshold = fixed_cfg.height * success_threshold
        else:
            raise NotImplementedError("Task not implemented")

        # 判断是否足够接近或在目标位置下方: Z位移 < 高度阈值
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        # 成功 = 对中 AND 足够接近
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        # 如果需要检查旋转(螺母任务),还要满足旋转角度要求
        if check_rot:
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _get_rewards(self):
        """
        更新奖励并计算成功统计信息。
        这是RL环境的核心函数,计算每个时间步的奖励。

        Returns:
            rew_buf: 奖励张量,shape: [num_envs]
        """

        # 算奖励
        rew_dict, rew_scales, curr_successes = self._get_factory_rew_dict()
        rew_buf = sum(rew_dict[k] * rew_scales[k] for k in rew_dict)
        # Only log episode success rates at the end of an episode.
        # 只在回合结束时记录成功率
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        # 记录回合首次成功的时间步
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        # 计算平均成功时间(只针对成功的回合)
        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        # 保存当前动作,用于下一步计算动作梯度惩罚
        self.prev_actions = self.actions.clone()
        total_reward_components = {}
        for rew_name, rew_scale in rew_scales.items():
            self.extras[f"logs_rwd_{rew_name}"] = (rew_dict[rew_name] * rew_scale).mean()
        # Save tactile images periodically when there is actual contact
        # 周期性保存触觉图像(用于调试)
        # self.tactile_save_counter += 1
        # if self.tactile_save_counter % self.tactile_save_interval == 0:
        #     self._save_tactile_images_during_episode()

        return rew_buf
    
    def _get_factory_rew_dict(self):
        '''
        返回奖励组成部分，奖励权重与瞬时成功标志（per-step success flag）
        '''
        rew_dict, rew_scales = {}, {}
        # 1. 获取所有任务都需要的基准位姿
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task, self.num_envs, self.device,
        )
        
        if self.cfg_task.use_decoupled_reward:

            # a. 计算解耦的误差
            xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
            z_dist = torch.abs(target_held_base_pos[:, 2] - held_base_pos[:, 2])

            # b. 计算解耦的XY和Z奖励
            xy_coef_a, xy_coef_b = self.cfg_task.xy_dist_coef
            rew_xy_align = factory_utils.squashing_fn(xy_dist, xy_coef_a, xy_coef_b)
            
            z_coef_a, z_coef_b = self.cfg_task.z_dist_coef
            rew_z_insert = factory_utils.squashing_fn(z_dist, z_coef_a, z_coef_b)
            
            # c. 创建XY位置门控
            gate_sharpness = 100.0 
            z_reward_mask = torch.exp(-gate_sharpness * xy_dist)
            if self.cfg.debug == True:
                print("="*20)
                print("z_reward_mask==",z_reward_mask)
            rew_z_insert_gated = rew_z_insert * z_reward_mask

            # d. 计算姿态奖励，对于需要方向对齐的任务
            if self.cfg_task.requires_orientation_logic:
                symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
                _, _, min_yaw_error = factory_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = factory_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )
            
                # e. 创建姿态门控
                orientation_threshold = np.deg2rad(1.0)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
                if self.cfg.debug == True:
                    print("="*20)
                    print("min_yaw_error,orientation_mask===",min_yaw_error,orientation_mask)
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
            offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
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
                "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
                "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
                "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
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
                _, _, min_yaw_error = factory_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = factory_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )

                rew_dict["orientation"] = orientation_rew
                rew_scales["orientation"] = self.cfg_task.orientation_reward_scale
                
                orientation_threshold = np.deg2rad(10.0)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
                # 门控应用于 kp_fine
                rew_dict["kp_fine"] *= orientation_mask
                
        return rew_dict, rew_scales, curr_successes
    
    def _update_rew_buf(self, curr_successes):
        """
        计算当前时间步的奖励。
        使用多种奖励项的组合:关键点奖励、动作惩罚、成功奖励等。

        Args:
            curr_successes: 当前成功掩码,shape: [num_envs]

        Returns:
            rew_buf: 奖励张量,shape: [num_envs]
        """
        rew_dict = {}

        # Keypoint rewards.
        # -- 关键点奖励(使用压缩函数) --
        # 压缩函数: 将距离映射到[0,1]范围,距离越小奖励越高
        def squashing_fn(x, a, b):
            # 双曲函数形式: 1 / (e^(ax) + b + e^(-ax))
            # a控制陡峭程度, b控制基线高度
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

        # 基线奖励: 使用较平缓的压缩函数,鼓励大致接近目标
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(self.keypoint_dist, a0, b0)

        # 粗对齐奖励: 使用中等陡峭的压缩函数,鼓励进入目标附近
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(self.keypoint_dist, a1, b1)

        # 精对齐奖励: 使用非常陡峭的压缩函数,鼓励精确对齐
        a2, b2 = self.cfg_task.keypoint_coef_fine
        rew_dict["kp_fine"] = squashing_fn(self.keypoint_dist, a2, b2)

        # Action penalties.
        # -- 动作惩罚 --
        # 动作幅度惩罚: 鼓励小幅度动作,节省能量
        rew_dict["action_penalty"] = torch.norm(self.actions, p=2)
        # 动作变化率惩罚: 鼓励平滑动作,避免抖动
        rew_dict["action_grad_penalty"] = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)

        # -- 接触奖励 --
        # 判断是否已经接触(使用较宽松的阈值)
        rew_dict["curr_engaged"] = (
            self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False).clone().float()
        )

        # -- 成功奖励 --
        rew_dict["curr_successes"] = curr_successes.clone().float()

        # -- 综合奖励 --
        # 奖励 = 关键点奖励 - 动作惩罚 + 接触奖励 + 成功奖励
        rew_buf = (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"]
            + rew_dict["curr_successes"]
        )

        # 记录各项奖励的平均值(用于监控训练)
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        重置指定环境。
        这是环境重置的入口函数,会执行完整的重置流程。

        Args:
            env_ids: 需要重置的环境ID

        注意: 我们假设所有环境总是同时重置
        """
        # 调用父类的重置函数
        super()._reset_idx(env_ids)

        # 将资产移动到默认位置
        self._set_assets_to_default_pose(env_ids)
        # 将Franka机器人移动到默认姿态
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        # 执行一步仿真(不应用动作)
        self.step_sim_no_action()

        # 随机化初始状态(这是重置的核心步骤)
        self.randomize_initial_state(env_ids)

        # Don't save at reset - sensors have no contact data yet
        # 不在重置时保存触觉图像 - 传感器还没有接触数据
        # self._save_tactile_images_on_reset(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """
        在随机化之前将资产移动到默认位姿。
        清空速度并重置到配置文件中定义的默认位置。

        Args:
            env_ids: 需要重置的环境ID
        """
        # -- 重置手持工件 --
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        # 加上环境原点偏移(每个环境有不同的空间位置)
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0  # 清空速度
        # 写入位姿和速度到仿真
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        # -- 重置固定工件 --
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0  # 清空速度
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(self, env_ids):
        """
        使用阻尼最小二乘法(DLS)逆运动学设置机器人关节位置。
        这是一个迭代式IK求解器,通过多次小幅度调整关节角度来达到目标位姿。

        Args:
            env_ids: 需要执行IK的环境ID

        Returns:
            pos_error: 最终的位置误差
            axis_angle_error: 最终的姿态误差(轴-角表示)
        """
        ik_time = 0.0
        # 在0.25秒内迭代调整关节位置,直到末端达到目标位姿
        while ik_time < 0.25:
            # (1) 计算当前位姿与目标位姿之间的误差
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            # 将位置误差和姿态误差拼接成6D位姿误差向量
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # (2) 求解阻尼最小二乘(DLS)逆运动学问题
            # 使用雅可比矩阵将任务空间的位姿误差转换为关节空间的角度增量
            # DLS方法比伪逆更鲁棒,可以避免在奇异点附近出现不稳定
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",  # 阻尼最小二乘法
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            # (3) 更新关节位置: 当前角度 + 增量
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            # 清空关节速度(重置时不希望有残留速度)
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            # 设置关节位置目标
            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # (4) 更新仿真中的关节状态
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # (5) 执行一步仿真并更新所有张量
            self.step_sim_no_action()
            ik_time += self.physics_dt

        # 返回最终的位姿误差(用于检查IK是否成功)
        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """
        获取手持工件相对于夹爪指尖的默认相对位姿。
        这个函数计算工件应该在夹爪中的哪个位置,不同任务有不同的抓取方式。

        Returns:
            held_asset_relative_pos: 手持工件相对于指尖的位置偏移
            held_asset_relative_quat: 手持工件相对于指尖的姿态偏移
        """
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "peg_insert_test":
            # 插销任务: 抓取插销的顶部
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            # Z方向偏移 = 插销高度 - 指垫长度
            # 这样指尖刚好接触到插销的顶部
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        else:
            raise NotImplementedError(f"Task '{self.cfg_task.name}' not implemented")

        # 默认姿态为单位四元数(无旋转)
        held_asset_relative_quat = self.identity_quat

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """
        将Franka机器人返回到默认关节位置。
        这个函数在环境重置时被调用,用于初始化机器人的姿态。

        Args:
            joints: 目标关节角度列表(7个手臂关节的弧度值)
            env_ids: 需要重置的环境ID
        """
        # 计算夹爪宽度: 根据手持工件的直径设置,乘以1.25留出一些余量
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        # 从默认状态克隆关节位置
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # 设置夹爪宽度(MIMIC关节,2个夹爪指)
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]  # 设置7个手臂关节角度
        # 清空速度和力矩
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        # 设置关节位置目标(用于PhysX的PD控制器)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        # 写入关节状态到仿真
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        # 清空关节力矩(重置时不施加任何力)
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        # 执行一步仿真,让机器人稳定在新的姿态
        self.step_sim_no_action()

    def step_sim_no_action(self):
        """
        执行一步仿真但不应用策略动作。
        用于环境重置和IK求解过程中,需要让仿真前进但不执行RL策略动作的场景。
        """
        # 将场景中所有物体的状态写入仿真
        self.scene.write_data_to_sim()
        # 执行一步物理仿真(不渲染画面)
        self.sim.step(render=False)
        # 从仿真读取最新的物体状态
        self.scene.update(dt=self.physics_dt)
        # 重新计算所有中间值(位姿、速度、雅可比等)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """
        随机化初始状态并执行回合级别的域随机化(Domain Randomization)。
        这是环境重置的核心函数,通过随机化各种参数来提高策略的泛化能力。
        整个过程包括:
        1. 随机化固定工件的位姿
        2. 使用IK将机器人移动到固定工件上方的随机位置
        3. 将手持工件放置在夹爪中的随机位置
        4. 闭合夹爪抓取工件

        Args:
            env_ids: 需要重置的环境ID
        """
        # -- 0. 禁用重力(在重置过程中,防止物体掉落) --
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # -- 1. 随机化固定工件的位姿 --
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a) 随机化位置
        # 生成[-1, 1]范围内的随机数
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        # 从配置中读取位置噪声幅度
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        # 将随机数缩放到噪声范围: random * noise_magnitude
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        # 应用位置随机化: 默认位置 + 随机偏移 + 环境原点
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b) 随机化姿态(绕Z轴旋转,即yaw角)
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)  # 初始yaw角
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)  # yaw角随机范围
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        # 计算随机的欧拉角: 初始角度 + 随机偏移
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # 只改变yaw,保持roll=0和pitch=0
        # 将欧拉角转换为四元数
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c) 清空速度
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d) 将新的状态写入仿真
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e) 为观测添加固定工件位置噪声
        # 这个噪声会在整个回合中保持不变,用于模拟传感器的系统误差
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # -- 计算用于观测的固定工件参考框架位置 --
        # 对于某些任务(如插销),我们使用工件顶端的位置作为观测参考点
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height  # 工件高度
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height  # 基座高度
        # 将局部坐标转换为世界坐标
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # -- 2. 使用IK将夹爪移动到固定工件上方的随机位置 --
        # 这一步很关键:通过IK求解器确保机器人能够到达目标位置
        # 如果IK失败(例如目标位置超出工作空间),会重新采样位置直到成功
        bad_envs = env_ids.clone()  # 记录IK失败的环境ID
        ik_attempt = 0  # IK尝试次数

        # 初始化夹爪朝下的姿态张量
        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # 循环直到所有环境的IK都成功
        while True:
            n_bad = bad_envs.shape[0]  # 当前失败的环境数量

            # (a) 计算目标位置:固定工件顶端上方
            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]  # 添加Z方向偏移(高度)

            # 添加随机扰动到目标位置
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) 计算随机的朝下姿态
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            # 添加姿态随机扰动
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            # 将欧拉角转换为四元数
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) 使用迭代式IK方法求解
            # 设置控制目标(期望的末端位姿)
            self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
            self.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            # 调用IK求解器
            pos_error, aa_error = self.set_pos_inverse_kinematics(env_ids=bad_envs)
            # 检查IK是否成功(误差是否足够小)
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3  # 位置误差 > 1mm
            angle_error = torch.norm(aa_error, dim=1) > 1e-3  # 角度误差 > 1e-3 rad
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # 如果所有环境的IK都成功了,退出循环
            if bad_envs.shape[0] == 0:
                break

            # 对于IK失败的环境,重置机器人到默认姿态,然后重新尝试
            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1

        self.step_sim_no_action()

        # -- 3. 随机化手持工件在夹爪中的位置 --
        # 这是域随机化的关键步骤:让工件在夹爪中的位置每次都不同,
        # 从而提高策略对抓取误差的鲁棒性

        # (a) 翻转夹爪的Z轴姿态(因为工件需要朝下)
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros_like(self.fingertip_midpoint_pos),
        )

        # (b) 获取默认的"夹爪相对于工件"的变换
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        # 计算逆变换:"工件相对于夹爪"
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        # (c) 计算工件在世界坐标系中的位置
        # 世界工件位姿 = 翻转后的夹爪位姿 * 工件在夹爪中的位姿
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # (d) 添加"工件在夹爪中"的位置随机化
        # 这模拟了抓取时的不确定性
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]

        held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        # 应用随机化
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.identity_quat,
            t2=self.held_asset_pos_noise,
        )

        # (e) 将手持工件移动到计算出的位置
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0  # 清空速度
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # -- 4. 闭合夹爪抓取工件 --
        # 设置用于快速重置的高增益控制器
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        # 在0.25秒内逐步闭合夹爪
        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # 闭合夹爪(目标宽度=0)
            self.ctrl_target_gripper_dof_pos = 0.0
            self.close_gripper_in_place()  # 保持末端位置不动,只闭合夹爪
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        # -- 5. 初始化有限差分所需的"上一时刻"状态 --
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # -- 6. 初始化动作为零(无运动) --
        # 这对于EMA平滑和动作惩罚的正确计算是必需的
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # -- 7. 反向计算初始状态对应的动作 --
        # 这一步是为了让动作表示与初始状态一致,避免第一步就出现大的动作梯度惩罚

        # (a) 设置动作参考框架:固定工件顶端(加上观测噪声)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # (b) 计算位置动作:当前末端位置相对于固定工件的位置
        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        # 归一化到动作空间[-1, 1]
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # (c) 计算旋转动作:当前yaw角相对于固定工件的yaw角
        # 先去掉180度的偏移(因为夹爪朝下)
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        # 计算相对于固定工件的yaw角
        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        # 将yaw角归一化到合理范围
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        # 将yaw角映射到动作空间[-1, 1]
        # 公式: (yaw + 180°) / 270° * 2 - 1
        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

        # -- 8. 清零初始速度(避免有限差分计算出错误的速度) --
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # -- 9. 恢复正常的控制器增益 --
        self._set_gains(self.default_gains)

        # -- 10. 重新启用重力 --
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _save_tactile_images_during_episode(self):
        """
        在回合期间保存触觉传感器图像(用于调试和可视化)。
        这个函数会定期保存触觉图像,用于分析触觉传感器的输出和接触模式。
        """
        # 只保存环境0的图像,避免文件过多
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
        """
        在环境重置时保存触觉传感器图像到磁盘(使用OpenCV)。
        这个函数用于可视化重置后的触觉传感器状态,帮助调试。
        """
        # 只保存第一个环境的图像,避免文件过多
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
