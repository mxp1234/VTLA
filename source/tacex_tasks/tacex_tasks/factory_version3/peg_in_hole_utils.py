import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

def get_keypoint_offsets(num_keypoints, device):
    """
    计算用于奖励函数的关键点（Keypoints）在局部坐标系下的偏移量。
    这些关键点被用来衡量两个物体之间的对齐程度。
    官方原始的关键点奖励会使用该函数，修改后的对齐插入解耦奖励则不会使用。

    Args:
        num_keypoints (int): 想要生成的关键点数量。
        device: PyTorch设备。

    Returns:
        torch.Tensor: 一个shape为 [num_keypoints, 3] 的张量，包含了所有关键点的局部坐标。
    """
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device) # 初始化一个全零张量
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0): 
    """
    根据给定的比例增益 (P-gain)，计算出对应的微分增益 (D-gain)。
    这个计算基于“临界阻尼” (Critical Damping) 的控制理论，
    这是一种能让系统最快达到稳定状态而又不会产生振荡的理想状态。

    Args:
        prop_gains (torch.Tensor): 任务空间的P-gain, shape [num_envs, 6]。
        rot_deriv_scale (float, optional): 一个额外的缩放因子，用于独立调整旋转部分的阻尼。

    Returns:
        torch.Tensor: 计算出的D-gain, shape [num_envs, 6]。
    """
    # 临界阻尼的PD控制器中，P增益(Kp)和D增益(Kd)的关系是: Kd = 2 * sqrt(Kp)。
    # （这来自于二阶系统的特征方程，当阻尼比ζ=1时，得到这个关系）。
    deriv_gains = 2 * torch.sqrt(prop_gains)
    # 对旋转部分的D-gain（索引3到5）进行额外的缩放。
    # 通常，旋转运动比平移运动需要更小的阻尼才能保持稳定。
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains

def wrap_yaw(angle):
    """
    一个数学工具，用于将偏航角（yaw）规范化到一个特定的范围内。
    这在处理周期性角度时很有用，避免因为 359度 和 1度 被认为是两个差异很大的值。
    """
    # `np.deg2rad(235)` 是一个特定的阈值。
    # 如果角度大于这个阈值，就减去 2*pi (360度)，把它“卷”回到负半轴。
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)

def set_friction(asset, value, num_envs):
    """
    一个封装函数，用于设置指定资产（Asset）所有碰撞体的物理材质的摩擦系数。
    """
    # 获取资产当前的物理材质属性。这是一个批处理操作
    materials = asset.root_physx_view.get_material_properties()
    # materials张量的第0列是静摩擦系数，第1列是动摩擦系数
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(num_envs, device="cpu")  # 需要提供一个环境ID列表
    asset.root_physx_view.set_material_properties(materials, env_ids) # 将修改后的材质属性写回仿真器
    
def set_body_inertias(robot, num_envs):
    """
    修改机器人连杆的惯性张量。
    注释中提到，这是为了模拟 `asset_options.armature` (电枢) 参数的效果。
    增加惯性张量对角线元素可以提高数值稳定性。
    """
    """Note: this is to account for the asset_options.armature parameter in IGE."""
    inertias = robot.root_physx_view.get_inertias() # 获取机器人所有连杆的惯性张量
    offset = torch.zeros_like(inertias)
    # inertias 的shape是 [num_envs, num_bodies, 10]，其中后10个值代表了质量、CoM、和惯性张量的元素。
    # offset[:, :, [0, 4, 8]] 对应于惯性张量 I_xx, I_yy, I_zz 三个对角线元素的位置。
    # 在对角线上增加一个小的数值。
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    # 将修改后的惯性张量写回仿真器
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(num_envs))
    
# =================================================================
#        核心几何计算函数 (Core Geometry Calculation)
# 以下函数是计算奖励和成功条件的基础。它们定义了任务的“几何目标”。
# =================================================================
def get_held_base_pos_local(num_envs, device):
    """
    计算手持工件的“基准点”(base) 在其自身局部坐标系下的位置。
    这个“基准点”是用于计算奖励和成功条件的核心参考点，例如插销的底部中心。
    """
    held_base_x_offset = 0.0

    held_base_z_offset = 0.0 # 基准点就在工件原点处

    held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset

    return held_base_pos_local

def get_held_base_pose(held_pos, held_quat, num_envs, device):
    """
    获取手持工件“基准点”在世界坐标系下的当前位姿。
    """
    # 1. 获取基准点在工件局部坐标系下的位置。
    held_base_pos_local = get_held_base_pos_local(num_envs, device)
    #    基准点的姿态与工件自身姿态一致，无相对旋转。
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    # 2. 使用坐标变换 (tf_combine)，将这个局部点转换到世界坐标系。
    #    世界坐标下的基准点位姿 = 世界坐标下的工件位姿 * 局部坐标下的基准点位姿
    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat

def get_target_held_base_pose(fixed_pos, fixed_quat, cfg_task, num_envs, device):
    """
    计算任务的“目标位姿”。这指的是手持工件的“基准点”最终应该到达的世界坐标位姿。
    """
    # 1. 定义一个“成功点”在固定工件局部坐标系下的位置
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if cfg_task.use_decoupled_reward:
        fixed_success_pos_local[:, 2] = 0.0 # 解耦奖励中，Z轴位置设为0.0米
    else:
        fixed_success_pos_local[:, 2] = 0.02 # 关键点奖励中，Z轴位置设为0.02米，为两阶段课程学习做准备
        
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    # 2. 使用坐标变换，将这个局部的成功点转换到世界坐标系。
    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat

def squashing_fn(x, a, b):
    """
    实现论文中提到的奖励塑形压缩函数。
    这是一个S型的函数，可以将一个无界的距离输入 `x` 映射到一个有界的奖励值。
    函数形式: r(x) = 1 / (exp(a*x) + b + exp(-a*x))
    
    Args:
        x (torch.Tensor): 输入的距离值。
        a (float): 控制函数曲线的陡峭程度。a越大，曲线在中心点附近越陡峭。
        b (float): 控制函数的最大值。最大值为 1 / (b + 2)。
    """
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

def collapse_obs_dict(obs_dict, obs_order):
    """
    一个简单的数据处理函数。
    将一个存储着观测张量的字典，按照给定的顺序 (`obs_order` 列表)，
    拼接 (concatenate) 成一个单一的、扁平的张量。
    这是将结构化的观测数据喂给神经网络之前的标准操作。
    """
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors

def compute_orientation_reward(min_yaw_error: torch.Tensor, coef: list):
    """根据给定的最小偏航角误差计算奖励。"""
    a, b = coef
    return squashing_fn(min_yaw_error, a, b)

def get_closest_symmetry_transform(held_quat, target_quat, symmetry_angles: list):
    """
    计算并返回相对于最近对称目标的变换信息。

    这是处理旋转对称性的核心函数。

    Args:
        held_quat: 手持工件的当前姿态。
        target_quat: 目标姿态 (0度方向)。
        symmetry_angles: 所有对称角度（弧度）的列表。

    Returns:
        tuple:
            - closest_target_quat (torch.Tensor): [N, 4], 最近的对称目标姿态。
            - relative_quat_to_closest (torch.Tensor): [N, 4], 从当前姿态到最近目标的相对旋转。
            - min_yaw_error_wrapped (torch.Tensor): [N], 到最近目标的最小偏航角误差（弧度）。
    """
    _, _, held_yaw = torch_utils.get_euler_xyz(held_quat)
    roll, pitch, target_yaw = torch_utils.get_euler_xyz(target_quat) # 保留目标的roll和pitch

    # --- 1. 找到最小误差和对应的目标 ---
    
    # 计算与所有对称目标的误差
    all_errors = []
    all_target_yaws = []
    for sym_angle in symmetry_angles:
        sym_target_yaw = target_yaw + sym_angle
        error = torch.abs(held_yaw - sym_target_yaw)
        # 处理周期性
        error = torch.min(error, 2 * np.pi - error)
        all_errors.append(error)
        all_target_yaws.append(sym_target_yaw)
        
    all_errors_tensor = torch.stack(all_errors) # Shape: [num_symmetries, num_envs]
    all_target_yaws_tensor = torch.stack(all_target_yaws) # Shape: [num_symmetries, num_envs]

    # 找到最小误差的索引
    # min_error_indices.shape: [num_envs]
    min_error_indices = torch.argmin(all_errors_tensor, dim=0)
    
    # 使用索引来选取每个环境对应的最小误差和最近的目标偏航角
    min_yaw_error_wrapped = torch.gather(all_errors_tensor, 0, min_error_indices.unsqueeze(0)).squeeze(0)
    closest_target_yaw = torch.gather(all_target_yaws_tensor, 0, min_error_indices.unsqueeze(0)).squeeze(0)

    # --- 2. 构建并返回变换信息 ---
    
    # a. 根据最近的目标偏航角，构建完整的“最近目标四元数”
    #    (保持原始目标的roll和pitch不变)
    closest_target_quat = torch_utils.quat_from_euler_xyz(roll, pitch, closest_target_yaw)

    # b. 计算从当前姿态到“最近目标姿态”的相对旋转
    #    q_rel = inverse(q_current) * q_closest_target
    held_quat_inv = torch_utils.quat_conjugate(held_quat)
    relative_quat_to_closest = torch_utils.quat_mul(held_quat_inv, closest_target_quat)

    return closest_target_quat, relative_quat_to_closest, min_yaw_error_wrapped


