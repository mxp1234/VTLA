# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: control module.

控制模块,实现操作空间控制(OSC)和逆运动学(IK)算法。
由基类、环境类和任务类导入使用,不直接执行。
"""

import math
import torch

import isaacsim.core.utils.torch as torch_utils # Isaac Sim的PyTorch工具库,用于四元数运算等

from isaaclab.utils.math import axis_angle_from_quat # 从四元数计算轴-角表示


def compute_dof_torque(
    cfg, # 环境的完整配置对象,用于获取控制器参数
    dof_pos, # 当前所有关节的位置(角度),shape: [num_envs, 9]
    dof_vel, # 当前所有关节的速度,shape: [num_envs, 9]
    fingertip_midpoint_pos, # 当前末端执行器的位置,shape: [num_envs, 3]
    fingertip_midpoint_quat, # 当前末端执行器的姿态(四元数),shape: [num_envs, 4]
    fingertip_midpoint_linvel, # 当前末端执行器的线速度,shape: [num_envs, 3]
    fingertip_midpoint_angvel, # 当前末端执行器的角速度,shape: [num_envs, 3]
    jacobian, # 当前姿态下的手臂雅可比矩阵,shape: [num_envs, 6, 7]
    arm_mass_matrix, # 当前姿态下的手臂质量(惯性)矩阵,shape: [num_envs, 7, 7]
    ctrl_target_fingertip_midpoint_pos, # 目标末端位置,shape: [num_envs, 3]
    ctrl_target_fingertip_midpoint_quat, # 目标末端姿态,shape: [num_envs, 4]
    task_prop_gains, # 任务空间的P-gain(比例增益),shape: [num_envs, 6]
    task_deriv_gains, # 任务空间的D-gain(微分增益),shape: [num_envs, 6]
    device, # PyTorch设备(e.g., 'cuda:0')
):
    """
    计算Franka机器人关节的驱动力矩,将指尖移动到目标位姿。
    这是一个基于动力学的操作空间控制器(OSC),实现了主任务控制和零空间控制。

    参考文献:
    1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    2) Modern Robotics (教科书)

    Args:
        cfg: 环境配置对象
        dof_pos: 关节位置
        dof_vel: 关节速度
        fingertip_midpoint_pos: 指尖中心位置
        fingertip_midpoint_quat: 指尖中心姿态
        fingertip_midpoint_linvel: 指尖中心线速度
        fingertip_midpoint_angvel: 指尖中心角速度
        jacobian: 雅可比矩阵
        arm_mass_matrix: 质量矩阵
        ctrl_target_fingertip_midpoint_pos: 目标位置
        ctrl_target_fingertip_midpoint_quat: 目标姿态
        task_prop_gains: 比例增益
        task_deriv_gains: 微分增益
        device: PyTorch设备

    Returns:
        dof_torque: 关节力矩,shape: [num_envs, 9]
        task_wrench: 任务空间扳手,shape: [num_envs, 6]
    """

    num_envs = cfg.scene.num_envs # 获取并行环境数量
    # 初始化关节力矩张量(全零),shape: [num_envs, 9] (7个手臂关节 + 2个夹爪关节)
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    # 初始化任务空间扳手张量(全零),shape: [num_envs, 6] (3个力 + 3个力矩)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    # --- 1. 计算任务空间误差 (Task-space Error) ---
    # 调用辅助函数计算当前位姿和目标位姿之间的误差
    # pos_error: [num_envs, 3], axis_angle_error: [num_envs, 3]
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric", # 使用几何雅可比矩阵
        rot_error_type="axis_angle", # 姿态误差使用轴-角表示
    )
    # 将位置误差和姿态误差拼接成6D位姿误差向量
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # --- 2. 计算主任务控制扳手 (Primary Task Wrench) ---
    # 实现任务空间的PD控制器: F = Kp * error - Kd * velocity
    # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    # 将PD控制扳手加到总扳手上
    task_wrench += task_wrench_motion

    # --- 3. 将任务空间扳手映射到关节空间力矩 (核心步骤) ---
    # OSC的核心公式: τ_task = J^T * F
    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2) # 雅可比矩阵的转置
    # 矩阵乘法: [num_envs, 7, 6] @ [num_envs, 6, 1] = [num_envs, 7, 1]
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
    # roboticsproceedings.org/rss07/p31.pdf

    # --- 4. 计算零空间控制力矩 (Null-space Control) ---
    # 目标: 在不影响主任务的前提下,让关节保持在舒适的姿态

    # useful tensors
    # (a) 计算中间矩阵
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix) # 质量矩阵的逆
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2) # 雅可比矩阵的转置
    # 操作空间惯性矩阵: Lambda = (J * M^-1 * J^T)^-1
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )  # ETH eq. 3.86; geometric Jacobian is assumed
    # 雅可比矩阵的加权伪逆: J_bar = Lambda * J * M^-1
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    # 从配置中读取期望的"舒适"关节姿态
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))

    # nullspace computation
    # (b) 计算次要任务的控制力
    # 误差 = 目标关节姿态 - 当前关节姿态 (处理角度的周期性)
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    # 将角度归一化到[-π, π]范围
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    # 关节空间PD控制: u_null = Kd * (-vel) + Kp * pos_error
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    # 转换为关节力矩
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)

    # (c) 将次要任务力矩投影到零空间
    # 零空间投影矩阵: (I - J^T * J_bar)
    # 这个矩阵会"过滤"掉任何影响主任务的力矩分量
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    # (d) 将零空间力矩叠加到总力矩上
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # TODO: Verify it's okay to no longer do gripper control here.
    # 对计算出的力矩进行裁剪,防止指令值过大(Franka的力矩限制约87Nm)
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """
    计算目标Franka指尖位姿与当前位姿之间的任务空间误差。

    Args:
        fingertip_midpoint_pos: 当前指尖位置
        fingertip_midpoint_quat: 当前指尖姿态(四元数)
        ctrl_target_fingertip_midpoint_pos: 目标指尖位置
        ctrl_target_fingertip_midpoint_quat: 目标指尖姿态(四元数)
        jacobian_type: 雅可比矩阵类型("geometric"表示几何雅可比矩阵)
        rot_error_type: 旋转误差类型("quat"或"axis_angle")

    Returns:
        pos_error: 位置误差,shape: [num_envs, 3]
        quat_error 或 axis_angle_error: 姿态误差,shape: [num_envs, 3或4]

    参考: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    """

    # -- 计算位置误差 --
    # 简单的向量减法
    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # -- 计算姿态误差 --
    # 四元数的误差计算比位置复杂,需要考虑四元数的特殊性质
    # Compute rot error
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        # (a) 检查最短路径:
        #     在四元数中,q和-q代表同一个旋转。为了避免走"远路",
        #     通过点积检查两个四元数是否在同一个半球。如果不在,就将其中一个取反
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        # (b) 计算差值四元数:
        #     q_error = q_target * inverse(q_current)
        #     四元数的逆 = 共轭 / 模长的平方
        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # 标量分量,应该接近1
        # 计算当前四元数的逆
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        # 计算差值四元数
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        # (c) 将差值四元数转换为轴-角表示法:
        #     这是为了得到一个物理意义明确的3D旋转误差向量
        #     其方向为旋转轴,模长为需要旋转的角度,可以直接用于PD控制
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def _get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """
    从给定的末端位姿误差计算关节位置的增量,使用指定的IK方法。
    这是运动学层面的逆解,不考虑动力学,主要用于重置时的IK求解。

    Args:
        delta_pose: 末端位姿误差,shape: [num_envs, 6]
        ik_method: IK算法("pinv"/"trans"/"dls"/"svd")
        jacobian: 雅可比矩阵,shape: [num_envs, 6, 7]
        device: PyTorch设备

    Returns:
        delta_dof_pos: 关节位置增量,shape: [num_envs, 7]

    参考文献:
    1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)
    """

    if ik_method == "pinv":  # Jacobian pseudoinverse
        # 雅可比伪逆法: Δq = J_pinv * Δx
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        # 雅可比转置法: Δq = J^T * Δx
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        # 阻尼最小二乘法: Δq = J^T * (J * J^T + λ^2 * I)^-1 * Δx
        # 这是最常用的方法,因为它在奇异点附近也很鲁棒
        lambda_val = 0.1  # 阻尼因子,防止在奇异点附近求解结果过大
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        # 自适应SVD法: 通过奇异值分解计算雅可比伪逆,并过滤过小的奇异值
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        # 将过小的奇异值对应的逆设为0,提高数值稳定性
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
):
    """
    将PD增益解释为任务空间增益并应用到任务空间误差上。
    实现任务空间的PD控制器。

    Args:
        delta_fingertip_pose: 末端位姿误差,shape: [num_envs, 6]
        fingertip_midpoint_linvel: 末端线速度,shape: [num_envs, 3]
        fingertip_midpoint_angvel: 末端角速度,shape: [num_envs, 3]
        task_prop_gains: P增益,shape: [num_envs, 6]
        task_deriv_gains: D增益,shape: [num_envs, 6]

    Returns:
        task_wrench: 任务空间扳手(力和力矩),shape: [num_envs, 6]
    """

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    # 对线性分量应用增益
    # F_linear = Kp_lin * pos_error + Kd_lin * (target_vel - current_vel)
    # 这里target_vel=0,希望末端稳定在目标位置
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rot error components
    # 对旋转分量应用增益
    # T_angular = Kp_rot * rot_error + Kd_rot * (target_ang_vel - current_ang_vel)
    # 这里target_ang_vel=0
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench
