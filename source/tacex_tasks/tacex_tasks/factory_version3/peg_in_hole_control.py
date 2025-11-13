import math
import torch

import isaacsim.core.utils.torch as torch_utils # 导入Isaac Sim的PyTorch工具库，主要用于四元数运算

from isaaclab.utils.math import axis_angle_from_quat # 导入Isaac Lab的数学工具库，这里用于从四元数计算轴-角表示

def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    """计算当前末端位姿与目标位姿之间的任务空间误差。"""

    # Compute pos error
    # 1. 计算位置误差：就是一个简单的向量减法。
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    # 2. 计算姿态误差：
    #    这部分比位置误差复杂，因为旋转不能简单地相减。
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        # (a) 检查最短路径：
        #     在四元数中，q 和 -q 代表同一个旋转。为了避免走“远路”，
        #     我们通过点积检查两个四元数是否在同一个半球。如果不在，就将其中一个取反。
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        # (b) 计算差值四元数：
        #     q_error = q_target * inverse(q_current)
        #     四元数的逆等于其共轭除以模长的平方，也即除以范数
        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ] # 计算当前四元数的范数（应该接近1）
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat # 计算当前四元数的共轭，共轭还要除以范数以得到逆
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1) # 计算当前四元数的逆
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv) # 计算差值四元数

        # Convert to axis-angle error
        # (c) 将差值四元数转换为轴-角表示法：
        #     这是为了得到一个物理意义明确的3D旋转误差向量，其方向为旋转轴，模长为需要旋转的角度。
        #     这个向量可以直接用于PD控制。
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error

def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
):
    """Interpret PD gains as task-space gains. Apply to task-space error."""
    """
    实现任务空间的PD控制器。
    Args:
        delta_fingertip_pose: 末端位姿误差，shape: [num_envs, 6]，前3维是位置误差，后3维是姿态误差（轴-角表示）
        fingertip_midpoint_linvel: 末端线速度，shape: [num_envs, 3]
        fingertip_midpoint_angvel: 末端角速度，shape: [num_envs, 3]
        task_prop_gains: 任务空间的P增益，shape: [num_envs, 6]
        task_deriv_gains: 任务空间的D增益，shape: [num_envs, 6]
    """

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # 分别对线性和旋转部分应用PD控制
    # 线性部分 (前3维)
    lin_error = delta_fingertip_pose[:, 0:3]
    # F_linear = Kp_lin * pos_error + Kd_lin * (target_vel - current_vel)
    # 这里 target_vel 是 0，希望末端稳定在目标位置
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # 旋转部分 (后3维)
    rot_error = delta_fingertip_pose[:, 3:6]
    # T_angular = Kp_rot * rot_error + Kd_rot * (target_ang_vel - current_ang_vel)
    # 这里 target_ang_vel 也是 0
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench

def compute_dof_torque(
    cfg, # 环境的完整配置对象，用于获取控制器参数 (如 kp_null, kd_null)
    dof_pos, # 当前所有关节的位置（角度），shape: [num_envs, 9]
    dof_vel, # 当前所有关节的速度，shape: [num_envs, 9]
    fingertip_midpoint_pos, # 当前末端执行器的位置，shape: [num_envs, 3]
    fingertip_midpoint_quat, # 当前末端执行器的姿态（四元数），shape: [num_envs, 4]
    fingertip_midpoint_linvel, # 当前末端执行器的线速度，shape: [num_envs, 3]
    fingertip_midpoint_angvel, # 当前末端执行器的角速度，shape: [num_envs, 3]
    jacobian, # 当前姿态下的手臂雅可比矩阵，shape: [num_envs, 6, 7]
    arm_mass_matrix, # 当前姿态下的手臂质量（惯性）矩阵，shape: [num_envs, 7, 7]
    ctrl_target_fingertip_midpoint_pos, # 目标末端位置，shape: [num_envs, 3]
    ctrl_target_fingertip_midpoint_quat,  # 目标末端姿态，shape: [num_envs, 4]
    task_prop_gains, # 任务空间的P-gain（比例增益），shape: [num_envs, 6]
    task_deriv_gains, # 任务空间的D-gain（微分增益），shape: [num_envs, 6]
    device,  # PyTorch设备 (e.g., 'cuda:0')
    dead_zone_thresholds=None, # 控制死区阈值
):
    """
    计算Franka机器人手臂关节的驱动力矩，以将指尖移动到目标位姿。
    这是一个典型的基于动力学的操作空间控制器（OSC）。
    """
    # References:
    # 1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # 2) Modern Robotics

    num_envs = cfg.scene.num_envs # 获取并行环境数量
    # 初始化一个零张量，用于存储最终计算出的9个关节的力矩（7手臂+2夹爪）
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    # 初始化一个零张量，用于存储在任务空间（末端）计算出的期望“扳手”（Wrench）
    # Wrench是一个6D向量，前3维是力(Force)，后3维是力矩(Torque)。
    task_wrench = torch.zeros((num_envs, 6), device=device)

     # --- 1. 计算任务空间误差 (Task-space Error) ---
    # 调用辅助函数计算当前位姿和目标位姿之间的误差。
    # pos_error: [num_envs, 3], axis_angle_error: [num_envs, 3]
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1) # 将位置误差和姿态误差拼接成一个6D的位姿误差向量。

    # --- 2. 计算主任务控制扳手 (Primary Task Wrench) ---
    # 这一步实现了一个任务空间的PD控制器。
    # F = Kp * position_error - Kd * velocity_error
    # 这里的速度误差是 (0 - current_velocity)，因为我们的目标速度是0（稳定地到达目标点）。
    # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    
    # 将计算出的PD控制扳手加到总扳手上，总扳手初始化为零
    task_wrench += task_wrench_motion

    # 如果设置了死区阈值，则应用死区处理，防止微小误差引起不必要的振动
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

     # --- 3. 将任务空间扳手映射到关节空间力矩 (核心步骤) ---
    # 这是OSC的核心公式之一: τ_task = J^T * F
    # τ_task: 由主任务产生的关节力矩，在关节空间
    # J^T: 雅可比矩阵的转置, shape: [num_envs, 7, 6]
    # F: 任务空间的扳手 (task_wrench), shape: [num_envs, 6]，在任务空间
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    # @ 表示批处理矩阵乘法。unsqueeze(-1)将F从[N,6]变为[N,6,1]以进行矩阵乘法。
    # .squeeze(-1)将结果从[N,7,1]变回[N,7]。
    # 结果是每个手臂关节（0-6）需要施加的力矩。
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
    # roboticsproceedings.org/rss07/p31.pdf

    # --- 4. 计算零空间控制力矩 (Null-space Control) ---
    # 目标是在不影响主任务（移动末端）的前提下，
    # 完成一个次要任务（例如，让关节保持在一个舒适的姿态，避免奇异点）。
   
    
    # useful tensors
    # (a) 计算一些有用的中间矩阵
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix) # 质量矩阵的逆
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2) # 雅可比矩阵的转置
    arm_mass_matrix_task = torch.inverse( # 操作空间惯性矩阵 Lambda = (J * M^-1 * J^T)^-1
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )  # ETH eq. 3.86; geometric Jacobian is assumed
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv # 这是雅可比矩阵的“加权伪逆”：J_bar = Lambda * J * M^-1
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1)) # 从配置中读取期望的“舒适”关节姿态
    # nullspace computation
    # (b) 计算次要任务（关节姿态保持）的控制力
    #     这是一个关节空间的PD控制器
    # 误差 = 目标关节姿态 - 当前关节姿态 (并处理角度的周期性)
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    # PD控制: u_null = Kp_null * pos_error - Kd_null * vel_error， 这里目标速度是0
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)  # 转换为关节力矩
    # (c) 将次要任务力矩投影到零空间
    #     τ_null = (I - J^T * J_bar) * τ_secondary
    #     (I - J^T * J_bar) 是一个零空间投影矩阵，它可以“过滤”掉任何会影响主任务的力矩分量。
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1) # (d) 将零空间力矩叠加到总力矩上
    # Note: the null-space torque does not affect end-effector motion (in theory)

    # --- 5. 返回最终结果 ---
    # 对计算出的力矩进行裁剪，防止指令值过大。
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0) # Franka的力矩限制大约是87Nm，这里留点余量
    return dof_torque, task_wrench

def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    """
    (此函数在IK重置时使用)
    根据给定的末端位姿误差 (delta_pose)，使用指定的IK方法计算关节位置的增量。
    这是运动学层面的逆解，不考虑动力学。
    """
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == "pinv":  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    # DLS (Damped Least Squares) 方法:
    # 这是最常用的一种，因为它很鲁棒。
    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1  # 0.1 # 阻尼因子，防止在奇异点附近求解结果过大
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        # 核心公式: Δq = J^T * (J * J^T + λ^2 * I)^-1 * Δx，也即再求伪逆的时候加了一个阻尼项，防止奇异
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device) # 注意这里的维度是6不是7
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos