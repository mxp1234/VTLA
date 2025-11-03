# Factory Environment 详细代码说明

## 文件概述
`factory_env.py` 是一个基于 Isaac Lab 的强化学习环境实现，用于工厂装配任务（如插销、齿轮啮合、螺母拧紧等）。该环境继承自 `DirectRLEnv`，使用 Franka 机械臂和触觉传感器完成精密装配任务。

---

## 类定义

### `FactoryEnv(DirectRLEnv)`

主环境类，继承自 Isaac Lab 的 `DirectRLEnv`。

---

## 核心函数详解

### 1. `__init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs)`

**功能**: 初始化工厂环境

**主要步骤**:
- 根据配置计算观察空间和状态空间的维度
- 将动作空间维度加入观察和状态空间
- 调用父类初始化
- 设置刚体惯性
- 初始化张量
- 设置默认动力学参数
- 计算中间值

**参数**:
- `cfg`: 环境配置对象
- `render_mode`: 渲染模式（可选）

---

### 2. `_set_body_inertias(self)`

**功能**: 设置刚体惯性参数

**说明**:
- 用于适配 IGE 中的 `armature` 参数
- 在惯性张量的对角线元素上增加 0.01 的偏移量
- 提高数值稳定性

**实现细节**:
```python
offset[:, :, [0, 4, 8]] += 0.01  # 对角线元素 [Ixx, Iyy, Izz]
```

---

### 3. `_set_default_dynamics_parameters(self)`

**功能**: 设置定义动态交互的参数

**主要设置**:
- **默认增益**: 任务的比例增益（PD 控制器的 P 参数）
- **位置阈值**: 位置动作的阈值范围
- **旋转阈值**: 旋转动作的阈值范围
- **摩擦系数**: 为机器人、抓取物体、固定物体设置摩擦系数

---

### 4. `_set_friction(self, asset, value)`

**功能**: 更新给定资产的材料摩擦属性

**参数**:
- `asset`: 要修改的资产对象
- `value`: 摩擦系数值

**设置内容**:
- `materials[..., 0]`: 静摩擦系数
- `materials[..., 1]`: 动摩擦系数

---

### 5. `_init_tensors(self)`

**功能**: 一次性初始化所有张量

**初始化的张量类别**:

#### (1) 基础张量
- `identity_quat`: 单位四元数 [1, 0, 0, 0]

#### (2) 控制目标
- `ctrl_target_joint_pos`: 目标关节位置
- `ctrl_target_fingertip_midpoint_pos`: 指尖中点目标位置
- `ctrl_target_fingertip_midpoint_quat`: 指尖中点目标姿态

#### (3) 固定资产
- `fixed_pos_action_frame`: 动作坐标系中的固定物体位置
- `fixed_pos_obs_frame`: 观察坐标系中的固定物体位置
- `init_fixed_pos_obs_noise`: 初始位置观察噪声

#### (4) 抓取资产
- `held_base_pos_local`: 抓取物体基座的局部位置
- `held_base_quat_local`: 抓取物体基座的局部姿态
- 根据任务类型设置不同的 z 轴偏移量

#### (5) 刚体索引
- `left_finger_body_idx`: 左指索引
- `right_finger_body_idx`: 右指索引
- `fingertip_body_idx`: 指尖中心索引

#### (6) 有限差分张量
- `prev_fingertip_pos`: 上一时刻指尖位置
- `prev_fingertip_quat`: 上一时刻指尖姿态
- `prev_joint_pos`: 上一时刻关节位置

#### (7) 关键点张量
- `keypoint_offsets`: 关键点偏移量
- `keypoints_held`: 抓取物体上的关键点
- `keypoints_fixed`: 固定物体上的关键点

#### (8) 成功判定
- `ep_succeeded`: 每个环境的成功标志
- `ep_success_times`: 成功所需时间步数

---

### 6. `_get_keypoint_offsets(self, num_keypoints)`

**功能**: 获取沿单位长度线段均匀分布的关键点，中心在 0

**参数**:
- `num_keypoints`: 关键点数量

**返回**:
- 形状为 `(num_keypoints, 3)` 的张量，z 轴方向均匀分布在 [-0.5, 0.5]

---

### 7. `_setup_scene(self)`

**功能**: 初始化仿真场景

**创建的对象**:
1. **地面平面**: 位于 z = -1.05
2. **桌子**: 从 USD 文件加载的 Seattle Lab Table
3. **机器人**: Franka 机械臂
4. **固定资产**: 根据任务配置（螺栓、齿轮底座等）
5. **抓取资产**: 机器人抓取的物体
6. **齿轮资产**: 如果是齿轮啮合任务，添加大小齿轮
7. **触觉传感器**: 左右两个 GelSight Mini 传感器
8. **光照**: 圆顶光，强度 2000.0

---

### 8. `_compute_intermediate_values(self, dt)`

**功能**: 从原始张量计算中间值，包括添加噪声

**计算内容**:

#### (1) 资产位姿
- `fixed_pos`, `fixed_quat`: 固定物体的位置和姿态
- `held_pos`, `held_quat`: 抓取物体的位置和姿态

#### (2) 指尖状态
- `fingertip_midpoint_pos/quat`: 指尖中点位姿
- `fingertip_midpoint_linvel/angvel`: 指尖线速度和角速度

#### (3) Jacobian 和质量矩阵
- `left_finger_jacobian`, `right_finger_jacobian`: 左右手指的 Jacobian
- `fingertip_midpoint_jacobian`: 指尖中点 Jacobian（两指平均）
- `arm_mass_matrix`: 机械臂质量矩阵

#### (4) 有限差分速度估计
- `ee_linvel_fd`: 通过有限差分计算的指尖线速度
- `ee_angvel_fd`: 通过有限差分计算的指尖角速度
- `joint_vel_fd`: 通过有限差分计算的关节速度

**说明**: 有限差分方法比直接读取速度更可靠

#### (5) 关键点计算
- 计算抓取物体和目标位置的关键点世界坐标
- 计算关键点距离作为奖励信号

---

### 9. `_get_observations(self)`

**功能**: 获取演员（策略）和评论家（价值函数）的输入（非对称评论家架构）

**观察字典 (obs_dict)** - 用于策略:
- `fingertip_pos`: 指尖位置
- `fingertip_pos_rel_fixed`: 指尖相对固定物体的位置（带噪声）
- `fingertip_quat`: 指尖姿态
- `ee_linvel`: 指尖线速度（有限差分）
- `ee_angvel`: 指尖角速度（有限差分）
- `prev_actions`: 上一步的动作

**状态字典 (state_dict)** - 用于评论家:
- 包含所有观察信息
- 额外包含：
  - `joint_pos`: 关节位置
  - `held_pos/quat`: 抓取物体位姿
  - `fixed_pos/quat`: 固定物体位姿
  - `task_prop_gains`: 任务比例增益
  - `pos_threshold`, `rot_threshold`: 动作阈值

**返回**:
```python
{
    "policy": obs_tensors,   # 策略网络输入
    "critic": state_tensors  # 价值网络输入
}
```

---

### 10. `_reset_buffers(self, env_ids)`

**功能**: 重置指定环境的缓冲区

**重置内容**:
- 成功标志归零
- 成功时间归零

---

### 11. `_pre_physics_step(self, action)`

**功能**: 在物理步进前应用策略动作（带平滑）

**平滑方法**: 指数移动平均 (EMA)
```python
self.actions = ema_factor * new_action + (1 - ema_factor) * old_actions
```

**作用**: 减少动作抖动，提高控制平滑性

---

### 12. `close_gripper_in_place(self)`

**功能**: 保持抓手当前位置不变，执行闭合动作

**步骤**:
1. 创建零动作向量
2. 设置抓手目标位置为当前位置
3. 设置抓手目标姿态为当前姿态
4. 限制姿态为垂直向下（roll=π, pitch=0）
5. 设置抓手 DOF 目标为闭合状态
6. 生成控制信号

**用途**: 重置过程中闭合抓手抓取物体

---

### 13. `_apply_action(self)`

**功能**: 将策略动作作为当前位置的增量目标应用

**处理流程**:

#### (1) 获取当前偏航角
```python
self.curr_yaw = torch.where(curr_yaw > 235°, curr_yaw - 2π, curr_yaw)
```

#### (2) 更新中间值
如果时间戳过期，重新计算速度

#### (3) 处理位置动作
```python
pos_actions = self.actions[:, 0:3] * self.pos_threshold
self.ctrl_target_pos = current_pos + pos_actions
```

#### (4) 处理旋转动作
- 对于单向旋转任务，限制 z 轴旋转为负值
- 转换轴角表示为四元数
- 应用旋转增量

#### (5) 限制位置范围
防止机械臂移动超过 5cm 远离基座

#### (6) 限制姿态
强制 roll=π, pitch=0（保持垂直向下）

#### (7) 生成控制信号

---

### 14. `_set_gains(self, prop_gains, rot_deriv_scale=1.0)`

**功能**: 使用临界阻尼设置机器人增益

**公式**:
- **比例增益**: `kp = prop_gains`
- **微分增益**: `kd = 2 * sqrt(kp)` （临界阻尼条件）
- **旋转微分增益**: 额外缩放 `1/rot_deriv_scale`

---

### 15. `generate_ctrl_signals(self)`

**功能**: 获取 Jacobian 并设置 Franka DOF 位置目标（手指）或力矩（机械臂）

**调用**: `fc.compute_dof_torque()` 计算关节力矩

**输入**:
- DOF 位置和速度
- 指尖位姿和速度
- Jacobian 矩阵
- 质量矩阵
- 目标位姿
- PD 增益

**输出**:
- `joint_torque`: 关节力矩
- `applied_wrench`: 应用的扳手力

**设置控制**:
- 机械臂关节：使用力矩控制
- 抓手关节：使用位置控制（PhysX PD 控制器）

---

### 16. `_get_dones(self)`

**功能**: 更新用于奖励和观察的中间值，判断是否结束

**返回**:
- `(time_out, time_out)` - 是否超时

**超时条件**:
```python
episode_length >= max_episode_length - 1
```

---

### 17. `_get_curr_successes(self, success_threshold, check_rot=False)`

**功能**: 获取当前时间步的成功掩码

**成功条件**:

#### (1) XY 平面对齐
```python
xy_dist < 0.0025  # 2.5mm
```

#### (2) Z 轴高度
- **插销/齿轮**: `z_disp < fixed_height * success_threshold`
- **螺母拧紧**: `z_disp < thread_pitch * success_threshold`

#### (3) 旋转检查（可选）
如果是螺母拧紧任务：
```python
curr_yaw < ee_success_yaw
```

---

### 18. `_get_rewards(self)`

**功能**: 更新奖励并计算成功统计

**步骤**:
1. 检查当前时间步是否成功
2. 调用 `_update_rew_buf()` 计算奖励
3. 记录成功率到 `extras["successes"]`
4. 记录首次成功时间到 `extras["success_times"]`
5. 更新上一步动作

---

### 19. `_update_rew_buf(self, curr_successes)`

**功能**: 计算当前时间步的奖励

**奖励组成**:

#### (1) 关键点奖励（使用挤压函数）
```python
squashing_fn(x, a, b) = 1 / (exp(a*x) + b + exp(-a*x))
```

三个级别：
- **基线**: `(a0, b0)` - 粗略接近
- **粗略**: `(a1, b1)` - 中等精度
- **精细**: `(a2, b2)` - 高精度

#### (2) 动作惩罚
- **动作幅度惩罚**: `||action||_2`
- **动作梯度惩罚**: `||action - prev_action||_2`

#### (3) 成功奖励
- **参与奖励**: 达到参与阈值
- **成功奖励**: 达到成功阈值

**总奖励**:
```python
reward = kp_baseline + kp_coarse + kp_fine
         - action_penalty * scale1
         - action_grad_penalty * scale2
         + curr_engaged + curr_successes
```

---

### 20. `_reset_idx(self, env_ids)`

**功能**: 重置指定环境

**步骤**:
1. 调用父类重置
2. 设置资产到默认位姿
3. 设置 Franka 到默认位姿
4. 执行一步仿真（无动作）
5. 随机化初始状态

---

### 21. `_get_target_gear_base_offset(self)`

**功能**: 获取目标齿轮相对于齿轮底座的偏移量

**支持的齿轮类型**:
- `gear_large`: 大齿轮
- `gear_medium`: 中齿轮
- `gear_small`: 小齿轮

**返回**: 3D 偏移向量

---

### 22. `_set_assets_to_default_pose(self, env_ids)`

**功能**: 在随机化之前将资产移动到默认位姿

**设置内容**:
1. **抓取资产**:
   - 位姿 = 默认位姿 + 环境原点
   - 速度 = 0
2. **固定资产**:
   - 位姿 = 默认位姿 + 环境原点
   - 速度 = 0

---

### 23. `set_pos_inverse_kinematics(self, env_ids)`

**功能**: 使用阻尼最小二乘法 (DLS) 逆运动学设置机器人关节位置

**迭代过程**:
- 时长: 0.25 秒
- 每次迭代:
  1. 计算位姿误差
  2. 求解 DLS 问题获得关节位置增量
  3. 更新关节位置
  4. 执行仿真步进

**收敛判断**:
- 位置误差 < 1e-3
- 角度误差 < 1e-3

**返回**: `(pos_error, axis_angle_error)`

---

### 24. `get_handheld_asset_relative_pose(self)`

**功能**: 获取抓取物体相对于指尖的默认相对位姿

**不同任务的配置**:

#### 插销任务 (peg_insert)
```python
z_offset = peg_height - fingerpad_length
```

#### 齿轮啮合 (gear_mesh)
```python
x_offset = gear_base_offset[0]
z_offset = gear_base_offset[2] + gear_height/2 * 1.1
```

#### 螺母拧紧 (nut_thread)
```python
pos = held_base_pos_local
quat = rotation(yaw=held_asset_rot_init)
```

**返回**: `(relative_pos, relative_quat)`

---

### 25. `_set_franka_to_default_pose(self, joints, env_ids)`

**功能**: 将 Franka 机械臂返回到默认关节位置

**设置内容**:
1. **抓手宽度**: `held_asset_diameter / 2 * 1.25`
2. **关节位置**: 使用提供的 `joints` 参数
3. **关节速度**: 0
4. **关节力矩**: 0

**执行**: 一步仿真以应用设置

---

### 26. `step_sim_no_action(self)`

**功能**: 执行一步仿真但不应用动作（用于重置）

**步骤**:
1. 将数据写入仿真
2. 执行物理步进（不渲染）
3. 更新场景
4. 计算中间值

---

### 27. `randomize_initial_state(self, env_ids)`

**功能**: 随机化初始状态并执行任何回合级别的随机化

这是最复杂的函数，包含完整的环境重置流程。

#### 阶段 1: 禁用重力
```python
physics_sim_view.set_gravity((0, 0, 0))
```

#### 阶段 2: 随机化固定资产位姿

**(a) 位置随机化**
```python
rand ∈ [-1, 1]
fixed_pos += rand * fixed_asset_init_pos_noise + env_origin
```

**(b) 姿态随机化**
```python
yaw = init_yaw + rand * yaw_range
quat = quat_from_euler(0, 0, yaw)
```

**(c) 观察噪声**
```python
fixed_pos_obs_noise = randn * fixed_asset_pos_rand
```

#### 阶段 3: 随机化机械臂位置

**(a) 计算目标位置**
```python
target_pos = fixed_tip_pos + hand_init_pos + rand_noise
target_pos.z += hand_init_height
```

**(b) 随机化姿态**
```python
hand_euler = hand_init_orn + rand * hand_init_orn_noise
hand_quat = quat_from_euler(hand_euler)
```

**(c) 迭代 IK 求解**
- 使用 DLS 方法
- 如果失败，重新随机化并重试
- 直到所有环境 IK 成功

#### 阶段 4: 添加侧翼齿轮（仅齿轮啮合任务）
如果配置开启，放置小齿轮和大齿轮

#### 阶段 5: 随机化抓取物体在手中的位置

**(a) 翻转抓手姿态**
```python
flip_quat = [0, 0, 1, 0]  # 180° z轴旋转
```

**(b) 计算资产在手中的变换**
```python
asset_in_hand = inverse(held_relative_pose)
held_in_world = fingertip_flipped * asset_in_hand
```

**(c) 添加位置噪声**
```python
held_pos_noise = rand * held_asset_pos_noise
# 齿轮任务：z方向只向下随机 [-1, 0]
```

#### 阶段 6: 闭合抓手

**(a) 设置快速重置增益**
```python
reset_gains = reset_task_prop_gains
```

**(b) 执行抓取**
- 时长: 0.25 秒
- 每步: 闭合抓手至 0.0

#### 阶段 7: 初始化动作和参考帧

**(a) 设置初始动作为零**
```python
actions = prev_actions = zeros
```

**(b) 反推初始动作**
- 位置动作: 相对于固定物体的归一化位置
- 偏航动作: 相对于固定物体的归一化偏航角

**(c) 零初始速度**
```python
ee_linvel = ee_angvel = 0
```

#### 阶段 8: 恢复重力
```python
physics_sim_view.set_gravity(cfg.sim.gravity)
```

---

## 关键设计特点

### 1. 非对称评论家架构
- **策略网络**: 使用带噪声的部分观察
- **价值网络**: 使用完整的特权状态信息

### 2. 有限差分速度估计
相比直接读取，提供更稳定的速度估计

### 3. 关键点奖励
使用多个关键点计算对齐误差，比单点更鲁棒

### 4. 临界阻尼控制
自动计算 PD 控制器的微分增益，确保平滑运动

### 5. 分层随机化
- 域随机化（摩擦、位姿噪声）
- 初始状态随机化
- 观察噪声

### 6. 姿态约束
限制机械臂姿态为垂直向下，简化学习问题

---

## 支持的任务类型

### 1. Peg Insert (插销)
- 将圆柱插入孔中
- 关键: XY 对齐 + Z 轴插入

### 2. Gear Mesh (齿轮啮合)
- 将齿轮啮合到齿轮底座
- 支持小/中/大齿轮
- 可选侧翼齿轮

### 3. Nut Thread (螺母拧紧)
- 将螺母拧到螺栓上
- 需要旋转对齐
- 检查偏航角成功条件

---

## 重要参数

### 控制参数
- `pos_action_threshold`: 位置动作范围
- `rot_action_threshold`: 旋转动作范围
- `pos_action_bounds`: 相对基座的最大移动范围
- `ema_factor`: 动作平滑系数

### 奖励参数
- `keypoint_coef_*`: 关键点奖励系数
- `action_penalty_scale`: 动作惩罚系数
- `action_grad_penalty_scale`: 动作梯度惩罚系数
- `success_threshold`: 成功判定阈值
- `engage_threshold`: 参与判定阈值

### 随机化参数
- `fixed_asset_init_pos_noise`: 固定物体位置噪声
- `held_asset_pos_noise`: 抓取物体位置噪声
- `hand_init_pos_noise`: 手部初始位置噪声
- `hand_init_orn_noise`: 手部初始姿态噪声

---

## 文件位置

`/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory/factory_env.py`

---

## 依赖项

- Isaac Lab (isaacsim, isaaclab)
- PyTorch
- TacEx (触觉传感器)
- NumPy
- Carb (NVIDIA Omniverse)

---

## 总结

`FactoryEnv` 是一个功能完整的强化学习环境，专门设计用于精密装配任务。它结合了：

1. **高级控制**: OSC (操作空间控制) + PD 控制
2. **多模态感知**: 位置、姿态、触觉传感器
3. **鲁棒学习**: 域随机化、非对称评论家
4. **任务通用性**: 支持多种装配任务

该环境特别适合研究需要高精度、接触丰富的机器人操作任务。
