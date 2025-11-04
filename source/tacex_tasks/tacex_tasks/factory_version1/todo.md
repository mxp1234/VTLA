优化1: 时间惩罚 (Time Penalty for Faster Convergence)

  问题: 当前奖励函数没有显式鼓励"快速完成任务",策略可能学到"慢慢移动"的保守策略。

  建议:
  # 在 peg_in_hole_env.py 的 _get_factory_rew_dict 中添加:

  # 1. 添加时间惩罚项
  time_penalty = -0.01  # 每步-0.01的惩罚

  # 2. 添加"提前成功"奖励
  early_success_bonus = torch.zeros_like(curr_successes, dtype=torch.float32)
  if torch.any(curr_successes):
      # 如果在t步成功,额外奖励 (max_steps - t) * bonus_scale
      remaining_steps = self.max_episode_length - self.episode_length_buf
      early_success_bonus[curr_successes] = remaining_steps[curr_successes] * 0.02

  rew_dict.update({
      "time_penalty": torch.ones(self.num_envs, device=self.device) * time_penalty,
      "early_success_bonus": early_success_bonus
  })
  rew_scales.update({
      "time_penalty": 1.0,
      "early_success_bonus": 10.0  # 高权重鼓励快速成功
  })

  预期效果:
  - 策略会学习更高效的轨迹
  - 可降低success_time 15-25%

  ---
  优化3: 速度奖励 (Velocity Bonus for Approach Phase)

  问题: 当前的 action_penalty_ee 惩罚所有动作幅度,这可能让策略过于保守。在接近阶段,大幅度动作应该被鼓励。

  建议:
  # 在 peg_in_hole_env.py 的 _get_factory_rew_dict 中修改:

  # 当前代码:
  action_penalty_ee = torch.norm(self.actions, p=2)

  # 优化后 (根据距离自适应调整惩罚):
  xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
  # 当距离较远时,减小动作惩罚,鼓励快速接近
  distance_factor = torch.clamp(xy_dist / 0.05, 0.0, 1.0)  # 归一化到[0,1]
  action_penalty_ee = torch.norm(self.actions, p=2) * (0.3 + 0.7 * (1 - distance_factor))
  # 远离时惩罚系数=0.3,接近时惩罚系数=1.0

  预期效果:
  - 接近阶段策略会采用更大的动作,加快收敛
  - 精细插入阶段仍保持小心
  - 可降低success_time 10-15%

  ---


  TODO: task-cfg继承关系混乱，重构一下。