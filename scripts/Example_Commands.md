# Example Commands
Some example commands for training etc.

To list all TacEx environments:
```bash
# Assuming you are in the TacEx root directory
isaaclab -p scripts/reinforcement_learning/list_envs.py
```

## Training

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-IK-v0 --num_envs 1024
```

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-Privileged-v0 --num_envs 1024
```

```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/train.py --task TacEx-Ball-Rolling-Privileged-without-Reach_v0 --num_envs 1024 --enable_cameras
```

```bash
isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0 --num_envs 1 --enable_cameras --checkpoint /workspace/tacex/logs/skrl/ball_rolling/2025-05-16_18-16-16_tactile_rgb_best/checkpoints/best_agent.pt
```

## Play
```bash
isaaclab -p ./scripts/reinforcement_learning/rsl_rl/play.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 23 --enable_cameras --load_run logs/skrl/ball_rolling/2025-04-08_22-55-53_improved_ppo_torch_base_env_cluster --checkpoint best_agent.pt
```

```bash
isaaclab -p ./scripts/reinforcement_learning/skrl/play.py --task TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0 --num_envs 23 --enable_cameras --checkpoint logs/skrl/ball_rolling/workspace/tacex/logs/skrl/ball_rolling/2025-05-16_18-16-16_tactile_rgb_best/checkpoints/best_agent.pt
```


## Other
You can activate tensorboard with
```bash
isaaclab -p -m tensorboard.main serve --logdir /workspace/tacex/logs/rsl_rl/ball_rolling
isaaclab -p -m tensorboard.main serve --logdir /workspace/tacex/logs/skrl/ball_rolling
```

You can debug RL training scripts by (for example) running the command
```bash
#python -m pip install --upgrade debugpy
lab -p -m debugpy --listen 3000 --wait-for-client _your_command_
```
and then attaching via VScode debugger.


## 训练peg insert 任务
  # 从最新 checkpoint 继续训练
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --checkpoint logs/rl_games/Factory/test/nn/Factory.pth

  # 或使用特定迭代的 checkpoint
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --checkpoint logs/rl_games/Factory/test/nn/Factory_1000.pth

  配置文件位置:
  IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml

  关键参数: (在 rl_games_ppo_cfg.yaml:74-76)
  config:
    max_epochs: 200              # 最大训练轮数(epoch)
    save_best_after: 10          # 训练10个epoch后开始保存最佳模型
    save_frequency: 100          # 每100个epoch保存一次checkpoint
    horizon_length: 128          # 每个epoch的步数

  计算总训练步数:
  总步数 = max_epochs × horizon_length × num_envs
        = 200 × 128 × 128  
        = 3,276,800 步

  2️⃣ 可视化设置

  方式1: 训练时不显示GUI (headless模式)
  # 默认就是 headless 模式(无GUI),更快
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --headless  # 默认已启用

  方式2: 训练时显示可视化
  # 不使用 headless 模式
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0
      # 不加 --headless 参数

  3️⃣ 录制视频设置

  命令行参数: (在 train.py:18-20)
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --video \                    # 启用视频录制
      --video_length 200 \          # 每个视频长度(步数)
      --video_interval 2000         # 每2000步录制一次

  视频保存位置:
  logs/rl_games/<config_name>/<experiment_name>/videos/train/

  4️⃣ 其他重要参数

  环境配置: (在 factory_env_cfg.py:96,119)
  episode_length_s = 10.0        # 每个episode长度(秒)
  num_envs = 128                 # 并行环境数量

  命令行覆盖:
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --num_envs 256 \              # 覆盖环境数量
      --max_iterations 500 \         # 覆盖最大训练轮数
      --seed 42                      # 设置随机种子

  🎯 完整训练命令示例

  基础训练(快速,无可视化)

  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --num_envs 256 \
      --headless

  训练+录制视频

  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task Isaac-Factory-PegInsert-Direct-v0 \
      --num_envs 128 \
      --video \
      --video_length 200 \
      --video_interval 2000 \
      --headless
## Tacex 的nut thread


 ## 推理+录制视频
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
      --task  TacEx-Factory-PegInsert-Direct-v0 \
      --num_envs 1 \
      --enable_cameras \
      --video \
      --video_length 200 \
     --checkpoint  /home/pi-zero/isaac-sim/TacEx/logs/rl_games/Factory/test/nn/last_Factory_ep_400_rew_344.56436.pth

 ## 触觉不引入 训练
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task  TacEx-Factory-PegInsert-Direct-v0 \
      --num_envs 128 \
      --enable_cameras \
      --wandb-project-name isaac_lab \
      --wandb-entity 2996124754-salesforce \
      --track
      --headless


## 触觉引入obs 训练
       ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task  TacEx-Factory-PegInsert-Tactile-v1 \
      --num_envs 128 \
      --enable_cameras \
      --wandb-project-name isaac_lab_tactile_v1 \
      --wandb-entity 2996124754-salesforce \
      --track
      --headless