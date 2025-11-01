#

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
<!-- [![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/) -->
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

**Keywords:** tactile sensing, gelsight, isaaclab, vision-based-tactile-sensor, vbts, reinforcement learning

> [!note]
> **Preview Release**:
>
> The framework is under active development and currently in its beta phase.
> If you encounter bugs or have suggestions on how the framework can be improved, please tell us about them (e.g. via [Issues](https://github.com/DH-Ng/TacEx/issues)/[Discussions](https://github.com/DH-Ng/TacEx/discussions)).


# TacEx - Tactile Extension for Isaac Sim/Isaac Lab
**TacEx** brings **Vision-Based Tactile Sensor (VBTS)** into Isaac Sim/Lab.

The framework integrates multiple simulation approaches for VBTS's and aims to be modular and extendable.
Components can be easily switched out, added and modified.

Currently, only the **GelSight Mini** is supported, but you can also easily add your own sensor (guide coming soon). We also plan to add more VBTS types later.

## **Main features**:
- [GPU accelerated Tactile RGB simulation](https://github.com/TimSchneider42/taxim) via [Taxim](https://github.com/Robo-Touch/Taxim)'s simulation approach
- Marker Motion Simulation via [FOTS](https://github.com/Rancho-zhao/FOTS)
- Integration of [UIPC](https://github.com/spiriMirror/libuipc) for GPU accelerated incremental potential contact to simulate FEM soft bodies, rigid bodies, cloth, etc. in a penetration-free and robust manner
- Marker Motion Simulation with FEM soft body based on the simulator used by the [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025) that leverages UIPC


Checkout the [website](https://sites.google.com/view/tacex) for showcases and the documentation for details, guides and tutorials.


## Installation
> [!NOTE]
> TacEx currently works with **Isaac Sim 4.5** and **IsaacLab 2.1.1**.
> The installation was tested on Ubuntu 22.04 with a 4090 GPU and Driver Version 550.163.01 + Cuda 12.4.

**0.** Make sure that you have **git-lfs**:

```bash
# Need it for the USD assets
git lfs install
```

**1.** Clone this repository and its submodules:
```bash
git clone --recurse-submodules https://github.com/DH-Ng/TacEx
cd TacEx
```

Then **install TacEx** [locally](docs/source/installation/Local-Installation.md)
or build a [Docker Container](docs/source/installation/Docker-Container-Setup.md).


## Contributing
Contributions of any kind are, of course, very welcome.
Be it suggestions, feedback, bug reports or pull requests.

Let's work together to advance tactile sensing in robotics!!!

## Citation
```bibtex
@article{nguyen2024tacexgelsighttactilesimulation,
      title={TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
      author={Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
      year={2024},
      eprint={2411.04776},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04776},
}
```

## Acknowledgements

TacEx is built upon code from
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/main)
- [Taxim](https://github.com/Robo-Touch/Taxim)
- [FOTS](https://github.com/Rancho-zhao/FOTS)
- [UIPC](https://github.com/spiriMirror/libuipc)
- [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025)


## VTLA command

 ### 推理+录制视频
 ```bash
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
      --task  TacEx-Factory-PegInsert-Direct-v0 \
      --num_envs 1 \
      --enable_cameras \
      --video \
      --video_length 200 \
     --checkpoint  /home/pi-zero/isaac-sim/TacEx/logs/rl_games/Factory/test/nn/last_Factory_ep_400_rew_344.56436.pth
```

 ### without tactile 训练
 ```bash
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task  TacEx-Factory-PegInsert-Direct-v0 \
      --num_envs 128 \
      --enable_cameras \
      --wandb-project-name isaac_lab \
      --wandb-entity 2996124754-salesforce \
      --track
      --headless
```

 ### with tactile 训练
```bash
       ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
      --task  TacEx-Factory-PegInsert-Tactile-v1 \
      --num_envs 128 \
      --enable_cameras \
      --wandb-project-name isaac_lab_tactile_v1 \
      --wandb-entity 2996124754-salesforce \
      --track
      --headless
```

使用[sparsh](https://github.com/facebookresearch/sparsh)的预训练触觉编码/解码器 last.ckpt：

[下载路径](https://huggingface.co/facebook/sparsh-gelsight-forcefield-decoder/tree/main/gelsight_t1_forcefield_dino_vitbase_bg/checkpoints)

存放目录：
```
/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt
```