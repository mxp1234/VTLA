"""
触觉图像预训练模型推理示例

模型输入输出规格:
- 输入: (B, C, H, W) 触觉图像，建议尺寸 224x224 或 384x384
- 输出:
  - depth: (B, 1, H, W) 深度图
  - shear: (B, 2, H, W) 剪切力图 (x, y方向)
"""

import torch
import numpy as np
import os
from PIL import Image

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载预训练权重
pth_path = os.path.join(script_dir, 'epoch-0021.pth')
checkpoint = torch.load(pth_path, map_location='cpu')

print("=" * 80)
print("模型权重加载成功")
print("=" * 80)

# 注意: 这里只有权重，需要先定义完整的模型结构才能加载
# 模型定义需要包含:
# 1. Vision Transformer backbone (输出768维特征)
# 2. DPT-style decoder (reassemble + fusion + probe)

print("\n模型输入输出规格:")
print("-" * 80)
print("输入:")
print("  - 形状: (batch_size, channels, height, width)")
print("  - 建议尺寸: 224x224 或 384x384")
print("  - 通道数: 3 (RGB) 或 1 (灰度)")
print("  - 数据范围: [0, 1] 归一化")
print()
print("输出:")
print("  1. 深度图 (depth)")
print("     - 形状: (batch_size, 1, height, width)")
print("     - 含义: 触觉传感器表面的深度/位移")
print()
print("  2. 剪切力图 (shear)")
print("     - 形状: (batch_size, 2, height, width)")
print("     - 含义: x和y方向的剪切力分量")
print()

print("=" * 80)
print("使用步骤:")
print("=" * 80)
print("1. 定义完整的模型架构 (包含 ViT backbone + DPT decoder)")
print("2. 加载权重: model.load_state_dict(checkpoint, strict=False)")
print("3. 准备输入图像并归一化")
print("4. 前向推理获得深度和剪切力输出")
print()

# 示例: 创建一个虚拟输入
print("=" * 80)
print("虚拟输入示例:")
print("=" * 80)

batch_size = 1
img_size = 224  # 或 384
channels = 3    # RGB 触觉图像

dummy_input = torch.randn(batch_size, channels, img_size, img_size)
print(f"输入形状: {dummy_input.shape}")
print(f"输入数据范围: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")
print()
print("注意: 实际使用时需要:")
print("  1. 从文件加载真实触觉图像")
print("  2. 进行预处理 (resize, normalize)")
print("  3. 使用完整模型进行推理")
