import numpy as np
import torch
import sys
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# 使用绝对路径加载 .pth 文件
pth_path = os.path.join(script_dir, 'epoch-0021.pth')
sd = torch.load(pth_path, map_location='cpu')
print(type(sd))
print("=" * 80)
print("模型权重键:")
print("=" * 80)
for key in sd.keys():
    print(f"{key}: shape = {sd[key].shape}")

print("\n" + "=" * 80)
print("模型结构分析:")
print("=" * 80)

# 分析关键层的输入输出维度
print("\n1. Norm 层:")
if 'model_task.norm.weight' in sd:
    norm_dim = sd['model_task.norm.weight'].shape[0]
    print(f"   - 特征维度: {norm_dim}")

print("\n2. Reassemble 层 (特征重组):")
for i in range(4):
    conv1_key = f'model_task.reassembles.{i}.resample.conv1.weight'
    if conv1_key in sd:
        w = sd[conv1_key]
        print(f"   - Reassemble {i}: 输入通道={w.shape[1]}, 输出通道={w.shape[0]}, 卷积核={w.shape[2]}x{w.shape[3]}")

print("\n3. Fusion 层 (特征融合):")
for i in range(4):
    conv1_key = f'model_task.fusions.{i}.res_conv1.conv1.weight'
    if conv1_key in sd:
        w = sd[conv1_key]
        print(f"   - Fusion {i}: 输入通道={w.shape[1]}, 输出通道={w.shape[0]}")

print("\n4. Probe 层 (输出头):")
if 'model_task.probe.dispconv.conv.weight' in sd:
    disp_w = sd['model_task.probe.dispconv.conv.weight']
    print(f"   - Dispconv (深度/位移输出): 输入通道={disp_w.shape[1]}, 输出通道={disp_w.shape[0]}")

if 'model_task.probe.shear_mlp.2.conv.weight' in sd:
    shear_w = sd['model_task.probe.shear_mlp.2.conv.weight']
    print(f"   - Shear MLP (剪切力输出): 输出通道={shear_w.shape[0]}")

print("\n" + "=" * 80)
print("推断:")
print("=" * 80)
print("这是一个触觉图像处理模型，可能基于 DPT (Dense Prediction Transformer)")
print("- 输入: 触觉传感器图像 (需要查看第一层卷积的输入通道数)")
print("- 输出: 深度图 (dispconv) 和剪切力 (shear_mlp)")
print("- 建议图像尺寸: 224x224 或 384x384 (Transformer 常用尺寸)")