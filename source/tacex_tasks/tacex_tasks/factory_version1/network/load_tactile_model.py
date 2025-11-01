"""
加载预训练的触觉编码器模型

使用方法:
python load_tactile_model.py
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from tactile_encoder import create_model, TactileForceFieldModel


def main():
    print("=" * 80)
    print("加载Sparsh预训练触觉编码器")
    print("=" * 80)

    # Checkpoint路径
    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt"

    print(f"\nCheckpoint: {checkpoint_path}")

    # 创建并加载模型
    print("\n1. 创建模型架构...")
    model = create_model(checkpoint_path)

    # 设置为评估模式
    model.eval()

    # 统计参数
    print(f"\n2. 模型参数统计:")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = encoder_params + decoder_params

    print(f"   Encoder: {encoder_params:,} 参数")
    print(f"   Decoder: {decoder_params:,} 参数")
    print(f"   Total:   {total_params:,} 参数")

    # 测试推理
    print(f"\n3. 测试推理...")
    print(f"   输入: (batch=2, channels=6, height=224, width=224)")

    x = torch.randn(2, 6, 224, 224)

    with torch.no_grad():
        outputs = model(x, mode='normal_shear')

    print(f"\n   输出:")
    for key, value in outputs.items():
        print(f"   - {key}: {value.shape}")
        print(f"     范围: [{value.min():.4f}, {value.max():.4f}]")

    # 保存简化版checkpoint
    # print(f"\n4. 保存简化版checkpoint...")
    # save_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/tactile_encoder_standalone.pth"

    # torch.save({
    #     'encoder': model.encoder.state_dict(),
    #     'decoder': model.decoder.state_dict(),
    #     'config': {
    #         'img_size': (224, 224),
    #         'in_chans': 6,
    #         'embed_dim': 768,
    #         'depth': 12,
    #         'num_heads': 12,
    #         'num_register_tokens': 1,
    #         'hooks': [2, 5, 8, 11],
    #     }
    # }, save_path)

    # print(f"   ✓ 已保存到: {save_path}")

    print("\n" + "=" * 80)
    print("加载成功!")
    print("=" * 80)
    print("""
下一步:
1. 使用模型进行推理:
   ```python
   from tactile_encoder import create_model
   model = create_model('tactile_encoder_standalone.pth')
   model.eval()
   outputs = model(tactile_images)
   ```

2. 提取特征编码器:
   ```python
   encoder = model.encoder
   features = encoder(tactile_images)
   ```

3. 单独使用decoder:
   ```python
   decoder = model.decoder
   outputs = decoder(encoder_activations)
   ```
    """)

    return model


if __name__ == "__main__":
    model = main()
