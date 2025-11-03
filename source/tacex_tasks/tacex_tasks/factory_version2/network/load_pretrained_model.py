"""
加载Sparsh预训练模型示例

文件说明:
1. last.ckpt - 完整训练checkpoint (包含encoder+decoder+训练状态)
2. epoch-0021.pth - 只包含decoder权重

使用场景:
- 如果要继续训练: 使用 last.ckpt
- 如果只要推理: 使用 last.ckpt 或 epoch-0021.pth + 单独的encoder
"""

import torch
import sys
sys.path.append('/home/pi-zero/isaac-sim/sparsh')

from tactile_ssl.downstream_task.forcefield_sl import ForceFieldModule, ForceFieldDecoder
from tactile_ssl.model.vision_transformer import vit_base
from functools import partial


def method1_load_complete_checkpoint():
    """
    方法1: 加载完整checkpoint (推荐)

    last.ckpt包含完整的模型权重，可以直接加载
    """
    print("=" * 80)
    print("方法1: 加载完整checkpoint (last.ckpt)")
    print("=" * 80)

    # 创建encoder
    encoder = vit_base(
        img_size=(224, 224),
        in_chans=6,
        pos_embed_fn='sinusoidal',
        num_register_tokens=1
    )

    # 创建decoder
    decoder = ForceFieldDecoder(
        image_size=(3, 224, 224),
        embed_dim='base',
        patch_size=16,
        resample_dim=128,
        hooks=[2, 5, 8, 11],
        reassemble_s=[4, 8, 16, 32]
    )

    # SSL配置
    ssl_config = {
        'img_sz': [224, 224],
        'pose_estimator': {'num_encoder_layers': 18},
        'loss': {
            'with_mask_supervision': False,
            'with_sl_supervision': False,
            'with_ssim': True,
            'disparity_smoothness': 1e-3,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }

    # 创建完整模型
    model = ForceFieldModule(
        model_encoder=encoder,
        model_task=decoder,
        optim_cfg=partial(torch.optim.Adam, lr=0.0001),
        scheduler_cfg=None,
        checkpoint_encoder=None,  # 不需要单独的encoder checkpoint
        checkpoint_task="/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt",
        train_encoder=False,
        encoder_type='dino',
        ssl_config=ssl_config
    )

    print("\n✓ 模型加载完成!")
    print("  - Encoder权重: 从last.ckpt加载")
    print("  - Decoder权重: 从last.ckpt加载")

    return model


def method2_load_separately():
    """
    方法2: 分别加载encoder和decoder

    如果你想使用不同的encoder和decoder checkpoint
    """
    print("\n" + "=" * 80)
    print("方法2: 分别加载encoder和decoder")
    print("=" * 80)

    # 加载last.ckpt获取encoder权重
    print("\n1. 从last.ckpt提取encoder权重...")
    last_ckpt = torch.load(
        '/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt',
        map_location='cpu'
    )

    # 提取encoder权重
    encoder_state_dict = {
        key.replace('model_encoder.', ''): value
        for key, value in last_ckpt['model'].items()
        if key.startswith('model_encoder.')
    }

    # 创建encoder并加载权重
    encoder = vit_base(
        img_size=(224, 224),
        in_chans=6,
        pos_embed_fn='sinusoidal',
        num_register_tokens=1
    )
    encoder.load_state_dict(encoder_state_dict, strict=False)
    print(f"   ✓ Encoder加载完成 ({len(encoder_state_dict)} keys)")

    # 加载epoch-0021.pth获取decoder权重
    print("\n2. 从epoch-0021.pth加载decoder权重...")
    epoch_ckpt = torch.load(
        '/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth',
        map_location='cpu'
    )

    # 提取decoder权重
    decoder_state_dict = {
        key.replace('model_task.', ''): value
        for key, value in epoch_ckpt.items()
    }

    # 创建decoder并加载权重
    decoder = ForceFieldDecoder(
        image_size=(3, 224, 224),
        embed_dim='base',
        patch_size=16,
        resample_dim=128,
        hooks=[2, 5, 8, 11],
        reassemble_s=[4, 8, 16, 32]
    )
    decoder.load_state_dict(decoder_state_dict, strict=False)
    print(f"   ✓ Decoder加载完成 ({len(decoder_state_dict)} keys)")

    # 组合成完整模型
    ssl_config = {
        'img_sz': [224, 224],
        'pose_estimator': {'num_encoder_layers': 18},
        'loss': {
            'with_mask_supervision': False,
            'with_sl_supervision': False,
            'with_ssim': True,
            'disparity_smoothness': 1e-3,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }

    model = ForceFieldModule(
        model_encoder=encoder,
        model_task=decoder,
        optim_cfg=partial(torch.optim.Adam, lr=0.0001),
        scheduler_cfg=None,
        checkpoint_encoder=None,
        checkpoint_task=None,  # 已经手动加载
        train_encoder=False,
        encoder_type='dino',
        ssl_config=ssl_config
    )

    print("\n✓ 模型组合完成!")

    return model


def method3_inference_only():
    """
    方法3: 仅推理模式

    直接加载权重到模型进行推理
    """
    print("\n" + "=" * 80)
    print("方法3: 推理模式 (最简单)")
    print("=" * 80)

    # 创建encoder和decoder
    encoder = vit_base(
        img_size=(224, 224),
        in_chans=6,
        pos_embed_fn='sinusoidal',
        num_register_tokens=1
    )

    decoder = ForceFieldDecoder(
        image_size=(3, 224, 224),
        embed_dim='base',
        patch_size=16,
        resample_dim=128,
        hooks=[2, 5, 8, 11],
        reassemble_s=[4, 8, 16, 32]
    )

    # 加载last.ckpt
    print("\n加载checkpoint...")
    checkpoint = torch.load(
        '/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt',
        map_location='cpu'
    )

    # 提取并加载encoder权重
    encoder_state_dict = {
        key.replace('model_encoder.', ''): value
        for key, value in checkpoint['model'].items()
        if key.startswith('model_encoder.')
    }
    encoder.load_state_dict(encoder_state_dict, strict=False)

    # 提取并加载decoder权重
    decoder_state_dict = {
        key.replace('model_task.', ''): value
        for key, value in checkpoint['model'].items()
        if key.startswith('model_task.')
    }
    decoder.load_state_dict(decoder_state_dict, strict=False)

    # 设置为评估模式
    encoder.eval()
    decoder.eval()

    print("✓ 模型加载完成，已设置为评估模式")
    print(f"  - Encoder参数: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  - Decoder参数: {sum(p.numel() for p in decoder.parameters()):,}")

    return encoder, decoder


def test_inference(encoder, decoder):
    """
    测试推理
    """
    print("\n" + "=" * 80)
    print("测试推理")
    print("=" * 80)

    # 创建随机输入 (batch_size=2, channels=6, height=224, width=224)
    x = torch.randn(2, 6, 224, 224)

    print(f"\n输入形状: {x.shape}")

    # 设置hooks来捕获中间特征
    encoder_activations = {}
    hooks = [2, 5, 8, 11]

    def get_activation(name):
        def hook(model, input, output):
            encoder_activations[name] = output
        return hook

    for h in hooks:
        encoder.blocks[h].register_forward_hook(get_activation(f"t{h}"))

    # 前向传播
    print("\n执行推理...")
    with torch.no_grad():
        # Encoder
        z = encoder(x)
        print(f"✓ Encoder输出形状: {z.shape}")

        # Decoder
        outputs = decoder(encoder_activations, mode='normal_shear')
        print(f"✓ Decoder输出:")
        for key, value in outputs.items():
            print(f"  - {key}: {value.shape}")

    print("\n✓ 推理测试成功!")


if __name__ == "__main__":
    print("\n╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "Sparsh模型加载示例" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝\n")

    # 选择一个方法
    print("可用方法:")
    print("1. 使用完整checkpoint (last.ckpt) - 推荐")
    print("2. 分别加载encoder和decoder")
    print("3. 仅推理模式 - 最简单\n")

    # 方法3: 推理模式 (最简单)
    encoder, decoder = method3_inference_only()

    # 测试推理
    test_inference(encoder, decoder)

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
你的checkpoint文件:
1. last.ckpt - 完整checkpoint
   ├─ model_encoder.* (174 keys) - ViT-Base encoder
   ├─ model_task.* (57 keys) - ForceField decoder
   └─ 训练状态 (optim, scheduler, etc.)

2. epoch-0021.pth - 只有decoder
   └─ model_task.* (57 keys) - ForceField decoder

建议:
- 推理: 使用last.ckpt (包含完整模型)
- 微调: 使用last.ckpt作为起点
- 替换decoder: 使用last.ckpt的encoder + epoch-0021.pth的decoder

代码位置:
- 加载逻辑: /home/pi-zero/isaac-sim/sparsh/tactile_ssl/downstream_task/sl_module.py
  - load_encoder() (第81-103行)
  - load_task() (第53-79行)
    """)
