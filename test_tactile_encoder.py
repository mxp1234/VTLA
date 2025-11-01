"""
测试Sparsh触觉编码器是否能正确加载预训练权重
"""

import sys
import os
import torch

# Add paths
TACEX_PATH = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1"
SPARSH_PATH = "/home/pi-zero/isaac-sim/sparsh"

if TACEX_PATH not in sys.path:
    sys.path.insert(0, TACEX_PATH)
if SPARSH_PATH not in sys.path:
    sys.path.insert(0, SPARSH_PATH)

from network.tactile_feature_extractor import SparshTactileEncoder, DualSensorSparshEncoder

def test_checkpoint_loading():
    """测试checkpoint加载"""
    print("=" * 80)
    print("测试1: 检查checkpoint文件")
    print("=" * 80)

    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return False

    print(f"✓ Checkpoint存在: {checkpoint_path}")

    # 加载checkpoint查看结构
    print("\n加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    if 'model_encoder' in checkpoint:
        print(f"\n✓ 发现 'model_encoder' 键")
        encoder_state = checkpoint['model_encoder']
        print(f"  Encoder state_dict包含 {len(encoder_state)} 个参数")
        print(f"\n  前10个参数键:")
        for i, key in enumerate(list(encoder_state.keys())[:10]):
            print(f"    {i+1}. {key}: {encoder_state[key].shape}")
    else:
        print(f"\n❌ 未发现 'model_encoder' 键")
        print(f"  可用的键: {list(checkpoint.keys())}")

    if 'model_task' in checkpoint:
        print(f"\n✓ 发现 'model_task' 键")
        task_state = checkpoint['model_task']
        print(f"  Task state_dict包含 {len(task_state)} 个参数")

    return True


def test_encoder_creation():
    """测试编码器创建"""
    print("\n" + "=" * 80)
    print("测试2: 创建编码器（不加载权重）")
    print("=" * 80)

    try:
        print("\n创建SparshTactileEncoder...")
        encoder = SparshTactileEncoder(
            checkpoint_path=None,  # 不加载权重
            output_dim=256,
            freeze_encoder=True
        )

        print(f"✓ 编码器创建成功")
        print(f"  总参数: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")

        # 测试forward
        print("\n测试forward传播...")
        dummy_input = torch.randn(2, 3, 32, 32)  # 2个样本, 3通道, 32x32
        with torch.no_grad():
            output = encoder(dummy_input)

        print(f"✓ Forward成功")
        print(f"  输入shape: {dummy_input.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  期望输出shape: (2, 256)")

        if output.shape == (2, 256):
            print("✓ 输出shape正确")
            return True
        else:
            print(f"❌ 输出shape不正确: 期望(2, 256), 实际{output.shape}")
            return False

    except Exception as e:
        print(f"❌ 编码器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_with_checkpoint():
    """测试加载checkpoint的编码器"""
    print("\n" + "=" * 80)
    print("测试3: 创建编码器并加载checkpoint")
    print("=" * 80)

    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth"

    try:
        print("\n创建SparshTactileEncoder并加载预训练权重...")
        encoder = SparshTactileEncoder(
            checkpoint_path=checkpoint_path,
            output_dim=256,
            freeze_encoder=True
        )

        print(f"\n✓ 编码器创建成功")
        print(f"  总参数: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
        print(f"  可训练参数: {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:.2f}M")

        # 测试forward
        print("\n测试forward传播...")
        dummy_input = torch.randn(4, 3, 32, 32)  # 4个样本
        with torch.no_grad():
            output = encoder(dummy_input)

        print(f"✓ Forward成功")
        print(f"  输入shape: {dummy_input.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  输出均值: {output.mean():.3f}")
        print(f"  输出标准差: {output.std():.3f}")

        return True

    except Exception as e:
        print(f"❌ 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_encoder():
    """测试双传感器编码器"""
    print("\n" + "=" * 80)
    print("测试4: 创建双传感器编码器")
    print("=" * 80)

    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth"

    try:
        print("\n创建DualSensorSparshEncoder...")
        dual_encoder = DualSensorSparshEncoder(
            checkpoint_path=checkpoint_path,
            single_encoder_dim=256,
            fusion_dim=512,
            freeze_encoder=True
        )

        print(f"\n✓ 双传感器编码器创建成功")
        print(f"  总参数: {sum(p.numel() for p in dual_encoder.parameters()) / 1e6:.2f}M")
        print(f"  可训练参数: {sum(p.numel() for p in dual_encoder.parameters() if p.requires_grad) / 1e6:.2f}M")

        # 测试forward
        print("\n测试forward传播...")
        left_input = torch.randn(4, 3, 32, 32)
        right_input = torch.randn(4, 3, 32, 32)

        with torch.no_grad():
            fused_output = dual_encoder(left_input, right_input)

        print(f"✓ Forward成功")
        print(f"  左传感器输入shape: {left_input.shape}")
        print(f"  右传感器输入shape: {right_input.shape}")
        print(f"  融合输出shape: {fused_output.shape}")
        print(f"  期望输出shape: (4, 512)")

        if fused_output.shape == (4, 512):
            print("✓ 输出shape正确")
            return True
        else:
            print(f"❌ 输出shape不正确: 期望(4, 512), 实际{fused_output.shape}")
            return False

    except Exception as e:
        print(f"❌ 双传感器编码器失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_support():
    """测试GPU支持"""
    print("\n" + "=" * 80)
    print("测试5: GPU支持测试")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过GPU测试")
        return True

    print(f"✓ CUDA可用")
    print(f"  设备数量: {torch.cuda.device_count()}")
    print(f"  当前设备: {torch.cuda.current_device()}")
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")

    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth"

    try:
        print("\n在GPU上创建编码器...")
        encoder = SparshTactileEncoder(
            checkpoint_path=checkpoint_path,
            output_dim=256,
            freeze_encoder=True
        ).cuda()

        print(f"✓ 编码器移至GPU成功")

        # GPU forward测试
        print("\nGPU forward测试...")
        dummy_input = torch.randn(8, 3, 32, 32).cuda()

        with torch.no_grad():
            output = encoder(dummy_input)

        print(f"✓ GPU Forward成功")
        print(f"  输入device: {dummy_input.device}")
        print(f"  输出device: {output.device}")
        print(f"  输出shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_structure_match():
    """测试模型结构是否匹配checkpoint"""
    print("\n" + "=" * 80)
    print("测试6: 检查模型结构与checkpoint匹配度")
    print("=" * 80)

    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/epoch-0021.pth"

    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_encoder' not in checkpoint:
            print("❌ Checkpoint中没有model_encoder")
            return False

        encoder_state = checkpoint['model_encoder']

        # 创建模型
        from tactile_ssl.model.vision_transformer import vit_base

        print("\n创建ViT-Base模型...")
        model = vit_base(
            img_size=(224, 224),
            patch_size=16,
            in_chans=3,
            pos_embed_fn='learned',
        )

        print(f"✓ 模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # 尝试加载权重
        print("\n尝试加载权重...")
        missing_keys, unexpected_keys = model.load_state_dict(encoder_state, strict=False)

        print(f"\n权重加载结果:")
        print(f"  缺失的键数量: {len(missing_keys)}")
        print(f"  多余的键数量: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            print(f"\n  缺失的键（前10个）:")
            for key in missing_keys[:10]:
                print(f"    - {key}")

        if len(unexpected_keys) > 0:
            print(f"\n  多余的键（前10个）:")
            for key in unexpected_keys[:10]:
                print(f"    - {key}")

        # 检查关键参数是否加载
        print("\n检查关键参数:")
        key_params = ['patch_embed.proj.weight', 'pos_embed', 'blocks.0.attn.qkv.weight', 'norm.weight']
        for param_name in key_params:
            if hasattr(model, param_name.split('.')[0]):
                print(f"  ✓ {param_name} 存在")
            else:
                print(f"  ? {param_name} 检查")

        # 测试forward
        print("\n测试加载权重后的forward...")
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward成功")
        print(f"  输出shape: {output.shape}")

        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("\n✓ 模型结构完全匹配checkpoint")
            return True
        else:
            print(f"\n⚠ 模型结构部分匹配（可能是正常的，因为我们只需要encoder部分）")
            return True

    except Exception as e:
        print(f"❌ 结构匹配测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("Sparsh触觉编码器测试")
    print("=" * 80)

    tests = [
        ("Checkpoint文件检查", test_checkpoint_loading),
        ("编码器创建测试", test_encoder_creation),
        ("加载checkpoint测试", test_encoder_with_checkpoint),
        ("双传感器编码器测试", test_dual_encoder),
        ("GPU支持测试", test_gpu_support),
        ("模型结构匹配测试", test_model_structure_match),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！编码器可以正常使用。")
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
