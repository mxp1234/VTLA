"""
Tactile Feature Extractor using Sparsh Pretrained Models for TacEx Factory Environment
使用Sparsh预训练模型提取触觉特征（ForceField版本）
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add tactile encoder path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tactile_encoder import TactileForceFieldModel


class SparshTactileEncoder(nn.Module):
    """
    使用Sparsh预训练ViT模型提取触觉特征
    从32x32触觉图像提取特征向量
    """
    def __init__(
        self,
        checkpoint_path=None,
        vit_model='base',
        input_size=32,
        output_dim=256,
        freeze_encoder=True,
        upsample_size=224,
    ):
        """
        Args:
            checkpoint_path: Path to pretrained checkpoint (.pth or .ckpt)
            vit_model: 'tiny', 'small', 'base', 'large'
            input_size: Input image size (32 for GelSightMini)
            output_dim: Output feature dimension
            freeze_encoder: Whether to freeze ViT encoder weights
            upsample_size: Size to upsample input before feeding to ViT (224 for standard ViT)
        """
        super().__init__()

        self.input_size = input_size
        self.upsample_size = upsample_size
        self.freeze_encoder = freeze_encoder

        # 上采样层：32x32 -> 224x224
        self.upsample = nn.Upsample(
            size=(upsample_size, upsample_size),
            mode='bilinear',
            align_corners=False
        )

        # 创建ViT encoder（根据配置文件，使用vit_base）
        print(f"[SparshTactileEncoder] Creating ViT-{vit_model} encoder...")
        if vit_model == 'base':
            self.encoder = vit_base(
                img_size=(upsample_size, upsample_size),
                patch_size=16,
                in_chans=3,  # RGB输入
                pos_embed_fn='learned',  # 使用learned positional embedding
            )
        else:
            raise NotImplementedError(f"ViT model {vit_model} not implemented yet")

        # 获取ViT的输出维度
        self.vit_embed_dim = VIT_EMBED_DIMS[f'vit_{vit_model}']  # 768 for base

        # 加载预训练权重（如果提供）
        if checkpoint_path is not None:
            self._load_pretrained_weights(checkpoint_path)

        # 冻结encoder参数（如果需要）
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"[SparshTactileEncoder] Encoder weights frozen")

        # 全局平均池化 + 投影到输出维度
        # ViT输出: (batch, num_patches, embed_dim)
        # 我们对所有patch tokens做平均池化，然后投影到output_dim
        self.projection = nn.Sequential(
            nn.Linear(self.vit_embed_dim, output_dim),
            nn.ReLU(),
        )

        print(f"[SparshTactileEncoder] Initialized:")
        print(f"  - Input size: {input_size}x{input_size}")
        print(f"  - Upsample size: {upsample_size}x{upsample_size}")
        print(f"  - ViT embed dim: {self.vit_embed_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Encoder frozen: {freeze_encoder}")

    def _load_pretrained_weights(self, checkpoint_path):
        """加载预训练权重"""
        print(f"[SparshTactileEncoder] Loading checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"[SparshTactileEncoder] WARNING: Checkpoint not found at {checkpoint_path}")
            return

        try:
            # 尝试加载checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 检查checkpoint结构
            if 'model_encoder' in checkpoint:
                # Sparsh的checkpoint格式
                encoder_state = checkpoint['model_encoder']
                print(f"[SparshTactileEncoder] Found 'model_encoder' in checkpoint")
            elif 'state_dict' in checkpoint:
                # Lightning checkpoint格式
                encoder_state = {}
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('model_encoder.'):
                        new_key = k.replace('model_encoder.', '')
                        encoder_state[new_key] = v
                print(f"[SparshTactileEncoder] Extracted encoder from 'state_dict'")
            else:
                # 直接是state_dict
                encoder_state = checkpoint
                print(f"[SparshTactileEncoder] Using checkpoint as state_dict directly")

            # 加载权重（忽略不匹配的键）
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                encoder_state, strict=False
            )

            if missing_keys:
                print(f"[SparshTactileEncoder] Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"[SparshTactileEncoder] Unexpected keys: {len(unexpected_keys)}")

            print(f"[SparshTactileEncoder] Successfully loaded pretrained weights")

        except Exception as e:
            print(f"[SparshTactileEncoder] Error loading checkpoint: {e}")
            print(f"[SparshTactileEncoder] Will use random initialization")

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 32, 32) - GelSightMini RGB images, normalized to [0, 1]
        Returns:
            features: (batch, output_dim) - Encoded features
        """
        # 上采样到ViT输入大小: (batch, 3, 32, 32) -> (batch, 3, 224, 224)
        x_upsampled = self.upsample(x)

        # 通过ViT encoder: (batch, 3, 224, 224) -> (batch, num_patches, embed_dim)
        # ViT forward返回patch tokens (不包括register tokens)
        with torch.set_grad_enabled(not self.freeze_encoder):
            vit_output = self.encoder(x_upsampled)  # (batch, num_patches, 768)

        # 全局平均池化: (batch, num_patches, 768) -> (batch, 768)
        pooled_features = vit_output.mean(dim=1)

        # 投影到输出维度: (batch, 768) -> (batch, output_dim)
        features = self.projection(pooled_features)

        return features


class DualSensorSparshEncoder(nn.Module):
    """
    融合左右两个GelSightMini传感器的特征
    每个传感器独立使用SparshTactileEncoder提取特征，然后融合
    """
    def __init__(
        self,
        checkpoint_path=None,
        single_encoder_dim=256,
        fusion_dim=512,
        freeze_encoder=True,
    ):
        super().__init__()

        # 为左右传感器创建独立的encoder
        # 注意: 实际应用中可以共享encoder权重，这里为了灵活性分开
        self.left_encoder = SparshTactileEncoder(
            checkpoint_path=checkpoint_path,
            output_dim=single_encoder_dim,
            freeze_encoder=freeze_encoder,
        )

        self.right_encoder = SparshTactileEncoder(
            checkpoint_path=checkpoint_path,
            output_dim=single_encoder_dim,
            freeze_encoder=freeze_encoder,
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(single_encoder_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        print(f"[DualSensorSparshEncoder] Initialized with fusion_dim={fusion_dim}")

    def forward(self, left_img, right_img):
        """
        Args:
            left_img: (batch, 3, 32, 32)
            right_img: (batch, 3, 32, 32)
        Returns:
            fused_features: (batch, fusion_dim)
        """
        left_feat = self.left_encoder(left_img)
        right_feat = self.right_encoder(right_img)

        # 拼接并融合
        combined = torch.cat([left_feat, right_feat], dim=1)
        fused = self.fusion(combined)

        return fused


class TactileForceFieldExtractor(nn.Module):
    """
    使用Sparsh预训练模型提取触觉force field特征（求和池化版本）

    输入: 左右触觉传感器图像 (224x224x3 each)
    输出: force field表征 (normal + shear的求和)
    """

    def __init__(
        self,
        checkpoint_path,
        device='cuda',
        freeze_model=True,
    ):
        """
        Args:
            checkpoint_path: 预训练模型路径 (last.ckpt)
            device: 'cuda' or 'cpu'
            freeze_model: 是否冻结模型参数
        """
        super().__init__()

        self.device = device

        # 加载预训练模型
        print(f"[TactileForceFieldExtractor] Loading checkpoint from: {checkpoint_path}")
        self.model = TactileForceFieldModel(
            img_size=(224, 224),
            in_chans=6,  # 双指拼接: left(3) + right(3)
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_register_tokens=1,
            hooks=[2, 5, 8, 11],
        )

        self.model.load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        # 冻结所有参数
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[TactileForceFieldExtractor] Model weights frozen")

        # 输出维度: 3个值 (normal_sum, shear_x_sum, shear_y_sum)
        # 注意：模型输入是6通道(left+right拼接)，输出是combined force field
        self.output_dim = 3

        print(f"[TactileForceFieldExtractor] Initialized:")
        print(f"  - Pooling method: sum")
        print(f"  - Output dimension: {self.output_dim}")
        print(f"  - Note: Force field is computed from combined left+right tactile input")

    def forward(self, tactile_left, tactile_right):
        """
        处理触觉图像，提取force field特征

        Args:
            tactile_left: (B, H, W, 3) 左指触觉图像，范围[0, 1]
            tactile_right: (B, H, W, 3) 右指触觉图像，范围[0, 1]

        Returns:
            features: (B, 3) 提取的force field特征
                [normal_sum, shear_x_sum, shear_y_sum]
                这是从combined left+right输入计算出的force field
        """
        # 准备输入: (B, H, W, 3) -> (B, 6, H, W)
        tactile_input = self._prepare_input(tactile_left, tactile_right)

        with torch.no_grad():
            # 获取force field (对于拼接的6通道输入)
            outputs = self.model(tactile_input, mode='normal_shear')

            # 提取特征（求和池化）
            features = self._sum_pool_force_field(outputs)

        return features

    def _prepare_input(self, tactile_left, tactile_right):
        """
        准备模型输入

        Args:
            tactile_left: (B, H, W, 3)
            tactile_right: (B, H, W, 3)

        Returns:
            input: (B, 6, H, W)
        """
        # 转换为CHW格式
        left = tactile_left.permute(0, 3, 1, 2)   # (B, 3, H, W)
        right = tactile_right.permute(0, 3, 1, 2)  # (B, 3, H, W)

        # 拼接
        tactile_input = torch.cat([left, right], dim=1)  # (B, 6, H, W)

        return tactile_input

    def _sum_pool_force_field(self, outputs):
        """
        对force field进行求和池化

        Args:
            outputs: dict with 'normal' (B, 1, 224, 224) and 'shear' (B, 2, 224, 224)

        Returns:
            features: (B, 3) [normal_sum, shear_x_sum, shear_y_sum]
        """
        normal = outputs['normal']  # (B, 1, 224, 224)
        shear = outputs['shear']    # (B, 2, 224, 224)

        # 求和池化
        normal_sum = normal.sum(dim=[2, 3])  # (B, 1)
        shear_sum = shear.sum(dim=[2, 3])     # (B, 2)

        features = torch.cat([normal_sum, shear_sum], dim=1)  # (B, 3)

        return features

    def get_output_dim(self):
        """获取输出维度"""
        return self.output_dim


def create_tactile_encoder(encoder_type='sparsh', **kwargs):
    """
    Args:
        encoder_type: 'sparsh' | 'dual_sparsh' | 'force_field'
        **kwargs:
    """
    if encoder_type == 'sparsh':
        return SparshTactileEncoder(**kwargs)
    elif encoder_type == 'dual_sparsh':
        return DualSensorSparshEncoder(**kwargs)
    elif encoder_type == 'force_field':
        return TactileForceFieldExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# =============== 测试代码 ===============
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试ForceField特征提取器
    print("\n=== Testing TactileForceFieldExtractor ===")
    checkpoint_path = "/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt"

    force_field_extractor = create_tactile_encoder(
        encoder_type='force_field',
        checkpoint_path=checkpoint_path,
        device=device,
        freeze_model=True
    )

    # 测试输入 (224x224 as configured in factory_env_cfg.py)
    batch_size = 4
    tactile_left = torch.randn(batch_size, 224, 224, 3).to(device)
    tactile_right = torch.randn(batch_size, 224, 224, 3).to(device)

    features = force_field_extractor(tactile_left, tactile_right)
    print(f"\nInput shapes:")
    print(f"  - tactile_left: {tactile_left.shape}")
    print(f"  - tactile_right: {tactile_right.shape}")
    print(f"\nOutput:")
    print(f"  - features: {features.shape}")
    print(f"  - output_dim: {force_field_extractor.get_output_dim()}")
    print(f"  - features sample: {features[0]}")

    print("\n✓ All tests passed!")

