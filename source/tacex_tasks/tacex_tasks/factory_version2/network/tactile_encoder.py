"""
触觉编码器架构 - 独立版本
基于Sparsh的ForceField模型，适配到TacEx环境

包含:
1. Vision Transformer Encoder (ViT-Base)
2. ForceField Decoder (DPT-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


# ============================================================================
# Vision Transformer Components
# ============================================================================

class PatchEmbed(nn.Module):
    """2D图像转换为patches"""
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class SinusoidalEmbed(nn.Module):
    """正弦位置编码"""
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        H, W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        num_patches = H * W

        # 创建位置编码
        pe = torch.zeros(num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, device):
        return self.pe.to(device)


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP层"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_bias=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer - Base版本"""
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embedding (sinusoidal)
        self.pos_embed = SinusoidalEmbed(img_size, [patch_size, patch_size], embed_dim)

        # Register tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add position encoding
        pos_encoding = self.pos_embed(x.device).unsqueeze(0)
        x = x + pos_encoding

        # Add register tokens
        if self.register_tokens is not None:
            x = torch.cat([
                self.register_tokens.expand(x.shape[0], -1, -1),
                x
            ], dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Return only patch tokens (skip register tokens)
        if self.register_tokens is not None:
            return x[:, self.num_register_tokens:, :]
        return x


# ============================================================================
# ForceField Decoder Components
# ============================================================================

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        return F.relu(self.conv(x))


class ResidualConvUnit(nn.Module):
    """残差卷积单元"""
    def __init__(self, features):
        super().__init__()
        self.conv1 = ConvBlock(features, features)
        self.conv2 = ConvBlock(features, features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class Fusion(nn.Module):
    """特征融合层"""
    def __init__(self, features):
        super().__init__()
        self.res_conv1 = ResidualConvUnit(features)
        self.res_conv2 = ResidualConvUnit(features)

    def forward(self, x, skip=None):
        # 如果没有skip，创建零tensor
        if skip is None:
            skip = torch.zeros_like(x)

        # ResConv1
        output = self.res_conv1(x)

        # 加上skip connection
        output = output + skip

        # ResConv2
        output = self.res_conv2(output)

        # 2倍上采样
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output


class Reassemble(nn.Module):
    """重组特征层"""
    def __init__(self, image_size, patch_size, scale, embed_dim, out_dim):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.image_size = image_size

        # 1x1卷积降维
        self.conv1 = nn.Conv2d(embed_dim, out_dim, 1)

        # 根据scale上采样或下采样
        # scale=4: 上采样4倍, scale=8: 上采样2倍, scale=16: 不变, scale=32: 下采样2倍
        if scale == 4:
            self.conv2 = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=4, stride=4)
        elif scale == 8:
            self.conv2 = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2)
        elif scale == 16:
            self.conv2 = nn.Identity()
        elif scale == 32:
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported scale: {scale}")

    def forward(self, x):
        # x: (B, N, C) -> (B, C, H, W)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # 降维
        x = self.conv1(x)

        # 上采样/下采样
        x = self.conv2(x)

        return x


class NormalShearHead(nn.Module):
    """Force field输出头"""
    def __init__(self, features=128):
        super().__init__()
        self.scale_flow = 20.0

        # Upconv layers
        self.upconv_0 = ConvBlock(features, features)
        self.upconv_1 = ConvBlock(features + features, features)  # concat input

        # Normal output
        self.dispconv = nn.Conv2d(features, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Shear output
        self.shear_mlp = nn.Sequential(
            ConvBlock(features, 64),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, mode='normal_shear'):
        # Upconv 0
        x0 = self.upconv_0(x)

        # Concat with input
        x_cat = torch.cat([x0, x], dim=1)

        # Upconv 1
        x1 = self.upconv_1(x_cat)

        # Upsample 2x
        x1 = F.interpolate(x1, scale_factor=2, mode='nearest')

        if mode in ['normal', 'normal_shear']:
            normal = self.sigmoid(self.dispconv(x1))

        if mode in ['shear', 'normal_shear']:
            shear = self.shear_mlp(x1) * self.scale_flow

        if mode == 'normal':
            return normal
        elif mode == 'shear':
            return shear
        else:  # normal_shear
            return torch.cat([normal, shear], dim=1)


class ForceFieldDecoder(nn.Module):
    """ForceField解码器"""
    def __init__(
        self,
        image_size=(3, 224, 224),
        embed_dim=768,
        patch_size=16,
        resample_dim=128,
        hooks=[2, 5, 8, 11],
        reassemble_s=[4, 8, 16, 32],
    ):
        super().__init__()
        self.hooks = hooks
        self.n_patches = (image_size[1] // patch_size) ** 2

        self.norm = nn.LayerNorm(embed_dim)

        # Reassemble layers
        self.reassembles = nn.ModuleList([
            Reassemble(image_size, patch_size, s, embed_dim, resample_dim)
            for s in reassemble_s
        ])

        # Fusion layers
        self.fusions = nn.ModuleList([
            Fusion(resample_dim) for _ in range(len(hooks))
        ])

        # Output head
        self.probe = NormalShearHead(features=resample_dim)

    def forward(self, encoder_activations, mode='normal_shear'):
        # Normalize activations
        sample_key = list(encoder_activations.keys())[0]
        start_idx = encoder_activations[sample_key].shape[1] - self.n_patches

        for b in encoder_activations.keys():
            encoder_activations[b] = self.norm(encoder_activations[b][:, start_idx:, :])

        # Progressive fusion
        previous_stage = None
        for i in range(len(self.fusions) - 1, -1, -1):
            hook_to_take = "t" + str(self.hooks[i])
            activation_result = encoder_activations[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        # Output
        y = self.probe(previous_stage, mode)

        outputs = {}
        if mode == 'normal_shear':
            outputs['normal'] = y[:, 0, :, :].unsqueeze(1)
            outputs['shear'] = y[:, 1:, :, :]
        elif mode == 'normal':
            outputs['normal'] = y
        elif mode == 'shear':
            outputs['shear'] = y
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return outputs


# ============================================================================
# 完整模型
# ============================================================================

class TactileForceFieldModel(nn.Module):
    """完整的触觉力场预测模型"""
    def __init__(
        self,
        img_size=(224, 224),
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_register_tokens=1,
        hooks=[2, 5, 8, 11],
    ):
        super().__init__()

        # Encoder
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_register_tokens=num_register_tokens,
        )

        # Decoder
        self.decoder = ForceFieldDecoder(
            image_size=(in_chans, img_size[0], img_size[1]),
            embed_dim=embed_dim,
            patch_size=16,
            resample_dim=128,
            hooks=hooks,
        )

        # Hook设置
        self.hooks = hooks
        self.encoder_activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册forward hooks"""
        def get_activation(name):
            def hook(model, input, output):
                self.encoder_activations[name] = output
            return hook

        for h in self.hooks:
            self.encoder.blocks[h].register_forward_hook(
                get_activation("t" + str(h))
            )

    def forward(self, x, mode='normal_shear'):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            mode: 'normal', 'shear', 或 'normal_shear'

        Returns:
            dict with 'normal' and/or 'shear' keys
        """
        # Encoder
        _ = self.encoder(x)

        # Decoder
        outputs = self.decoder(self.encoder_activations, mode)

        return outputs

    def load_checkpoint(self, ckpt_path):
        """加载checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # 获取state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 分离encoder和decoder权重
        encoder_dict = {}
        decoder_dict = {}

        for key, value in state_dict.items():
            if key.startswith('model_encoder.'):
                new_key = key.replace('model_encoder.', '')
                encoder_dict[new_key] = value
            elif key.startswith('model_task.'):
                new_key = key.replace('model_task.', '')
                decoder_dict[new_key] = value

        # 加载权重
        if encoder_dict:
            self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"✓ Loaded encoder: {len(encoder_dict)} keys")

        if decoder_dict:
            self.decoder.load_state_dict(decoder_dict, strict=False)
            print(f"✓ Loaded decoder: {len(decoder_dict)} keys")

        return self


def create_model(checkpoint_path=None):
    """创建模型并加载权重"""
    model = TactileForceFieldModel(
        img_size=(224, 224),
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_register_tokens=1,
        hooks=[2, 5, 8, 11],
    )

    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)

    return model


if __name__ == "__main__":
    # 测试模型
    print("创建模型...")
    model = create_model()

    print(f"\n模型参数统计:")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = encoder_params + decoder_params

    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total:   {total_params:,}")

    # 测试前向传播
    print(f"\n测试前向传播...")
    x = torch.randn(2, 6, 224, 224)
    with torch.no_grad():
        outputs = model(x, mode='normal_shear')

    print(f"✓ 输入: {x.shape}")
    for key, value in outputs.items():
        print(f"✓ 输出 {key}: {value.shape}")
