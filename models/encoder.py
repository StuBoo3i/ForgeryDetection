import torch
import torch.nn as nn
import timm
from config.config import cfg

class ChannelAttention(nn.Module):
    """通道注意力模块，用于多尺度特征加权融合"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DINOEncoder(nn.Module):
    """DINO预训练ViT编码器，取前4个Block，冻结权重，多尺度特征融合"""
    def __init__(self):
        super().__init__()
        # 加载DINO预训练ViT-B/16
        self.vit = timm.create_model(
            cfg.vit_model_name,
            pretrained=True,
            num_classes=0,
            img_size=cfg.img_size
        )
        # 仅保留前4个Transformer Block
        self.patch_embed = self.vit.patch_embed
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = nn.ModuleList([
            self.vit.blocks[i] for i in cfg.vit_extract_layers
        ])
        self.norm = self.vit.norm
        
        # 多尺度特征融合模块
        self.num_layers = len(cfg.vit_extract_layers)
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(cfg.vit_embed_dim) for _ in range(self.num_layers)
        ])
        self.fusion_conv = nn.Conv2d(
            cfg.vit_embed_dim * self.num_layers,
            cfg.vit_embed_dim,
            kernel_size=1,
            stride=1
        )
        
        # 全程冻结编码器权重
        if cfg.vit_freeze:
            self._freeze_weights()
        
        self.to(cfg.device)

    def _freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        """
        前向传播
        :param x: 输入图像 [B, 3, 224, 224]
        :return z: 融合后的隐表示 [B, 768, 14, 14]
        """
        # ViT Patch Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 提取前4个Block的输出
        layer_outputs = []
        for blk in self.blocks:
            x = blk(x)
            layer_outputs.append(x)
        
        # 多尺度特征融合：通道注意力加权 + concat + 维度映射
        fused_features = []
        for feat, attn in zip(layer_outputs, self.channel_attentions):
            # 转换为空间格式 [B, 196, 768] -> [B, 768, 14, 14]
            b, n, c = feat.shape
            h = w = int(n ** 0.5)
            feat_spatial = feat.permute(0, 2, 1).reshape(b, c, h, w)
            # 通道注意力加权
            weighted_feat = attn(feat_spatial)
            fused_features.append(weighted_feat)
        
        # 拼接所有层特征
        concat_feat = torch.cat(fused_features, dim=1)
        # 映射到768通道
        z = self.fusion_conv(concat_feat)
        
        return z