import torch
import torch.nn as nn
from config.config import cfg

class CrossAttention(nn.Module):
    """双向交叉注意力模块"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        """
        交叉注意力前向
        :param query: 查询特征 [B, N, C]
        :param key_value: 键值特征 [B, N, C]
        :return out: 注意力输出 [B, N, C]
        """
        B, N, C = query.shape
        
        # 归一化
        query = self.norm(query)
        key_value = self.norm(key_value)
        
        # 投影并分头
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 输出融合
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out + query  # 残差连接

class DualBranchAlignment(nn.Module):
    """双分支特征对齐与双向交叉注意力融合"""
    def __init__(self):
        super().__init__()
        # Δ残差分支：4层CNN + Patch Embedding
        self.delta_branch = nn.Sequential(
            # 前3层3×3卷积，步长2，下采样
            nn.Conv2d(cfg.in_channels, cfg.delta_branch_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.delta_branch_channels[0]),
            nn.GELU(),
            nn.Conv2d(cfg.delta_branch_channels[0], cfg.delta_branch_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.delta_branch_channels[1]),
            nn.GELU(),
            nn.Conv2d(cfg.delta_branch_channels[1], cfg.delta_branch_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.delta_branch_channels[2]),
            nn.GELU(),
            # Patch Embedding：映射到14×14×768
            nn.Conv2d(cfg.delta_branch_channels[2], cfg.align_out_dim, kernel_size=16, stride=16)
        )
        
        # Z特征分支：线性变换层
        self.z_branch = nn.Sequential(
            nn.LayerNorm(cfg.align_out_dim),
            nn.Linear(cfg.align_out_dim, cfg.align_out_dim),
            nn.GELU()
        )
        
        # 双向交叉注意力
        self.cross_attn_delta2z = CrossAttention(cfg.align_out_dim, cfg.cross_attn_num_heads)
        self.cross_attn_z2delta = CrossAttention(cfg.align_out_dim, cfg.cross_attn_num_heads)
        
        # 融合输出层
        self.fusion_out = nn.Sequential(
            nn.Linear(cfg.align_out_dim * 2, cfg.align_out_dim),
            nn.LayerNorm(cfg.align_out_dim),
            nn.GELU()
        )
        
        self.to(cfg.device)

    def forward(self, delta, z):
        """
        前向传播
        :param delta: 像素级残差 [B, 3, 224, 224]
        :param z: ViT隐表示 [B, 768, 14, 14]
        :return fused_feat: 融合特征 [B, 14, 14, 768]
        """
        B, C, H, W = z.shape
        # Δ分支特征提取与对齐
        delta_feat = self.delta_branch(delta)  # [B, 768, 14, 14]
        delta_feat = delta_feat.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        # Z分支特征变换
        z_feat = z.flatten(2).transpose(1, 2)  # [B, 196, 768]
        z_feat = self.z_branch(z_feat)
        
        # 双向交叉注意力融合
        delta_attn_out = self.cross_attn_delta2z(delta_feat, z_feat)
        z_attn_out = self.cross_attn_z2delta(z_feat, delta_feat)
        
        # 特征拼接与最终融合
        concat_feat = torch.cat([delta_attn_out, z_attn_out], dim=-1)
        fused_feat = self.fusion_out(concat_feat)
        
        # 转换为空间格式 [B, 196, 768] -> [B, 14, 14, 768]
        fused_feat = fused_feat.reshape(B, H, W, C)
        
        return fused_feat