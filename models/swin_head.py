import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock
from config.config import cfg

class SwinCorrelationHead(nn.Module):
    """Swin Transformer像素级相关性建模 + 分类/定位输出头"""
    def __init__(self):
        super().__init__()
        self.window_size = cfg.swin_window_size
        self.embed_dim = cfg.swin_embed_dim
        self.num_heads = cfg.swin_num_heads
        
        # 2个连续的Swin Transformer Block，交替W-MSA和SW-MSA
        self.swin_blocks = nn.ModuleList()
        for i in range(cfg.swin_depth):
            self.swin_blocks.append(SwinTransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0 if i % 2 == 0 else self.window_size // 2,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm
            ))
        
        # 二分类头
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 像素级定位头
        self.loc_head = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.to(cfg.device)

    def forward(self, fused_feat):
        """
        前向传播
        :param fused_feat: 融合特征 [B, 14, 14, 768]
        :return cls_out: 二分类概率 [B, 2]
        :return loc_out: 像素级伪造概率图 [B, 1, 224, 224]
        """
        B, H, W, C = fused_feat.shape
        
        # Swin Transformer前向
        x = fused_feat.reshape(B, H*W, C)
        for blk in self.swin_blocks:
            x = blk(x, (H, W))
        
        # 分类头输出
        cls_out = self.cls_head(x.transpose(1, 2))
        
        # 定位头输出：patch级特征上采样到224×224
        loc_feat = self.loc_head(x)  # [B, 196, 1]
        loc_feat = loc_feat.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, 14, 14]
        loc_out = nn.functional.interpolate(
            loc_feat,
            size=(cfg.img_size, cfg.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        return cls_out, loc_out