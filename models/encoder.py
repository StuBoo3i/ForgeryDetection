import torch
import torch.nn as nn
import timm
from config.config import cfg

class DINOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载DINO预训练ViT-B/16
        self.vit = timm.create_model(
            cfg.vit_model_name,
            pretrained=True,
            num_classes=0,
            img_size=cfg.img_size
        )
        
        self.patch_embed = self.vit.patch_embed
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        
        self.blocks = nn.ModuleList([
            self.vit.blocks[i] for i in cfg.vit_extract_layers
        ])
        
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
        :param x:  [B, 3, 224, 224]
        :return z:  [B, 768, 14, 14]
        """
        # ViT Patch Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        # 转换逻辑：[B, 196, 768] -> [B, 768, 14, 14]
        b, n, c = x.shape
        h = w = int(n ** 0.5)
        z = x.permute(0, 2, 1).reshape(b, c, h, w)
        
        return z