import torch
import torch.nn as nn
from models.encoder import DINOEncoder
from models.decoder import ReconstructionDecoder
from models.feature_alignment import DualBranchAlignment
from models.mutual_info import InfoNCEMutualInfoEstimator
from models.swin_head import SwinCorrelationHead
from config.config import cfg

class ForgeryDetector(nn.Module):
    """端到端伪造检测与定位模型"""
    def __init__(self, freeze_encoder=True, freeze_decoder=True):
        super().__init__()
        # 核心模块
        self.encoder = DINOEncoder()
        self.decoder = ReconstructionDecoder()
        self.feature_alignment = DualBranchAlignment()
        self.mi_estimator = InfoNCEMutualInfoEstimator()
        self.swin_head = SwinCorrelationHead()
        
        # 权重冻结控制
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.to(cfg.device)

    def load_decoder_weights(self, weight_path):
        """加载预训练的解码器权重"""
        decoder_state_dict = torch.load(weight_path, map_location=cfg.device)
        self.decoder.load_state_dict(decoder_state_dict)
        # 冻结解码器权重
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        print(f"解码器权重加载完成: {weight_path}")

    def forward(self, x):
        """
        完整前向传播
        :param x: 输入图像 [B, 3, 224, 224]
        :return cls_out: 真实/伪造二分类概率 [B, 2]
        :return loc_out: 像素级伪造概率图 [B, 1, 224, 224]
        :return mi_score: 互信息分数 [B]
        :return delta: 像素级残差 [B, 3, 224, 224]
        """
        # 1. ViT编码器提取隐表示Z
        z = self.encoder(x)
        
        # 2. 解码器重构图像，计算残差Δ
        x_recon = self.decoder(z)
        delta = x - x_recon
        
        # 3. 双分支特征对齐与融合
        fused_feat = self.feature_alignment(delta, z)
        
        # 4. 互信息估计
        delta_feat = self.feature_alignment.delta_branch(delta).flatten(2).transpose(1, 2)
        z_feat = z.flatten(2).transpose(1, 2)
        mi_score, _ = self.mi_estimator(delta_feat, z_feat)
        
        # 5. Swin Transformer相关性建模与输出
        cls_out, loc_out = self.swin_head(fused_feat)
        
        return cls_out, loc_out, mi_score, delta