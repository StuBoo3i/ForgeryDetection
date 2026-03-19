import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg

class InfoNCEMutualInfoEstimator(nn.Module):
    """基于InfoNCE的互信息估计器"""
    def __init__(self):
        super().__init__()
        self.temperature = cfg.infonce_temperature
        self.to(cfg.device)

    def forward(self, delta_feat, z_feat):
        """
        计算InfoNCE损失与互信息下界
        :param delta_feat: 对齐后的Δ特征 [B, 196, 768]
        :param z_feat: 对齐后的Z特征 [B, 196, 768]
        :return mi_score: 互信息下界分数 [B]
        :return infonce_loss: InfoNCE损失
        """
        B, N, C = delta_feat.shape
        
        # 全局平均池化得到样本级特征
        delta_global = delta_feat.mean(dim=1)  # [B, 768]
        z_global = z_feat.mean(dim=1)  # [B, 768]
        
        # L2归一化
        delta_norm = F.normalize(delta_global, dim=-1)
        z_norm = F.normalize(z_global, dim=-1)
        
        # 计算余弦相似度矩阵 [B, B]
        sim_matrix = torch.matmul(delta_norm, z_norm.T) / self.temperature
        
        # 正样本：对角线元素（同图像的Δ与Z）
        pos_sim = torch.diag(sim_matrix)
        
        # InfoNCE损失计算
        infonce_loss = -torch.mean(pos_sim - torch.logsumexp(sim_matrix, dim=-1))
        
        # 互信息下界代理指标：正样本余弦相似度
        mi_score = pos_sim
        
        return mi_score, infonce_loss