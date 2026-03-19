import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision.models import vgg16
from config.config import cfg

class ReconstructionLoss(nn.Module):
    """重构损失：MSE + LPIPS + VGG感知损失，权重0.6/0.3/0.1"""
    def __init__(self):
        super().__init__()
        self.mse_weight = cfg.recon_train["mse_weight"]
        self.lpips_weight = cfg.recon_train["lpips_weight"]
        self.vgg_weight = cfg.recon_train["vgg_weight"]
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # LPIPS损失
        self.lpips_loss = lpips.LPIPS(net='vgg').to(cfg.device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        
        # VGG感知损失
        self.vgg = vgg16(pretrained=True).features[:16].to(cfg.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()
        
        self.to(cfg.device)

    def forward(self, x_gt, x_recon):
        """
        计算总重构损失
        :param x_gt: 真实图像 [B, 3, 224, 224]
        :param x_recon: 重构图像 [B, 3, 224, 224]
        :return total_loss: 总重构损失
        :return loss_dict: 各分项损失字典
        """
        # 归一化到[0,1]适配LPIPS和VGG
        x_gt_norm = (x_gt - x_gt.min()) / (x_gt.max() - x_gt.min() + 1e-8)
        x_recon_norm = (x_recon - x_recon.min()) / (x_recon.max() - x_recon.min() + 1e-8)
        
        # 各分项损失
        mse = self.mse_loss(x_gt, x_recon)
        lpips = torch.mean(self.lpips_loss(x_gt_norm, x_recon_norm))
        vgg_feat_gt = self.vgg(x_gt_norm)
        vgg_feat_recon = self.vgg(x_recon_norm)
        vgg_loss = self.mse_loss(vgg_feat_gt, vgg_feat_recon)
        
        # 总损失
        total_loss = self.mse_weight * mse + self.lpips_weight * lpips + self.vgg_weight * vgg_loss
        
        loss_dict = {
            "mse": mse.item(),
            "lpips": lpips.item(),
            "vgg": vgg_loss.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict