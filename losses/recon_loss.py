import torch
import torch.nn as nn
from config.config import cfg

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_weight = cfg.recon_train["mse_weight"]
        
        self.mse_loss = nn.MSELoss()
        
        self.to(cfg.device)

    def forward(self, x_gt, x_recon):
        mse = self.mse_loss(x_gt, x_recon)
        
        total_loss = self.mse_weight * mse
        
        loss_dict = {
            "mse": mse.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict