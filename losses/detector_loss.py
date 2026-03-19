import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg

class DetectorLoss(nn.Module):
    """检测总损失：分类损失 + 互信息专项损失 + 定位Dice损失"""
    def __init__(self):
        super().__init__()
        self.cls_weight = cfg.cls_loss_weight
        self.mi_weight = cfg.mi_loss_weight
        self.cls_loss = nn.CrossEntropyLoss()
        self.mi_bce_loss = nn.BCELoss()
        self.to(cfg.device)

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice损失，用于像素级定位任务"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, cls_out, mi_score, labels, loc_out=None, loc_gt=None):
        """
        计算总检测损失
        :param cls_out: 分类输出 [B, 2]
        :param mi_score: 互信息分数 [B]
        :param labels: 真实/伪造标签 [B]，0=真实，1=伪造
        :param loc_out: 定位输出 [B, 1, 224, 224]，可选
        :param loc_gt: 定位真值 [B, 1, 224, 224]，可选
        :return total_loss: 总损失
        :return loss_dict: 各分项损失字典
        """
        # 分类损失
        loss_cls = self.cls_loss(cls_out, labels)
        
        # 互信息专项损失：伪造样本MI尽可能大，真实样本MI尽可能小
        mi_norm = torch.sigmoid(mi_score)
        loss_mi = self.mi_bce_loss(mi_norm, labels.float())
        
        # 总损失
        total_loss = self.cls_weight * loss_cls + self.mi_weight * loss_mi
        loss_dict = {
            "cls_loss": loss_cls.item(),
            "mi_loss": loss_mi.item(),
            "total_loss": total_loss.item()
        }
        
        # 加入定位Dice损失
        if loc_out is not None and loc_gt is not None:
            loss_dice = self.dice_loss(loc_out, loc_gt)
            total_loss += 0.3 * loss_dice
            loss_dict["dice_loss"] = loss_dice.item()
            loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict