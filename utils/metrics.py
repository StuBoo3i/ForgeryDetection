import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(y_true, y_pred):
    """计算二分类AUC"""
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5
    return auc

def calculate_dice(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def calculate_miou(pred, target, smooth=1e-6):
    """计算mIoU"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou