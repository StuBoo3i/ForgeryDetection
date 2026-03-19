import os
import torch
from config.config import cfg

def save_weights(model, save_path, only_trainable=False):
    """保存模型权重"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if only_trainable:
        state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    print(f"权重已保存至: {save_path}")

def freeze_model(model):
    """冻结模型所有权重"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def unfreeze_model(model):
    """解冻模型所有权重"""
    for param in model.parameters():
        param.requires_grad = True
    model.train()