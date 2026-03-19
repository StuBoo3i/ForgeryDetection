import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config.config import cfg
from models.forgery_detector import ForgeryDetector
from losses.detector_loss import DetectorLoss
from data.datasets import build_detector_dataloaders
from utils.logger import get_logger
from utils.weights_utils import save_weights
from utils.metrics import calculate_auc

def train_detector():
    # 初始化日志与tensorboard
    logger = get_logger("detector_train", cfg.detector_train["log_path"])
    writer = SummaryWriter(cfg.detector_train["log_path"])
    logger.info("="*50)
    logger.info("开始阶段2：伪造检测模型训练")
    logger.info(f"训练配置: {cfg.detector_train}")
    logger.info("="*50)
    
    # 初始化模型，加载预训练解码器
    model = ForgeryDetector(freeze_encoder=True, freeze_decoder=True)
    model.load_decoder_weights(cfg.detector_train["decoder_weight_path"])
    
    # 初始化损失函数、优化器（仅更新可训练参数）
    criterion = DetectorLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.detector_train["lr"],
        weight_decay=cfg.detector_train["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.detector_train["epochs"], eta_min=1e-6
    )
    
    # 加载数据
    train_loader, val_loader = build_detector_dataloaders()
    
    # 训练状态初始化
    best_val_auc = 0.0
    global_step = 0
    log_interval = cfg.detector_train["log_interval"]
    
    # 训练循环
    for epoch in range(cfg.detector_train["epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{cfg.detector_train['epochs']}")
        model.train()
        train_loss_sum = 0.0
        
        # 训练步
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            optimizer.zero_grad()
            
            # 前向传播
            cls_out, loc_out, mi_score, _ = model(imgs)
            
            # 计算损失
            loss, loss_dict = criterion(cls_out, mi_score, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 日志更新
            train_loss_sum += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss_dict["total_loss"], cls_loss=loss_dict["cls_loss"], mi_loss=loss_dict["mi_loss"])
            
            # 间隔日志
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"Step {global_step} | 总损失: {loss_dict['total_loss']:.4f} | 分类损失: {loss_dict['cls_loss']:.4f} | 互信息损失: {loss_dict['mi_loss']:.4f}")
                writer.add_scalar("Train/cls_loss", loss_dict["cls_loss"], global_step)
                writer.add_scalar("Train/mi_loss", loss_dict["mi_loss"], global_step)
                writer.add_scalar("Train/total_loss", loss_dict["total_loss"], global_step)
        
        # 平均训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        logger.info(f"训练平均损失: {avg_train_loss:.6f}")
        
        # 验证步
        model.eval()
        all_labels = []
        all_preds = []
        val_loss_sum = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for imgs, labels in pbar:
                imgs = imgs.to(cfg.device)
                labels = labels.to(cfg.device)
                
                cls_out, _, mi_score, _ = model(imgs)
                loss, loss_dict = criterion(cls_out, mi_score, labels)
                
                val_loss_sum += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(cls_out[:, 1].cpu().numpy())  # 伪造类概率
        
        # 计算验证指标
        avg_val_loss = val_loss_sum / len(val_loader)
        val_auc = calculate_auc(np.array(all_labels), np.array(all_preds))
        
        logger.info(f"验证平均损失: {avg_val_loss:.6f} | 验证AUC: {val_auc:.4f}")
        writer.add_scalar("Val/total_loss", avg_val_loss, epoch+1)
        writer.add_scalar("Val/auc", val_auc, epoch+1)
        
        # 学习率调整
        scheduler.step()
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_weights(model, cfg.detector_train["save_path"], only_trainable=False)
            logger.info(f"最佳模型已更新，验证AUC: {best_val_auc:.4f}")
    
    logger.info("\n" + "="*50)
    logger.info(f"检测模型训练完成，最佳验证AUC: {best_val_auc:.4f}")
    logger.info("="*50)
    writer.close()

if __name__ == "__main__":
    train_detector()