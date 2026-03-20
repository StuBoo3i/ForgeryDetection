import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config.config import cfg
from models.encoder import DINOEncoder
from models.decoder import ReconstructionDecoder
from losses.recon_loss import ReconstructionLoss
from data.datasets import build_recon_dataloaders
from utils.logger import get_logger
from utils.weights_utils import save_weights
from utils.image_saver import save_reconstructed_images 

def train_decoder():
    # 初始化日志与tensorboard
    logger = get_logger("recon_train", cfg.recon_train["log_path"])
    writer = SummaryWriter(cfg.recon_train["log_path"])
    logger.info("="*50)
    logger.info("开始阶段1：重构解码器训练")
    logger.info(f"训练配置: {cfg.recon_train}")
    logger.info("="*50)
    
    # 初始化模型
    encoder = DINOEncoder()
    decoder = ReconstructionDecoder()
    encoder.eval()
    
    # 初始化损失函数、优化器
    criterion = ReconstructionLoss()
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg.recon_train["lr"],
        weight_decay=cfg.recon_train["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 加载数据
    train_loader, val_loader = build_recon_dataloaders()
    
    # 训练状态初始化
    best_val_loss = float('inf')
    early_stop_counter = 0
    global_step = 0
    
    # 训练循环
    for epoch in range(cfg.recon_train["epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{cfg.recon_train['epochs']}")
        decoder.train()
        train_loss_sum = 0.0
        
        # 训练步
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            
            # 前向传播
            with torch.no_grad():
                z = encoder(batch)
            x_recon = decoder(z)
            
            # 计算损失
            loss, loss_dict = criterion(batch, x_recon)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 日志更新
            train_loss_sum += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss_dict["total"])
            
            # tensorboard记录
            writer.add_scalar("Train/mse_loss", loss_dict["mse"], global_step)
            writer.add_scalar("Train/lpips_loss", loss_dict["lpips"], global_step)
            writer.add_scalar("Train/vgg_loss", loss_dict["vgg"], global_step)
            writer.add_scalar("Train/total_loss", loss_dict["total"], global_step)
        
        # 平均训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        logger.info(f"训练平均损失: {avg_train_loss:.6f}")
        
        # 验证步
        decoder.eval()
        val_loss_sum = 0.0
        # 【新增】用于保存最后一个epoch的重构图像
        last_recon_batch = None
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(cfg.device)
                z = encoder(batch)
                x_recon = decoder(z)
                loss, loss_dict = criterion(batch, x_recon)
                val_loss_sum += loss.item()
                pbar.set_postfix(loss=loss_dict["total"])
                
                # 【新增】保存第一个验证batch的重构结果，用于最后epoch的可视化
                if batch_idx == 0:
                    last_recon_batch = x_recon
        
        avg_val_loss = val_loss_sum / len(val_loader)
        logger.info(f"验证平均损失: {avg_val_loss:.6f}")
        writer.add_scalar("Val/total_loss", avg_val_loss, epoch+1)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型与早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_weights(decoder, cfg.recon_train["save_path"])
            early_stop_counter = 0
            logger.info(f"最佳模型已更新，验证损失: {best_val_loss:.6f}")
        else:
            early_stop_counter += 1
            logger.info(f"早停计数: {early_stop_counter}/{cfg.recon_train['early_stop_patience']}")
            if early_stop_counter >= cfg.recon_train["early_stop_patience"]:
                logger.info("验证损失连续10轮未下降，触发早停")
                # 【新增】早停触发前，保存当前重构图像
                save_reconstructed_images(
                    last_recon_batch,
                    save_root=cfg.recon_train.get("recon_save_root", "./recon_outputs"),
                    prefix=f"epoch_{epoch+1}_earlystop"
                )
                break
    
    # 【新增】正常训练结束（未早停），保存最后一个epoch的重构图像
    if early_stop_counter < cfg.recon_train["early_stop_patience"]:
        save_reconstructed_images(
            last_recon_batch,
            save_root=cfg.recon_train.get("recon_save_root", "./recon_outputs"),
            prefix=f"epoch_{cfg.recon_train['epochs']}_final"
        )
    
    logger.info("\n" + "="*50)
    logger.info(f"重构解码器训练完成，最佳验证损失: {best_val_loss:.6f}")
    logger.info("="*50)
    writer.close()

if __name__ == "__main__":
    train_decoder()