import os
import sys
import argparse  # 新增：命令行参数解析
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

# 新增：命令行参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="Train Reconstruction Decoder (Stage 1)")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay for optimizer")
    parser.add_argument("--data_root", type=str, default=None, help="Root path of training data")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs (weights, logs, images)")
    parser.add_argument("--device", type=str, default=None, help="Training device (e.g., cuda:0, cpu)")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Early stop patience epochs")
    return parser.parse_args()

# 新增：用命令行参数覆盖配置文件
def override_config_with_args(args):
    if args.batch_size is not None:
        cfg.recon_train["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg.recon_train["epochs"] = args.epochs
    if args.lr is not None:
        cfg.recon_train["lr"] = args.lr
    if args.weight_decay is not None:
        cfg.recon_train["weight_decay"] = args.weight_decay
    if args.data_root is not None:
        cfg.recon_train["data_root"] = args.data_root
    if args.output_dir is not None:
        cfg.recon_train["output_dir"] = args.output_dir
        # 同步更新日志、权重、重构图像的保存路径
        cfg.recon_train["log_path"] = os.path.join(args.output_dir, "logs")
        cfg.recon_train["save_path"] = os.path.join(args.output_dir, "weights", "decoder_best.pth")
        cfg.recon_train["recon_save_root"] = os.path.join(args.output_dir, "recon_images")
        # 自动创建目录
        os.makedirs(cfg.recon_train["log_path"], exist_ok=True)
        os.makedirs(os.path.dirname(cfg.recon_train["save_path"]), exist_ok=True)
        os.makedirs(cfg.recon_train["recon_save_root"], exist_ok=True)
    if args.device is not None:
        cfg.device = args.device
    if args.early_stop_patience is not None:
        cfg.recon_train["early_stop_patience"] = args.early_stop_patience

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
            
            # tensorboard记录（注意：若已切换为仅MSE损失，需删除lpips/vgg的记录）
            writer.add_scalar("Train/mse_loss", loss_dict["mse"], global_step)
            if "lpips" in loss_dict:
                writer.add_scalar("Train/lpips_loss", loss_dict["lpips"], global_step)
            if "vgg" in loss_dict:
                writer.add_scalar("Train/vgg_loss", loss_dict["vgg"], global_step)
            writer.add_scalar("Train/total_loss", loss_dict["total"], global_step)
        
        # 平均训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        logger.info(f"训练平均损失: {avg_train_loss:.6f}")
        
        # 验证步
        decoder.eval()
        val_loss_sum = 0.0
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
                logger.info("验证损失连续多轮未下降，触发早停")
                save_reconstructed_images(
                    last_recon_batch,
                    save_root=cfg.recon_train.get("recon_save_root", "./recon_outputs"),
                    prefix=f"epoch_{epoch+1}_earlystop"
                )
                break
    
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
    args = parse_args()
    override_config_with_args(args)
    train_decoder()