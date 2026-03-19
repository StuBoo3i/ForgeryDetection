import os
import torch

class Config:
    # 全局基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    img_size = 224
    in_channels = 3
    
    # ViT编码器配置
    vit_model_name = "vit_base_patch16_224.dino"  # DINO自监督预训练模型
    vit_embed_dim = 768
    vit_extract_layers = [0, 1, 2, 3]  # 取前4个Transformer Block
    vit_freeze = True  # 编码器全程冻结
    
    # 重构解码器配置
    decoder_in_dim = 768
    decoder_channels = [512, 256, 128, 64, 3]
    decoder_act = "GELU"
    decoder_out_act = "Tanh"
    
    # 特征对齐模块配置
    align_out_dim = 768
    delta_branch_channels = [64, 128, 256]
    cross_attn_num_heads = 12
    
    # 互信息估计器配置
    infonce_temperature = 0.07
    mi_loss_weight = 0.5  # β
    cls_loss_weight = 1.0  # α
    
    # Swin Transformer配置
    swin_window_size = 7
    swin_num_heads = 12
    swin_embed_dim = 768
    swin_depth = 2  # 2个Swin Block
    
    # 阶段1：解码器训练配置
    recon_train = {
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 50,
        "early_stop_patience": 10,
        "mse_weight": 0.6,
        "lpips_weight": 0.3,
        "vgg_weight": 0.1,
        "data_root": "./data/recon_dataset",
        "save_path": "./weights/decoder.pth",
        "log_path": "./logs/recon_train"
    }
    
    # 阶段2：检测模型训练配置
    detector_train = {
        "batch_size": 32,
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "epochs": 30,
        "data_root": "./data/forgery_pair_dataset",
        "decoder_weight_path": "./weights/decoder.pth",
        "save_path": "./weights/best_model.pth",
        "log_path": "./logs/detector_train",
        "log_interval": 100
    }
    
    # 数据预处理配置
    pixel_norm_range = [-1, 1]  # 与Tanh输出匹配
    vit_norm_mean = [0.485, 0.456, 0.406]
    vit_norm_std = [0.229, 0.224, 0.225]

# 全局配置实例
cfg = Config()