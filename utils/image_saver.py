import os
from datetime import datetime
import torch
import numpy as np
from PIL import Image

def save_reconstructed_images(tensor, save_root, prefix="recon"):
    """
    将解码器输出的批量重构图像保存为 PNG 格式，存储在带时间戳的文件夹中
    :param tensor: 解码器输出张量 [B, 3, 224, 224]，范围 [-1, 1]
    :param save_root: 保存根目录，如 "./recon_outputs"
    :param prefix: 文件名前缀，如 "epoch_50"
    """
    # 1. 创建带时间戳的保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_root, f"recon_images_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 遍历批量中的每张图像并保存
    tensor = tensor.detach().cpu()  # 移除梯度，转回 CPU
    batch_size = tensor.shape[0]
    
    for i in range(batch_size):
        # 单张图像张量处理
        img_tensor = tensor[i]
        # 数值范围转换：[-1, 1] → [0, 1] → [0, 255]
        img_tensor = (img_tensor + 1) / 2
        img_tensor = img_tensor.clamp(0, 1)
        # 维度转换：[3, 224, 224] (NCHW) → [224, 224, 3] (HWC)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # 保存为 PNG
        img_pil = Image.fromarray(img_np)
        save_path = os.path.join(save_dir, f"{prefix}_img_{i:03d}.png")
        img_pil.save(save_path, "PNG")
    
    print(f"已保存 {batch_size} 张重构图像至: {save_dir}")