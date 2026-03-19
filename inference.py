import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from config.config import cfg
from models.forgery_detector import ForgeryDetector

def parse_args():
    parser = argparse.ArgumentParser(description="伪造检测推理")
    parser.add_argument("--img_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model_path", type=str, default=cfg.detector_train["save_path"], help="模型权重路径")
    parser.add_argument("--save_result", action="store_true", help="是否保存检测结果")
    parser.add_argument("--save_dir", type=str, default="./inference_results", help="结果保存目录")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.vit_norm_mean, std=cfg.vit_norm_std)
    ])
    
    # 加载模型
    model = ForgeryDetector(freeze_encoder=True, freeze_decoder=True)
    model.load_state_dict(torch.load(args.model_path, map_location=cfg.device))
    model.eval()
    print(f"模型加载完成: {args.model_path}")
    
    # 加载图像
    img = Image.open(args.img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(cfg.device)
    
    # 推理
    with torch.no_grad():
        cls_out, loc_out, _, delta = model(img_tensor)
    
    # 结果解析
    real_prob = cls_out[0, 0].item()
    fake_prob = cls_out[0, 1].item()
    pred_label = "伪造" if fake_prob > 0.5 else "真实"
    
    print("="*50)
    print(f"检测结果: {pred_label}")
    print(f"真实概率: {real_prob:.4f}")
    print(f"伪造概率: {fake_prob:.4f}")
    print("="*50)
    
    # 结果可视化
    if args.save_result:
        os.makedirs(args.save_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(args.img_path))[0]
        
        # 原图、残差图、伪造概率热力图
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"输入图像 | {pred_label} (fake: {fake_prob:.4f})")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        delta_np = delta[0].permute(1, 2, 0).cpu().numpy()
        delta_np = (delta_np - delta_np.min()) / (delta_np.max() - delta_np.min())
        plt.imshow(delta_np)
        plt.title("像素级残差Δ")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        heatmap = loc_out[0, 0].cpu().numpy()
        plt.imshow(img)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.title("伪造区域热力图")
        plt.axis("off")
        
        save_path = os.path.join(args.save_dir, f"{img_name}_result.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"检测结果已保存至: {save_path}")
        plt.close()

if __name__ == "__main__":
    main()