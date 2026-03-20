import torch
import torch.nn as nn
from config.config import cfg

class ReconstructionDecoder(nn.Module):
    """重构解码器，完全匹配文档设计规范，适配ViT编码器输出，实现真实图像精准重构"""
    def __init__(self):
        super().__init__()
        self.in_dim = 768  
        self.channel_seq = [512, 256, 128, 64, 3]
        
        self.adapt = nn.Conv2d(
            in_channels=self.in_dim,
            out_channels=self.channel_seq[0],
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.deconv_layers = nn.ModuleList()
        in_channels = self.channel_seq[0]
        for out_channels in self.channel_seq[1:-1]:
            self.deconv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ))
            in_channels = out_channels
        
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, self.channel_seq[-1], kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Tanh()
        )
        
        self.to(cfg.device)

    def forward(self, z):
        if z.dim() != 4:
            raise ValueError(f"输入隐表示必须为4维张量，当前维度为{z.dim()}")
        
        if z.shape[1] == 14 and z.shape[2] == 14 and z.shape[3] == 768:
            z = z.permute(0, 3, 1, 2)
        
        if z.shape[1] != self.in_dim or z.shape[2] != 14 or z.shape[3] != 14:
            raise ValueError(f"输入隐表示必须为[B,768,14,14]格式，当前格式为{z.shape}")
        
        x = self.adapt(z)
        for deconv in self.deconv_layers:
            x = deconv(x)
        x_recon = self.output_layer(x)
        
        return x_recon