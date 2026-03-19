import torch
import torch.nn as nn
from config.config import cfg

class ReconstructionDecoder(nn.Module):
    """重构解码器，适配ViT编码器输出，实现真实图像精准重构"""
    def __init__(self):
        super().__init__()
        # 特征适配层
        self.adapt = nn.Conv2d(
            cfg.decoder_in_dim,
            cfg.decoder_channels[0],
            kernel_size=1,
            stride=1
        )
        
        # 反卷积上采样层
        self.deconv_layers = nn.ModuleList()
        in_channels = cfg.decoder_channels[0]
        for i in range(1, len(cfg.decoder_channels)-1):
            out_channels = cfg.decoder_channels[i]
            self.deconv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ))
            in_channels = out_channels
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, cfg.decoder_channels[-1], kernel_size=2, stride=2),
            nn.Conv2d(cfg.decoder_channels[-1], cfg.decoder_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self.to(cfg.device)

    def forward(self, z):
        """
        前向传播
        :param z: ViT编码器输出的隐表示 [B, 768, 14, 14]
        :return x_recon: 重构图像 [B, 3, 224, 224]
        """
        x = self.adapt(z)
        for deconv in self.deconv_layers:
            x = deconv(x)
        x_recon = self.output_layer(x)
        return x_recon