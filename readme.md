# Pair-based Forgery Detection Model: End-to-End AI Image Forgery Detection and Localization

# Pair-based Forgery Detection Model

[[![Image](bd2ad2ad-06c0-4b2b-a714-ab06e902108d)](https://www.python.org/)](https://www.python.org/)

[[![Image](1aca93b1-c9a7-4652-8352-2628ccdbb367)](https://pytorch.org/)](https://pytorch.org/)

![Image](https://p11-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/rc/online_import/217e6cf567854123b35c75d63532c117~tplv-noop.jpeg?rk3s=49177a0b&x-expires=1773894791&x-signature=M%2B5aiavWmjHQib7ZfSCgKjlYmAE%3D&resource_key=2463c452-b3c6-43da-8d00-385329d30298&resource_key=2463c452-b3c6-43da-8d00-385329d30298)

本项目是基于**配对式伪造范式**构建的端到端AI图像伪造检测与定位模型，完整复现了工业级伪造检测落地方案，可精准检测DeepFake人脸伪造、AI生成图像、局部编辑篡改、GAN风格迁移等主流伪造类型，同时输出像素级伪造区域定位结果。

## ✨ 核心特性

- **配对式范式设计**：基于同源<真实原图-伪造图像>配对样本训练，彻底消除语义差异对检测结果的干扰，大幅提升模型泛化性

- **自监督特征提取**：采用DINO预训练ViT-B/16浅层编码器，对伪造artifact的边缘、纹理、高频统计异常敏感度远高于有监督预训练模型

- **重构残差分支**：仅在真实自然图像上训练的重构解码器，精准分离真实图像可建模特征与伪造图像不可重构的异常信息，保证残差信号纯度

- **互信息差异建模**：基于InfoNCE框架构建互信息估计器，挖掘真实/伪造样本在「隐表示-残差」间的相关性差异，强化伪造特征辨识度

- **端到端多任务输出**：同时支持**真实/伪造二分类**与**像素级伪造区域定位**，兼顾全局判别与局部定位能力

- **强鲁棒性设计**：内置JPEG压缩、高斯模糊、噪声干扰等数据增强，适配真实场景下的后处理操作，模型抗干扰能力强

## 📁 项目结构

```Plain Text

ForgeryDetection/
├── config/                 # 全局配置模块
│   └── config.py           # 超参数、路径、模型结构配置
├── models/                 # 核心模型实现
│   ├── encoder.py          # DINO预训练ViT特征编码器
│   ├── decoder.py          # 图像重构解码器
│   ├── feature_alignment.py# 双分支特征对齐与双向交叉注意力融合
│   ├── mutual_info.py      # InfoNCE互信息估计器
│   ├── swin_head.py        # Swin Transformer相关性建模与输出头
│   └── forgery_detector.py # 端到端检测模型整体封装
├── data/                   # 数据集与数据加载
│   └── datasets.py         # 重构数据集、配对伪造数据集实现
├── losses/                 # 自定义损失函数
│   ├── recon_loss.py       # 重构损失（MSE+LPIPS+VGG感知损失）
│   └── detector_loss.py    # 检测损失（分类损失+互信息专项损失+Dice损失）
├── utils/                  # 工具函数
│   ├── metrics.py          # 评价指标（AUC/mIoU/Dice系数）
│   ├── logger.py           # 训练日志工具
│   └── weights_utils.py    # 权重加载、冻结与保存工具
├── train/                  # 训练脚本
│   ├── train_decoder.py    # 阶段1：重构解码器训练
│   └── train_detector.py   # 阶段2：检测模型训练
├── inference.py            # 单图推理与可视化脚本
├── requirements.txt        # 项目依赖清单
└── README.md
```

## 🔧 环境要求

- 操作系统：Linux / Windows / macOS

- Python 版本：3.8 及以上

- 依赖框架：

    - PyTorch >= 2.0.0

    - TorchVision >= 0.15.0

    - timm >= 0.9.0

    - 其他依赖详见 `requirements.txt`

- 硬件建议：

    - 训练：NVIDIA GPU 显存 >= 12GB（推荐24GB及以上）

    - 推理：NVIDIA GPU 显存 >= 4GB 或 CPU

### 环境安装

```Bash

# 克隆项目
git clone https://github.com/your-username/ForgeryDetection.git
cd ForgeryDetection

# 安装依赖
pip install -r requirements.txt
```

## 📦 数据准备

本项目分为两个训练阶段，对应两套数据集规范：

### 1. 重构解码器训练数据集

仅需真实自然图像，无需伪造样本，用于训练解码器对真实图像分布的适配能力。

```Plain Text

./data/recon_dataset/
├── 自定义文件夹1/
│   ├── img_001.jpg
│   ├── img_002.png
│   └── ...
├── 自定义文件夹2/
│   └── ...
└── ...
```

推荐数据集：ImageNet-1K、COCO、FFHQ、CelebA-HQ等通用真实图像数据集。

### 2. 检测模型训练配对数据集

严格遵循**同源配对**规范，真实原图与伪造图像必须一一对应、语义完全对齐，仅存在伪造操作带来的像素差异。

```Plain Text

./data/forgery_pair_dataset/
├── train/
│   ├── real/  # 真实原图X1
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── fake/  # 同源伪造图X2，与real中文件同名
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
└── val/
    ├── real/
    └── fake/
```

支持伪造类型：DeepFake人脸替换、AI局部编辑/填充、ControlNet全图生成、GAN风格迁移/属性篡改等。

## 🏋️ 模型训练

本项目采用两阶段训练范式，需先训练重构解码器，再训练检测模型。

### 阶段1：重构解码器训练

冻结ViT编码器权重，仅训练重构解码器，学习真实自然图像的分布特征。

```Bash

python train/train_decoder.py
```

- 训练配置：优化器AdamW，初始学习率1e-4，训练轮次50epoch，验证损失连续10轮不下降自动早停

- 输出：训练完成后，解码器权重自动保存至 `./weights/decoder.pth`

### 阶段2：伪造检测模型训练

冻结ViT编码器与预训练解码器，仅训练特征对齐、融合、Swin Transformer与输出头模块。

```Bash

python train/train_detector.py
```

- 训练配置：优化器AdamW，初始学习率5e-5，训练轮次30epoch，基于验证集AUC保存最佳模型

- 输出：最佳模型权重自动保存至 `./weights/best_model.pth`

- 日志：训练过程与指标自动记录至 `./logs/` 目录，支持TensorBoard可视化

## 🔍 推理与可视化

使用训练完成的模型对单张图像进行伪造检测，输出分类结果、像素级伪造热力图与残差特征图。

```Bash

# 基础推理
python inference.py --img_path 你的测试图像路径

# 推理并保存可视化结果
python inference.py --img_path 你的测试图像路径 --save_result --save_dir ./inference_results
```

### 推理参数说明

|参数|说明|默认值|
|---|---|---|
|`--img_path`|待检测图像路径（必填）|-|
|`--model_path`|模型权重文件路径|`./weights/best_model.pth`|
|`--save_result`|是否保存可视化结果|不开启|
|`--save_dir`|可视化结果保存目录|`./inference_results`|
### 输出结果

1. 控制台输出：真实/伪造判别结果、对应概率值

2. 可视化结果（开启`--save_result`后）：

    - 输入原图+判别结果

    - 像素级残差特征图

    - 伪造区域热力图叠加效果

## 📊 评价指标

|任务类型|核心指标|说明|
|---|---|---|
|二分类任务|AUC|模型整体判别能力，越接近1性能越好|
|定位任务|mIoU / Dice系数|像素级伪造区域定位精度，越接近1性能越好|
|鲁棒性评价|AUC下降幅度|后处理/对抗攻击后的AUC衰减，幅度越小鲁棒性越强|
## 📌 核心模块设计详解

1. **ViT编码器模块**：采用DINO自监督预训练ViT-B/16，取前4个Transformer Block提取低级视觉特征，通过通道注意力加权融合多尺度输出，全程冻结权重保证特征稳定性。

2. **重构解码器模块**：对称式反卷积上采样结构，匹配ViT编码器的特征维度与空间尺度，仅在真实图像上训练，对伪造artifact无法精准重构，输出高纯度残差信号。

3. **双分支特征对齐模块**：分别对残差信号与ViT隐表示做维度与空间尺度对齐，通过双向交叉注意力完成跨特征融合，直接建模二者的相关性。

4. **互信息估计器**：基于InfoNCE框架，以同图像的残差-隐表示为正样本，跨图像的为负样本，优化真实/伪造样本的互信息差异，强化伪造特征区分度。

5. **Swin Transformer模块**：2个连续的Swin Block，交替使用窗口注意力与滑动窗口注意力，建模全局与局部像素间的统计差异，兼顾计算效率与长程依赖捕捉能力。

## 🤝 贡献指南

欢迎提交Issue与PR参与项目贡献：

1. Fork 本仓库

2. 新建你的功能分支 (`git checkout -b feature/AmazingFeature`)

3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)

4. 推送到分支 (`git push origin feature/AmazingFeature`)

5. 新建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

本项目仅用于学术研究与合规的图像取证场景，严禁用于非法用途。使用本项目产生的任何法律责任，均由使用者自行承担。