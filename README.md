==========================================
CIFAR-10 图像分类项目报告

1. 项目概述
本训练项目基于 PyTorch Lightning 框架，使用 ResNet-18 模型对 CIFAR-10 数据集进行分类任务。

2. 核心方法
为了追求更高的泛化性能，项目采用了以下方法：
- 架构：对 32x32 小图修改了 ResNet18 的起始卷积层与池化层。
- 算法：用的 SGD (LR=0.1, Momentum=0.9)。
- 数据增强：引入 AutoAugment (CIFAR10 Policy) 自动数据增强技术。
- 使用 CosineAnnealingLR。

3. 性能表现
- 训练轮次：150 Epochs
- 验证集最高准确率: 96.30%
- 测试集最终准确率: 95.44%
- 最佳模型位置: lightning_logs/version_1/checkpoints/

4. 环境 (requirements.txt)
- Python 3.8
- PyTorch / PyTorch Lightning
- Torchvision
- Torchmetrics
