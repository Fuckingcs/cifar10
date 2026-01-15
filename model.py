import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy


class LitResNet(pl.LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.1, max_epochs: int = 150):
        super().__init__()
        self.save_hyperparameters()

        # 加载预训练的 ResNet18
        self.model = models.resnet18(weights=None)

        # 针对 CIFAR-10 修改第一层卷积
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 跳过 maxpool 层以保留空间分辨率
        self.model.maxpool = nn.Identity()

        # 修改最后的全连接层为 10 类输出
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        # 初始化准确率指标
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # 记录指标
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # 记录指标
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=0
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
