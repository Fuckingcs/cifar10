import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR10 的均值和标准差
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        # 训练集的强力数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # 测试集只做标准化
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def prepare_data(self):
        """下载数据"""
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """设置数据集并划分验证集"""
        if stage == "fit" or stage is None:
            full_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.train_transform
            )

            # 划分 90% 训练集，10% 验证集
            total_size = len(full_dataset)
            val_size = int(total_size * 0.1)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # 验证集使用 test_transform（不进行数据增强）
            self.val_dataset.dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.test_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
