import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import CIFAR10DataModule
from model import LitResNet


def main():
    # 设置 ModelCheckpoint：监控 val_acc，保存最好的模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='best-{epoch:02d}-{val_acc:.4f}',
        save_top_k=1,
        verbose=True
    )

    # 初始化 Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=150,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # 初始化模型
    model = LitResNet(num_classes=10, learning_rate=0.1, max_epochs=150)

    # 初始化数据模块
    dm = CIFAR10DataModule(data_dir='./data', batch_size=128)

    # 开始训练
    trainer.fit(model, datamodule=dm)

    # 训练完成后使用最佳权重进行测试
    trainer.test(model, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
