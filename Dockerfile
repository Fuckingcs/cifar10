# 使用官方 PyTorch 镜像作为基础镜像（CUDA 11.8 + cuDNN 8，适配 RTX 4060）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制训练相关文件
COPY dataset.py .
COPY model.py .
COPY train.py .

# 设置启动命令
CMD ["python", "train.py"]

# 重要提示：
# 由于 CIFAR-10 数据集较大，运行容器时请使用 -v 参数挂载本地 data 目录，避免重复下载：
# docker run --gpus all -v $(pwd)/data:/app/data -it cifar10:latest
