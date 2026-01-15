# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch Lightning implementation for training a ResNet18 model on the CIFAR-10 dataset. The project follows modern deep learning best practices with modular architecture, automatic checkpointing, and containerization support.

## Core Architecture

### Main Components

- **[train.py](train.py)**: Training orchestration script using PyTorch Lightning Trainer
  - Configures model checkpointing to save best model based on validation accuracy
  - Sets up GPU training with mixed precision
  - Automatically runs testing with best checkpoint after training

- **[model.py](model.py)**: ResNet18 model implementation adapted for CIFAR-10
  - `LitResNet` class: PyTorch LightningModule with modified ResNet18 architecture
  - Changes first conv layer to 3x3 kernel (instead of 7x7) for 32x32 inputs
  - Removes maxpooling layer to preserve spatial resolution
  - Implements custom training/validation/test steps with accuracy metrics

- **[dataset.py](dataset.py)**: Data handling and preprocessing
  - `CIFAR10DataModule`: PyTorch Lightning DataModule for CIFAR-10 dataset
  - Strong data augmentation for training (RandomCrop, RandomHorizontalFlip, AutoAugment)
  - Standard normalization for validation/testing
  - Automatic download to `./data/` directory
  - 90/10 train/validation split from training data

## Development Commands

### Training
```bash
# Install dependencies
pip install -r requirements.txt

# Run training (150 epochs, GPU accelerated)
python train.py
```

### Docker Development
```bash
# Build container
docker build -t cifar10:latest .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data -it cifar10:latest
```

### Expected Output
- Training logs saved to `./lightning_logs/`
- Best model checkpoints saved with pattern `best-{epoch:02d}-{val_acc:.4f}`
- Final test accuracy reported after training

## Model Configuration

- **Architecture**: Modified ResNet18
  - Input: 3x32x32 (CIFAR-10 images)
  - First conv: 3x3 kernel, stride 1, padding 1
  - No maxpooling layer
  - Output: 10 classes

- **Training Setup**
  - Optimizer: SGD with momentum=0.9, weight_decay=5e-4
  - Learning rate: 0.1 (cosine annealing to 0)
  - Batch size: 128
  - Epochs: 150
  - Metrics: Training/Validation/Test accuracy

- **Data Augmentation**
  - Training: RandomCrop(32, padding=4), RandomHorizontalFlip, AutoAugment(CIFAR10 policy)
  - Validation/Test: Only normalization (CIFAR-10 specific mean/std)

## File Structure

```
.
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── lightning_logs/          # Training logs and checkpoints
├── train.py                 # Main training script
├── model.py                 # Model definition
├── dataset.py              # Data loading and preprocessing
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # Project documentation
```

## Important Notes

- The project uses PyTorch Lightning for high-level abstractions
- GPU acceleration is required for training (accelerator='gpu', devices=1)
- Checkpoints are automatically managed and the best model is used for final testing
- Data is automatically downloaded to the `./data/` directory on first run
- The implementation follows CIFAR-10 specific best practices for architecture and augmentation