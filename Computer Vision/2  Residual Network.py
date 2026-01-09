import os
import pickle
import argparse
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.ao.nn.quantized.functional import conv1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -----------------------------
# Data loading (compatible with CIFAR-10 python pickles)
# -----------------------------
def _unpickle(file_path: str):
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict


def _load_batch(data_dir: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    batch_path = os.path.join(data_dir, filename)
    batch_dict = _unpickle(batch_path)

    images = batch_dict['data']
    labels = batch_dict['labels']

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = images.astype(np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)

    return images, labels


def load_numpy_cifar(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_dir = os.path.abspath(base_dir)
    data_dir = os.path.join(base_dir, 'cifar-10-batches-py')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"找不到目录: {data_dir}，请检查文件结构。"
            f" (cwd={os.getcwd()})"
        )

    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(data_dir, f'data_batch_{i}')
        train_images.append(imgs)
        train_labels.append(lbls)
    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    x_test, y_test = _load_batch(data_dir, 'test_batch')
    return x_train, y_train, x_test, y_test


class NumpyCIFARDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        """
        :param images: 输入图像，形状 (N, H, W, C)，值在 [0, 1] 之间
        :param labels: 对应标签，形状 (N,)
        :param augment: 是否应用数据增强
        """
        self.images = images
        self.labels = labels
        if augment:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # expects HWC float in [0,1]
                transforms.RandomHorizontalFlip(), # 随机水平翻转
                transforms.RandomRotation(10), # 随机旋转
                transforms.RandomResizedCrop(size=32, scale=(0.9, 1.1), ratio=(0.9, 1.1)), # 随机裁剪并调整大小
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img, label


class InceptionNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        """
        :param in_channels: 输入渠道
        :param out_channels: 输出渠道为 out_channels * 4
        :param stride: 步伐幅度默认为1
        """
        super().__init__()

        # 每个分支输出 out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)

        # 池化后用 1x1 卷积调整通道数
        self.pool_conv = nn.Conv2d(in_channels, out_channels, stride = 1, kernel_size=1, bias=False)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)


    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        xp = self.pool_conv(self.MaxPool(x))

        # 拼接通道
        x_new = torch.cat([x1, x3, x5, xp], dim=1)
        return x_new


class ResidualNetwork(nn.Module):
    """
    残差网络
    输入 nh * nw * in_channels，
    输出 nh * nw * out_channels，
    """
    def __init__(self, in_channels:int, out_channels:int, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut（同时处理通道/步幅差异）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# -----------------------------
# Model (CNN)
# -----------------------------
class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 入口：Inception 将 3 通道 -> 64 通道（每分支 16，拼接后 4*16=64）
        self.stem = nn.Sequential(
            InceptionNetwork(in_channels=3, out_channels=16, stride=1),  # out: 64ch, 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 64 (保持分辨率)
        self.stage1 = nn.Sequential(
            ResidualNetwork(64, 64, stride=1),
            ResidualNetwork(64, 64, stride=1),
        )

        # Stage 2: 下采样到 16x16，通道 64 -> 128
        self.stage2 = nn.Sequential(
            ResidualNetwork(64, 128, stride=2),  # 32x32 -> 16x16
            ResidualNetwork(128, 128, stride=1),
        )

        # Stage 3: 下采样到 8x8，通道 128 -> 256
        self.stage3 = nn.Sequential(
            ResidualNetwork(128, 256, stride=2),  # 16x16 -> 8x8
            ResidualNetwork(256, 256, stride=1),
        )

        # 头部：全局平均池化 + 全连接
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

# -----------------------------
# Training / Evaluation
# -----------------------------
from contextlib import nullcontext

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    autocast_ctx = (
        torch.cuda.amp.autocast if (scaler is not None and device.type == "cuda")
        else nullcontext
    )

    for images, labels in tqdm(loader, desc="Train ", leave=False):
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size

        optimizer.zero_grad()
        with autocast_ctx():
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


# === 配置部分 ===
class Args:
    # 注意：相对路径会受当前工作目录(cwd)影响，这里用脚本所在目录拼默认路径，避免从哪里运行都找不到。
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # 数据目录
    batch_size = 128         # 批次大小
    epochs = 40              # 训练轮数
    lr = 1e-3               # 学习率
    weight_decay = 5e-4      # 权重衰减
    patience = 10            # 早停耐心值
    lr_patience = 5          # 学习率调整耐心值
    min_lr = 1e-6           # 最小学习率

# ====================================
def main():
    args = Args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    x_train, y_train, x_test, y_test = load_numpy_cifar(args.data_dir)

    train_ds = NumpyCIFARDataset(x_train, y_train, augment=True)
    test_ds = NumpyCIFARDataset(x_test, y_test, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model / optimizer / loss
    # 模型、损失函数、优化器和学习率调度器

    # 将模型移动到GPU（如果可用）
    model = CifarCNN(num_classes=len(CLASSES)).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    # 采用 ReduceLROnPlateau，根据验证集损失调整学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)

    # Training with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nFinal evaluation on test set:")
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {val_loss:.4f} | acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()