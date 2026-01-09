import os
import pickle
import argparse
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim
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


# -----------------------------
# Model (CNN)
# -----------------------------
class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    执行一个训练周期（Epoch）的函数。
    :param model: 要训练的神经网络模型
    :param loader: 数据加载器 (DataLoader)，用于分批次提供数据
    :param criterion: 损失函数 (Loss Function)，比如 CrossEntropyLoss
    :param optimizer: 优化器 (Optimizer)，比如 SGD 或 Adam，用于更新参数
    :param device: 设备 (CPU 或 cuda)，指定在哪里计算
    :param scaler: 混合精度训练的缩放器 (可能为 None)
    """

    # 1. 开启训练模式
    # 这非常重要！这会启用 Dropout（随机丢弃神经元）和 BatchNorm（批归一化）的训练行为。
    # 如果不写这行，模型可能会像在测试时一样运行，导致无法正常训练。
    model.train()

    # 初始化统计变量，用于记录这一轮的总体表现
    running_loss = 0.0  # 累计总损失
    correct = 0  # 累计预测正确的数量
    total = 0  # 累计处理的图片总数

    # 2. 遍历数据加载器
    # tqdm 是那个进度条工具，desc="Train" 会在进度条前面显示 "Train"
    for images, labels in tqdm(loader, desc="Train", leave=False):

        # 3. 将数据移动到计算设备 (如 GPU)
        images, labels = images.to(device), labels.to(device)

        # 4. 清空梯度
        # 在 PyTorch 中，梯度是会累加的。
        # 每次计算新的一批数据前，必须把上一次留下的梯度清零，否则梯度会乱掉。
        # set_to_none=True 比 =0 稍微快一点点。
        optimizer.zero_grad(set_to_none=True)

        # 5. 前向传播 (Forward Pass)
        # torch.amp.autocast: 这是一个上下文管理器。
        # 如果 scaler 存在 (enabled=True)，它会自动把部分计算转为 float16 (半精度) 以加速。
        outputs = model(images)  # 模型预测
        loss = criterion(outputs, labels)  # 把预测结果和真实标签对比，计算"损失"(误差大小)

        # 6. 反向传播 (Backward Pass) 和 参数更新 (Optimizer Step)
        if scaler:
            # === 如果开启了混合精度训练 (AMP) ===
            # 将 loss 放大 (scale)，防止半精度下梯度太小变成 0
            scaler.scale(loss).backward()
            # 更新模型参数
            scaler.step(optimizer)
            # 更新缩放因子，为下一批次做准备
            scaler.update()
        else:
            # === 普通训练模式 ===
            loss.backward()  # 根据误差计算每个参数的梯度 (求导)
            optimizer.step()  # 根据梯度更新参数

        # 7. 统计指标
        # loss.item() 取出的是一个标量数值（平均 Loss）。
        # 乘以 labels.size(0) (即 batch_size) 是为了还原出这一批次的总 Loss，方便后面求全局平均。
        running_loss += loss.item() * labels.size(0)

        # 计算准确率
        # outputs 是一个概率分布 (比如 [0.1, 0.8, 0.1])
        # .max(1) 找出每一行概率最大的那个类别下标 (preds)
        _, preds = outputs.max(1)

        # 比较预测值 (preds) 和 真实值 (labels) 是否相等，累加正确的数量
        correct += preds.eq(labels).sum().item()

        # 累加处理的总样本数
        total += labels.size(0)

    # 8. 返回本轮的平均损失和平均准确率
    return running_loss / total, correct / total


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


# === 替代 argparse 的简单配置部分 ===
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
    amp = False              # 是否使用混合精度 (True/False)


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

    # 混合精度训练的梯度缩放器
    """
    通常神经网络计算使用的是 float32（32位浮点数）。
    为了加速，我们可以让显卡使用 float16（16位浮点数）。它的计算速度快，占内存少。
    这个函数将帮助我们在使用 float16 时，动态调整梯度的大小，防止数值溢出。
    """
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # Training with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
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