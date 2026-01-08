import glob
import os
import cv2
import torch
import numpy as np

from torch import nn
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Args:
    images_dir = r'D:\__SelfCoding\Deep_learning\Computer Vision\data\coco128\images\train2017'
    labels_dir = r"D:\__SelfCoding\Deep_learning\Computer Vision\data\coco128\labels\train2017"
    batch_size = 8
    device_state = 1
    device = torch.device("cuda")


def load_mydata(
    data_dir: str,
    label_dir: str,
    img_size,
    split_ratio: float = 0.9,
    shuffle_seed: int = 42 ):

    """
    :param data_dir: 文件地址
    :param img_size: 需要返回的图片大小
    :param split_ratio: 划分数据集和测试集的比例
    :return: X_train, y_train, X_val, y_val
    """

    img_paths = sorted(
        glob.glob(os.path.join(data_dir, "*.jpg")) +
        glob.glob(os.path.join(data_dir, "*.png")) +
        glob.glob(os.path.join(data_dir, "*.jpeg"))
    )

    pairs: List[tuple] = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(label_dir, stem + ".txt")
        print(lp)
        if os.path.isfile(lp):
            pairs.append((ip, lp))

    # Shuffle data
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pairs)

    # 划分训练集和验证集
    split_idx = int(len(pairs) * split_ratio)

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # 目标图片大小
    target_h = target_w = img_size

    def load_split(split_pairs):
        imgs, labels = [], []
        for ip, lp in split_pairs:
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            imgs.append(img.astype(np.uint8))

            with open(lp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if lines:
                arr = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = map(float, parts)
                    arr.append([cls, cx, cy, w, h])
                arr = np.array(arr, dtype=np.float32) if arr else np.zeros((0, 5), dtype=np.float32)
            else:
                arr = np.zeros((0, 5), dtype=np.float32)
            labels.append(arr)

        X = np.stack(imgs, axis=0) if imgs else np.empty((0, target_h, target_w, 3), dtype=np.uint8)
        y = np.array(labels, dtype=object)
        return X, y

    X_train, y_train = load_split(train_pairs)
    X_val, y_val = load_split(val_pairs)
    return X_train, y_train, X_val, y_val


class MyDataset(Dataset):
    def __init__(self, np_X, np_y):
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.images = np_X
        self.labels = np_y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = self.trans(img)
        label = self.trans(label)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label


class MyModule(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

    def forward(self, x):
        return x

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors):
        super().__init__()
        pass

    def forward(self, x):
        pass

    @staticmethod
    def _make_grid(nx: int = 10, ny: int = 10) -> torch.Tensor:
        pass


class Network(nn.Module):
    def __init__(self, num_classes=10, anchors=3):
        super().__init__()
        pass

    def forward(self, x):
        pass

def main():
    train_image, train_labels, val_image, val_labels = (
        load_mydata(Args.images_dir, Args.labels_dir, img_size=640, split_ratio=0.9))

    """
    Train images shape: (115, 640, 640, 3)
    Train labels shape: (115,)
    Validation images shape: (13, 640, 640, 3)
    Validation labels shape: (13,)
    """

    """
        dataset	数据集对象	必填
        batch_size	每批多少样本	训练时必用
        shuffle	是否打乱	训练集设 True
        num_workers	多进程加载数据 CPU 多时设大一点，注意每个epoch都会销毁重建进程
        pin_memory=True 加速GPU
    """

    train_ds = MyDataset(train_image, train_labels)
    val_ds = MyDataset(val_image, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=Args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=Args.device_state == 1,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        val_ds,
        batch_size=Args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=Args.device_state == 1,
    )



if __name__ == "__main__":
    main()