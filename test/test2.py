import glob
import os
import cv2
import torch
import numpy as np

from torch import nn
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------
# 配置
# --------------------
class Args:
    images_dir = r'D:\__SelfCoding\Deep_learning\Computer Vision\data\coco128\images\train2017'
    labels_dir = r"D:\__SelfCoding\Deep_learning\Computer Vision\data\coco128\labels\train2017"
    batch_size = 8
    img_size = 640
    split_ratio = 0.9
    device_state = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    num_workers = 4
    lr = 1e-3
    num_classes = 1
    anchors = [(10, 13), (16, 30), (33, 23)]
    stride = 32  # 经过 5 次 2x2 pool，640 -> 20，640/20=32


# --------------------
# 数据加载
# --------------------
def load_mydata(
    data_dir: str,
    label_dir: str,
    img_size,
    split_ratio: float = 0.9,
    shuffle_seed: int = 42):

    img_paths = sorted(
        glob.glob(os.path.join(data_dir, "*.jpg")) +
        glob.glob(os.path.join(data_dir,  "*.png")) +
        glob.glob(os.path.join(data_dir, "*.jpeg"))
    )

    pairs: List[tuple] = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(lp):
            pairs.append((ip, lp))

    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pairs)

    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    target_h = target_w = img_size

    def load_split(split_pairs):
        imgs, labels = [], []
        for ip, lp in split_pairs:
            # 1) 读取图像（默认 BGR 三通道）
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 2) BGR -> RGB（很多模型/可视化期望 RGB）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3) 按目标尺寸缩放 (target_w, target_h)
            img = cv2.resize(img, dsize=(target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 4) 保存到列表，确保是 uint8
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
        y = labels  # list of (n,5) arrays
        return X, y

    X_train, y_train = load_split(train_pairs)
    X_val, y_val = load_split(val_pairs)
    return X_train, y_train, X_val, y_val

class MyDataset(Dataset):
    def __init__(self, np_X, list_y):
        self.trans = transforms.ToTensor()
        self.images = np_X
        self.labels = list_y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # HWC, uint8
        labels = self.labels[idx]  # ndarray shape (n,5) [cls, cx, cy, w, h] (normalized 0~1)
        img = self.trans(img)  # CHW, float32 in [0,1]
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return img, labels

def collate_fn(batch):
    # batch: 长度 = B 的列表，每个元素是 (img, targets)

    # 1) 拆分图片和标注，得到两个元组 imgs, targets
    imgs, targets = list(zip(*batch))

    # 2) 把所有图片堆叠到 batch 维，得到 (B, C, H, W)
    imgs = torch.stack(imgs, 0)

    new_targets = []
    for i, t in enumerate(targets):
        # 3) 如果这个样本没有标注，跳过
        if t.numel() == 0:
            continue
        # 4) 给每条标注加一列“batch 索引” i，形状 (n, 1)
        batch_col = torch.full((t.shape[0], 1), i, dtype=t.dtype)
        # 5) 拼成 (n, 6): [batch_id, cls, cx, cy, w, h]
        new_targets.append(torch.cat((batch_col, t), dim=1))

    # 6) 把所有样本的标注在 0 维拼成一个大表 (M, 6)，M 为本 batch 的总框数
    if len(new_targets):
        new_targets = torch.cat(new_targets, dim=0)
    else:
        # 整个 batch 都没标注时，返回空张量
        new_targets = torch.zeros((0, 6), dtype=torch.float32)

    # 7) 返回图片张量 (B, C, H, W) 和合并后的标注 (M, 6)
    return imgs, new_targets

class YOLOLayer(nn.Module):
    """Decode raw preds to pixel-space boxes."""
    def __init__(self, anchors, num_classes=1, stride=32):
        """
        anchors : List of tuple = [(10, 13), (16, 30), (33, 23)]
        num_classes: number of classes
        """
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2) # (nA, 2)
        self.num_anchors = self.anchors.size(0)
        self.num_classes = num_classes
        self.stride = stride
        self.num_outputs = 5 + num_classes
        self.grid = None

    def forward(self, x):
        # x: (B, nA*(5+nc), ny, nx)
        b, _, ny, nx = x.shape
        device = x.device
        x = x.view(b, self.num_anchors, self.num_outputs, ny, nx)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  
        # (B, nA, ny, nx, no)

        if (self.grid is None
                or self.grid.shape[2:4] != (ny, nx)
                or self.grid.device != device):
            self.grid = self._make_grid(nx, ny, device)

        
        anchors = (
            (self.anchors.to(device) / self.stride).
            view(1, self.num_anchors, 1, 1, 2)
        )

        # raw predictions
        raw = x

        # decoded
        xy = torch.sigmoid(raw[..., 0:2]) + self.grid  # center
        wh = torch.exp(raw[..., 2:4]) * anchors        # wh
        conf = torch.sigmoid(raw[..., 4:5])
        cls = torch.sigmoid(raw[..., 5:])

        decoded = torch.cat((xy * self.stride, wh * self.stride, conf, cls), dim=-1)
        # shapes:
        # decoded: (B, nA, ny, nx, 5+nc) in pixel coords for xywh
        # raw    : same shape, before sigmoid/exp on each part
        return decoded, raw

    @staticmethod
    def _make_grid(nx: int, ny: int, device=None) -> torch.Tensor:
        yv, xv = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32, device=device),
            torch.arange(nx, dtype=torch.float32, device=device),
            indexing="ij"
        )
        grid = torch.stack((xv, yv), dim=-1)
        return grid.view(1, 1, ny, nx, 2)


class Network(nn.Module):
    def __init__(self, num_classes=1, anchors=None, stride=32):
        super().__init__()
        if anchors is None:
            anchors = [(10, 13), (16, 30), (33, 23)]
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.stride = stride
        self.yolo = YOLOLayer(anchors, num_classes=num_classes, stride=stride)

        # 简化的特征提取网络
        def conv_bn_act(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_bn_act(3, 32),
            nn.MaxPool2d(2),
            conv_bn_act(32, 64),
            nn.MaxPool2d(2),
            conv_bn_act(64, 128),
            nn.MaxPool2d(2),
            conv_bn_act(128, 256),
            nn.MaxPool2d(2),
            conv_bn_act(256, 512),
            nn.MaxPool2d(2),
            conv_bn_act(512, 128),
            conv_bn_act(128, 128),
        )
        self.prediction = nn.Conv2d(128, self.num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        preds = self.prediction(feats)
        decoded, raw = self.yolo(preds)
        return decoded, raw  # decoded for inference, raw for loss



def main():
    """
    训练集图像形状: (115, 640, 640, 3)
    训练集标签数量: 115
    """
    pass



def main2():
    device = Args.device
    train_image, train_labels, val_image, val_labels = (
        load_mydata(Args.images_dir, Args.labels_dir, img_size=Args.img_size, split_ratio=Args.split_ratio))

    train_ds = MyDataset(train_image, train_labels)
    val_ds = MyDataset(val_image, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=Args.batch_size,
        shuffle=True,
        num_workers=Args.num_workers,
        pin_memory=Args.device_state == 1,
        persistent_workers=Args.num_workers > 0,
        prefetch_factor=2 if Args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=Args.device_state == 1,
        collate_fn=collate_fn,
    )

    """
    每一个 batch：
    imgs: (B, 3, 640, 640) float32 in [0,1]
    targets: (M, 6) float32: [batch_id, cls, cx, cy, w, h] (normalized 0 ~ 1)
    """

    model = Network(num_classes=Args.num_classes, anchors=Args.anchors, stride=Args.stride).to(device)
    criterion = YoloLoss(anchors=Args.anchors, num_classes=Args.num_classes, stride=Args.stride)
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.lr)

    model.train()
    loss = None
    loss_items = None
    for epoch in range(Args.epochs):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)  # (n,6): b,cls,cx,cy,w,h (normalized)

            decoded, raw = model(imgs)            # decoded unused in loss here
            loss, loss_items = criterion(raw, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{Args.epochs} | "
              f"loss={loss.item():.4f} | "
              f"l_box={loss_items['l_box']:.4f} l_wh={loss_items['l_wh']:.4f} "
              f"l_obj={loss_items['l_obj']:.4f} l_cls={loss_items['l_cls']:.4f}")

    # 简单验证（仅前向，不计算 mAP）
    model.eval()
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            decoded, _ = model(imgs)
            # decoded: (B, nA, ny, nx, 5+nc) in pixel coords
            # 在此可添加 NMS / 置信度过滤

if __name__ == "__main__":
    main()