import glob
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

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
    """
    collate_fn 用于 DataLoader，将单个样本列表合并为一个 batch。
    """
    # batch 是 DataLoader 传入的单个 batch 列表，每个元素为 (img, targets)
    # imgs: 元组[Tensor]，targets: 元组[Tensor或空]
    imgs, targets = list(zip(*batch))  

    # 将图片张量按 batch 维堆叠 -> (B, C, H, W)
    imgs = torch.stack(imgs, 0)

    new_targets = []
    for i, t in enumerate(targets):
        # 如果该样本没有标注（空），跳过
        if t.numel() == 0:
            continue
        # 给每条标注前面加一列 batch 索引，形状 (n,1)
        batch_col = torch.full((t.shape[0], 1), i, dtype=t.dtype)
        # 接后形状 (n,6): [batch_id, cls, cx, cy, w, h]，坐标仍为归一化格式
        new_targets.append(torch.cat((batch_col, t), dim=1))

    if len(new_targets):
        # 将不同样本的标注在 0 维拼接 -> (M,6)
        new_targets = torch.cat(new_targets, dim=0)
    else:
        # 若整个 batch 没有任何标注，返回空张量
        new_targets = torch.zeros((0, 6), dtype=torch.float32)

    # 返回堆叠后的图片和合并后的 targets（含 batch 索引）
    return imgs, new_targets

# --------------------
# 模型
# --------------------
class YOLOLayer(nn.Module):
    """Decode raw preds to pixel-space boxes."""
    def __init__(self, anchors, num_classes=1, stride=32):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)
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
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (B, nA, ny, nx, no)

        if (self.grid is None
                or self.grid.shape[2:4] != (ny, nx)
                or self.grid.device != device):
            self.grid = self._make_grid(nx, ny, device)

        anchors = (self.anchors.to(device) / self.stride).view(1, self.num_anchors, 1, 1, 2)

        # raw predictions
        raw = x

        # decoded
        xy = torch.sigmoid(raw[..., 0:2]) + self.grid  # center
        wh = torch.exp(raw[..., 2:4]) * anchors        # wh
        conf = torch.sigmoid(raw[..., 4:5]) # 检测置信度
        cls = torch.sigmoid(raw[..., 5:]) # 类别置信度

        decoded = torch.cat((xy * self.stride, wh * self.stride, conf, cls), dim=-1)
        # shapes:
        # decoded: (B, nA, ny, nx, 5+nc) in pixel coords for xywh
        # raw    : same shape, before sigmoid/exp on each part
        
        breakpoint()
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


# --------------------
# 目标分配与损失
# --------------------
def build_targets(
    targets,      # (n,6): b,cls,cx,cy,w,h in normalized coords
    anchors,      # tensor (nA,2) pixel anchors
    stride,       # int
    feat_h, feat_w,
    num_classes,
    device,
):
    """
    返回：
    tbox:   (B, nA, feat_h, feat_w, 4)
    tconf:  (B, nA, feat_h, feat_w, 1)
    tcls:   (B, nA, feat_h, feat_w, num_classes)
    """
    tbox = torch.zeros((targets[:,0].max().int().item()+1 if targets.numel() else 1,
                        anchors.shape[0], feat_h, feat_w, 4), device=device)
    tconf = torch.zeros((tbox.shape[0], anchors.shape[0], feat_h, feat_w, 1), device=device)
    tcls = torch.zeros((tbox.shape[0], anchors.shape[0], feat_h, feat_w, num_classes), device=device)

    if targets.numel() == 0:
        return tbox, tconf, tcls

    gxy = targets[:, 2:4] * torch.tensor([feat_w, feat_h], device=device)  # center in grid coords
    gwh = targets[:, 4:6] * torch.tensor([feat_w * stride, feat_h * stride], device=device) / stride  # normalize to anchor space
    gij = gxy.long()
    gi, gj = gij[:, 0], gij[:, 1]

    # anchor matching by ratio
    anchor_wh = anchors.to(device) / stride  # to grid units
    ratios = gwh[:, None, :] / anchor_wh[None]  # (nT, nA, 2)
    inv_ratios = anchor_wh[None] / (gwh[:, None, :] + 1e-9)
    max_ratios = torch.max(ratios, inv_ratios).max(dim=2).values  # (nT, nA)
    best_anchors = max_ratios.argmin(dim=1)  # choose best anchor

    for ti, a in enumerate(best_anchors):
        b = int(targets[ti, 0])
        cls = int(targets[ti, 1])
        if gj[ti] < feat_h and gi[ti] < feat_w:
            tbox[b, a, gj[ti], gi[ti], 0:2] = gxy[ti] - gij[ti]          # offset within cell (0~1)
            tbox[b, a, gj[ti], gi[ti], 2:4] = torch.log(
                targets[ti, 4:6] * torch.tensor([feat_w * stride, feat_h * stride], device=device) / anchors[a] + 1e-16
            )
            tconf[b, a, gj[ti], gi[ti], 0] = 1.0
            if num_classes > 1:
                tcls[b, a, gj[ti], gi[ti], cls] = 1.0
    return tbox, tconf, tcls


class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, stride):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_classes = num_classes
        self.stride = stride
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_pos = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.l1 = nn.SmoothL1Loss()

    def forward(self, raw_pred, targets):
        # raw_pred: (B, nA, ny, nx, 5+nc) BEFORE sigmoid/exp
        device = raw_pred.device
        b, nA, ny, nx, no = raw_pred.shape

        anchors = self.anchors.to(device)
        tbox, tconf, tcls = build_targets(
            targets, anchors, self.stride, ny, nx, self.num_classes, device
        )

        # predicted parts
        p_xy = torch.sigmoid(raw_pred[..., 0:2])
        p_wh = raw_pred[..., 2:4]  # to be compared with logged targets
        p_obj = raw_pred[..., 4:5]
        p_cls = raw_pred[..., 5:]

        # box loss: offset + wh log-space
        l_box = self.l1(p_xy[tconf.bool().expand_as(p_xy)], tbox[..., 0:2][tconf.bool().expand_as(tbox[..., 0:2])]) \
            if tconf.sum() > 0 else torch.tensor(0., device=device)
        l_wh = self.l1(p_wh[tconf.bool().expand_as(p_wh)], tbox[..., 2:4][tconf.bool().expand_as(tbox[..., 2:4])]) \
            if tconf.sum() > 0 else torch.tensor(0., device=device)

        # obj loss
        l_obj = self.bce(p_obj, tconf)

        # cls loss
        if self.num_classes > 1:
            l_cls = self.bce(p_cls, tcls)
        else:
            # binary: treat as confidence of class 0 = 1 when object exists
            l_cls = torch.tensor(0., device=device)

        loss = l_box + l_wh + l_obj + l_cls
        return loss, {"l_box": l_box.item(), "l_wh": l_wh.item(),
                      "l_obj": l_obj.item(), "l_cls": l_cls.item() if isinstance(l_cls, torch.Tensor) else l_cls}
    


# --------------------
# 训练与验证（简化）
# --------------------
def main():
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

    model = Network(num_classes=Args.num_classes, anchors=Args.anchors, stride=Args.stride).to(device)
    criterion = YoloLoss(anchors=Args.anchors, num_classes=Args.num_classes, stride=Args.stride)
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.lr)

    model.train()

    for epoch in range(Args.epochs):
        tpdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Args.epochs}")

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)  # (n,6): b,cls,cx,cy,w,h (normalized)

            decoded, raw = model(imgs)            # decoded unused in loss here
            loss, loss_items = criterion(raw, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpdm.set_postfix(loss=f"{loss.item():.4f}",
                           l_box=f"{loss_items['l_box']:.4f}",
                           l_wh=f"{loss_items['l_wh']:.4f}",
                           l_obj=f"{loss_items['l_obj']:.4f}",
                           l_cls=f"{loss_items['l_cls']:.4f}")

        print(f"Epoch {epoch+1}/{Args.epochs} | "
              f"loss={loss.item():.4f} | "
              f"l_box={loss_items['l_box']:.4f} l_wh={loss_items['l_wh']:.4f} "
              f"l_obj={loss_items['l_obj']:.4f} l_cls={loss_items['l_cls']:.4f}")




if __name__ == "__main__":
    main()