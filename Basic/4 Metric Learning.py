from typing import Tuple, List
import numpy as np
import torch

from torch import Tensor
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import datasets as torchvisionDatasets

from tqdm import tqdm
from os import path


class Args():
    workers = 4              # 数据加载线程数
    data_dir = path.join(path.dirname(path.abspath(__file__)), "data")  # 数据目录
    model_dir = path.join(path.dirname(path.abspath(__file__)), r"models/4 Metric Learning.pth") # 模型保存目录
    batch_size = 100         # 批次大小
    epochs = 60              # 训练轮数
    lr = 1e-3               # 学习率
    weight_decay = 5e-4      # 权重衰减
    patience = 10            # 早停耐心值
    lr_patience = 5          # 学习率调整耐心值
    min_lr = 1e-6           # 最小学习率
    loss_gate = 0.001
args = Args()

class TripletDataset(Dataset):
    def __init__(self, dataset):
        super(TripletDataset, self).__init__()

        self.dataset = dataset
        # 预先整理索引，提高 __getitem__ 效率
        self.labels = np.array(dataset.targets)
        self.index_pool = {i: np.where(self.labels == i)[0] for i in range(10)}

    def __getitem__(self, index):
            # Anchor
            img_a, label_a = self.dataset[index]
            
            # Positive: 从同类中随机选一个不同的
            pos_idx = index
            while pos_idx == index:
                pos_idx = np.random.choice(self.index_pool[label_a])
            img_p, _ = self.dataset[pos_idx]
            
            # Negative: 从异类中随机选一个
            neg_label = np.random.choice([l for l in range(10) if l != label_a])
            neg_idx = np.random.choice(self.index_pool[neg_label])
            img_n, _ = self.dataset[neg_idx]
            
            return img_a, img_p, img_n
    
    def __len__(self) -> int:
        return len(self.dataset)

class MyModel(nn.Module):
    def __init__(self, embedding_size: int = 64):
        super(MyModel, self).__init__()
        self.embedding_size = embedding_size
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fullc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.convnet(x)
        x = x.view(x.size(0), -1)  
        x = self.fullc(x)
        x = nn.functional.normalize(x, p=2, dim=1)  
        return x


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])

    train_dataset = torchvisionDatasets.MNIST(
        root=args.data_dir, 
        train=True, 
        download=True,
        transform=transform
    )

    train_dataset = TripletDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,  
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=True
    )

    model = MyModel(embedding_size=64).cuda()
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    for _ in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            optimizer.zero_grad()
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{_+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.model_dir)

def test():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])

    test_dataset = torchvisionDatasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_dataset = TripletDataset(test_dataset)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=False,
        pin_memory=False
    )
    
    model = MyModel(embedding_size=64).cuda()
    model.load_state_dict(torch.load(args.model_dir))
    model.eval()

    total_loss = 0.0
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    with torch.no_grad():
        for anchor, positive, negative in tqdm(test_loader):
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = criterion(anchor_output, positive_output, negative_output)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    # Test Loss: 0.0158

if __name__ == "__main__":
    # main()
    test()