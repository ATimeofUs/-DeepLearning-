import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def main():
    # 使用 TensorBoard 记录数据
    # tensorboard --logdir="D:\__SelfCoding\Deep_learning\logs"
    writer = SummaryWriter(log_dir=r"D:\__SelfCoding\Deep_learning\logs")
    img = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
         [[255, 255, 0], [0, 255, 255], [255, 0, 255]]], dtype=np.uint8)

    # 注意需要指定 dataformats='HWC'
    writer.add_image('img', img, 0, dataformats='HWC')

    for i in range(100):
        y = i * i
        x = i
        # 记录标量数据
        writer.add_scalar('y=x', y, x)

    writer.close()

if __name__ == '__main__':
    main()