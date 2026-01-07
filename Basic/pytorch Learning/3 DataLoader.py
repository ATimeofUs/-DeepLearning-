import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from matplotlib import pyplot as plt
from PIL import Image


def main():
    """
        dataset	数据集对象	必填
        batch_size	每批多少样本	训练时必用
        shuffle	是否打乱	训练集设 True
        num_workers	多进程加载数据 CPU 多时设大一点，注意每个epoch都会销毁重建进程
        pin_memory=True 加速GPU
    """

    data_loader = DataLoader(

    )


if __name__ == "__main__":
    main()