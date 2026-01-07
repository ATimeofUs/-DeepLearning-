import torch
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

size = (224, 224)

# 随机裁剪
torchvision.transforms.RandomCrop(size, padding = None, pad_if_needed = False, fill = 0, padding_mode ='constant')
# 中心裁剪
torchvision.transforms.CenterCrop(size)
# 随机缩放裁剪
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# 五点裁剪
torchvision.transforms.FiveCrop(size)
# 随机旋转
torchvision.transforms.RandomRotation(1, expand=False, center=None)
# 转为Tensor数据
torchvision.transforms.ToTensor()
# 归一化
torchvision.transforms.Normalize(0, 1)
# pad填充图片
torchvision.transforms.Pad(1, fill=0, padding_mode='constant')
# 调整图片色彩饱和度、对比度、亮度、色调
torchvision.transforms.ColorJitter(brightness=0, contrast=1, saturation=0, hue=0)
# 转为灰度图
torchvision.transforms.Grayscale(num_output_channels=1)

# 组合多个变换操作
image_transform = transforms.Compose([
    transforms.RandomCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def main():
    img = Image.open(r"D:\27019\Pictures\尻图\大西王\f47bd2f5e0ccc0363f1f3bebb2012d45.jpg")
    img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.FiveCrop(size)
    ])

    # FiveCrop 返回的是一个包含5张裁剪图的元组
    img = transform(img)

    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        axs[i].imshow(img[i].permute(1, 2, 0))
        axs[i].axis('off')
    plt.show()

def main2():
    img = Image.open(r"D:\27019\Pictures\尻图\大西王\f47bd2f5e0ccc0363f1f3bebb2012d45.jpg")
    img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.3)
    ])

    img = transform(img)

    plt.imshow(img.permute(1, 2, 0)) # permute用于重新排列维度，将通道维度放到最后
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # main()
    main2()