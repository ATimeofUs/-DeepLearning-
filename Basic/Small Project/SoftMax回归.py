import torch
import torchvision
from d2l.tensorflow import download, Accumulator
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt

def get_fashion_mnist_labels(labels):
    """返回 Fashion-MNIST 数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    res = []
    for i in range(len(labels)):
        res.append(text_labels[int(labels[i])])
    return res

def show_image(imgs, num_row, num_col, titles=None, scale=1.5):
    """按网格显示图像，并在每张图下方显示标签"""
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    # 自动处理单通道 (N, 1, H, W) -> (N, H, W)
    if imgs.ndim == 4 and imgs.shape[1] == 1:
        imgs = imgs.squeeze(1)

    figsize = (num_col * scale, num_row * scale)
    fig, axes = plt.subplots(num_row, num_col, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.axis("off")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        # 关键点：将 labels (titles) 设置到对应的子图上
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.show()

def get_dataloader_workers():
    """
    使用 0 个进程来读取数据

    """
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载 Fashion-MNIST 数据集，将其加载到内存中，并返回训练集和测试集的迭代器
    """

    # 1. 组装预处理流程

    # ToTensor() 包含两个步骤：
    # 数值归一化 原始像素值范围是 [0, 255] -> [0, 1]
    # 通道预处理 原始图片通常是 (H, W, C)（高度、宽度、通道） -> (C, H, W)

    trans = [transforms.ToTensor()]
    if resize:
        # 如果有 resize 需求，插入到最前面（Resize 作用于 PIL Image）
        trans.insert(0, transforms.Resize(resize))

    # 将列表转化为 torchvision 的 Compose 对象，意思就是将列表trans的各个元素按顺序串联起来，形成一个新的变换操作
    # 比如在这里，如果resize != None，先 Resize（调整图片大小一致） 再 ToTensor
    trans = transforms.Compose(trans)

    # 2. 加载数据集
    # 训练集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    # 测试集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    # 3. 定义数据加载器
    # 这里我们调用之前讨论过的获取线程数的逻辑
    workers = get_dataloader_workers()
    # timer = d2l.Timer()

    train_iter = data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)  # pin_memory 对你的高性能显卡有加速作用

    test_iter = data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_iter, test_iter

def softmax(X):
    """
    X 是一个二维张量，表示批量数据，他的行数=批量大小，列数=类别数（10）
    将每一行的元素取指数，然后除以该行元素指数和，得到归一化的概率分布
    """
    X_exp = torch.exp(X)
    temp = X_exp.sum(1, keepdim=True)
    return X_exp / temp  # 这里应用了广播机制

def corss_enropy(y_hat, y):
    """
    交叉熵损失函数
    y_hat: 预测的概率分布，形状为（批量大小，类别数）
    y: 真实标签，形状为（批量大小，）
    """
    return -torch.log(y_hat[range(len(y_hat)), y])

def net(x):
    """
    SoftMax 回归模型
    X = 输入张量，形状为（批量大小，784）
    W = 权重矩阵，形状为（784，10）
    b = 偏置向量，形状为（10，）
    该函数返回预测的概率分布，形状为（批量大小，10）
    """
    return softmax(torch.matmul(x.reshape((-1, W.shape[0])), W) + b)

def accuracy(y_hat, y):
    """
    计算预测正确的数量
    y_hat: 预测的概率分布，形状为（批量大小，类别数）
    y: 真实标签，形状为（批量大小，）
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    preds = torch.argmax(y_hat, dim=1)
    cmp = preds.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(data_iter, net):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(accuracy(net(x, y), y.numel()))
    return metric[0] / metric[1]

if __name__ == "__main__":
    d2l.use_svg_display()
    batch_size = 256
    num_input = 28 * 28 # 每张图片的像素数，也是输入数
    num_output = 10     # 输出类别数

    """
        mnist_train[i][0] → 图片i 张量，形状 [1, 28, 28]。
        mnist_train[i][1] → 图片i 标签，整数（比如 5 表示数字 5）。
    """

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # SoftMax 回归参数
    # W 是权重矩阵 784 * 10 ，b 是偏置向量 10 * 1
    W = torch.normal(0, 0.01, size=(num_input, num_output), requires_grad=True)
    b = torch.zeros(num_output,  requires_grad=True)

"""帮我修改这个代码。使得新手也能很好理解，减少使用复杂的简洁语法、需要详细的注释"""