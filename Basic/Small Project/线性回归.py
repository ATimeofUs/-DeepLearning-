# python
import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l
from pyexpat import features

# 选择设备：优先 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 使用随机函数，生成数据集（在 device 上）
def synthetic_data(w, b, num_examples):
    """生成 y = xw + b + 随机噪声（所有张量在 device 上）"""
    x = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape, device=device)
    return x, y.reshape((-1, 1))

# 可视化数据（把张量移到 CPU 再转 numpy）
def show_feature_label():
    plt.figure(figsize=(6, 4))
    plt.scatter(feature[:, 0].detach().cpu().numpy(), labels.detach().cpu().numpy(),
                s=10, c='blue', label='feature 1')
    plt.scatter(feature[:, 1].detach().cpu().numpy(), labels.detach().cpu().numpy(),
                s=10, c='red', label='feature 2')
    plt.xlabel('Feature values')
    plt.ylabel('Labels')
    plt.legend()
    plt.title('Synthetic data: features vs labels')
    plt.show()

# 小批量随机梯度下降（使用全局 device 索引）
def data_iterator(batch_size, features, labels):
    """将原来的数据 切成很多小份（返回在 device 上的批次）"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)],
                                     dtype=torch.long, device=device)
        yield features[batch_indices], labels[batch_indices]

# 线性回归模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b

# 损失函数
def loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()

# 优化函数
def sgd(params, batch_size, lr=0.01):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

if __name__ == '__main__':
    ans_w = torch.tensor([2, -3.4], device=device)
    ans_b = torch.tensor(4.2, device=device)
    num = 500

    feature, labels = synthetic_data(ans_w, ans_b, num)
    # show_feature_label()

    batch_size = 20

    # 初始化 w b，在 device 上并开启梯度
    w = torch.zeros(size=(2, 1), requires_grad=True, device=device)
    b = torch.zeros(size=(1, 1), requires_grad=True, device=device)

    lr = 0.03
    num_epochs = 6  # 训练30个周期
    net = linreg
    loss_fn = loss

    print('training...')

    for epoch in range(num_epochs):
        for X, y in data_iterator(batch_size, feature, labels):
            y_hat = net(X, w, b)
            l = loss_fn(y_hat, y)
            l.backward()
            sgd([w, b], batch_size, lr)
        with torch.no_grad():
            train_l = loss_fn(net(feature, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l):f}')

    # 输出估计误差（把结果移到 CPU 打印更方便）
    print(f'w的估计误差: {(ans_w - w.reshape(ans_w.shape)).cpu()}')
    print(f'b的估计误差: {(ans_b - b).cpu()}')
