import torch

def temp1():
    x = torch.arange(5, requires_grad=True, dtype=torch.float)
    y = torch.dot(x, x)

    # Tensor.backward()
    # 只能在标量 1 x 1（单个数值）输出上直接调用
    print(x.grad) # 这个时候为空
    y.backward() # 反向传播
    print(x.grad) # 这个时候才有

    # 梯度清空
    x.grad.zero_()

    y = x.sum()
    y.backward()
    print(x.grad)

if __name__ == '__main__':
    temp1()