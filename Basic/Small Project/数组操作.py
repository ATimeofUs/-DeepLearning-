import torch

def temp1():
    # x是一个张量
    # 逐渐递增
    x = torch.arange(12)
    print(x)

    # 1
    x = torch.zeros(2,2)
    print(x)

    # 0
    x = torch.ones(2,2)
    print(x)

    # 特定值
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x)

def temp2():
    # 运算
    x = torch.arange(3, dtype=torch.float)
    y = torch.tensor([5,12,2])

    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)

def temp3():
    a = torch.arange(3).reshape(3, 1)
    b = torch.arange(3).reshape(1, 3)

    # 广播机制，注意出错
    print(a + b)

if __name__ == "__main__":
    # temp1()
    # temp2()
    temp3()