import os
import pandas as pd
import torch

def temp1():
    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_file = os.path.join(".", "data", "test1.csv")  # 改成csv后缀

    with open(data_file, "w") as f:
        f.write("Name,Phone,age\n")
        f.write("ZCR,NaN,1\n")
        f.write("Alice,002,2\n")
        f.write("NaN,003,3\n")
        f.write("Carl,004,4\n")
        f.write("Carl,005,5\n")

    data = pd.read_csv(data_file)
    # print(data)

    # 处理缺失数据，可以把nan看做同一类
    X = data[["Name", "Phone"]]  # 输入特征
    Y = data["age"]              # 输出标签

    print(X)
    X = pd.get_dummies(X, dummy_na=True)
    print(X)

def temp2():
    A = torch.arange(10).reshape(2, 5)
    print("this is A")
    print(A)
    sum_a = A.sum(dim=1,keepdim=True) # 保持维度，可以更方便使用广播机制
    print("this is sum_a")
    print(sum_a)

    print(A / sum_a)

def temp3():
    pass


if __name__ == "__main__":
    temp2()
