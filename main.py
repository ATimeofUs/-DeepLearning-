import torch

# 创建一个有 27 个元素的张量
a = torch.arange(27)

print(a)
a = a.view(1, 3, 3, 3)
print(a)
