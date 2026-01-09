# Litter Yolo 代码解析

## class Args:

配置文件

## class load_mydata

加载数据

**glob.glob** Return a list of paths matching a pathname pattern.



**os.path.join** 路径整合，兼容不同系统

**os.path.basename** Returns the final component of a pathname 返回文件名

**os.path.splitext** Split the extension from a pathname. 分割扩展名



**cv2.imread** Loads an image from a file.

**cv2.cvtColor** Converts an image from one color space to another.



## collate_fn

- DataLoader 每次会取出若干个「单个样本」组成一个 batch。默认的合并逻辑只会把同结构的张量直接堆叠。
- 目标检测等任务，标注（targets）是“变长”的：同一张图里目标数不同，不能直接堆叠成规则张量。
- 因此我们自定义 `collate_fn`，决定“单个样本列表”如何合并为一个 batch，使模型能一次看到 (Batch, C, H, W) 的图片张量，以及拼起来的目标框。



### 

```
B = list(zip(*A))
```

可以理解为转置操作



### 

```python
a = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]

for i in enumerate(a):

  print(i)

输出：
(0, [1, 2, 3, 4])
(1, [5, 6, 7, 8])
(2, [9, 10, 11, 12])
```



### 

`numel()` 是 PyTorch 张量的一个方法，返回张量中元素的总数（number of elements）。



### 所以这个函数在干嘛？

原始 targets（按样本存储，变长）：
- 第 0 张图：
  ```
  [[1.0000, 0.5000, 0.5000],
   [3.0000, 0.1000, 0.2000]]
  ```
- 第 1 张图：`[]`（无标注）
- 第 2 张图：
  ```
  [[2.0000, 0.9000, 0.9000]]
  ```

转换后 new_targets（按 batch 展平，每行: [batch_id, cls, cx, cy, w, h]）：
- 来自第 0 张图 (batch_id = 0)：
  ```
  [0.0000, 1.0000, 0.5000, 0.5000, 0.2000, 0.3000]
  [0.0000, 3.0000, 0.1000, 0.2000, 0.4000, 0.4000]
  ```
- （第 1 张图无框，跳过）
- 来自第 2 张图 (batch_id = 2)：
  ```
  [2.0000, 2.0000, 0.9000, 0.9000, 0.1000, 0.1000]
  ```

合在一起就是：
```
tensor([[0.0000, 1.0000, 0.5000, 0.5000, 0.2000, 0.3000],
        [0.0000, 3.0000, 0.1000, 0.2000, 0.4000, 0.4000],
        [2.0000, 2.0000, 0.9000, 0.9000, 0.1000, 0.1000]])
```



```python
[
tensor([[1.0000, 0.5000, 0.5000, 0.2000, 0.3000],
        [3.0000, 0.1000, 0.2000, 0.4000, 0.4000]]), 
 tensor([], size=(0, 5)), 
tensor([[2.0000, 0.9000, 0.9000, 0.1000, 0.1000]])
]
# 加上ID，并且去掉空
[tensor([[0.0000, 1.0000, 0.5000, 0.5000, 0.2000, 0.3000],
        [0.0000, 3.0000, 0.1000, 0.2000, 0.4000, 0.4000]]), tensor([[2.0000, 2.0000, 0.9000, 0.9000, 0.1000, 0.1000]])]
```



## class YOLOLayer

### 参数

​    self.anchors = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)

​    self.num_anchors = self.anchors.size(0)

​    self.num_classes = num_classes

​    self.stride = stride

​    self.num_outputs = 5 + num_classes

​    self.grid = None



### 矩阵大小

| 名称 / 步骤                 | 形状                   | 说明                                |
| --------------------------- | ---------------------- | ----------------------------------- |
| 输入 x                      | (B, nA*(5+nc), ny, nx) | 原始特征图输出                      |
| reshape                     | (B, nA, no, ny, nx)    | view(b, nA, 5+nc, ny, nx)           |
| permute → raw               | (B, nA, ny, nx, no)    | permute(0,1,3,4,2)                  |
| grid（缓存）                | (1, 1, ny, nx, 2)      | `_make_grid` 生成的网格坐标         |
| anchors（归一化后）         | (1, nA, 1, 1, 2)       | anchors/stride，匹配特征图尺度      |
| raw[..., 0:2] (tx, ty)      | (B, nA, ny, nx, 2)     | 中心偏移（logits）                  |
| raw[..., 2:4] (tw, th)      | (B, nA, ny, nx, 2)     | 宽高（logits）                      |
| raw[..., 4:5] (obj)         | (B, nA, ny, nx, 1)     | 目标置信度（logits）                |
| raw[..., 5:] (cls)          | (B, nA, ny, nx, nc)    | 类别（logits）                      |
| xy = sigmoid(tx, ty) + grid | (B, nA, ny, nx, 2)     | 特征图坐标系下的中心                |
| wh = exp(tw, th) * anchors  | (B, nA, ny, nx, 2)     | 特征图坐标系下的宽高                |
| conf = sigmoid(obj)         | (B, nA, ny, nx, 1)     | 目标置信度                          |
| cls = sigmoid(cls logits)   | (B, nA, ny, nx, nc)    | 类别概率                            |
| decoded = concat(xy*stride, |                        | 拼接后的输出：                      |
| wh*stride, conf, cls)       | (B, nA, ny, nx, 5+nc)  | xy/wh 已乘 stride 转到像素坐标      |
| 返回 raw（未解码）          | (B, nA, ny, nx, 5+nc)  | 与 decoded 同形状，未做 sigmoid/exp |
