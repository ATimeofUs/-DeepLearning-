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

`YOLOLayer` 这层的作用：把网络最后一层卷积输出的“原始预测张量”（raw logits）**整理形状并解码成可解释的检测结果**（边框位置/尺寸、置信度、类别概率），同时把**未解码的 raw** 也返回给 loss 用。

------

**输入是什么**

- `x` ：来自 Network.prediction 的输出，形状是 $(B,  nA×(5+nc),  ny,  nx)$

  其中：

  - `B`：batch size
  - `nA`：anchor 数（你这里是 3）
  - `nc`：类别数（你这里 `num_classes=1`）
  - `ny, nx`：特征图高宽（你的 backbone 经过 5 次池化，`640 -> 20`，所以通常是 `20×20`）

`x` 的通道维里，每个 anchor 对应一组 `(5+nc)`：

- 0:2：tx, ty（中心点偏移的 raw）
- 2:4：tw, th（宽高的 raw）
- 4：objectness（有无目标）
- 5:：分类 raw（nc 个）

------

**输出是什么**
`forward()` 返回两个东西：`decoded, raw`

1. `decoded`：**解码后的预测（用于推理/可视化）**形状：$(B,  nA,  ny,  nx,  5+nc)$

   内容是：

   - `[x, y, w, h]`：**像素坐标系**下的 `xywh`（代码里乘了 `stride`，所以是像素尺度）
   - `conf`：objectness，做了 `sigmoid`
   - `cls`：类别概率（这里用 `sigmoid`，更像多标签/二分类风格；如果多类单标签通常用 softmax）

2. `raw`：**未解码的原始预测（用于计算 loss）**
   形状同 `decoded`：$(B,  nA,  ny,  nx,  5+nc)$

   但内部值还是 logits / log-space（例如 wh 还没 exp，obj/cls 还没 sigmoid）。

------

**它内部做了哪些关键变换（直观理解）**

- reshape + permute：把 `(B, nA*(5+nc), ny, nx)` 变成 `(B, nA, ny, nx, 5+nc)`，方便按 anchor/网格处理
- 构建 `grid`：每个网格点的左上角坐标 `(i,j)`
- 解码公式（你这份实现）：
  - $xy=σ(txy)+grid$（得到网格坐标系下中心点）
  - $wh=exp⁡(twh)⋅anchor$（得到网格坐标系下宽高）
  - 然后整体乘以 `stride` 变成像素尺度



### 参数

​    `self.anchors` = torch.tensor(anchors, dtype=torch.float32).view(-1, 2)

​    `self.num_anchors` = self.anchors.size(0)

   ` self.num_classes` = num_classes

  ` self.stride` = stride

   ` self.num_outputs` = 5 + num_classes

​    `self.grid` = None



### 矩阵大小

| 名称 / 步骤                                                  | 形状                   | 说明                                         |
| ------------------------------------------------------------ | ---------------------- | -------------------------------------------- |
| 输入 x                                                       | (B, nA*(5+nc), ny, nx) | 原始特征图输出                               |
| reshape                                                      | (B, nA, no, ny, nx)    | view(b, nA, 5+nc, ny, nx)                    |
| permute → raw                                                | (B, nA, ny, nx, no)    | permute(0,1,3,4,2)                           |
| grid（缓存）                                                 | (1, 1, ny, nx, 2)      | `_make_grid` 生成的网格坐标                  |
| anchors（归一化后）                                          | (1, nA, 1, 1, 2)       | anchors/stride，匹配特征图尺度               |
| raw[..., 0:2] (tx, ty)                                       | (B, nA, ny, nx, 2)     | 中心偏移（logits）                           |
| raw[..., 2:4] (tw, th)                                       | (B, nA, ny, nx, 2)     | 宽高（logits）                               |
| raw[..., 4:5] (obj)                                          | (B, nA, ny, nx, 1)     | 目标置信度（logits）                         |
| raw[..., 5:] (cls)                                           | (B, nA, ny, nx, nc)    | 类别（logits）                               |
| xy = <span style="background:#33FF33;">sigmoid</span>(tx, ty) + grid | (B, nA, ny, nx, 2)     | 特征图坐标系下的中心                         |
| wh = exp(tw, th) * anchors                                   | (B, nA, ny, nx, 2)     | 特征图坐标系下的宽高                         |
| conf = <span style="background:#33FF33;">sigmoid</span>(obj) | (B, nA, ny, nx, 1)     | 目标置信度 confidence                        |
| cls = <span style="background:#33FF33;">sigmoid</span>(cls logits) | (B, nA, ny, nx, nc)    | 类别概率 class probabilities                 |
| decoded = concat(xy`*`stride, wh`*`stride, conf, cls)        | (B, nA, ny, nx, 5+nc)  | 拼接后的输出：xy/wh 已乘 stride 转到像素坐标 |
| 返回 raw（未解码）                                           | (B, nA, ny, nx, 5+nc)  | 与 decoded 同形状，未做 sigmoid/exp          |



### grid

基本形状 

```python
 torch.Size([1, 1, 2, 3, 2])
 For example:
 tensor([[[0., 0.],
         [1., 0.],
         [2., 0.]],

        [[0., 1.],
         [1., 1.],
         [2., 1.]]])
```

用法

```python
xy = torch.sigmoid(raw[..., 0:2]) + self.grid 
```

注意 `raw.shape` = [B, nA, ny, nx, no + 5]

所以 `raw[..., 0:2]` = [B, nA, ny, nx, 2]

`grid` = [1, 1, ny, nx, 2]

 通过广播机制，`gird`能运用于每一个`batch`中，将所有相对坐标映射到绝对坐标上



### anchors

理解为“锚”框的大小比例

传入的形状

> For example

```
anchors = [(10, 13), (16, 30), (33, 23)]
```

> resize后

基本形状

```python
size([1, self.num_anchors, 1, 1, 2])
```





## build_targets

| 名称/步骤                                                 | 形状                                   | 含义/单位                                |
| --------------------------------------------------------- | -------------------------------------- | ---------------------------------------- |
| targets                                                   | $(n, 6)$                               | $[b, cls, cx, cy, w, h]$，归一化坐标     |
| anchors                                                   | $(nA, 2)$                              | 先验宽高，单位：像素                     |
| tbox                                                      | $(B, nA, feat_h, feat_w, 4)$           | 目标框真值（网格坐标系，wh 为 log 空间） |
| tconf                                                     | $(B, nA, feat_h, feat_w, 1)$           | 目标存在标记（0/1）                      |
| tcls                                                      | $(B, nA, feat_h, feat_w, num_classes)$ | one-hot 类别（num_classes=1 时全 0）     |
| gxy = targets[:,2:4]*[feat_w,feat_h]                      | $(n, 2)$                               | 归一化中心 → 网格坐标                    |
| gwh = targets[:,4:6]*[feat_w*stride,feat_h*stride]/stride | $(n, 2)$                               | 归一化 wh → 网格尺度（已除 stride）      |
| gij = gxy.long()                                          | $(n, 2)$                               | 网格整索引 (gi, gj)                      |
| gi, gj                                                    | $(n,)$                                 | x/y 整索引                               |
| anchor_wh = anchors/stride                                | $(nA, 2)$                              | anchor 转为网格尺度                      |
| ratios = gwh[:,None,:]/anchor_wh[None]                    | $(n, nA, 2)$                           | 目标 wh 与 anchor 比例                   |
| inv_ratios = anchor_wh[None]/gwh                          | $(n, nA, 2)$                           | 反比例                                   |
| max_ratios = max(ratios, inv_ratios).max(-1)              | $(n, nA)$                              | 取更大的比例，衡量匹配程度               |
| best_anchors = argmin(max_ratios, dim=1)                  | $(n,)$                                 | 为每个目标选最匹配的 anchor              |







## YoloLoss

- 总损失：
  $ L = L_{\text{box}} + L_{\text{wh}} + L_{\text{obj}} + L_{\text{cls}} $

- 仅在正样本位置（obj\_mask = tconf.squeeze(-1) > 0）计算：
  $ L_{\text{box}} = \text{SmoothL1}( \sigma(\mathbf{p}_{xy}) , \mathbf{t}_{xy} ) $
  
  $ L_{\text{wh}}  = \text{SmoothL1}( \mathbf{p}_{wh} , \mathbf{t}_{wh} ) $

  其中 $\sigma$ 是 sigmoid，$\mathbf{t}_{wh}$ 是对真实框 wh 做 $\log(\frac{w,h}{\text{anchor}})$ 的结果。

- 目标存在性（含正负样本）：
  $ L_{\text{obj}} = \text{BCEWithLogits}( \mathbf{p}_{obj}, \mathbf{t}_{conf} ) $

- 类别（多类时；若 num\_classes=1，则 $L_{\text{cls}} = 0$）：
  $ L_{\text{cls}} = \text{BCEWithLogits}( \mathbf{p}_{cls}, \mathbf{t}_{cls} ) $

- 输出的 logits/预测分量：
  - $\mathbf{p}_{xy} = \sigma(\text{raw}[...,0:2])$
  - $\mathbf{p}_{wh} = \text{raw}[...,2:4]$（与 $\log$ 目标比）
  - $\mathbf{p}_{obj} = \text{raw}[...,4:5]$
  - $\mathbf{p}_{cls} = \text{raw}[...,5:]$（多标签 sigmoid）



$BCEWithLogits$:

![image-20260109150632887](D:\__SelfCoding\Deep_learning\markdown\note\little_yolo代码精读.assets\image-20260109150632887.png)



