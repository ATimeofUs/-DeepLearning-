"""
回归

nn.MSELoss：最常用 L2 回归；对异常值敏感。
nn.L1Loss：L1 回归；对异常值更鲁棒，收敛略慢。
nn.SmoothL1Loss / nn.HuberLoss：介于 L1/L2 之间，既平滑又抗异常值；检测回归头常用。
nn.LogCoshLoss：更平滑的回归损失，对极端误差较稳。
二分类 / 多分类（独立样本）

nn.BCEWithLogitsLoss：二分类/多标签分类首选（内置 sigmoid+交叉熵），数值稳定。
nn.BCELoss：需要你先手动 sigmoid；数值不如上者稳定。
nn.CrossEntropyLoss：多类单标签分类（softmax+NLL）；可配 weight 处理类不平衡，label_smoothing 抑制过拟合。
nn.NLLLoss：若你已在模型中输出 log_softmax，则用它。
序列 / 语言

nn.CrossEntropyLoss：序列分类/序列标注常用。
nn.CTCLoss：对齐未知的序列到序列任务（语音识别、OCR）。
nn.Transformer 相关任务仍常用交叉熵（可结合 label smoothing / mask）。
语义分割 / 像素分类

nn.CrossEntropyLoss：像素级多类单标签分割。
Dice / IoU 类损失（需自定义或第三方）：应对前景稀疏、类别不平衡。
Focal Loss（通常自定义）：降低易分类样本权重，专注难样本；目标检测、分割常用。
BCEWithLogits + Dice 组合：多标签/多类别分割常见套路。
检测 / 框回归

分类头：nn.CrossEntropyLoss 或 Focal Loss。
框回归：nn.SmoothL1Loss、nn.L1Loss、nn.GIoULoss/DIoU/CIoU（需自定义）。IoU 家族在大重叠时更有意义。
度量学习 / 排序

nn.TripletMarginLoss / nn.TripletMarginWithDistanceLoss：三元组度量学习。
nn.CosineEmbeddingLoss：用余弦相似度对 pair 做正负判别。
nn.MarginRankingLoss：排序/对比任务。
生成对抗 (GAN)

判别器/生成器常用：BCEWithLogitsLoss（原版 GAN）、MSELoss（LSGAN）、
nn.HingeEmbeddingLoss（hinge GAN）。WGAN/WGAN-GP 则是自定义 critic 损失 + 梯度惩罚。
自监督 / 对比学习（常自定义）

InfoNCE / NT-Xent：基于温度缩放的对比交叉熵；需自己实现。
BYOL/SimSiam：负样本自由，带 predictor，配合回归式相似度损失。
不平衡与正则化技巧

类不平衡：weight（CrossEntropy）、pos_weight（BCEWithLogits）、Focal Loss。
Label smoothing：缓解过拟合、提升校准；CrossEntropy 的参数 label_smoothing。
温度缩放（自定义）：调节 softmax 置信度。
组合损失：分类 + 回归/IoU，多任务加权可采用动态权重（例如 uncertainty weighting）。
数值与实现注意

优先用带 logits 的版本（如 BCEWithLogitsLoss, CrossEntropyLoss）以获更好数值稳定性。
确保 target dtype：分类用 long（CrossEntropy/NLL），二分类/多标签用 float。
注意 ignore_index（分割、填充位置）和 reduction（mean/sum/none）。
混合精度下保持 logits 在合理范围，必要时梯度裁剪。
多标签问题：不要用 softmax+CrossEntropy，应使用 sigmoid+BCEWithLogits。
"""

from torch import nn

def main():
    nn.NLLLoss()

if __name__ == '__main__':
    main()