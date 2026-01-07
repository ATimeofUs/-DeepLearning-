from torch import nn

class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # features 负责提取特征：卷积 + 激活 + BN + 池化 + Dropout
        self.features = nn.Sequential(
            # Block 1：输入 (3x32x32) -> 输出 (64x16x16)
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),          # 空间下采样
            nn.Dropout(0.25),
            # Block 2：输出 (128x8x8)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 3：输出 (256x4x4)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # classifier 将特征映射到类别空间
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 展平成 (batch, 256*4*4)
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # 更高的 dropout 减少过拟合
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),  # 输出 logits
        )

    def forward(self, x):
        # 前向：先提取特征，再分类
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # 构建模型示例（默认10类）
    CiferCNN = CifarCNN(num_classes=10)

if __name__ == "__main__":
    main()