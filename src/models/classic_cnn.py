import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCNN(nn.Module):
    """
    纯 CNN 基线模型，不含 RNN 结构。

    输入是 28×28 的流量灰度图，先展平成 784 字节的一维序列，
    然后通过三层 1D 卷积逐级提取特征，最后用全连接层做二分类。
    这个模型只靠卷积的局部感受野来捕捉流量模式，推理速度很快，
    适合作为对比基线。
    """
    def __init__(self, num_classes=2):
        super(ClassicCNN, self).__init__()

        self.cnn = nn.Sequential(
            # 第 1 层: 1 → 32 通道，尺寸减半 784→392
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),

            # 第 2 层: 32 → 64 通道，尺寸再减半 392→196
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),

            # 第 3 层: 64 → 128 通道，尺寸减半 196→98
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128 * 98, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
