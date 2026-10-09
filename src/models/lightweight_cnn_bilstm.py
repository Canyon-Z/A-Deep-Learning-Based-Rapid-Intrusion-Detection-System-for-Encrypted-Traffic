import torch
import torch.nn as nn


class Lightweight_CNN_BiLSTM(nn.Module):
    """
    CNN + BiLSTM 的轻量化版本。

    相比标准版 CNN_BiLSTM，卷积通道数从 16/32 减到 8/16，LSTM 隐层
    从 128 降到 32。参数量约为标准版的 1/5，推理更快、内存占用更低，
    适合部署在边缘设备上。

    参数:
        num_classes: 分类类别数
        hidden_dim:  LSTM 隐层维度，默认 32
    """
    def __init__(self, num_classes, hidden_dim=32):
        super(Lightweight_CNN_BiLSTM, self).__init__()

        self.cnn = nn.Sequential(
            # 第 1 层卷积: 1→8 通道，池化后 784→392
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),

            # 第 2 层卷积: 8→16 通道，池化后 392→196
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm_input_size = 16
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)

        # 卷积后形状: (batch, 16, 196)
        x = self.cnn(x)

        # 转置为 LSTM 输入: (batch, 序列长=196, 特征=16)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
