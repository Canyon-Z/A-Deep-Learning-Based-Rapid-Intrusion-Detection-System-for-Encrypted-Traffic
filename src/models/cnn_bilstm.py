import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
    """
    CNN + 双向 LSTM 混合模型，是系统的默认检测模型。

    28×28 的流量灰度图先展平成一维序列，经过两层 1D 卷积提取局部
    字节模式，再交给双向 LSTM 捕捉传输层的前后依赖关系。卷积负责
    "看局部"，LSTM 负责"理解上下文"，最后取最后一个时间步的隐状态
    做分类。

    参数:
        num_classes: 分类类别数
        hidden_dim:  LSTM 隐层维度，默认 128
    """
    def __init__(self, num_classes, hidden_dim=128):
        super(CNN_BiLSTM, self).__init__()

        self.cnn = nn.Sequential(
            # 第 1 层卷积: 1→16 通道，池化后 784→392
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),

            # 第 2 层卷积: 16→32 通道，池化后 392→196
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm_input_size = 32
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)

        # 卷积提取局部特征: (batch, 32, 196)
        x = self.cnn(x)

        # 调整为 LSTM 需要的形状: (batch, 序列长度=196, 特征=32)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)

        # 取最后一个时间步的输出做分类
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
