"""基于流统计特征（CICFlowMeter 风格）的 MLP 分类器。

和 src/models/mlp.py 的区别：那个吃 28x28 字节图像展平的 784 维，这个直接吃
流统计特征（包长、到达间隔、流持续时间、上下行字节数等几十维物理量）。两条
分支的输入表示完全不同，是为了对比"字节图像 vs 流特征"的跨域鲁棒性。

特征维数不写死，由训练脚本按实际 CSV 的列数传入。
"""

import torch
import torch.nn as nn


class FlowMLP(nn.Module):
    """全连接分类器：BatchNorm 输入归一化 + 若干隐藏层 + Dropout。"""

    def __init__(self, input_dim=78, hidden_sizes=(256, 128), num_classes=2, dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
