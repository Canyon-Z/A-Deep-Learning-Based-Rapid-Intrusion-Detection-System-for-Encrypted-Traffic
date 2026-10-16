"""纯 BiLSTM 模型：把 28×28 图像按行拆成 28 个时间步，每步 28 维特征。

代码审查第 31 条：这份网络原先有 3 份拷贝（src/training/run_training_all.py、
src/training/train_pure_bilstm.py、web/backend/main.py），其中 web 端那份默认
hidden_dim=128 且没有 dropout，与训练权重不一致 —— 训练时的 nn.LSTM 带 dropout、
构造时 hidden_dim=64，state_dict 键名相同但默认值不同，靠调用方显式传参才碰巧对上。
现在统一以本文件为唯一定义，训练脚本与 web 后端都从这里导入。
"""
import torch.nn as nn


class PureBiLSTM(nn.Module):
    """纯 BiLSTM 模型：把 28×28 图像按行拆成 28 个时间步，每步 28 维特征。"""

    def __init__(self, input_size=28, hidden_dim=64, num_layers=2, num_classes=2):
        super(PureBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1)                # [B, 1, 28, 28] → [B, 28, 28]
        output, _ = self.bilstm(x)      # [B, 28, hidden*2]
        last_state = output[:, -1, :]   # 取最后一个时间步
        last_state = self.dropout(last_state)
        return self.fc(last_state)
