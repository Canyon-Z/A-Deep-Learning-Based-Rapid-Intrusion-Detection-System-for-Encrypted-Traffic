import torch
import torch.nn as nn


class TrafficTransformer(nn.Module):
    """
    基于 Transformer Encoder 的流量分类模型。

    把 28×28 流量灰度图的每一行看作一个 28 维的 token（共 28 个 token），
    通过线性嵌入 + 位置编码后送入多层 Transformer Encoder，用自注意力机制
    捕捉流量字节之间的长距离依赖关系。

    相比 CNN 的局部感受野，Transformer 可以同时关注整条流的所有位置，
    对模式分散的恶意流量可能更有优势。不过参数量大，需要更多数据。

    参数:
        input_dim:   每个 token 的维度，默认 28（对应每行 28 个字节）
        d_model:     Transformer 内部特征维度，必须能被 nhead 整除
        nhead:       多头注意力的头数
        num_layers:  Transformer Encoder 层数
        num_classes: 分类类别数
        dropout:     Dropout 比例
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(TrafficTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        # 兼容两种输入: (batch, seq, dim) 或 (batch, 1, seq, dim)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        x = self.embedding(x)

        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        output = self.transformer_encoder(x)

        # 对所有 token 取平均作为整体表示
        output = output.mean(dim=1)

        output = self.classifier_dropout(output)
        return self.classifier(output)
