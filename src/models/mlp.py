import torch.nn as nn


class MLP(nn.Module):
    """
    多层感知机基线模型。

    把 28×28 的流量图拉直成 784 维向量，通过三层全连接做分类。
    结构简单、训练快，适合验证"更复杂的模型是否真的有必要"。

    参数:
        input_size:   输入维度，默认 784 (28×28)
        hidden_sizes: 各隐层的神经元数，默认 [512, 256, 128]
        num_classes:  分类类别数
    """
    def __init__(self, input_size=28 * 28, hidden_sizes=None, num_classes=2):
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
