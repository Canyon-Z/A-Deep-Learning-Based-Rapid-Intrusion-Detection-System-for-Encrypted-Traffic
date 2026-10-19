"""流特征输入层的域自适应归一化。

跨天失效的实测证据（`logs/pipeline_b2_summary.txt`、`logs/pipeline_b3_summary.txt`）：

  A  15-02+21-02 → 02-03   准确率 72.60%  检出 0.00%  AUC 0.5001
  F  02-03+15-02 → 21-02   准确率 34.41%  检出 0.00%  AUC 0.2420
  G  02-03+21-02 → 15-02   准确率 94.94%  检出 0.00%  AUC 0.2969

三个方向的训练天互不相同，结论却完全一致，且 AUC 有两个显著低于 0.5（反向排序）。
这说明问题不在"选了哪几天训练"，而在特征本身携带了**域指纹**——包长、包间隔、
速率这类绝对数值在不同天差一个数量级，模型学到的是"这几天长什么样"而不是
"恶意流量长什么样"。

本模块提供两种"去域指纹"的归一化手段：

  robust  全局中位数 + IQR。对离群值稳健，但统计量仍来自训练集，换天依旧会偏。
  batch   批内中位数 + MAD。完全不用训练集统计量，只用当前这一批流量自己的分布。

batch 模式的假设是：部署时同一时段的流量属于同一个域，于是「比本批中位数大多少倍」
这种相对量比绝对数值更稳定。代价是推理必须成批进行（单条流算不出批内统计量），
且批大小太小时统计量不稳（见 min_batch）。
"""

import torch
import torch.nn as nn

# MAD -> 标准差的换算常数（正态分布下 1.4826 * MAD ≈ sigma）
MAD_TO_SIGMA = 1.4826


class FlowInputNorm(nn.Module):
    """把「批内稳健归一化」做成模型最前置的一层（无参数、无状态）。

    mode='batch'  用当前 batch 每列的中位数与 MAD 归一化
    mode='global' 直接透传（统计量已在数据侧做完）
    """

    def __init__(self, mode='batch', min_batch=32, scale_floor=1e-3):
        super().__init__()
        self.mode = mode
        self.min_batch = int(min_batch)
        self.scale_floor = float(scale_floor)

    def extra_repr(self):
        return f"mode={self.mode}, min_batch={self.min_batch}"

    def forward(self, x):
        if self.mode != 'batch':
            return x
        if x.size(0) < self.min_batch:
            # 批太小算不出稳定中位数，原样返回（评估的最后一个小批常见）
            return x
        med = x.median(dim=0).values
        mad = (x - med).abs().median(dim=0).values
        scale = (MAD_TO_SIGMA * mad).clamp_min(self.scale_floor)
        return (x - med) / scale
