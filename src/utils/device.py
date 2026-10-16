"""设备解析工具：6 个训练脚本原先各自复制了一份相同实现（代码审查第 40 条）。"""
import torch


def resolve_device(requested_device: str) -> torch.device:
    """解析用户传入的设备参数，自动判断是否可用 CUDA。

    规则（与历史实现逐字保持一致，避免改变训练行为）：
    - 'auto'：有 CUDA 用 CUDA，否则 CPU；
    - 'cuda' 但当前环境没有 CUDA：打印警告并回退 CPU；
    - 其它未知取值：打印警告并按 'auto' 处理。
    """
    requested = requested_device.lower().strip()
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        print('警告: 当前环境没有可用 CUDA，已自动回退到 CPU。')
        return torch.device('cpu')
    if requested not in ('cpu', 'cuda'):
        print(f"警告: 未知 device='{requested_device}'，已使用 auto。")
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)
