from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def find_malware_index(class_names: Sequence[str], default_index: int = 1) -> int:
    malware_idx = default_index
    for idx, class_name in enumerate(class_names):
        lowered = str(class_name).lower()
        if 'malware' in lowered or 'malicious' in lowered:
            return idx
    return malware_idx


def compute_class_weights(
    label_counts: torch.Tensor,
    class_names: Sequence[str],
    malware_weight_multiplier: float = 1.0,
    max_ratio_warning: float = 10.0,
) -> torch.Tensor:
    num_classes = int(label_counts.numel())
    total_samples = float(label_counts.sum().item())
    class_weights = total_samples / (num_classes * label_counts.clamp(min=1.0))

    if malware_weight_multiplier and malware_weight_multiplier > 1.0 and num_classes > 0:
        malware_idx = find_malware_index(class_names, default_index=1 if num_classes > 1 else 0)
        class_weights[malware_idx] = class_weights[malware_idx] * float(malware_weight_multiplier)

    weight_min = float(class_weights.min().item()) if class_weights.numel() else 0.0
    weight_max = float(class_weights.max().item()) if class_weights.numel() else 0.0
    if weight_min > 0 and (weight_max / weight_min) > max_ratio_warning:
        print(
            f"警告: 类别权重最大/最小比值过大 ({weight_max / weight_min:.2f} > {max_ratio_warning:.2f})，"
            "可能导致训练不稳定或误报上升。"
        )

    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'sum':
            return focal_loss.sum()
        if self.reduction == 'none':
            return focal_loss
        return focal_loss.mean()


def build_criterion(
    loss_function: str,
    class_weights: torch.Tensor,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> nn.Module:
    loss_name = str(loss_function).strip().lower()
    if loss_name == 'focal':
        return FocalLoss(gamma=focal_gamma, alpha=class_weights.to(device))
    return nn.CrossEntropyLoss(weight=class_weights.to(device))