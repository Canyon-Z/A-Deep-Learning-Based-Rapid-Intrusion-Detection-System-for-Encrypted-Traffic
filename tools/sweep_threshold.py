"""阈值扫描：在弱排序模型上找可用的工作点。

B5 的结论是 AUC 只有 0.50~0.61，但 P3 在 AUC 0.4969 的情况下仍检出 31.52% ——
这说明概率分布里有一段区间是可用的，只是默认的 argmax（阈值 0.5）没落在上面。

这个脚本对指定权重重跑一遍测试集，扫 0.05~0.95 的判决阈值，输出 TPR/FPR 曲线，
并给出几个候选工作点：

  - 误报 <= 1%  时的最大检出
  - 误报 <= 5%  时的最大检出
  - F1 最大的点
  - Youden J 最大的点（TPR - FPR）

用法：
    python -u tools/sweep_threshold.py --ckpt checkpoints/flow_P3_all_batch.pth \
        --test-days 02-03 --norm-mode batch --drop-features "Dst Port" \
        --log-features IAT,Duration,Pkts/s
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.flow_input_norm import FlowInputNorm
from src.models.flow_mlp import FlowMLP
from src.utils.device import resolve_device
from tools.train_flow_model import align_columns, load_day, to_binary


def main():
    parser = argparse.ArgumentParser(description="扫描判决阈值，找可用工作点")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--csv-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'cse_csv'))
    parser.add_argument('--test-days', type=str, required=True)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    feature_names = ckpt['feature_names']
    hidden = tuple(ckpt.get('hidden_sizes', (256, 128)))
    norm = ckpt.get('normalization') or {}
    norm_mode = norm.get('mode', 'global')
    print(f"[ckpt] {os.path.basename(args.ckpt)} | {len(feature_names)} 特征 | "
          f"归一化 {norm_mode} | 训练天 {ckpt.get('train_days')}")

    test_days = [d.strip() for d in args.test_days.split(',') if d.strip()]
    frames, labels = [], []
    for day in test_days:
        df, lab = load_day(args.csv_dir, day, args.max_rows)
        frames.append(df)
        labels.append(to_binary(lab))
    frames = [df[feature_names] for df in frames]
    X = np.vstack([df.to_numpy(dtype=np.float32) for df in frames])
    y = np.concatenate(labels)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    print(f"[data] 测试 {X.shape}，恶意占比 {y.mean() * 100:.2f}%")

    # 按 ckpt 里记录的归一化口径还原输入
    if norm_mode == 'global':
        mean = np.array(norm['mean'], dtype=np.float32).reshape(1, -1)
        std = np.array(norm['std'], dtype=np.float32).reshape(1, -1)
        std[std < 1e-6] = 1.0
        X = ((X - mean) / std).astype(np.float32)
    elif norm_mode == 'robust':
        mean = np.array(norm['mean'], dtype=np.float32).reshape(1, -1)
        std = np.array(norm['std'], dtype=np.float32).reshape(1, -1)
        std[std < 1e-6] = 1.0
        X = ((X - mean) / std).astype(np.float32)
    # batch 模式：数据侧不动，模型内 FlowInputNorm 处理

    hidden_sizes = list(hidden)
    base = FlowMLP(input_dim=len(feature_names), hidden_sizes=hidden_sizes, num_classes=2)
    model = nn.Sequential(FlowInputNorm(mode='batch'), base) if norm_mode == 'batch' else base
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            trues.append(yb.numpy())
    prob = np.concatenate(probs)
    true = np.concatenate(trues)

    n_pos = int(true.sum())
    n_neg = int(len(true) - n_pos)
    auc = roc_auc_score(true, prob) if n_pos and n_neg else None
    print(f"[base] 阳性 {n_pos} / 阴性 {n_neg} | AUC {auc:.4f}\n" if auc else "")

    print(f"{'阈值':>6}{'检出TPR':>10}{'误报FPR':>10}{'精确率':>9}{'F1':>9}{'YoudenJ':>9}{'告警数':>10}")
    print('-' * 66)
    rows = []
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = prob >= thr
        tp = int((pred & (true == 1)).sum())
        fp = int((pred & (true == 0)).sum())
        tpr = tp / n_pos * 100 if n_pos else 0.0
        fpr = fp / n_neg * 100 if n_neg else 0.0
        precision = tp / max(tp + fp, 1) * 100
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-9)
        youden = tpr - fpr
        rows.append({'thr': round(float(thr), 2), 'tpr': tpr, 'fpr': fpr,
                     'precision': precision, 'f1': f1, 'youden': youden,
                     'alerts': tp + fp})
        print(f"{thr:>6.2f}{tpr:>9.2f}%{fpr:>9.2f}%{precision:>8.2f}%{f1:>8.2f}{youden:>9.2f}{tp + fp:>10}")

    def best(key, limit=None):
        pool = [r for r in rows if (limit is None or r[key] >= limit)]
        if not pool:
            return None
        return max(pool, key=lambda r: r['tpr'] if key != 'f1' and key != 'youden' else r[key])

    print('\n=== 候选工作点 ===')
    for label, limit in (('误报 <= 1%', 1.0), ('误报 <= 3%', 3.0), ('误报 <= 5%', 5.0)):
        pool = [r for r in rows if r['fpr'] <= limit]
        if pool:
            b = max(pool, key=lambda r: r['tpr'])
            print(f"{label}: 阈值 {b['thr']:.2f} -> 检出 {b['tpr']:.2f}% / "
                  f"误报 {b['fpr']:.2f}% / 精确率 {b['precision']:.2f}% / F1 {b['f1']:.2f}")
        else:
            print(f"{label}: 该误报约束下无可用阈值")
    bf1 = max(rows, key=lambda r: r['f1'])
    print(f"F1 最大: 阈值 {bf1['thr']:.2f} -> 检出 {bf1['tpr']:.2f}% / 误报 {bf1['fpr']:.2f}% / F1 {bf1['f1']:.2f}")
    byj = max(rows, key=lambda r: r['youden'])
    print(f"Youden J 最大: 阈值 {byj['thr']:.2f} -> 检出 {byj['tpr']:.2f}% / 误报 {byj['fpr']:.2f}% / J {byj['youden']:.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
