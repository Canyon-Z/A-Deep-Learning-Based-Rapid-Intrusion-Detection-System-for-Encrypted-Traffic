"""流特征自编码器：只学「正常流量长什么样」，用重构误差当异常分。

为什么要有这条分支（而不是继续调二分类）：

B2→B5 + 阈值扫描已经把「良性/恶意二分类」这条路线推到头了：
  - 三个训练天方向全部检出 0%（B3）
  - 三种归一化口径最好 AUC 0.51（B4）
  - 剔域指纹特征后 AUC 0.61，但阈值扫描显示概率全部压在低位、没有任何可用工作点（B5 + sweep）
AUC 0.61 只是「形式上比随机好」，实际输出无法用。

自编码器换的是问题定义，不是模型结构：
  - 二分类要学「恶意与良性的决策边界」，而这个边界换一天就失效
  - 自编码器只学「正常流量的分布形状」，训练与执行都在**同一个网络**里，不存在跨域迁移

异常分 = 重构误差。恶意流量偏离正常分布 → 误差大 → 告警。
代价：检出率做不到 99%，且「长得像正常的攻击」（慢速渗出、正常协议的内嵌载荷）会漏。

关键设计：
  - 训练**只用良性样本**（无监督），恶意样本一个都不喂
  - 重构误差用逐样本 MSE，并在训练集良性的 99 分位处设一个参考阈值
  - 评估时报告 AUC / 各误报约束下的检出率，与二分类分支用同一套口径对比

用法：
    python -u tools/train_autoencoder.py --train-days 15-02,21-02 --test-days 02-03
    python -u tools/train_autoencoder.py --train-days 15-02,21-02 --test-days 02-03 --epochs 40
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.device import resolve_device
from tools.train_flow_model import align_columns, load_day, to_binary


class FlowAE(nn.Module):
    """对称瓶颈自编码器。输入维度由 CSV 列的数目决定，不写死。"""

    def __init__(self, input_dim, hidden=(128, 32, 128), dropout=0.1):
        super().__init__()
        h1, bottleneck, h3 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, bottleneck),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h3, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruct_error(self, x):
        """逐样本重构误差（MSE，按特征维取平均），作为异常分。"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1)


def pick_workpoints(scores, y_true):
    """给定异常分与真值，输出几个误报约束下的最大检出。"""
    if len(set(y_true.tolist())) < 2:
        return []
    fpr, tpr, thr = roc_curve(y_true, scores)
    out = []
    for limit in (1.0, 3.0, 5.0, 10.0):
        mask = fpr * 100 <= limit
        if not mask.any():
            out.append((limit, None, None, None))
            continue
        idx = np.argmax(np.where(mask, tpr, -1))
        out.append((limit, float(tpr[idx] * 100), float(fpr[idx] * 100), float(thr[idx])))
    return out


def main():
    parser = argparse.ArgumentParser(description="流特征自编码器（异常检测范式）")
    parser.add_argument('--csv-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'cse_csv'))
    parser.add_argument('--train-days', type=str, required=True,
                        help="训练天（只取其中的良性样本训练）")
    parser.add_argument('--test-days', type=str, required=True, help="测试天（含良性+恶意）")
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='128,32,128')
    parser.add_argument('--drop-features', type=str, default='',
                        help="剔除的高漂移特征，逗号分隔（建议 Dst Port）")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--metrics-out', type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_days = [d.strip() for d in args.train_days.split(',') if d.strip()]
    test_days = [d.strip() for d in args.test_days.split(',') if d.strip()]
    print(f"设备：{device}")
    print(f"训练天：{train_days}（只用良性）  测试天：{test_days}")

    t0 = time.time()
    train_frames, train_labels = [], []
    test_frames, test_labels = [], []
    for day in train_days:
        df, lab = load_day(args.csv_dir, day, args.max_rows)
        train_frames.append(df)
        train_labels.append(to_binary(lab))
    for day in test_days:
        df, lab = load_day(args.csv_dir, day, args.max_rows)
        test_frames.append(df)
        test_labels.append(to_binary(lab))

    train_frames, feature_names = align_columns(train_frames)
    if args.drop_features:
        drops = [f.strip() for f in args.drop_features.split(',') if f.strip()]
        kept = [f for f in feature_names if f not in drops]
        if len(kept) != len(feature_names):
            print(f"[feat ] 剔除 {len(feature_names) - len(kept)} 个高漂移特征 -> 剩 {len(kept)} 个")
            feature_names = kept
            train_frames = [df[feature_names] for df in train_frames]
    test_frames = [df[feature_names] for df in test_frames]

    X_train_all = np.vstack([df.to_numpy(dtype=np.float32) for df in train_frames])
    y_train_all = np.concatenate(train_labels)
    X_test = np.vstack([df.to_numpy(dtype=np.float32) for df in test_frames])
    y_test = np.concatenate(test_labels)

    ok_tr = np.isfinite(X_train_all).all(axis=1)
    ok_te = np.isfinite(X_test).all(axis=1)
    X_train_all, y_train_all = X_train_all[ok_tr], y_train_all[ok_tr]
    X_test, y_test = X_test[ok_te], y_test[ok_te]
    print(f"[data ] 训练 {X_train_all.shape}（恶意 {int(y_train_all.sum())}，将被丢弃），"
          f"测试 {X_test.shape}，测试恶意占比 {y_test.mean() * 100:.2f}%")

    # 只用良性样本训练 —— 这是异常检测范式的核心，喂进恶意样本就等于又回到二分类
    X_normals = X_train_all[y_train_all == 0]
    print(f"[data ] 实际用于训练的正常样本 {X_normals.shape}")

    # 标准化：用正常样本自己的统计量（部署时也只可能拿到正常样本）
    mean = X_normals.mean(axis=0, keepdims=True)
    std = X_normals.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_normals_n = ((X_normals - mean) / std).astype(np.float32)
    X_test_n = ((X_test - mean) / std).astype(np.float32)

    # 划一份正常样本做早期验证（不涉及恶意）
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_normals_n))
    n_val = max(1, int(len(X_normals_n) * 0.05))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def make_loader(X, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_normals_n[tr_idx], True)
    val_loader = make_loader(X_normals_n[val_idx], False)

    h1, bneck, h3 = (int(v) for v in args.hidden.split(','))
    model = FlowAE(input_dim=len(feature_names), hidden=(h1, bneck, h3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val, best_state, best_epoch, no_improve = float('inf'), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                val_loss += float(criterion(model(xb), xb)) * xb.size(0)
                val_seen += xb.size(0)
        val_loss /= max(val_seen, 1)
        print(f"▶ Epoch [{epoch + 1}/{args.epochs}] 训练重构 {running / max(seen, 1):.6f} | "
              f"验证重构 {val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"⏹ 早停触发：{best_epoch} 轮后连续 {no_improve} 轮无改善")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n最佳验证重构误差 {best_val:.6f}（第 {best_epoch} 轮）")

    # 用正常样本的 99 分位作为「默认阈值」参考
    model.eval()
    ref_loader = make_loader(X_normals_n[val_idx], False)
    ref_scores = []
    with torch.no_grad():
        for (xb,) in ref_loader:
            ref_scores.append(model.reconstruct_error(xb.to(device)).cpu().numpy())
    ref_scores = np.concatenate(ref_scores)
    thr99 = float(np.percentile(ref_scores, 99))
    print(f"[threshold] 正常样本重构误差：中位数 {np.median(ref_scores):.6f}，"
          f"99 分位 {thr99:.6f}")

    # 测试集上的异常分 = 重构误差（越大越异常）
    test_scores = []
    with torch.no_grad():
        for xb, _yb in DataLoader(TensorDataset(torch.from_numpy(X_test_n),
                                               torch.from_numpy(y_test)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0):
            test_scores.append(model.reconstruct_error(xb.to(device)).cpu().numpy())
    scores = np.concatenate(test_scores)

    auc = roc_auc_score(y_test, scores) if len(set(y_test.tolist())) > 1 else None
    print(f"[test ] 异常分 AUC {auc:.4f}" if auc else "")

    print(f"\n{'误报约束':>10}{'检出TPR':>10}{'误报FPR':>10}{'对应阈值':>14}")
    print('-' * 46)
    for limit, tpr, fpr, thr in pick_workpoints(scores, y_test):
        if tpr is None:
            print(f"{limit:>9.1f}%{'  无可用点':>14}")
        else:
            print(f"{limit:>9.1f}%{tpr:>9.2f}%{fpr:>9.2f}%{thr:>14.6f}")

    # 默认阈值（正常 99 分位）下的表现
    pred = scores >= thr99
    tp = int((pred & (y_test == 1)).sum())
    fp = int((pred & (y_test == 0)).sum())
    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)
    print(f"\n[默认阈值=正常99分位] 检出 {tp / max(n_pos, 1) * 100:.2f}% | "
          f"误报 {fp / max(n_neg, 1) * 100:.2f}% | 告警数 {tp + fp} "
          f"（精确率 {tp / max(tp + fp, 1) * 100:.2f}%）")

    out = args.out or os.path.join(_PROJECT_ROOT, 'checkpoints', 'flow_ae.pth')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': len(feature_names),
        'feature_names': feature_names,
        'hidden': [h1, bneck, h3],
        'normalization': {'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()},
        'threshold_p99': thr99,
        'train_days': train_days,
        'test_days': test_days,
    }, out)
    print(f"权重已保存 -> {out}")

    metrics_out = args.metrics_out or os.path.join(
        _PROJECT_ROOT, 'logs', 'flow_ae_metrics.json')
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, 'w', encoding='utf-8') as fh:
        json.dump({
            'train_days': train_days,
            'test_days': test_days,
            'feature_count': len(feature_names),
            'normal_train_samples': int(len(X_normals)),
            'test_samples': int(len(y_test)),
            'test_malware_ratio': round(float(y_test.mean()) * 100, 2),
            'best_val_recon': best_val,
            'best_epoch': best_epoch,
            'threshold_p99': thr99,
            'auc': auc,
            'workpoints': [
                {'fpr_limit': limit, 'tpr': tpr, 'fpr': fpr, 'threshold': thr}
                for limit, tpr, fpr, thr in pick_workpoints(scores, y_test)
            ],
            'default_threshold': {
                'tpr': tp / max(n_pos, 1) * 100,
                'fpr': fp / max(n_neg, 1) * 100,
                'precision': tp / max(tp + fp, 1) * 100,
                'alerts': tp + fp,
            },
            'elapsed_sec': round(time.time() - t0, 1),
        }, fh, ensure_ascii=False, indent=2)
    print(f"指标已写入 -> {metrics_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
