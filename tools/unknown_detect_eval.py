"""离线验证「目标外（out-of-distribution）检测」能不能靠类中心距离实现。

思路：六分类模型只学过 6 个类（Benign / Bot / DDoS-HOIC / DoS-GoldenEye /
DoS-Slowloris / DDoS-LOIC-UDP）。遇到没见过的攻击（Brute Force、XSS、SQLi、
Infiltration、Web Attack），正经做法不是让它硬塞进最像的那一类，而是识别出
「这行不属于任何已知类」并单列。

判据用的是「到最近类中心的距离」：如果一行样本在标准化空间里离它被判的那一类的
中心比训练样本通常的距离远得多，它就不该被当作该类的普通成员。

本脚本回答两个问题：
  1. 已知类的距离分布长什么样（用来定阈值）
  2. 未见过的攻击类会不会被判成「远」（即判据是否有效）

用法：
    python tools/unknown_detect_eval.py
    python tools/unknown_detect_eval.py --known-sample 200000 --ood-sample 40000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.flow_mlp import FlowMLP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'flow_multiclass.pth')
CSV_DIR = os.path.join(ROOT, 'data', 'cse_csv')

# 训练时用到的三天。只有这些文件里的行才算「已知分布」。
KNOWN_FILES = [
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
]

# 训练集里出现过的 Label 值（六类的原始写法）。
KNOWN_LABELS = {
    'Benign',
    'Bot',
    'DDOS attack-HOIC',
    'DoS attacks-GoldenEye',
    'DoS attacks-Slowloris',
    'DDOS attack-LOIC-UDP',
}


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f'找不到模型权重：{path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def build_model(ckpt: dict) -> FlowMLP:
    class_names = ckpt.get('class_names') or []
    model = FlowMLP(input_dim=len(ckpt['feature_names']), num_classes=max(2, len(class_names)))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def collect_frames(files: list[str], max_rows: int) -> pd.DataFrame:
    frames = []
    for name in files:
        path = os.path.join(CSV_DIR, name)
        if not os.path.exists(path):
            print(f'  [跳过] 缺文件 {name}')
            continue
        df = pd.read_csv(path, nrows=max_rows, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        print(f'  [读入] {name}: {len(df):,} 行')
    if not frames:
        raise SystemExit('没有任何可用的 CSV')
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--known-sample', type=int, default=120000,
                    help='每天读取多少行作为「已知类」样本')
    ap.add_argument('--ood-sample', type=int, default=40000,
                    help='每天读取多少行，用于挑出「未见过的攻击类」')
    ap.add_argument('--ood-file', default='Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
                    help='专门提供未见攻击类型的 CSV（默认用含 Infilteration 的那天）')
    ap.add_argument('--output', default=os.path.join(ROOT, 'logs', 'unknown_detect_eval.json'))
    args = ap.parse_args()

    ckpt = load_checkpoint(CHECKPOINT)
    feature_names = list(ckpt['feature_names'])
    class_names = list(ckpt['class_names'])
    mean = np.asarray(ckpt['normalization']['mean'], dtype=np.float32)
    std = np.asarray(ckpt['normalization']['std'], dtype=np.float32)
    temperature = float(ckpt.get('temperature') or 1.0)
    model = build_model(ckpt)

    print('=' * 68)
    print('六分类模型类中心距离评估（目标外检测）')
    print('=' * 68)
    print(f'模型: {CHECKPOINT}')
    print(f'类别: {class_names}')
    print(f'特征: {len(feature_names)} 维 · 温度: {temperature:.4f}')

    print('\n[1/4] 读取已知分布数据 …')
    known_df = collect_frames(KNOWN_FILES, args.known_sample)

    print('\n[2/4] 过滤与标准化 …')
    known_df = known_df[known_df['Label'].isin(KNOWN_LABELS)].copy()
    missing = [c for c in feature_names if c not in known_df.columns]
    if missing:
        raise SystemExit(f'CSV 缺少特征列：{missing[:5]}')
    feature_block = known_df[feature_names].apply(pd.to_numeric, errors='coerce')
    feature_block = feature_block.replace([np.inf, -np.inf], np.nan)
    keep = feature_block.notna().all(axis=1)
    rows_before = len(known_df)
    known_df = known_df[keep].reset_index(drop=True)
    Xk = feature_block[keep].to_numpy(dtype=np.float32)
    Xk_norm = (Xk - mean) / std
    print(f'  可用样本 {len(Xk_norm):,} 行（剔除含 NaN/Inf 的 {rows_before - len(Xk_norm):,} 行）')

    # ---- 类中心：用已知分布的每类均值 ----
    label_to_class = {lab: i for i, lab in enumerate(class_names)}
    label_map = {
        'Benign': 'Benign',
        'Bot': 'Bot',
        'DDOS attack-HOIC': 'DDoS-HOIC',
        'DoS attacks-GoldenEye': 'DoS-GoldenEye',
        'DoS attacks-Slowloris': 'DoS-Slowloris',
        'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
    }
    centers = np.zeros((len(class_names), len(feature_names)), dtype=np.float32)
    for raw, mapped in label_map.items():
        idx = label_to_class.get(mapped)
        if idx is None:
            continue
        mask = (known_df['Label'].to_numpy() == raw)[: len(Xk_norm)]
        if mask.sum() == 0:
            continue
        centers[idx] = Xk_norm[mask].mean(axis=0)
    print(f'  类中心已算：{[(c, int((known_df["Label"] == r).sum())) for r, c in label_map.items()]}')

    def nearest_distance(X: np.ndarray) -> np.ndarray:
        # 到每个类中心的欧氏距离取最小。用矩阵化写法避免逐样本循环。
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return d.min(axis=1)

    print('\n[3/4] 已知类的距离分布 …')
    dist_known = nearest_distance(Xk_norm)
    pct = [50, 90, 99, 99.9, 99.99]
    percentiles = {p: float(np.percentile(dist_known, p)) for p in pct}
    print(f'  均值 {dist_known.mean():.4f} · 最大 {dist_known.max():.4f}')
    for p in pct:
        print(f'  p{p:<6} = {percentiles[p]:.4f}')

    print('\n[4/4] 未见过的攻击类 …')
    # 三天训练数据里没有任何训练集之外的攻击类型（全是那 6 类），
    # 所以未见样本必须从专门的文件里取：Thursday-01-03 带 Infilteration。
    ood_sources = [args.ood_file] if args.ood_file else KNOWN_FILES
    ood_df = collect_frames(ood_sources, args.ood_sample)
    ood_df = ood_df[~ood_df['Label'].isin(KNOWN_LABELS)].copy()
    ood_df = ood_df.dropna(subset=['Label'])
    if ood_df.empty:
        print('  这三天的 CSV 里没有训练集之外的攻击类型')
        return 1

    print('  未见类别原始分布：')
    for lab, cnt in ood_df['Label'].value_counts().items():
        print(f'    {lab}: {cnt:,}')

    # 同样是脏行问题：这些 CSV 里混着重复表头，某一列会冒出 'ACK Flag Cnt' 之类的字符串。
    # 先 to_numeric 把非数值压成 NaN，再按行剔除，否则 to_numpy(float32) 直接抛 ValueError。
    ood_block = ood_df[feature_names].apply(pd.to_numeric, errors='coerce')
    ood_block = ood_block.replace([np.inf, -np.inf], np.nan)
    ood_keep = ood_block.notna().all(axis=1)
    ood_df = ood_df[ood_keep].reset_index(drop=True)
    Xo = ood_block[ood_keep].to_numpy(dtype=np.float32)
    Xo_norm = (Xo - mean) / std
    dist_ood = nearest_distance(Xo_norm)

    # 判据：距离超过已知类 p99.9 就当作目标外
    threshold = percentiles[99.9]
    trigger_known = float((dist_known > threshold).mean() * 100)
    trigger_ood = float((dist_ood > threshold).mean() * 100)

    print(f'\n  阈值取已知类 p99.9 = {threshold:.4f}')
    print(f'  已知类误触发率: {trigger_known:.3f}%  (期望≈0.1%)')
    print(f'  未见类触发率  : {trigger_ood:.3f}%  (越高越好)')
    print(f'  未见类平均距离: {dist_ood.mean():.4f}（已知类 {dist_known.mean():.4f}）')

    # 顺带看模型的软判定：未见类会被硬塞进哪一类
    with torch.no_grad():
        logits = model(torch.from_numpy(Xo_norm))
        probs = torch.softmax(logits / temperature, dim=1).numpy()
    pred = probs.argmax(axis=1)
    print('\n  未见类被硬塞进的类别分布：')
    for i, name in enumerate(class_names):
        cnt = int((pred == i).sum())
        if cnt:
            print(f'    {name}: {cnt:,} ({cnt / len(pred) * 100:.1f}%)')

    result = {
        'checkpoint': CHECKPOINT,
        'class_names': class_names,
        'known_samples': int(len(Xk_norm)),
        'ood_samples': int(len(Xo_norm)),
        'known_distance': {
            'mean': float(dist_known.mean()),
            'max': float(dist_known.max()),
            'percentiles': {str(p): v for p, v in percentiles.items()},
        },
        'ood_distance': {
            'mean': float(dist_ood.mean()),
            'max': float(dist_ood.max()),
        },
        'threshold_p99_9': float(threshold),
        'trigger_rate_known_pct': trigger_known,
        'trigger_rate_ood_pct': trigger_ood,
        'ood_hard_assignment': {name: int((pred == i).sum()) for i, name in enumerate(class_names)},
        'ood_label_distribution': {k: int(v) for k, v in ood_df['Label'].value_counts().items()},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已写入 {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
