"""为「未知攻击」判据计算六分类模型的类中心与距离统计。

后端要判断一行流特征「属不属于模型学过的六类」，需要一个参照：该行在标准化
空间里到最近类中心的距离。这个距离在训练分布内通常很小，遇到没见过的攻击
类型时会显著变大。

本脚本一次性把参照数据算好存成 npz，后端启动时只加载，不在每次请求里重算：

    checkpoints/flow_class_centers.npz
        centers       (6, 78)  每类在标准化空间里的均值向量
        class_names   (6,)     与 centers 行号对应
        mean / std    (78,)    标准化参数（与模型权重里的完全一致）
        thresholds    (5,)     [p50, p90, p99, p99.9, p99.99] 的距离分位
        percentiles   (5,)     对应的百分位数值

用法：
    python tools/build_flow_centers.py
    python tools/build_flow_centers.py --sample 400000
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
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'flow_multiclass.pth')
OUT_NPZ = os.path.join(ROOT, 'checkpoints', 'flow_class_centers.npz')
CSV_DIR = os.path.join(ROOT, 'data', 'cse_csv')

# 只有这三天的数据被用于训练，所以它们才能代表「训练分布」。
TRAIN_FILES = [
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
]

# CSV 里的原始 Label 到模型类别名的映射
LABEL_MAP = {
    'Benign': 'Benign',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS-HOIC',
    'DoS attacks-GoldenEye': 'DoS-GoldenEye',
    'DoS attacks-Slowloris': 'DoS-Slowloris',
    'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
}

PERCENTILES = [50, 90, 99, 99.9, 99.99]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=300000,
                    help='每个 CSV 最多读取多少行（None 表示全部）')
    ap.add_argument('--clusters', type=int, default=8,
                    help='每个类别切成多少个中心（k-means）。单中心对多峰类别不适用')
    ap.add_argument('--output', default=OUT_NPZ)
    args = ap.parse_args()

    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f'找不到模型权重：{CHECKPOINT}')

    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    feature_names = list(ckpt['feature_names'])
    class_names = list(ckpt['class_names'])
    mean = np.asarray(ckpt['normalization']['mean'], dtype=np.float32)
    std = np.asarray(ckpt['normalization']['std'], dtype=np.float32)

    print('=' * 68)
    print('构建类中心与距离分布')
    print('=' * 68)
    print(f'类别 {len(class_names)} 个: {class_names}')
    print(f'特征 {len(feature_names)} 维')

    frames = []
    for name in TRAIN_FILES:
        path = os.path.join(CSV_DIR, name)
        if not os.path.exists(path):
            print(f'  [跳过] 缺文件 {name}')
            continue
        nrows = args.sample if args.sample > 0 else None
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        print(f'  [读入] {name}: {len(df):,} 行')
    if not frames:
        raise SystemExit('没有可用的训练 CSV')

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all['Label'].isin(set(LABEL_MAP.keys()))].copy()

    block = df_all[feature_names].replace([np.inf, -np.inf], np.nan)
    keep = block.notna().all(axis=1)
    df_all = df_all[keep].reset_index(drop=True)
    X = block[keep].to_numpy(dtype=np.float32)
    X_norm = (X - mean) / std
    print(f'\n可用样本 {len(X_norm):,} 行')

    class_index = {name: i for i, name in enumerate(class_names)}
    counts = {}
    cluster_blocks = []
    cluster_owner = []
    labels_arr = df_all['Label'].to_numpy()
    print(f'\n每类切成最多 {args.clusters} 个子中心（k-means）：')
    for raw, mapped in LABEL_MAP.items():
        idx = class_index.get(mapped)
        if idx is None:
            print(f'  [警告] 模型里没有类别 {mapped}，跳过')
            continue
        mask = (labels_arr == raw)
        n = int(mask.sum())
        counts[mapped] = n
        if n == 0:
            print(f'  [警告] {mapped} 没有样本，跳过')
            continue
        sub = X_norm[mask]
        k = max(1, min(args.clusters, n))
        if k == 1:
            block = sub.mean(axis=0, keepdims=True)
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3,
                                 batch_size=4096, max_iter=100)
            km.fit(sub)
            block = km.cluster_centers_.astype(np.float32)
        cluster_blocks.append(block)
        cluster_owner.extend([idx] * len(block))
        print(f'  {mapped}: {n:,} 样本 -> {len(block)} 个中心')

    # 单一类中心对多峰类别不成立：Benign 混着几十种正常业务、Slowloris 跨多个时间片，
    # 均值会落在「谁都不像」的空地上，于是大量正常行被判成离中心很远。
    # 改成每类多个 k-means 子中心，距离取到最近子中心，判据才有意义。
    centers = np.vstack(cluster_blocks).astype(np.float32)
    cluster_owner_arr = np.array(cluster_owner, dtype=np.int32)

    dist = np.empty(len(X_norm), dtype=np.float32)
    CHUNK = 20000
    for start in range(0, len(X_norm), CHUNK):
        block = X_norm[start:start + CHUNK]
        d = np.linalg.norm(block[:, None, :] - centers[None, :, :], axis=2)
        dist[start:start + CHUNK] = d.min(axis=1)
    thresholds = np.array([np.percentile(dist, p) for p in PERCENTILES], dtype=np.float32)

    print('\n训练分布内的距离统计：')
    print(f'  均值 {dist.mean():.4f} · 中位 {np.median(dist):.4f} · 最大 {dist.max():.4f}')
    for p, t in zip(PERCENTILES, thresholds):
        print(f'  p{p:<6} = {t:.4f}')

    np.savez_compressed(
        args.output,
        centers=centers,
        cluster_owner=cluster_owner_arr,
        class_names=np.array(class_names),
        feature_names=np.array(feature_names),
        mean=mean,
        std=std,
        thresholds=thresholds,
        percentiles=np.array(PERCENTILES, dtype=np.float32),
    )
    print(f'\n已写入 {args.output}')

    side = os.path.splitext(args.output)[0] + '.json'
    with open(side, 'w', encoding='utf-8') as f:
        json.dump({
            'class_names': class_names,
            'class_counts': counts,
            'distance': {
                'mean': float(dist.mean()),
                'median': float(np.median(dist)),
                'max': float(dist.max()),
                'percentiles': {str(p): float(t) for p, t in zip(PERCENTILES, thresholds)},
            },
        }, f, ensure_ascii=False, indent=2)
    print(f'已写入 {side}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
