"""从特征缓存估计「到训练分布的距离」判据所需的统计量。

为什么需要它：级联的置信度判据实测不够用——拿 8 份分布外的工控协议 pcap
（BACnet/DNP3/Modbus/Zigbee）去跑，模型给出的批次平均置信度是 0.9585~0.9998，
甚至比同源流量还自信。softmax 完全无法表达「这是我没见过的东西」。

这里用**对角马氏距离**当分布外指标：

    z_i  = (x_i - mu_i) / sigma_i
    dist = mean_i(z_i^2)

mu/sigma 从训练集（USTC 会话图，展平成 784 维）估计。推理时每个会话算一个 dist，
超过训练集 dist 的 99 分位就认为落在分布外，可以触发级联升级。

用对角形式（只存均值与方差，784×2 个 float ≈ 6KB）而不是完整协方差（784×784
≈ 2.4MB 且需要远超样本量才估得准），是因为 28x28 的像素间相关性对本判据贡献不大，
而鲁棒性和体积都更可控。

输出：checkpoints/dist_stats.npz，含 mean / std / p90 / p99 / p999 / count

用法：
    python -u tools/compute_dist_stats.py
    python -u tools/compute_dist_stats.py --cache-dir data/USTC-TFC2016-master/.feature_cache
"""

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def iter_batches(npz_files, batch_sessions=20000):
    """按批从 npz 缓存里读会话图，避免一次性把 140 万张图读进内存。"""
    for path in npz_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                sessions = data['sessions']
        except Exception as e:
            print(f"[skip] {os.path.basename(path)}: {e}")
            continue
        if sessions.ndim == 2:
            sessions = np.expand_dims(sessions, axis=0)
        n = sessions.shape[0]
        for start in range(0, n, batch_sessions):
            yield sessions[start:start + batch_sessions]


def main():
    parser = argparse.ArgumentParser(description="估计对角马氏距离的均值/方差")
    parser.add_argument('--cache-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'USTC-TFC2016-master',
                                             '.feature_cache'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'checkpoints', 'dist_stats.npz'))
    parser.add_argument('--max-sessions', type=int, default=600000,
                        help="最多用多少个会话估计（够用即可，默认 60 万）")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.cache_dir, '*.npz')))
    if not npz_files:
        print(f"在 {args.cache_dir} 里找不到 npz 缓存")
        return 1
    print(f"[find] {len(npz_files)} 个 npz 缓存")

    dim = 28 * 28
    count = 0
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)

    for batch in iter_batches(npz_files):
        if count >= args.max_sessions:
            break
        take = min(batch.shape[0], args.max_sessions - count)
        flat = batch[:take].reshape(take, -1).astype(np.float64) / 255.0
        total += flat.sum(axis=0)
        total_sq += (flat ** 2).sum(axis=0)
        count += take
        if count % 100000 < take:
            print(f"  已累计 {count} 个会话")

    if count == 0:
        print("没有读到任何会话")
        return 1

    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0

    print(f"[stats] 用 {count} 个会话估计：mean 范围 [{mean.min():.4f}, {mean.max():.4f}]，"
          f"std 中位数 {np.median(std):.4f}")

    # 在同一个训练集上算 dist 的分布，作为「分布内」的参考分位
    print("[stats] 计算训练集自身距离分位 ...")
    dists = []
    seen = 0
    for batch in iter_batches(npz_files):
        if seen >= count:
            break
        take = min(batch.shape[0], count - seen)
        flat = batch[:take].reshape(take, -1).astype(np.float64) / 255.0
        z = (flat - mean) / std
        dists.append((z ** 2).mean(axis=1))
        seen += take
    dists = np.concatenate(dists)
    p90, p99, p999 = (float(np.percentile(dists, q)) for q in (90, 99, 99.9))
    print(f"[stats] 训练集距离：中位 {np.median(dists):.4f} | p90 {p90:.4f} | "
          f"p99 {p99:.4f} | p99.9 {p999:.4f} | 最大 {dists.max():.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32),
             p90=p90, p99=p99, p999=p999, count=count)
    print(f"[done] 已保存 -> {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
