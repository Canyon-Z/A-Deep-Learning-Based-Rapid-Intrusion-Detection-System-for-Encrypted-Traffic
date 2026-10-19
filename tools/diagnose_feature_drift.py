"""特征跨天漂移诊断：找出哪些特征把「哪一天」写进了数据。

背景（B4 的实验结论，见 `logs/pipeline_b4_summary.txt`）：

  global   mean/std      → 检出 0.00%  AUC 0.3784
  batch    批内 median/MAD → 检出 31.64%  AUC 0.5064  （TP 从 0 变成 90,541）

批内归一化确实救回了一部分检出，但 AUC 仍趴在 0.51 —— 说明它只解掉了一部分域指纹，
剩下那些高漂移的特征还在把「这是哪一天」喂给模型。

这个脚本量化每个特征的跨天漂移，用来做特征筛选：

  ks        两样本 KS 统计量（0=两天分布一致，1=完全分离）
  day_auc   只用这一个特征区分「训练天 vs 测试天」的 AUC。
            越接近 1 说明它越像一个日期标签 —— 模型学到它就是在学域，不是在学恶意。
  shift     测试天中位数 / 训练天中位数（绝对量的偏移倍数，>1 表示放大，<1 表示缩小）

判读：
  day_auc > 0.85  强烈建议剔除（几乎等价于日期标签）
  0.70 ~ 0.85     建议做稳健变换或按批归一化
  < 0.70          可保留

用法：
    python -u tools/diagnose_feature_drift.py --train-days 15-02,21-02 --test-days 02-03
    python -u tools/diagnose_feature_drift.py --train-days 15-02,21-02 --test-days 02-03 --top 30
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.metrics import roc_auc_score

# 复用训练脚本里的读盘/对齐逻辑，保证两边的特征口径完全一致
from tools.train_flow_model import load_day, align_columns


def ks_statistic(a, b):
    """两样本 KS 统计量（不依赖 scipy）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    a.sort()
    b.sort()
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side='right') / a.size
    cdf_b = np.searchsorted(b, grid, side='right') / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def main():
    parser = argparse.ArgumentParser(description="诊断流特征的跨天漂移")
    parser.add_argument('--csv-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'cse_csv'))
    parser.add_argument('--train-days', type=str, required=True)
    parser.add_argument('--test-days', type=str, required=True)
    parser.add_argument('--max-rows', type=int, default=200000,
                        help="每天最多读多少行（诊断够用，默认 20 万）")
    parser.add_argument('--top', type=int, default=25, help="打印漂移最严重的前 N 个特征")
    parser.add_argument('--out', type=str, default=None,
                        help="结果 JSON，默认 logs/feature_drift_<train>_vs_<test>.json")
    args = parser.parse_args()

    train_days = [d.strip() for d in args.train_days.split(',') if d.strip()]
    test_days = [d.strip() for d in args.test_days.split(',') if d.strip()]

    train_frames, test_frames = [], []
    for day in train_days:
        df, _lab = load_day(args.csv_dir, day, args.max_rows)
        train_frames.append(df)
    for day in test_days:
        df, _lab = load_day(args.csv_dir, day, args.max_rows)
        test_frames.append(df)

    train_frames, feature_names = align_columns(train_frames)
    test_frames = [df[feature_names] for df in test_frames]

    X_train = np.vstack([df.to_numpy(dtype=np.float32) for df in train_frames])
    X_test = np.vstack([df.to_numpy(dtype=np.float32) for df in test_frames])
    X_train = X_train[np.isfinite(X_train).all(axis=1)]
    X_test = X_test[np.isfinite(X_test).all(axis=1)]
    print(f"[data ] 训练 {X_train.shape}（{len(train_days)} 天），"
          f"测试 {X_test.shape}（{len(test_days)} 天）")

    n_feat = len(feature_names)
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    rows = []
    for i, name in enumerate(feature_names):
        xtr = X_train[:, i].astype(np.float64)
        xte = X_test[:, i].astype(np.float64)

        ks = ks_statistic(xtr, xte)
        # 单特征区分「哪一天」的能力：把两天拼起来当二分类标签
        col = np.concatenate([xtr, xte])
        try:
            if np.nanstd(col) < 1e-12:
                day_auc = 0.5
            else:
                day_auc = float(roc_auc_score(y, col))
                day_auc = max(day_auc, 1.0 - day_auc)   # 方向无关，只看可分性
        except ValueError:
            day_auc = 0.5

        med_tr = float(np.median(xtr))
        med_te = float(np.median(xte))
        if abs(med_tr) > 1e-9:
            shift = med_te / med_tr
        else:
            shift = float('inf') if abs(med_te) > 1e-9 else 1.0

        rows.append({
            'feature': name,
            'ks': round(ks, 4),
            'day_auc': round(day_auc, 4),
            'median_train': med_tr,
            'median_test': med_te,
            'median_shift': round(shift, 4) if np.isfinite(shift) else None,
        })

    rows.sort(key=lambda r: (-r['day_auc'], -r['ks']))

    print(f"\n{'特征':<32}{'day_auc':>9}{'KS':>9}{'训练中位数':>14}{'测试中位数':>14}{'倍数':>10}  建议")
    print('-' * 104)
    for r in rows[:args.top]:
        if r['day_auc'] > 0.85:
            advice = '剔除'
        elif r['day_auc'] > 0.70:
            advice = '变换/批归一'
        else:
            advice = '保留'
        ratio = r['median_shift']
        ratio_txt = f"{ratio:.3g}" if ratio is not None else 'n/a'
        print(f"{r['feature'][:31]:<32}{r['day_auc']:>9.4f}{r['ks']:>9.4f}"
              f"{r['median_train']:>14.4g}{r['median_test']:>14.4g}{ratio_txt:>10}  {advice}")

    drop = [r['feature'] for r in rows if r['day_auc'] > 0.85]
    mid = [r['feature'] for r in rows if 0.70 < r['day_auc'] <= 0.85]
    print(f"\n[summary] 共 {n_feat} 个特征："
          f"建议剔除 {len(drop)} 个，需变换 {len(mid)} 个，可保留 {n_feat - len(drop) - len(mid)} 个")
    if drop:
        print(f"[summary] 剔除列表：{', '.join(drop)}")

    out = args.out or os.path.join(
        _PROJECT_ROOT, 'logs',
        f"feature_drift_{'_'.join(train_days)}_vs_{'_'.join(test_days)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({
            'train_days': train_days,
            'test_days': test_days,
            'feature_count': n_feat,
            'features': rows,
            'drop_candidates': drop,
            'transform_candidates': mid,
        }, fh, ensure_ascii=False, indent=2)
    print(f"[summary] 结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
