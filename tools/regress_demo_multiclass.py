"""用 data/cse_csv/demo_mixed_6class.csv 回归六分类链路。

为什么单独写一个脚本而不是只靠 tests/：单元测试只锁「下限」（≥95%），
但修复报告要的是**可比的具体数字**（修复前 65.79%、P0-1 修复后 98.69%），
并且重训之后还要能回头看一眼「每个类掉了多少」。所以这里把整体准确率、
未知攻击占比、逐类 recall/precision、混淆矩阵、三档分布、OOD 阈值与
当前生效的标准化来源一起落成 JSON。

走的是 web/backend/main.py 的 `_classify_flow_matrix`——与 /analyze-flow、
/analyze（pcap）完全同一条链路。自己另写一份推理会把口径差异算进「回归」里。

用法：
    python tools/regress_demo_multiclass.py --tag after_block_retrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 演示集里的原始标签名 → 模型 class_names（'DOS attack-HOIC' 是同一类的手误写法）
LABEL_MAP = {
    'Benign': 'Benign',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS-HOIC',
    'DOS attack-HOIC': 'DDoS-HOIC',
    'DoS attacks-GoldenEye': 'DoS-GoldenEye',
    'DoS attacks-Slowloris': 'DoS-Slowloris',
    'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
}


def load_demo(csv_path, feature_names):
    """返回 (X, y, 脏行数)。脏行指「重复表头」——数值列取值等于列名本身。"""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    num = df[feature_names].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    noise = (num.isna() & df[feature_names].notna()).any(axis=1)
    X = num.loc[~noise].to_numpy(dtype=np.float32)
    y = df.loc[~noise, 'Label'].map(
        lambda s: LABEL_MAP.get(str(s).strip(), str(s).strip())).to_numpy()
    return X, y, int(noise.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description='demo_mixed_6class.csv 六分类回归')
    ap.add_argument('--csv', default=os.path.join(ROOT, 'data', 'cse_csv',
                                                 'demo_mixed_6class.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'logs',
                                                 'demo_multiclass_regression.json'))
    ap.add_argument('--tag', default='', help='写入产物的标记，便于区分重训前后')
    args = ap.parse_args()

    import web.backend.main as main

    if main.flow_multiclass is None or main.flow_multiclass_meta is None:
        raise SystemExit('六分类权重没加载，无法回归')
    if not os.path.exists(args.csv):
        raise SystemExit(f'找不到演示集 {args.csv}')

    meta = main.flow_multiclass_meta
    feature_names = list(meta['feature_names'])
    class_names = list(meta['class_names'])
    X, y, noise = load_demo(args.csv, feature_names)

    _, detail = main._classify_flow_matrix(X)
    if detail is None:
        raise SystemExit('_classify_flow_matrix 返回空，六分类链路不可用')

    idx_of = {c: i for i, c in enumerate(detail['class_names'])}
    pred_idx = np.asarray(detail['pred'])
    pred = np.array([detail['class_names'][i] for i in pred_idx])
    conf = np.asarray(detail['conf'])
    is_unknown = np.asarray(detail['is_unknown'])

    acc = float((pred == y).mean()) * 100.0
    per_class = {}
    for name in class_names:
        truth = y == name
        hit = pred == name
        if truth.any():
            per_class[name] = {
                'support': int(truth.sum()),
                'recall': round(float((pred[truth] == name).mean()) * 100.0, 2),
                'precision': round(float((y[hit] == name).mean()) * 100.0, 2) if hit.any() else None,
            }
        else:
            per_class[name] = {'support': 0, 'recall': None, 'precision': None}

    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y, pred):
        ti, pi = idx_of.get(t), idx_of.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    tiers = {}
    for tier in ('auto', 'review', 'manual'):
        m = np.asarray(detail['tier']) == tier
        tiers[tier] = int(m.sum())

    # 「未知攻击」在回归里单独统计：它是 B1 阻塞项的核心症状（44.3% → 0.86%）
    unknown_ratio = float(is_unknown.mean()) * 100.0
    # 判成未知时 argmax 仍照算，但那部分准确率不能混进「模型认得出」的口径
    known_mask = ~is_unknown
    acc_known = float((pred[known_mask] == y[known_mask]).mean()) * 100.0 \
        if known_mask.any() else None

    calib = getattr(main, '_CALIBRATION_STATUS', None) or {}
    centers = getattr(main, 'flow_centers', None) or {}

    result = {
        'tag': args.tag,
        'csv': os.path.relpath(args.csv, ROOT).replace('\\', '/'),
        'rows': int(len(y)),
        'header_noise_rows': noise,
        'class_names': class_names,
        'overall_accuracy': round(acc, 2),
        'accuracy_excluding_unknown': round(acc_known, 2) if acc_known is not None else None,
        'unknown_ratio': round(unknown_ratio, 2),
        'unknown_count': int(is_unknown.sum()),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'tiers': tiers,
        'mean_confidence': round(float(conf.mean()), 4),
        'temperature': meta.get('temperature'),
        'ood_enabled': bool(detail.get('ood_enabled')),
        'ood_disabled_reason': detail.get('ood_disabled_reason'),
        'ood_threshold': detail.get('ood_threshold'),
        'centers_space_matches_active': centers.get('space_matches_active'),
        'checkpoint_normalization_source': calib.get('active_source'),
        'split_strategy': meta.get('split_strategy'),
    }

    print(f"演示集 {result['csv']}：{result['rows']} 行（脏行 {noise}）")
    print(f"整体准确率 {result['overall_accuracy']:.2f}%  "
          f"（排除未知后 {result['accuracy_excluding_unknown']}%）")
    print(f"判为未知攻击 {result['unknown_count']} 条（{result['unknown_ratio']:.2f}%）")
    print(f"OOD 阈值 {result['ood_threshold']} · 启用={result['ood_enabled']} · "
          f"类中心同空间={result['centers_space_matches_active']}")
    print(f"当前标准化来源 {result['checkpoint_normalization_source']} · "
          f"权重切分口径 {result['split_strategy']}")
    print('\n逐类：')
    for name in class_names:
        r = per_class[name]
        print(f"  {name:<18}支持 {r['support']:>5}  recall {r['recall']}  precision {r['precision']}")
    print(f"\n三档：auto {tiers['auto']} / review {tiers['review']} / manual {tiers['manual']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f'\n已写入 {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
