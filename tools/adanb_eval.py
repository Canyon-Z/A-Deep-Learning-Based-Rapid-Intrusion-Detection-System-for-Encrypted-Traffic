"""验证 AdaBN（自适应批归一化）能否救回跨天退化的流特征模型。

问题背景：模型在训练分布（2 月某两天）上验证准确率 99.99%，换到没见过的
3 月 1 日数据上恶意检出率是 0% —— 同一套 78 维流特征，两天的均值/方差整体
平移，网络第一层看到的输入分布就和训练时不一样了。

AdaBN 的思路：BatchNorm 的 running_mean/running_var 本质上在描述「这批数据
长什么样」。把模型在目标域的无标签数据上跑几遍 forward（只更新 BN 统计量，
不动任何权重），统计量就重新对齐到目标域。

它的工程价值在于**不需要标签**：部署到新环境，先让它看几分钟正常流量把统计
量调过来，再开始判定。

用法：
    # 严格跨域：模型只在 15-02+21-02 训过，拿 02-03 测
    python tools/adanb_eval.py --checkpoint checkpoints/flow_N1_global.pth \\
        --target-file Friday-02-03-2018_TrafficForML_CICFlowMeter.csv

    # 扫一遍校准比例，看需要多少正常流量才够
    python tools/adanb_eval.py --sweep-ratio 0.05,0.1,0.2,0.4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.flow_mlp import FlowMLP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, 'data', 'cse_csv')

# 六分类模型的原始 Label → 类别名
LABEL_TO_CLASS = {
    'Benign': 'Benign',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS-HOIC',
    'DoS attacks-GoldenEye': 'DoS-GoldenEye',
    'DoS attacks-Slowloris': 'DoS-Slowloris',
    'DDOS attack-LOIC-UDP': 'DDoS-LOIC-UDP',
}


def load_frame(name: str) -> pd.DataFrame:
    path = name if os.path.isabs(name) else os.path.join(CSV_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f'找不到 CSV：{path}')
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_label_map(class_names: list[str], raw_labels: pd.Series) -> np.ndarray:
    """把 CSV 的 Label 列映射成模型类别下标；无法识别的行标 -1。"""
    stripped = raw_labels.astype(str).str.strip()
    if class_names == ['Benign', 'Malware']:
        # 二分类模型：Benign 是 0，其余已知攻击一律算 1；标签缺失或认不出的行标 -1
        known = {
            'Benign', 'Bot',
            'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DoS attacks-Hulk',
            'DoS attacks-SlowHTTPTest', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP',
            'DDoS attacks-LOIC-HTTP', 'Brute Force -Web', 'Brute Force -XSS',
            'SQL Injection', 'Infilteration', 'FTP-BruteForce', 'SSH-Bruteforce',
        }
        valid = stripped.isin(known)
        y = np.where(stripped == 'Benign', 0, 1).astype(np.int64)
        return np.where(valid, y, -1).astype(np.int64)
    index = {name: i for i, name in enumerate(class_names)}
    mapped = stripped.map(LABEL_TO_CLASS).map(index)
    return mapped.fillna(-1).to_numpy(dtype=np.int64)


def prepare(df: pd.DataFrame, feature_names: list[str], class_names: list[str]):
    block = df[feature_names].apply(pd.to_numeric, errors='coerce')
    block = block.replace([np.inf, -np.inf], np.nan)
    keep = block.notna().all(axis=1)
    X = block[keep].to_numpy(dtype=np.float32)
    y = build_label_map(class_names, df.loc[keep, 'Label'])
    return X, y


def reset_bn_stats(model: nn.Module) -> None:
    """BN 统计量清零；momentum=None 让 PyTorch 用累积平均而不是滑动平均。"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None


def settle_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 0.1


@torch.no_grad()
def calibrate(model: nn.Module, X_calib: np.ndarray, rounds: int, batch_size: int = 512) -> None:
    """在目标域无标签数据上前向若干轮，只让 BN 统计量跟着动。"""
    model.eval()
    reset_bn_stats(model)
    for _ in range(rounds):
        for start in range(0, len(X_calib), batch_size):
            model(torch.from_numpy(X_calib[start:start + batch_size]))
    settle_bn(model)
    model.eval()


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, temperature: float, batch_size: int = 512):
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[start:start + batch_size])) / temperature
        chunks.append(torch.softmax(logits, dim=1).numpy())
    prob = np.vstack(chunks)
    return prob.argmax(axis=1), prob


def binary_report(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """把预测压成「Benign vs 攻击」，报告实际最关心的四个数。"""
    tp = int(((pred > 0) & (y_true > 0)).sum())
    fp = int(((pred > 0) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true > 0)).sum())
    total, attack, benign = tp + fp + tn + fn, tp + fn, tn + fp
    return {
        'samples': total,
        'accuracy': round((tp + tn) / total * 100, 4) if total else 0.0,
        'detection_rate': round(tp / attack * 100, 4) if attack else 0.0,
        'false_positive_rate': round(fp / benign * 100, 4) if benign else 0.0,
        'precision': round(tp / (tp + fp) * 100, 4) if (tp + fp) else 0.0,
        'tn': tn, 'fp': fp, 'tp': tp, 'fn': fn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.join('checkpoints', 'flow_N1_global.pth'),
                    help='二分类或六分类的流特征权重')
    ap.add_argument('--target-file', default='Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    ap.add_argument('--calib-ratio', type=float, default=0.2,
                    help='用目标域前多少比例的数据做无标签校准')
    ap.add_argument('--calib-all', action='store_true',
                    help='用不带标签的整批数据校准（默认只用良性行）。'
                         '注意：训练时 BN 统计量来自良性+恶意的混合 batch，'
                         '只用纯良性数据校准会把分布整体推偏，导致全判恶意。')
    ap.add_argument('--sweep-ratio', default='', help='逗号分隔的多个校准比例，做一次扫描')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--eval-limit', type=int, default=60000)
    ap.add_argument('--output', default=os.path.join(ROOT, 'logs', 'adanb_eval.json'))
    args = ap.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    feature_names = list(ckpt['feature_names'])
    class_names = list(ckpt['class_names'])
    mean = np.asarray(ckpt['normalization']['mean'], dtype=np.float32)
    std = np.asarray(ckpt['normalization']['std'], dtype=np.float32)
    temperature = float(ckpt.get('temperature') or 1.0)
    hidden = tuple(ckpt.get('hidden_sizes', (256, 128)))

    def build():
        m = FlowMLP(input_dim=ckpt['input_dim'], hidden_sizes=hidden,
                    num_classes=ckpt['num_classes'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        return m

    print('=' * 72)
    print('AdaBN 跨域适配评估')
    print('=' * 72)
    print(f'权重      : {os.path.relpath(ckpt_path, ROOT)}')
    print(f'类别      : {class_names}   特征数 {len(feature_names)}')
    print(f'目标域    : {args.target_file}')

    df = load_frame(args.target_file)
    X_raw, y_all = prepare(df, feature_names, class_names)
    n_labelled = int((y_all >= 0).sum())
    n_attack = int((y_all > 0).sum())
    print(f'可用样本  : {len(X_raw):,} 行（带标签 {n_labelled:,}，其中攻击 {n_attack:,}）')

    X_n = ((X_raw - mean) / std).astype(np.float32)

    # 评估集：带标签、且不属于校准集的那部分
    benign_idx = np.where(y_all == 0)[0]
    ratios = [float(v) for v in args.sweep_ratio.split(',')] if args.sweep_ratio else [args.calib_ratio]
    ratios = [r for r in ratios if 0 < r <= 1]

    print('\n[基线] 不做任何适配，直接用训练好的权重 …')
    base_model = build()
    pred0, _ = predict(base_model, X_n, temperature)
    base_labelled = y_all >= 0
    base = binary_report(y_all[base_labelled], pred0[base_labelled])
    print(f'  准确率 {base["accuracy"]:.2f}% · 检出 {base["detection_rate"]:.2f}%'
          f' · 误报 {base["false_positive_rate"]:.2f}% · 精确率 {base["precision"]:.2f}%'
          f'   (TP={base["tp"]} FP={base["fp"]} TN={base["tn"]} FN={base["fn"]})')

    results = []
    for ratio in ratios:
        pool = np.arange(len(X_n)) if args.calib_all else benign_idx
        n_calib = max(1, int(len(pool) * ratio))
        calib_idx = pool[:n_calib]
        eval_mask = np.ones(len(X_n), dtype=bool)
        eval_mask[calib_idx] = False
        eval_mask &= (y_all >= 0)
        idx_all = np.where(eval_mask)[0]
        # 缺陷 D-19：过去是 `idx_all[:eval_limit]` 顺序截断。CICIDS 的 CSV 按采集时间
        # 排序，前 30000 行往往全属于同一个类别（Friday 数据前段全是 DoS），于是
        # 评估集只剩攻击样本、而基线权重在该域又全判良性 —— accuracy 与 FPR 双双为 0，
        # 看起来像「评估脚本没接上标签」，其实是指标在单类子集上的必然结果。
        # 改成按类别分层的定长抽样：两类各占一半配额，样本不足时让另一边补足。
        if args.eval_limit and len(idx_all) > args.eval_limit:
            _pos = idx_all[y_all[idx_all] > 0]
            _neg = idx_all[y_all[idx_all] == 0]
            _half = max(1, args.eval_limit // 2)
            _n_pos = min(len(_pos), _half)
            _n_neg = min(len(_neg), args.eval_limit - _n_pos)
            _n_pos = min(len(_pos), args.eval_limit - _n_neg)
            _rs = np.random.RandomState(42)
            _parts = []
            if _n_pos:
                _parts.append(_rs.choice(_pos, _n_pos, replace=False))
            if _n_neg:
                _parts.append(_rs.choice(_neg, _n_neg, replace=False))
            idx = np.sort(np.concatenate(_parts)) if _parts else np.array([], dtype=int)
        else:
            idx = idx_all
        X_eval, y_eval = X_n[idx], y_all[idx]
        if len(idx) == 0:
            print(f'\n[AdaBN {ratio:.0%}] 校准后没有可评估样本，跳过')
            continue
        _eval_attack = int((y_eval > 0).sum())
        print(f'  评估集 {len(idx):,} 行（攻击 {_eval_attack:,} / 良性 {len(idx) - _eval_attack:,}）')

        print(f'\n[AdaBN {ratio:.0%}] 用 {n_calib:,} 行目标域良性流量校准 BN（{args.rounds} 轮前向）…')
        model = build()
        calibrate(model, X_n[calib_idx], args.rounds)
        pred1, _ = predict(model, X_eval, temperature)
        ada = binary_report(y_eval, pred1)
        print(f'  准确率 {ada["accuracy"]:.2f}% · 检出 {ada["detection_rate"]:.2f}%'
              f' · 误报 {ada["false_positive_rate"]:.2f}% · 精确率 {ada["precision"]:.2f}%'
              f'   (TP={ada["tp"]} FP={ada["fp"]} TN={ada["tn"]} FN={ada["fn"]})')
        results.append({'calib_ratio': ratio, 'calib_rows': int(n_calib),
                        'eval_samples': int(len(idx)), **ada})

    print('\n' + '=' * 72)
    print('对比（评估集随校准比例变化，故同时看检出率与误报率）')
    print('=' * 72)
    print(f'{"配置":<16}{"检出率":>10}{"误报率":>10}{"精确率":>10}{"准确率":>10}')
    print(f'{"基线":<16}{base["detection_rate"]:>9.2f}%{base["false_positive_rate"]:>9.2f}%'
          f'{base["precision"]:>9.2f}%{base["accuracy"]:>9.2f}%')
    for row in results:
        print(f'{("AdaBN " + format(row["calib_ratio"], ".0%")):<16}'
              f'{row["detection_rate"]:>9.2f}%{row["false_positive_rate"]:>9.2f}%'
              f'{row["precision"]:>9.2f}%{row["accuracy"]:>9.2f}%')

    payload = {
        'checkpoint': os.path.relpath(ckpt_path, ROOT),
        'target_file': args.target_file,
        'class_names': class_names,
        'rounds': args.rounds,
        'baseline': base,
        'adanb': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n结果已写入 {os.path.relpath(args.output, ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
