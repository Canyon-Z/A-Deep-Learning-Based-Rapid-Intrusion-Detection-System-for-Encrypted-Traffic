"""目标域少量有标签样本微调：回答「实际部署时要标注多少条才够用」。

背景：源域（2 月两天）训练出来的模型，换到没见过的 3 月 1 日数据上检出率 0%，
AdaBN、批内标准化、特征剔除、AutoEncoder 全部试过，都没救回来。根本原因是
跨天的流特征分布整体漂移，且类条件分布重叠。

那么工程上唯一确定有效的办法，就是让模型看目标环境自己的数据。这个脚本量的是
「最少需要多少条人工标注」—— 这是部署方案能不能落地的前提。

做法：
1. 加载源域权重（默认 flow_N1_global.pth，只在 15-02+21-02 上训过）
2. 从目标域按标签比例分层抽 N 条当作「人工标注预算」
3. 在这 N 条上微调（Adam，小学习率，几轮）
4. 在目标域剩余样本上评估检出率/误报率

用法：
    python tools/finetune_local.py --n-labels 200,500,1000,5000
    python tools/finetune_local.py --n-labels 1000 --epochs 10 --lr 5e-4
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

KNOWN_LABELS = {
    'Benign', 'Bot',
    'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DoS attacks-Hulk',
    'DoS attacks-SlowHTTPTest', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP',
    'DDoS attacks-LOIC-HTTP', 'Brute Force -Web', 'Brute Force -XSS',
    'SQL Injection', 'Infilteration', 'FTP-BruteForce', 'SSH-Bruteforce',
}


def load_frame(name: str) -> pd.DataFrame:
    path = name if os.path.isabs(name) else os.path.join(CSV_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f'找不到 CSV：{path}')
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare(df: pd.DataFrame, feature_names: list[str]):
    block = df[feature_names].apply(pd.to_numeric, errors='coerce')
    block = block.replace([np.inf, -np.inf], np.nan)
    keep = block.notna().all(axis=1)
    X = block[keep].to_numpy(dtype=np.float32)
    raw = df.loc[keep, 'Label'].astype(str).str.strip()
    valid = raw.isin(KNOWN_LABELS)
    y = np.where(raw == 'Benign', 0, 1).astype(np.int64)
    y = np.where(valid, y, -1)
    return X, y


def stratified_pick(y: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    """按类别比例抽取标注预算，保证攻击样本不会被抽空。"""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    chosen = []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        take = max(1, int(round(n_total * cnt / len(y))))
        take = min(take, len(idx))
        chosen.append(rng.choice(idx, take, replace=False))
    return np.concatenate(chosen)


def binary_report(y_true: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred > 0) & (y_true > 0)).sum())
    fp = int(((pred > 0) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true > 0)).sum())
    total, attack, benign = tp + fp + tn + fn, tp + fn, tn + fp
    return {
        'accuracy': round((tp + tn) / total * 100, 4) if total else 0.0,
        'detection_rate': round(tp / attack * 100, 4) if attack else 0.0,
        'false_positive_rate': round(fp / benign * 100, 4) if benign else 0.0,
        'precision': round(tp / (tp + fp) * 100, 4) if (tp + fp) else 0.0,
        'tn': tn, 'fp': fp, 'tp': tp, 'fn': fn,
    }


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(X), batch_size):
        out.append(model(torch.from_numpy(X[s:s + batch_size])).argmax(dim=1).numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def finetune(model: nn.Module, X: np.ndarray, y: np.ndarray,
             epochs: int, lr: float, batch_size: int = 64) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练集里攻击占比可能很低，用类别权重避免被良性样本淹没
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    for epoch in range(epochs):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:      # BatchNorm 在单样本 batch 上会报错
                continue
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        print(f'    epoch {epoch + 1}/{epochs}  loss={total / len(Xt):.4f}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.join('checkpoints', 'flow_N1_global.pth'))
    ap.add_argument('--target-file', default='Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    ap.add_argument('--n-labels', default='200,500,1000,5000',
                    help='要试的标注预算，逗号分隔')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-limit', type=int, default=60000)
    ap.add_argument('--save-prefix', default='',
                    help='给定时把微调结果存成 <prefix>_<标注数>.pth，可直接被后端加载')
    ap.add_argument('--output', default=os.path.join(ROOT, 'logs', 'finetune_local.json'))
    args = ap.parse_args()

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    feature_names = list(ckpt['feature_names'])
    mean = np.asarray(ckpt['normalization']['mean'], dtype=np.float32)
    std = np.asarray(ckpt['normalization']['std'], dtype=np.float32)
    hidden = tuple(ckpt.get('hidden_sizes', (256, 128)))

    def build():
        m = FlowMLP(input_dim=ckpt['input_dim'], hidden_sizes=hidden,
                    num_classes=ckpt['num_classes'])
        m.load_state_dict(ckpt['model_state_dict'])
        return m

    print('=' * 72)
    print('目标域少量标注微调实验')
    print('=' * 72)
    print(f'源域权重: {os.path.relpath(ckpt_path, ROOT)}')
    print(f'目标域  : {args.target_file}')

    df = load_frame(args.target_file)
    X_raw, y_all = prepare(df, feature_names)
    labelled = np.where(y_all >= 0)[0]
    print(f'可用样本: {len(labelled):,} 行（攻击 {int((y_all[labelled] > 0).sum()):,}）')

    X_n = ((X_raw - mean) / std).astype(np.float32)

    print('\n[基线] 源域权重直接用在目标域 …')
    m0 = build()
    pred0 = evaluate(m0, X_n[labelled][:args.eval_limit])
    base = binary_report(y_all[labelled][:args.eval_limit], pred0)
    print(f'  准确率 {base["accuracy"]:.2f}% · 检出 {base["detection_rate"]:.2f}%'
          f' · 误报 {base["false_positive_rate"]:.2f}% · 精确率 {base["precision"]:.2f}%')

    budgets = [int(v) for v in args.n_labels.split(',') if v.strip()]
    results = []
    for n in budgets:
        if n >= len(labelled):
            print(f'\n[标注 {n:,} 条] 超过可用样本数，跳过')
            continue
        pick = stratified_pick(y_all[labelled], n)
        pick_idx = labelled[pick]
        rest_mask = np.ones(len(y_all), dtype=bool)
        rest_mask[pick_idx] = False
        rest_mask &= (y_all >= 0)
        rest_idx = np.where(rest_mask)[0][:args.eval_limit]
        n_attack = int((y_all[pick_idx] > 0).sum())
        print(f'\n[标注 {n:,} 条] 其中攻击 {n_attack:,} 条；微调 {args.epochs} 轮 …')
        model = build()
        finetune(model, X_n[pick_idx], y_all[pick_idx], args.epochs, args.lr)
        pred = evaluate(model, X_n[rest_idx])
        rep = binary_report(y_all[rest_idx], pred)
        print(f'  准确率 {rep["accuracy"]:.2f}% · 检出 {rep["detection_rate"]:.2f}%'
              f' · 误报 {rep["false_positive_rate"]:.2f}% · 精确率 {rep["precision"]:.2f}%')
        results.append({'n_labels': n, 'n_attack_labels': n_attack,
                        'eval_samples': int(len(rest_idx)), **rep})

        if args.save_prefix:
            save_path = os.path.join(ROOT, 'checkpoints', f'{args.save_prefix}_{n}.pth')
            payload_ckpt = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
            payload_ckpt['model_state_dict'] = model.state_dict()
            payload_ckpt['local_finetune'] = {
                'source_checkpoint': os.path.relpath(ckpt_path, ROOT),
                'target_file': args.target_file,
                'n_labels': n,
                'n_attack_labels': n_attack,
                'epochs': args.epochs,
                'lr': args.lr,
                'metrics': rep,
            }
            torch.save(payload_ckpt, save_path)
            print(f'  权重已保存 → {os.path.relpath(save_path, ROOT)}')

    print('\n' + '=' * 72)
    print('结果汇总')
    print('=' * 72)
    print(f'{"标注预算":<14}{"检出率":>10}{"误报率":>10}{"精确率":>10}{"准确率":>10}')
    print(f'{"基线(0)":<14}{base["detection_rate"]:>9.2f}%{base["false_positive_rate"]:>9.2f}%'
          f'{base["precision"]:>9.2f}%{base["accuracy"]:>9.2f}%')
    for row in results:
        print(f'{format(row["n_labels"], ","):<14}'
              f'{row["detection_rate"]:>9.2f}%{row["false_positive_rate"]:>9.2f}%'
              f'{row["precision"]:>9.2f}%{row["accuracy"]:>9.2f}%')

    payload = {
        'checkpoint': os.path.relpath(ckpt_path, ROOT),
        'target_file': args.target_file,
        'epochs': args.epochs,
        'lr': args.lr,
        'baseline': base,
        'finetuned': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n结果已写入 {os.path.relpath(args.output, ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
