"""训练流统计特征（CICFlowMeter 风格）模型，并与字节图像分支做跨域对比。

为什么要有这条分支：项目原有的 6 个模型都吃 28x28 字节图像，在 CSE-CIC-IDS2018
上交叉验证时 AUC 只有 0.33~0.61（接近随机）。这一条分支改吃流统计特征，用来验证
"换表示能不能换来跨域鲁棒性"。

划分方式是**按天划分**（不是随机切行）：训练用若干天的 CSV，测试用另外几天。
同一天的流量在时间上连续、主机和攻击活动重合，随机切行会让训练集和测试集共享
同一批会话上下文，指标会虚高 —— 这是这个项目已经踩过一次的坑。

用法：
    python tools/train_flow_model.py --train-days 15-02,21-02 --test-days 02-03
    python tools/train_flow_model.py --train-days 15-02,21-02 --test-days 02-03 --epochs 30
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.flow_mlp import FlowMLP
from src.models.flow_input_norm import FlowInputNorm
from src.utils.device import resolve_device

BENIGN_LABEL = 'BENIGN'


def load_day(csv_dir, day, max_rows=None, verbose=True, with_meta=False):
    """读一天的 CSV，返回 (DataFrame, 标签数组原文)。列名会去掉官方文件里的前导空格。

    ``with_meta=True`` 时额外返回一个与 DataFrame 行对齐的元信息表（``timestamp``、
    ``day``），供多分类脚本做「时间块切分」。默认 ``False``，返回结构不变，
    二分类训练脚本无需改动。
    """
    import pandas as pd

    pattern = os.path.join(csv_dir, f'*{day}*.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"在 {csv_dir} 里找不到匹配 {day} 的 CSV（模式 {pattern}）")
    path = matches[0]

    df = pd.read_csv(path, low_memory=False, nrows=max_rows)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = None
    for c in df.columns:
        if c.lower() in ('label', 'labels'):
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"{os.path.basename(path)} 里找不到 Label 列，现有列：{list(df.columns)[:6]}…")

    raw_labels = df[label_col].astype(str).str.strip()
    df = df.drop(columns=[label_col])

    # 元信息必须在丢掉时间列**之前**取，且用同一份 index，保证逐行对齐。
    # CICIDS2018 的时间戳是 dd/mm/yyyy，必须显式 dayfirst=True —— 省掉它 pandas
    # 会按 month-first 猜，'15/02/2018' 直接变成 NaT，'02/03' 与 '03/02' 还会互换。
    meta = None
    if with_meta:
        meta = pd.DataFrame(index=df.index)
        ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
        if ts_col is not None:
            meta['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce', dayfirst=True)
        else:
            meta['timestamp'] = pd.NaT
        meta['day'] = day

    # Timestamp 之类的时间列对模型没有意义（而且会让模型学到"哪天"）
    df = df.drop(columns=[c for c in df.columns if c.lower() in ('timestamp', 'flow id', 'source ip', 'destination ip', 'src ip', 'dst ip')], errors='ignore')
    df = df.select_dtypes(include=[np.number])
    # 官方 CSV 里有 inf（Flow Bytes/s 除零等），必须清掉
    df = df.replace([np.inf, -np.inf], np.nan)

    if verbose:
        n_attack = int((raw_labels.str.upper() != BENIGN_LABEL).sum())
        print(f"[load] {os.path.basename(path)}: {len(df)} 行 / {df.shape[1]} 个数值特征；"
              f"恶意 {n_attack} ({n_attack / max(len(df), 1) * 100:.2f}%)")

    if with_meta:
        return df, raw_labels, meta
    return df, raw_labels


def align_columns(frames):
    """按列名交集对齐（官方不同天 CSV 的列数/列名略有出入），返回对齐后的帧列表与列名。"""
    common = None
    for df in frames:
        cols = set(df.columns)
        common = cols if common is None else (common & cols)
    common = sorted(common)
    print(f"[align] 各天共有 {len(common)} 个数值特征列")
    return [df[common] for df in frames], common


def to_binary(labels):
    """BENIGN -> 0，其余一律 -> 1（多类攻击合并成"恶意"）。"""
    return np.array([0 if str(v).strip().upper() == BENIGN_LABEL else 1 for v in labels], dtype=np.int64)


def evaluate(model, loader, device, name):
    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            all_pred.append(pred.cpu().numpy())
            all_true.append(yb.numpy())
            all_prob.append(prob.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, zero_division=0) * 100,
        'recall_malware': (tp / (tp + fn) * 100) if (tp + fn) else 0.0,
        'recall_benign': (tn / (tn + fp) * 100) if (tn + fp) else 0.0,
        'fpr': (fp / (fp + tn) * 100) if (fp + tn) else 0.0,
        'fnr': (fn / (fn + tp) * 100) if (fn + tp) else 0.0,
        'auc': roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'samples': int(len(y_true)),
    }
    auc_txt = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else '-'
    print(f"[{name}] 准确率 {metrics['accuracy']:.2f}% | 恶意检出 {metrics['recall_malware']:.2f}% | "
          f"误报 {metrics['fpr']:.2f}% | F1 {metrics['f1_score']:.2f}% | AUC {auc_txt}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="训练流统计特征模型（CSE-CIC-IDS2018 CSV）")
    parser.add_argument('--csv-dir', type=str, default=os.path.join(_PROJECT_ROOT, 'data', 'cse_csv'))
    parser.add_argument('--train-days', type=str, required=True, help="训练用日期，逗号分隔，如 15-02,21-02")
    parser.add_argument('--test-days', type=str, required=True, help="测试用日期，逗号分隔（跨天验证）")
    parser.add_argument('--max-rows', type=int, default=None, help="每天最多读多少行（调试用）")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='256,128', help="隐藏层宽度，逗号分隔")
    parser.add_argument('--patience', type=int, default=5, help="早停耐心值（0 表示关闭）")
    parser.add_argument('--class-weight', type=str, default='balanced',
                        choices=['balanced', 'none'],
                        help="类别加权：balanced=按频次反比（默认），none=不加权（对照用）")
    parser.add_argument('--norm-mode', type=str, default='global',
                        choices=['global', 'robust', 'batch'],
                        help="输入归一化方式：global=训练集 mean/std（原实现）；"
                             "robust=训练集 median/IQR（抗离群）；"
                             "batch=模型内按批内 median/MAD（不使用训练集统计量，用于跨域）")
    # 特征筛选与变换：口径来自 tools/diagnose_feature_drift.py 的 day_auc 排序
    # （day_auc 越接近 1，说明这个特征越像"日期标签"，模型学它等于在学域而不是学恶意）。
    parser.add_argument('--drop-features', type=str, default='',
                        help="剔除的高漂移特征，逗号分隔，如 \"Dst Port\"")
    parser.add_argument('--log-features', type=str, default='',
                        help="做 log1p 变换的特征名子串，逗号分隔，如 \"IAT,Duration,Pkts/s\"")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--out', type=str, default=None, help="权重输出路径")
    parser.add_argument('--metrics-out', type=str, default=None, help="指标 JSON 输出路径")
    args = parser.parse_args()

    train_days = [d.strip() for d in args.train_days.split(',') if d.strip()]
    test_days = [d.strip() for d in args.test_days.split(',') if d.strip()]
    device = resolve_device(args.device)
    print(f"设备：{device}")
    print(f"训练天：{train_days}   测试天：{test_days}\n")

    t0 = time.time()
    train_frames, train_labels, test_frames, test_labels = [], [], [], []
    for day in train_days:
        df, lab = load_day(args.csv_dir, day, args.max_rows)
        train_frames.append(df)
        train_labels.append(to_binary(lab))
    for day in test_days:
        df, lab = load_day(args.csv_dir, day, args.max_rows)
        test_frames.append(df)
        test_labels.append(to_binary(lab))

    train_frames, feature_names = align_columns(train_frames)

    # 高漂移特征剔除：必须在拼矩阵之前做，否则 X_train 的列顺序与 feature_names 对不上。
    # 诊断依据见 logs/feature_drift_*.json：Dst Port 的 day_auc 高达 0.90、
    # 中位数在两天之间差 101 倍（80 -> 8080），它几乎是"这是哪一天"的直接编码。
    if args.drop_features:
        drops = [f.strip() for f in args.drop_features.split(',') if f.strip()]
        missing = [f for f in drops if f not in feature_names]
        if missing:
            print(f"[feat ] 警告：以下特征不在列里，已忽略：{missing}")
        kept = [f for f in feature_names if f not in drops]
        if len(kept) == len(feature_names):
            print("[feat ] 提示：--drop-features 没有匹配到任何列，特征数未变")
        else:
            removed = len(feature_names) - len(kept)
            print(f"[feat ] 剔除 {removed} 个高漂移特征 -> 剩 {len(kept)} 个")
            feature_names = kept
            # 训练侧必须同步切片，否则 X_train 仍是 78 列而 X_test 是 77 列，
            # 后面的标准化与模型输入会对不上（踩过一次：broadcast (1044525,77) vs (1,78)）。
            train_frames = [df[feature_names] for df in train_frames]

    test_frames = [df[feature_names] for df in test_frames]

    X_train = np.vstack([df.to_numpy(dtype=np.float32) for df in train_frames])
    y_train = np.concatenate(train_labels)
    X_test = np.vstack([df.to_numpy(dtype=np.float32) for df in test_frames])
    y_test = np.concatenate(test_labels)

    # 对数变换：包间隔（IAT）/ 流时长 / 速率这类特征跨天相差一到两个数量级，
    # 直接喂线性层会让"绝对量级"主导梯度。log1p 把乘性偏移压成加性偏移，
    # 配合后面的标准化，跨天的相对形貌才可比。这些量本身非负，先截到 0 再取对数。
    if args.log_features:
        pats = [p.strip().lower() for p in args.log_features.split(',') if p.strip()]
        idxs = [i for i, n in enumerate(feature_names) if any(p in n.lower() for p in pats)]
        if idxs:
            X_train[:, idxs] = np.log1p(np.clip(X_train[:, idxs], 0.0, None))
            X_test[:, idxs] = np.log1p(np.clip(X_test[:, idxs], 0.0, None))
            shown = [feature_names[i] for i in idxs]
            print(f"[feat ] 对数变换 {len(idxs)} 个特征：{shown[:6]}{' …' if len(shown) > 6 else ''}")
        else:
            print(f"[feat ] 提示：--log-features {args.log_features!r} 没有匹配到任何列")

    # 训练集里剔除含 NaN 的行；测试集同样处理（用各自的掩码，保证统计口径诚实）
    train_ok = np.isfinite(X_train).all(axis=1)
    test_ok = np.isfinite(X_test).all(axis=1)
    dropped_train = int((~train_ok).sum())
    dropped_test = int((~test_ok).sum())
    X_train, y_train = X_train[train_ok], y_train[train_ok]
    X_test, y_test = X_test[test_ok], y_test[test_ok]
    print(f"[clean] 丢弃含 NaN/inf 的行：训练 {dropped_train}，测试 {dropped_test}")
    print(f"[data ] 训练 {X_train.shape}，测试 {X_test.shape}，"
          f"训练恶意占比 {y_train.mean() * 100:.2f}%，测试恶意占比 {y_test.mean() * 100:.2f}%")

    # 输入归一化。跨天失效的实测证据（pipeline_b2/b3）指向"特征携带域指纹"，
    # 所以这里提供三种口径做对照：
    #   global  训练集 mean/std —— 原实现，也是跨天实验里崩到 AUC 0.24 的配置
    #   robust  训练集 median/IQR —— 抗离群值，但统计量仍被训练域约束
    #   batch   不用训练集统计量，由模型内的 FlowInputNorm 按批内 median/MAD 归一化
    if args.norm_mode == 'global':
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        norm_note = 'global mean/std（训练集统计量）'
    elif args.norm_mode == 'robust':
        # 注意 dtype：np.median / np.percentile 会返回 float64，直接参与 float32 的
        # X_train 运算会把整块数据升成 double，进模型就报
        # "expected scalar type Double but found Float"，所以这里显式落回 float32。
        mean = np.median(X_train, axis=0, keepdims=True).astype(np.float32)
        q1 = np.percentile(X_train, 25, axis=0, keepdims=True).astype(np.float32)
        q3 = np.percentile(X_train, 75, axis=0, keepdims=True).astype(np.float32)
        iqr = q3 - q1
        iqr[iqr < 1e-6] = 1.0
        std = iqr
        X_train = ((X_train - mean) / std).astype(np.float32)
        X_test = ((X_test - mean) / std).astype(np.float32)
        norm_note = 'global median/IQR（训练集统计量，抗离群）'
    else:
        # batch 模式：数据侧不做任何缩放，交给模型内的 FlowInputNorm。
        # 这里保留占位统计量只是为了 ckpt 字段兼容（推理侧会读 norm_mode 判断）。
        mean = np.zeros((1, X_train.shape[1]), dtype=np.float32)
        std = np.ones((1, X_train.shape[1]), dtype=np.float32)
        norm_note = 'per-batch median/MAD（模型内计算，不使用训练集统计量）'
    print(f"[norm ] 归一化方式：{norm_note}")

    # 从训练集里划一小部分做验证（按行随机切不可怕，因为验证只用于早停）
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_train))
    n_val = max(1, int(len(X_train) * 0.1))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train[tr_idx], y_train[tr_idx], True)
    val_loader = make_loader(X_train[val_idx], y_train[val_idx], False)
    test_loader = make_loader(X_test, y_test, False)

    hidden = tuple(int(h) for h in args.hidden.split(',') if h.strip())
    base_model = FlowMLP(input_dim=len(feature_names), hidden_sizes=hidden, num_classes=2)
    if args.norm_mode == 'batch':
        # 把批内归一化前置进模型：训练与推理共用同一套统计逻辑，
        # 避免部署时外层预处理与训练时不一致而悄悄跑偏。
        model = nn.Sequential(FlowInputNorm(mode='batch'), base_model)
    else:
        model = base_model
    model = model.to(device)
    # 类别权重：CIC-IDS2018 的攻击流量占比很高（有的时段良性不到 1%），不加权模型会
    # 退化成"全判恶意"；但反过来加权过头会把模型推向"全判良性"（第一次跑就是这样，
    # 两次跨天实验的 TP 都是 0）。所以 --class-weight none 是必须留着的对照组。
    class_counts = np.bincount(y_train, minlength=2).astype(np.float64)
    class_pct = [round(c / max(class_counts.sum(), 1) * 100, 2) for c in class_counts.tolist()]
    print(f"[weight] 训练集类别计数 {class_counts.tolist()}（占比 {class_pct}%）")
    if args.class_weight == 'balanced':
        class_weights = len(y_train) / (2.0 * np.maximum(class_counts, 1.0))
        print(f"[weight] 启用反比加权 -> {[round(w, 4) for w in class_weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    else:
        print("[weight] 不加权（对照）")
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                total += yb.size(0)
        val_acc = correct / max(total, 1) * 100
        print(f"▶ Epoch [{epoch + 1}/{args.epochs}] 训练损失 {running / max(seen, 1):.4f} | "
              f"验证准确率 {val_acc:.2f}%", flush=True)

        if val_acc > best_acc:
            best_acc, best_epoch, no_improve = val_acc, epoch + 1, 0
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"⏹ 早停触发：{best_epoch} 轮后连续 {no_improve} 轮没有提升")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n最佳验证准确率 {best_acc:.2f}%（第 {best_epoch} 轮）")
    test_metrics = evaluate(model, test_loader, device, f"测试 {','.join(test_days)}")

    out = args.out or os.path.join(_PROJECT_ROOT, 'checkpoints', 'flow_mlp.pth')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': len(feature_names),
        'feature_names': feature_names,
        'hidden_sizes': list(hidden),
        'num_classes': 2,
        'class_names': ['Benign', 'Malware'],
        'normalization': {'mode': args.norm_mode,
                          'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()},
        'train_days': train_days,
        'test_days': test_days,
    }, out)
    print(f"权重已保存 -> {out}")

    metrics_out = args.metrics_out or os.path.join(_PROJECT_ROOT, 'logs', 'flow_model_metrics.json')
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, 'w', encoding='utf-8') as fh:
        json.dump({
            'train_days': train_days,
            'test_days': test_days,
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch,
            'test': test_metrics,
            'elapsed_sec': round(time.time() - t0, 1),
        }, fh, ensure_ascii=False, indent=2)
    print(f"指标已写入 -> {metrics_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
