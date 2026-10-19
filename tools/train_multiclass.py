"""多分类 + 置信度校准 + 高风险分流（CSE-CIC-IDS2018 流特征）。

与 train_flow_model.py 的区别：
  - 那个做「良性 vs 恶意」二分类，并按**天**切分来测跨域
  - 这个做**多分类**（Benign + 各种具体攻击类型），并按**时间顺序随机切分**

为什么不按天切分：实测标签与日期死绑（HOIC/LOIC-UDP 只在 21-02，GoldenEye/Slowloris
只在 15-02，Bot 只在 02-03）。跨天多分类要求模型认识测试天里根本不存在的攻击类型，
数学上不成立，任何「跨天多分类 90%」都不可信。多分类的正确评估口径是**同分布留出**。

三个产出：
  1. per-class 精确率/召回率/F1 —— 回答「能不能把攻击类型分开」
  2. 温度缩放校准 + 可靠性曲线 —— 回答「模型说的 0.9 到底是不是 90% 把握」
  3. 三档分流统计 —— 回答「高风险要人工复核多少条」

用法：
    python -u tools/train_multiclass.py --days 15-02,21-02,02-03
    python -u tools/train_multiclass.py --days 15-02,21-02,02-03 --epochs 25
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.flow_mlp import FlowMLP
from src.utils.device import resolve_device
from tools.train_flow_model import align_columns, load_day

# 官方 Label 列里的大小写/前后空格不统一，统一归一化到 6 个类别。
# 'Label' / 'nan' 是 CSV 表头重复与空行产生的垃圾标签，直接丢弃。
LABEL_ALIASES = {
    'BENIGN': 'Benign',
    'BOT': 'Bot',
    'DDOS ATTACK-HOIC': 'DDoS-HOIC',
    'DDOS ATTACK-LOIC-UDP': 'DDoS-LOIC-UDP',
    'DOS ATTACKS-GOLDENEYE': 'DoS-GoldenEye',
    'DOS ATTACKS-SLOWLORIS': 'DoS-Slowloris',
}
CLASS_NAMES = ['Benign', 'Bot', 'DDoS-HOIC', 'DoS-GoldenEye', 'DoS-Slowloris', 'DDoS-LOIC-UDP']
MIN_CLASS_SAMPLES = 500  # 低于这个数的类不参与训练，避免单类欠拟合拖垮整体


def load_multiclass(csv_dir, days, max_rows=None, with_meta=False):
    """按天读 CSV，返回 (X, y, class_names, 丢弃统计)。

    ``with_meta=True`` 时额外返回 ``meta``（包含 ``timestamp`` 与 ``day`` 两个数组），
    行序与清洗后的 X/y 完全一致，供时间块切分使用。
    """
    frames, labels_raw, metas = [], [], []
    for day in days:
        if with_meta:
            df, lab, meta = load_day(csv_dir, day, max_rows, with_meta=True)
            metas.append(meta)
        else:
            df, lab = load_day(csv_dir, day, max_rows)
        frames.append(df)
        labels_raw.append(lab)
    frames, feature_names = align_columns(frames)

    X = np.vstack([df.to_numpy(dtype=np.float32) for df in frames])
    raw = np.concatenate([np.asarray(l, dtype=object) for l in labels_raw])
    if with_meta:
        import pandas as pd
        meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    else:
        meta = None

    norm = np.array([str(v).strip().strip('\ufeff').upper() for v in raw], dtype=object)
    mapped = np.array([LABEL_ALIASES.get(v, None) for v in norm], dtype=object)
    keep = np.array([m is not None for m in mapped])
    dropped = int((~keep).sum())
    if dropped:
        print(f"[clean] 丢弃无法映射的标签行 {dropped}（如表头重复 'Label'/空值）")
    X, mapped = X[keep], mapped[keep]
    if with_meta and meta is not None:
        meta = meta.loc[keep].reset_index(drop=True)

    ok = np.isfinite(X).all(axis=1)
    if (~ok).any():
        print(f"[clean] 丢弃含 NaN/inf 的行 {int((~ok).sum())}")
    X, mapped = X[ok], mapped[ok]
    if with_meta and meta is not None:
        meta = meta.loc[ok].reset_index(drop=True)

    # 类别太少的不参与训练（LOIC-UDP 只有 1730 条，合并/剔除都会扭曲结论，这里剔除并记录）
    counts = {c: int((mapped == c).sum()) for c in CLASS_NAMES}
    usable = [c for c in CLASS_NAMES if counts.get(c, 0) >= MIN_CLASS_SAMPLES]
    skipped = {c: counts.get(c, 0) for c in CLASS_NAMES if c not in usable}
    if skipped:
        print(f"[clean] 样本不足 {MIN_CLASS_SAMPLES} 的类不参与训练：{skipped}")
    sel = np.array([m in usable for m in mapped])
    X, mapped = X[sel], mapped[sel]
    if with_meta and meta is not None:
        meta = meta.loc[sel].reset_index(drop=True)

    class_names = usable
    idx_map = {c: i for i, c in enumerate(class_names)}
    y = np.array([idx_map[m] for m in mapped], dtype=np.int64)

    print("[data ] 类别分布：")
    for c in class_names:
        n = int((y == idx_map[c]).sum())
        print(f"    {c:<18}{n:>9}  ({n / len(y) * 100:5.2f}%)")
    if with_meta:
        return X, y, class_names, feature_names, skipped, meta
    return X, y, class_names, feature_names, skipped


def block_split(y, meta, test_size=0.2, val_size=0.15, block_seconds=60.0,
                block_rows=2000, seed=42):
    """类内时间块切分：同一个时间块内的流必须落在同一侧。

    **为什么不能按行随机切分（缺陷 D-08）**：`train_test_split(stratify=y)` 会把
    同一次攻击里相邻的流随机分到训练集和测试集两侧。这些流来自同一条 pcap、
    同一个五元组、时间上只差几十毫秒，特征几乎完全相同 —— 模型在测试集上
    认出来的其实是「训练集里的孪生样本」，于是 test_accuracy 冲到 99.94%，
    而真实跨环境泛化接近 0（见 logs/pipeline_b*_summary.txt）。

    这里的做法：
      1. 按类分组（类别之间天然分属不同攻击时段，跨类不必共享块）；
      2. 类内按时间戳排序，每 ``block_seconds`` 秒（时间戳缺失时退回每
         ``block_rows`` 行）切成一个块；
      3. 以**块**为单位随机分配给 train / val / test，保证每类三侧都非空。

    这样同一次攻击的连续流不会横跨切分边界，测试集的数字才是「没见过的样本」上的数字。
    """
    import pandas as pd

    n = len(y)
    assign = np.empty(n, dtype=object)
    ts = None
    if meta is not None and 'timestamp' in getattr(meta, 'columns', []):
        ts = pd.to_datetime(meta['timestamp'], errors='coerce').to_numpy()
    rng = np.random.RandomState(seed)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        cls_ts = ts[idx] if ts is not None else None
        if cls_ts is not None and not np.isnat(cls_ts).all():
            order = idx[np.argsort(np.where(pd.isna(cls_ts), pd.NaT, cls_ts), kind='stable')]
            raw_i = pd.to_datetime(pd.Series(ts[order])).astype('int64').to_numpy()
            nat_mask = raw_i == np.iinfo(np.int64).min   # pandas 的 NaT 哨兵值
            secs = raw_i.astype(float) / 1e9
            finite = ~nat_mask
            if finite.any():
                base = secs[finite][0]
                blk = np.where(finite, np.floor((secs - base) / max(block_seconds, 1e-6)), -1)
            else:
                order, blk = idx, np.arange(len(idx)) // max(int(block_rows), 1)
        else:
            order = idx
            blk = np.arange(len(order)) // max(int(block_rows), 1)

        uniq = np.unique(blk)
        n_blocks = len(uniq)
        n_test = max(1, int(round(n_blocks * test_size)))
        n_val = max(1, int(round(n_blocks * val_size)))
        # 块数太少时保住至少一个训练块，否则该类无法学习
        while n_test + n_val >= n_blocks and (n_test + n_val) > 0:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        if n_blocks == 1:
            n_test, n_val = 0, 0

        shuffled = uniq[rng.permutation(n_blocks)]
        test_blocks = set(shuffled[:n_test].tolist())
        val_blocks = set(shuffled[n_test:n_test + n_val].tolist())
        for pos, b in enumerate(blk):
            row = order[pos]
            assign[row] = 'test' if b in test_blocks else ('val' if b in val_blocks else 'train')

    unassigned = pd.isna(assign)
    if unassigned.any():
        assign[unassigned] = 'train'
    return assign



def fit_temperature(logits, y, device):
    """在验证集上拟合单一温度参数 T，让 softmax 输出接近真实正确率。

    神经网络普遍过度自信（说 0.99 时实际可能只有 0.85），温度缩放是最轻量、
    最不容易过拟合的校准方法：只学一个标量，且不改变 argmax（准确率不变）。
    """
    logits = torch.as_tensor(logits, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    T = torch.ones(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def reliability_table(prob, y, bins=10):
    """可靠性曲线：每档置信区间的实际准确率。校准好的模型，两列应该接近。

    缺陷 D-20：过去空箱直接 `continue` 跳过，于是校准前（6 箱，首箱 0.4-0.5）与
    校准后（7 箱，首箱 0.3-0.4）得到两个**箱集合不同**的表，并排画曲线时同名箱
    其实对应不同区间，看起来像校准把低置信样本「变多了」。这里固定输出全部箱，
    空箱保留 `count=0` 且数值为 None，两表即可逐行对齐。
    """
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        n = int(mask.sum())
        rows.append({
            'bin': f'{lo:.1f}-{hi:.1f}',
            'count': n,
            'avg_confidence': round(float(conf[mask].mean()), 4) if n else None,
            'actual_accuracy': round(float(correct[mask].mean()), 4) if n else None,
            'empty': n == 0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="多分类 + 置信度校准 + 高风险分流")
    parser.add_argument('--csv-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'cse_csv'))
    parser.add_argument('--days', type=str, required=True, help="参与的数据天，逗号分隔")
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=str, default='256,128')
    parser.add_argument('--test-size', type=float, default=0.2)
    # 缺陷 D-08：默认改为 block（类内时间块切分）。random 仍保留，但会明确警告
    # 它会产生数据泄漏 —— 之前 99.94% 的 test_accuracy 就是这个口径下的产物。
    parser.add_argument('--split', type=str, default='block',
                        choices=['block', 'random'],
                        help='切分方式：block=类内时间块（无泄漏，默认）；'
                             'random=按行随机分层（有泄漏，仅用于复现旧数字）')
    parser.add_argument('--block-seconds', type=float, default=60.0,
                        help='时间块跨度（秒），只有落在同一块内的流才共享 train/test 归属')
    parser.add_argument('--block-rows', type=int, default=2000,
                        help='时间戳缺失时改用「每 N 行一块」')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--metrics-out', type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    # Windows 控制台默认 GBK，日志里那几行带 '▶' / '⏹' / '⚠' 的进度输出会把
    # print 打成 UnicodeEncodeError 直接中断训练（重定向到文件时同样触发）。
    # 只影响本进程的终端编码，不改任何产物内容。
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001 — 老环境没有该接口时保持原样
            pass
    days = [d.strip() for d in args.days.split(',') if d.strip()]
    print(f"设备：{device}")
    if args.split == 'block':
        print(f"数据天：{days}（类内时间块切分：{args.block_seconds:g}s 一块，"
              f"同块流不跨 train/test）\n")
    else:
        print(f"数据天：{days}（⚠ 按行随机切分：同一条 pcap 的流会同时进训练集与测试集，"
              f"指标存在泄漏，不可作为泛化能力引用）\n")

    t0 = time.time()
    X, y, class_names, feature_names, skipped, meta = load_multiclass(
        args.csv_dir, days, args.max_rows, with_meta=(args.split == 'block'))

    if args.split == 'block':
        assign = block_split(y, meta, test_size=args.test_size, val_size=0.15,
                             block_seconds=args.block_seconds,
                             block_rows=args.block_rows, seed=42)
        tr = assign == 'train'
        va = assign == 'val'
        te = assign == 'test'
        X_train, y_train = X[tr], y[tr]
        X_val, y_val = X[va], y[va]
        X_test, y_test = X[te], y[te]
        # 每类必须三侧都有样本，否则 per-class 指标（尤其 recall）无从计算
        missing = [c for i, c in enumerate(class_names)
                   if (y_test == i).sum() == 0 or (y_train == i).sum() == 0]
        if missing:
            raise SystemExit(
                f"[split] 这些类在训练集或测试集里为空：{missing}。"
                f"调小 --block-seconds / --block-rows 让块更多，或补更多数据。")
        print(f"[split] 时间块切分：训练 {X_train.shape} / 验证 {X_val.shape} / 测试 {X_test.shape}")
    else:
        # 分层切分，保证每个类在训练/验证/测试里都有
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.15, stratify=y_tmp, random_state=42)
        print(f"[split] 按行随机切分（有泄漏）：训练 {X_train.shape} / 验证 {X_val.shape} / "
              f"测试 {X_test.shape}")

    # 标准化用训练集统计量
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    def loader(Xa, ya, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = loader(X_train, y_train, True)
    val_loader = loader(X_val, y_val)
    test_loader = loader(X_test, y_test)

    hidden = tuple(int(h) for h in args.hidden.split(',') if h.strip())
    model = FlowMLP(input_dim=len(feature_names), hidden_sizes=hidden,
                    num_classes=len(class_names)).to(device)

    # 类别不平衡：按频次反比加权（HOIC 与 LOIC 的样本量差 400 倍）
    counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float64)
    weights = len(y_train) / (len(class_names) * np.maximum(counts, 1.0))
    print(f"[weight] 类别权重 {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
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
                correct += int((model(xb).argmax(dim=1) == yb).sum())
                total += yb.size(0)
        val_acc = correct / max(total, 1) * 100
        print(f"▶ Epoch [{epoch + 1}/{args.epochs}] 训练损失 {running / max(seen, 1):.4f} | "
              f"验证准确率 {val_acc:.2f}%", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch, no_improve = val_acc, epoch + 1, 0
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"⏹ 早停触发：{best_epoch} 轮后连续 {no_improve} 轮无提升")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n最佳验证准确率 {best_acc:.2f}%（第 {best_epoch} 轮）")

    def collect(loader_):
        model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in loader_:
                logits_all.append(model(xb.to(device)).cpu().numpy())
                y_all.append(yb.numpy())
        return np.concatenate(logits_all), np.concatenate(y_all)

    val_logits, val_y = collect(val_loader)
    test_logits, test_y = collect(test_loader)

    # ---- 1. 多分类指标 ----
    def softmax_np(z, temperature=1.0):
        z = z / temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    pred_raw = val_logits.argmax(axis=1)
    pred_test = test_logits.argmax(axis=1)
    report = classification_report(test_y, pred_test, target_names=class_names,
                                   digits=4, zero_division=0)
    print("\n=== 测试集 per-class 指标（未校准，argmax 不受校准影响）===")
    print(report)

    cm = confusion_matrix(test_y, pred_test, labels=list(range(len(class_names))))
    print("=== 混淆矩阵（行=真实，列=预测）===")
    header = '            ' + ''.join(f'{c[:9]:>11}' for c in class_names)
    print(header)
    for i, c in enumerate(class_names):
        print(f'{c[:11]:<12}' + ''.join(f'{cm[i, j]:>11}' for j in range(len(class_names))))

    # ---- 2. 温度校准 ----
    T = fit_temperature(val_logits, val_y, device)
    print(f"\n[calib] 拟合温度 T = {T:.4f}"
          f"（T>1 表示原模型过度自信，需要压平概率）")
    prob_before = softmax_np(test_logits, 1.0)
    prob_after = softmax_np(test_logits, T)
    nll = nn.CrossEntropyLoss()
    nll_before = float(nll(torch.from_numpy(test_logits), torch.from_numpy(test_y)))
    nll_after = float(nll(torch.from_numpy(test_logits / T), torch.from_numpy(test_y)))
    print(f"[calib] 测试集 NLL：校准前 {nll_before:.4f} -> 校准后 {nll_after:.4f}")
    print(f"[calib] 测试集准确率（校准不改变 argmax）：{float((pred_test == test_y).mean()) * 100:.2f}%")

    rel_before = reliability_table(prob_before, test_y)
    rel_after = reliability_table(prob_after, test_y)

    def _rel_cell(value):
        # 固定 10 箱之后空箱的数值是 None，直接格式化会抛 TypeError
        # （"'NoneType' object has no attribute '__format__'"，训练在写 JSON 之前就中断）。
        # 用 '-' 占位，与三档分流表里空档位的写法一致。
        return f'{value:>12.4f}' if value is not None else f'{"-":>12}'

    print(f"\n=== 可靠性曲线（校准前）===")
    print(f"{'置信区间':<12}{'样本数':>10}{'平均置信':>12}{'实际准确率':>12}")
    for r in rel_before:
        print(f"{r['bin']:<12}{r['count']:>10}"
              f"{_rel_cell(r['avg_confidence'])}{_rel_cell(r['actual_accuracy'])}")
    print(f"\n=== 可靠性曲线（校准后）===")
    print(f"{'置信区间':<12}{'样本数':>10}{'平均置信':>12}{'实际准确率':>12}")
    for r in rel_after:
        print(f"{r['bin']:<12}{r['count']:>10}"
              f"{_rel_cell(r['avg_confidence'])}{_rel_cell(r['actual_accuracy'])}")

    # ---- 3. 三档分流 ----
    conf = prob_after.max(axis=1)
    tiers = {
        '自动处置（置信 >= 0.90）': conf >= 0.90,
        '告警待确认（0.50 ~ 0.90）': (conf >= 0.50) & (conf < 0.90),
        '人工复核（置信 < 0.50）': conf < 0.50,
    }
    print(f"\n=== 三档分流（校准后置信度，测试集 {len(conf)} 条）===")
    print(f"{'档位':<28}{'样本数':>10}{'占比':>9}{'其中判对':>11}")
    tier_rows = []
    for name, mask in tiers.items():
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<28}{0:>10}{0:>8.2f}%{'-':>11}")
            continue
        acc = float((pred_test[mask] == test_y[mask]).mean()) * 100
        print(f"{name:<28}{n:>10}{n / len(conf) * 100:>8.2f}%{acc:>10.2f}%")
        tier_rows.append({'tier': name, 'count': n,
                          'ratio': round(n / len(conf) * 100, 2),
                          'accuracy': round(acc, 2)})

    out = args.out or os.path.join(_PROJECT_ROOT, 'checkpoints', 'flow_multiclass.pth')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': len(feature_names),
        'feature_names': feature_names,
        'hidden_sizes': list(hidden),
        'num_classes': len(class_names),
        'class_names': class_names,
        'normalization': {'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()},
        'temperature': T,
        # 让推理端能看出这份权重的校准/归一化属于哪个口径
        'split_strategy': 'block' if args.split == 'block' else 'random',
        'normalization_scope': 'train_split',
        'days': days,
    }, out)
    print(f"\n权重已保存 -> {out}")

    metrics_out = args.metrics_out or os.path.join(
        _PROJECT_ROOT, 'logs', 'flow_multiclass_metrics.json')
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, 'w', encoding='utf-8') as fh:
        json.dump({
            'days': days,
            # 缺陷 D-08：把切分口径写进产物，前端/报告才能判断这组数字是否含泄漏。
            'split_strategy': 'block' if args.split == 'block' else 'random',
            'leakage_controlled': args.split == 'block',
            'block_seconds': args.block_seconds if args.split == 'block' else None,
            'block_rows': args.block_rows if args.split == 'block' else None,
            'class_names': class_names,
            'skipped_classes': skipped,
            'train_samples': int(len(y_train)),
            'val_samples': int(len(y_val)),
            'test_samples': int(len(y_test)),
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch,
            'test_accuracy': round(float((pred_test == test_y).mean()) * 100, 2),
            'test_macro_f1': round(float(f1_score(test_y, pred_test, average='macro',
                                                 zero_division=0)) * 100, 2),
            'confusion_matrix': cm.tolist(),
            'temperature': T,
            'nll_before': nll_before,
            'nll_after': nll_after,
            'reliability_before': rel_before,
            'reliability_after': rel_after,
            'tiers': tier_rows,
            'elapsed_sec': round(time.time() - t0, 1),
        }, fh, ensure_ascii=False, indent=2)
    print(f"指标已写入 -> {metrics_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
