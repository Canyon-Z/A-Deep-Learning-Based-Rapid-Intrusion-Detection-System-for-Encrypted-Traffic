"""在现成的 28x28 PNG 数据集上交叉验证已训练模型。

与 evaluate_external.py 的分工：
  - evaluate_external.py  从 pcap 现场提取会话（特征表示与训练一致，只换数据来源）
  - 本脚本                直接读 28x28 PNG（数据来源与特征表示都换了）

项目里 data/processed/CSE-CIC-IDS2018 就是后一种：它是 CSE-CIC-IDS2018 的流量经
pcap2png.py 切出的图，和训练用的 FeatureExtractor 会话矩阵不是同一套表示。所以
这里的结果应当读作「换一套来源与表示之后，模型还剩多少判别力」，不等于纯数据域迁移。

它存在的理由：data/test_ustc 与训练集是同源副本（逐文件 SHA-256 完全一致），
data/test_ics 只有 4 个恶意会话，两者都无法用来判断真实的跨域检出能力。
CSE-CIC-IDS2018 是项目里唯一没有被训练管道碰过的第三方数据。

用法：
    python tools/evaluate_external_images.py --data-root data/processed/CSE-CIC-IDS2018
    python tools/evaluate_external_images.py --data-root <dir> --models CNN_BiLSTM,MLP
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.classic_cnn import ClassicCNN
from src.models.cnn_bilstm import CNN_BiLSTM
from src.models.lightweight_cnn_bilstm import Lightweight_CNN_BiLSTM
from src.models.mlp import MLP
from src.models.pure_bilstm import PureBiLSTM
from src.models.transformer import TrafficTransformer
from src.preprocessing.data_loader import TrafficDataset
from src.training.train import evaluate_model
from src.utils.device import resolve_device

MODEL_ORDER = ['CNN_BiLSTM', 'Classic_CNN', 'Lightweight_CNN_BiLSTM',
               'Pure_BiLSTM', 'MLP', 'Transformer']

# 模型名 -> (候选权重文件名（按优先级）, 构造函数)
MODEL_SPECS = {
    'CNN_BiLSTM': (
        ['final_model.pth', 'cnn_bilstm.pth'],
        lambda nc: CNN_BiLSTM(num_classes=nc, hidden_dim=64),
    ),
    'Classic_CNN': (
        ['classic_cnn.pth'],
        lambda nc: ClassicCNN(num_classes=nc),
    ),
    'Lightweight_CNN_BiLSTM': (
        ['lightweight.pth'],
        lambda nc: Lightweight_CNN_BiLSTM(num_classes=nc, hidden_dim=32),
    ),
    'Pure_BiLSTM': (
        ['pure_bilstm.pth'],
        lambda nc: PureBiLSTM(num_classes=nc, hidden_dim=64, num_layers=2),
    ),
    'MLP': (
        ['mlp.pth'],
        lambda nc: MLP(input_size=28 * 28, hidden_sizes=[512, 256, 128], num_classes=nc),
    ),
    'Transformer': (
        ['transformer.pth'],
        lambda nc: TrafficTransformer(input_dim=28, d_model=64, nhead=4, num_layers=2,
                                      num_classes=nc, dropout=0.3),
    ),
}


def collect_images(data_root, max_samples=None, verbose=True):
    """按子目录名收集 28x28 PNG，类别顺序按字母序（与训练口径一致：Benign=0, Malware=1）。"""
    classes = sorted(d for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.'))
    paths, labels = [], []
    for idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_root, cls_name)
        files = sorted(glob.glob(os.path.join(cls_dir, '**', '*.png'), recursive=True))
        # 各目录下的文件名前缀是恶意家族名（Bot_/Zeus_...），只用于展示
        families = sorted({os.path.basename(f).split('_')[0] for f in files})
        if verbose:
            print(f"[collect] {cls_name}: {len(files)} 张 png，家族 {families[:8]}"
                  f"{' ...' if len(families) > 8 else ''}")
        if max_samples is not None:
            per_class_cap = max(1, max_samples // max(1, len(classes)))
            files = files[:per_class_cap]
        paths.extend(files)
        labels.extend([idx] * len(files))

    if max_samples is not None and len(paths) > max_samples:
        order = np.random.RandomState(42).permutation(len(paths))[:max_samples]
        paths = [paths[i] for i in order]
        labels = [labels[i] for i in order]
    return paths, labels, classes


def load_state_dict(checkpoints_dir, candidates):
    """按优先级挑一个能干净加载的权重（missing/unexpected 最少）。"""
    probe = None
    best = None
    for name in candidates:
        path = os.path.join(checkpoints_dir, name)
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        if probe is None:
            probe = state
            best = (path, state, 0)
            continue
        score = 0
        best_score = best[2]
        if score < best_score:
            best = (path, state, score)
    return best


def main():
    parser = argparse.ArgumentParser(description="在 28x28 PNG 数据集上交叉验证已训练模型")
    parser.add_argument('--data-root', type=str, required=True,
                        help="图像数据集根目录（下面有 Benign / Malware 子目录）")
    parser.add_argument('--models', type=str, default='all',
                        help="要评估的模型，逗号分隔；all 表示全部")
    parser.add_argument('--checkpoints-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'checkpoints'))
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-samples', type=int, default=None, help="最多评估多少张图（调试用）")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', type=str, default=None,
                        help="结果 JSON 路径，默认 logs/external_eval_img_<数据集名>.json")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        print(f"错误：目录不存在 {data_root}")
        return 1

    device = resolve_device(args.device)
    print(f"评估数据集：{data_root}")
    print(f"评估设备：{device}")

    paths, labels, classes = collect_images(data_root, args.max_samples)
    if not paths:
        print("错误：没有找到任何 png 图片")
        return 1
    print(f"[collect] 图像样本 {len(paths)} 张")

    # 类别顺序必须与训练一致（Benign=0, Malware=1）；顺序不对则直接报错而不是悄悄算错
    if classes != ['Benign', 'Malware']:
        print(f"错误：类别目录为 {classes}，期望 ['Benign', 'Malware']（下标顺序必须与训练一致）")
        return 1
    counts = {c: int(sum(1 for l in labels if l == i)) for i, c in enumerate(classes)}
    print(f"类别分布：{counts}")

    dataset = TrafficDataset(paths, labels, transform=transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    if args.models.strip().lower() == 'all':
        model_names = MODEL_ORDER
    else:
        model_names = [m.strip() for m in args.models.split(',') if m.strip()]

    criterion = nn.CrossEntropyLoss()
    class_names = ['Benign', 'Malware']
    results = {}

    for name in model_names:
        if name not in MODEL_SPECS:
            print(f"[skip] 未知模型 {name}")
            continue
        candidates, builder = MODEL_SPECS[name]
        loaded = load_state_dict(args.checkpoints_dir, candidates)
        if loaded is None:
            print(f"[skip] {name}: 在 {args.checkpoints_dir} 里找不到 {candidates}")
            continue
        ckpt_path, state_dict, _ = loaded
        model = builder(len(class_names))
        model.load_state_dict(state_dict)
        model.to(device)

        metrics = evaluate_model(model, loader, criterion, device, class_names)
        if metrics is None:
            print(f"[skip] {name}: 评估返回空")
            continue
        results[name] = {
            'checkpoint': os.path.basename(ckpt_path),
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'recall_malware': metrics.get('recall_malware'),
            'recall_benign': metrics.get('recall_benign'),
            'fpr': metrics.get('fpr'),
            'fnr': metrics.get('fnr'),
            'auc': metrics.get('auc'),
        }
        print(f"[done] {name} -> 准确率 {metrics.get('accuracy', 0):.2f}% / "
              f"恶意检出 {metrics.get('recall_malware', 0):.2f}% / "
              f"误报 {metrics.get('fpr', 0):.2f}%")

    if not results:
        print("没有成功评估任何模型")
        return 1

    header = f"{'模型':<24}{'准确率':>9}{'精确率':>9}{'召回率':>9}{'恶意检出':>10}{'误报率':>9}{'F1':>9}{'AUC':>9}"
    line = '-' * (len(header) + 8)
    print(f"\n===== {os.path.basename(data_root)} 交叉验证结果（{len(paths)} 张图）=====")
    print(header)
    print(line)
    for name in model_names:
        m = results.get(name)
        if not m:
            continue
        auc = m.get('auc')
        auc_txt = f"{auc:.4f}" if isinstance(auc, (int, float)) else '-'
        print(f"{name:<24}{m.get('accuracy', 0):>8.2f}%{m.get('precision', 0):>8.2f}%"
              f"{m.get('recall', 0):>8.2f}%{m.get('recall_malware', 0):>9.2f}%"
              f"{m.get('fpr', 0):>8.2f}%{m.get('f1_score', 0):>8.2f}%{auc_txt:>9}")

    output = args.output or os.path.join(
        _PROJECT_ROOT, 'logs', f"external_eval_img_{os.path.basename(data_root)}.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            'data_root': data_root,
            'sample_count': len(paths),
            'class_counts': counts,
            'models': results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
