"""跨数据集（外部来源）评估脚本。

把 checkpoints 里训练好的模型拿到**不同来源**的数据集上跑一遍，看它们离开训练
分布之后还剩多少检出率。USTC-TFC2016 上 99.9% 的准确率只说明模型在同源划分上
拟合得好，不能代表真实网络里的泛化能力 —— 这正是之前 val 99.9% / test 87.8%
那种落差的来源。

用法：
    python tools/evaluate_external.py --data-root data/test_ics
    python tools/evaluate_external.py --data-root data/test_ustc --models CNN_BiLSTM,MLP
    python tools/evaluate_external.py --data-root data/test_ics --max-samples 20000

结果同时打印成表格，并写入 logs/external_eval_<数据集名>.json。
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
from src.preprocessing.feature_extraction import FeatureExtractor
from src.training.train import evaluate_model
from src.utils.device import resolve_device

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


def collect_sessions(data_root, extractor, max_samples=None, verbose=True):
    """把 data_root 下每个类别目录里的 pcap 全部转成 28x28 会话图。

    返回 (images, labels, classes)。labels 里的下标与 classes 的顺序对应，
    类别名按字母序排列 —— 与训练时 get_dataloaders 的口径一致（Benign=0, Malware=1）。
    """
    classes = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)) and not name.startswith('.')
    )
    images, labels = [], []
    started = time.time()
    reached_limit = False
    for class_idx, class_name in enumerate(classes):
        if reached_limit:
            break
        class_dir = os.path.join(data_root, class_name)
        pcaps = sorted(glob.glob(os.path.join(class_dir, '**', '*.pcap*'), recursive=True))
        if verbose:
            print(f"[collect] {class_name}: {len(pcaps)} 个 pcap")
        for pcap_file in pcaps:
            sessions, _timestamps, _skip = extractor.pcap_to_sessions(pcap_file)
            for _key, packet_list in sessions.items():
                image = extractor.process_session(packet_list)
                if image is None:
                    continue
                images.append(image)
                labels.append(class_idx)
                if max_samples and len(images) >= max_samples:
                    reached_limit = True
                    break
            if reached_limit:
                break
    if verbose:
        print(f"[collect] 会话样本 {len(images)} 个，耗时 {time.time() - started:.1f}s")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int64), classes


def build_model(model_name, num_classes):
    return MODEL_SPECS[model_name][1](num_classes)


def resolve_checkpoint(model_name, checkpoints_dir):
    for filename in MODEL_SPECS[model_name][0]:
        candidate = os.path.join(checkpoints_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_weights(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)


def format_table(rows):
    header = ('模型', '准确率', '检出率', '漏报率', '误报率', 'F1', 'AUC')
    widths = [24, 10, 10, 10, 10, 10, 9]
    lines = [''.join(h.ljust(w) for h, w in zip(header, widths)),
             '-' * sum(widths)]
    for row in rows:
        lines.append(''.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="在外部来源数据集上评估已训练模型")
    parser.add_argument('--data-root', type=str, required=True,
                        help="外部数据集根目录（下面有 Benign / Malware 子目录）")
    parser.add_argument('--models', type=str, default='all',
                        help="要评估的模型，逗号分隔；all 表示全部（默认 all）")
    parser.add_argument('--checkpoints-dir', type=str, default=None,
                        help="权重目录，默认 <项目根>/checkpoints")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-samples', type=int, default=None,
                        help="最多评估多少个会话（调试用）")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', type=str, default=None,
                        help="结果 JSON 路径，默认 logs/external_eval_<数据集名>.json")
    args = parser.parse_args()

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(_PROJECT_ROOT, args.data_root)
    if not os.path.isdir(data_root):
        print(f"数据集目录不存在：{data_root}")
        return 1

    checkpoints_dir = args.checkpoints_dir or os.path.join(_PROJECT_ROOT, 'checkpoints')
    dataset_tag = os.path.basename(os.path.normpath(data_root))

    if args.models.strip().lower() == 'all':
        selected = list(MODEL_SPECS.keys())
    else:
        selected = [name.strip() for name in args.models.split(',') if name.strip()]
        unknown = [name for name in selected if name not in MODEL_SPECS]
        if unknown:
            print(f"未知模型：{unknown}，可选：{list(MODEL_SPECS.keys())}")
            return 1

    device = resolve_device(args.device)
    print(f"评估数据集：{data_root}")
    print(f"评估设备：{device}")

    extractor = FeatureExtractor(mask_headers=True, mask_fill=0, max_packets=28, bytes_per_packet=28)
    images, labels, classes = collect_sessions(data_root, extractor, max_samples=args.max_samples)
    if len(images) == 0:
        print("没有采集到任何会话样本，检查 pcap 目录结构。")
        return 1

    counts = {name: int((labels == idx).sum()) for idx, name in enumerate(classes)}
    print(f"类别分布：{counts}")

    dataset = TrafficDataset(images, labels.tolist(), transform=transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    rows = []
    results = {}
    for model_name in selected:
        checkpoint = resolve_checkpoint(model_name, checkpoints_dir)
        if checkpoint is None:
            print(f"[skip] {model_name}: 没找到权重文件 {MODEL_SPECS[model_name][0]}")
            continue
        try:
            model = build_model(model_name, len(classes))
            load_weights(model, checkpoint)
        except Exception as error:
            print(f"[skip] {model_name}: 权重与结构不匹配（{error}）")
            continue

        model.to(device)
        metrics = evaluate_model(model, loader, criterion, device, classes)
        if metrics is None:
            print(f"[skip] {model_name}: 评估未返回指标")
            continue

        results[model_name] = {
            'checkpoint': os.path.relpath(checkpoint, _PROJECT_ROOT).replace('\\', '/'),
            'accuracy': round(float(metrics['accuracy']), 4),
            'precision': round(float(metrics['precision']), 4),
            'recall': round(float(metrics['recall']), 4),
            'f1_score': round(float(metrics['f1_score']), 4),
            'detection_rate': round(float(metrics['detection_rate']), 4),
            'fnr': round(float(metrics['fnr']), 4),
            'fpr': round(float(metrics['fpr']), 4),
            'auc': round(float(metrics['auc']), 4) if metrics.get('auc') is not None else None,
            'recall_malware': round(float(metrics['recall_malware']), 4),
            'recall_benign': round(float(metrics['recall_benign']), 4),
            'sample_count': int(metrics['sample_count']),
        }
        rows.append((
            model_name,
            f"{metrics['accuracy']:.2f}%",
            f"{metrics['detection_rate']:.2f}%",
            f"{metrics['fnr']:.2f}%",
            f"{metrics['fpr']:.2f}%",
            f"{metrics['f1_score']:.2f}%",
            f"{metrics['auc']:.4f}" if metrics.get('auc') is not None else 'N/A',
        ))
        print(f"[done] {model_name} -> 准确率 {metrics['accuracy']:.2f}% / 检出率 {metrics['detection_rate']:.2f}%")

    if not rows:
        print("没有任何模型完成评估。")
        return 1

    print()
    print(f"===== {dataset_tag} 外部评估结果（{len(images)} 个会话）=====")
    print(format_table(rows))

    output_path = args.output or os.path.join(_PROJECT_ROOT, 'logs', f'external_eval_{dataset_tag}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        'dataset': dataset_tag,
        'data_root': os.path.relpath(data_root, _PROJECT_ROOT).replace('\\', '/'),
        'classes': classes,
        'class_counts': counts,
        'total_sessions': int(len(images)),
        'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': results,
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n结果已写入 {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
