import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path to avoid import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.data_loader import get_dataloaders
from src.training.train import train_model
from src.training.loss_utils import build_criterion, compute_class_weights
from src.models.mlp import MLP

def resolve_device(requested_device: str) -> torch.device:
    requested = requested_device.lower().strip()
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        print('警告: 当前环境没有可用 CUDA，已自动回退到 CPU。')
        return torch.device('cpu')
    if requested not in ('cpu', 'cuda'):
        print(f"警告: 未知 device='{requested_device}'，已使用 auto。")
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="训练 MLP (多层感知机) 模型")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_data_root = os.path.join(project_root, 'data', 'USTC-TFC2016-master')
    parser.add_argument('--data-root', type=str, default=default_data_root,
                        help="包含 Benign 和 Malware 文件夹的数据集根目录")
    parser.add_argument('--dataset', type=str, default=None, choices=['ustc', 'ics', 'mixed', 'combined'],
                        help="使用预定义数据集路径：ustc|ics|mixed|combined。若同时指定 --data-root，优先使用 --data-root。")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮数")
    parser.add_argument('--batch-size', type=int, default=32, help="批次大小")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率")
    parser.add_argument('--truncate-len', type=int, default=784, help="会话截断长度")
    parser.add_argument('--mask-headers', action='store_true', default=True, help="是否隐藏MAC/IP头部")
    parser.add_argument('--no-mask-headers', dest='mask_headers', action='store_false', help="关闭MAC/IP隐藏")
    parser.add_argument('--mask-fill', type=int, default=0, help="隐藏时使用的填充值(0-255)")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="训练设备")
    parser.add_argument('--malware-weight-multiplier', type=float, default=1.0,
                        help="对 Malware 类的权重乘子 (>1 提高对误判 Malware 的惩罚)")
    parser.add_argument('--loss-function', type=str, default='ce', choices=['ce', 'focal'],
                        help="训练损失函数：ce=CrossEntropyLoss, focal=FocalLoss")
    parser.add_argument('--focal-gamma', type=float, default=2.0, help="Focal Loss 的 gamma 参数")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    # If dataset not specified and default data_root, ask user to choose
    if args.dataset is None and os.path.abspath(args.data_root) == os.path.abspath(default_data_root):
        print("请选择训练数据集：\n 1) ustc (默认 USTC-TFC2016-master)\n 2) ics (ICS-Pcaps-master)\n 3) mixed (data/mixed_train_pcap)\n 4) combined (data/combined_train_pcap)")
        choice = input("输入编号并回车 (默认 1): ").strip()
        mapping = {'1': 'ustc', '2': 'ics', '3': 'mixed', '4': 'combined'}
        args.dataset = mapping.get(choice, 'ustc')
        print(f"已选择数据集: {args.dataset}")

    if args.dataset is not None:
        parent_root = os.path.dirname(project_root)
        mapped = {
            'ustc': os.path.join(project_root, 'data', 'USTC-TFC2016-master'),
            'ics': os.path.join(parent_root, 'ICS-Pcaps-master'),
            'mixed': os.path.join(project_root, 'data', 'mixed_train_pcap'),
            'combined': os.path.join(project_root, 'data', 'combined_train_pcap')
        }
        chosen = mapped.get(args.dataset)
        if os.path.abspath(args.data_root) == os.path.abspath(default_data_root):
            if chosen is not None:
                data_root = os.path.abspath(chosen)
        else:
            print(f"注意: 同时提供了 --dataset {args.dataset} 和自定义 --data-root，优先使用 --data-root: {args.data_root}")

    if not os.path.exists(data_root):
        print(f"错误: 找不到数据集目录 '{data_root}'。请确保它与此脚本在同一目录下。")
        return

    device = resolve_device(args.device)
    print(f"使用的训练设备: {device}")

    # ==========================
    # 步骤 1: 数据加载 (复用特征提取)
    # ==========================
    print(f"\n--- 步骤 1: 开始从目录 {data_root} 的 transform 文件夹加载 PCAP 并处理 ---")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_root,
        batch_size=args.batch_size,
        truncate_len=args.truncate_len,
        mask_headers=args.mask_headers,
        mask_fill=args.mask_fill,
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 在提供的数据集中找不到任何被转换成功的有效数据。")
        return

    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"\n数据集加载成功！类别: {class_names}, 总类别数量: {num_classes}")

    # 根据训练集标签分布自动计算类别权重
    train_labels = getattr(train_loader.dataset, 'labels', [])
    if len(train_labels) > 0:
        label_tensor = torch.tensor(train_labels, dtype=torch.long)
        label_counts = torch.bincount(label_tensor, minlength=num_classes).float()
        class_weights = compute_class_weights(
            label_counts=label_counts,
            class_names=class_names,
            malware_weight_multiplier=args.malware_weight_multiplier,
        )
        print(f"训练集类别样本数: {[int(v) for v in label_counts.tolist()]}")
        print(f"自动类别权重: {[round(float(v), 4) for v in class_weights.tolist()]}")
    else:
        class_weights = torch.ones(num_classes)
        print("警告: 无法读取训练集标签分布，类别权重将使用全1。")

    # ==========================
    # 步骤 2: 初始化模型
    # ==========================
    model_name = "MLP"
    print(f"\n=============================================")
    print(f"  正在训练模型: {model_name}")
    print(f"=============================================")
    
    model = MLP(num_classes=num_classes)
    
    save_path = os.path.join(project_root, "checkpoints", "mlp.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = build_criterion(args.loss_function, class_weights, device, focal_gamma=args.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    config = {
        'model': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'truncate_len': args.truncate_len,
        'mask_headers': args.mask_headers,
        'mask_fill': args.mask_fill,
        'learning_rate': args.lr,
        'class_names': class_names,
        'num_classes': num_classes,
        'loss_function': args.loss_function,
        'focal_gamma': args.focal_gamma,
    }

    # ==========================
    # 步骤 3: 训练模型
    # ==========================
    trained_model = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=args.epochs,
        device=str(device),
        config=config
    )

    # ==========================
    # 步骤 4: 保存模型权重
    # ==========================
    print(f"\n-> [{model_name}] 训练完成，正在保存模型权重至 {save_path} ...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names
    }, save_path)
    print(f"-> [{model_name}] 保存成功！\n")


if __name__ == "__main__":
    main()
