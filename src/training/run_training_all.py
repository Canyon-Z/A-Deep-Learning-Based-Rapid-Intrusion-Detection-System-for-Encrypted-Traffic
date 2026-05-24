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

from src.models.cnn_bilstm import CNN_BiLSTM
from src.models.classic_cnn import ClassicCNN
from src.models.lightweight_cnn_bilstm import Lightweight_CNN_BiLSTM
# 新增模型
from src.models.transformer import TrafficTransformer
from src.models.mlp import MLP
# 为了纯 BiLSTM 模型，可以直接导入或在这里简单定义一个 Wrapper
class PureBiLSTM(nn.Module):
    def __init__(self, input_size=28, hidden_dim=64, num_layers=2, num_classes=2):
        super(PureBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=True, 
                              dropout=0.5 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.5)
        # BiLSTM output will be concatenated (hidden_dim * 2)
        # Sequence length after flatten is image width (28)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 1, 28, 28] -> remove channel dim -> [batch, 28, 28]
        x = x.squeeze(1)
        # lstm returns: output, (h_n, c_n)
        output, _ = self.bilstm(x)
        # use the last sequence state output[:, -1, :] 
        last_state = output[:, -1, :]
        last_state = self.dropout(last_state)
        out = self.fc(last_state)
        return out


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
    parser = argparse.ArgumentParser(description="一键训练所有的加密流量入侵检测模型")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_data_root = os.path.join(project_root, 'data', 'USTC-TFC2016-master')
    default_checkpoint_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'checkpoints'
    )

    parser.add_argument('--data-root', type=str, default=default_data_root,
                        help="包含 Benign 和 Malware 文件夹的数据集根目录")
    parser.add_argument('--dataset', type=str, default=None, choices=['ustc', 'ics', 'mixed', 'combined'],
                        help="使用预定义数据集路径：ustc|ics|mixed|combined。若同时指定 --data-root，优先使用 --data-root。")
    parser.add_argument('--epochs', type=int, default=50, help="每个模型的训练轮数")
    parser.add_argument('--batch-size', type=int, default=32, help="批次大小")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'CNN_BiLSTM', 'Classic_CNN', 'Lightweight_CNN_BiLSTM', 'Pure_BiLSTM', 'Transformer'], help="指定要训练的模型名称，默认为全部")
    parser.add_argument('--cache-dir', type=str, default=None, help="特征缓存目录，默认使用 <data-root>/.feature_cache")
    parser.add_argument('--no-cache', action='store_true', help="禁用特征缓存（每次都从PCAP重新提取）")
    parser.add_argument('--rebuild-cache', action='store_true', help="重建缓存，忽略已有缓存文件")
    parser.add_argument('--cache-compress', action='store_true', help="使用 np.savez_compressed 压缩缓存，节省磁盘空间")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="训练设备")
    parser.add_argument('--malware-weight-multiplier', type=float, default=1.0,
                        help="对 Malware 类的权重乘子 (>1 提高对误判 Malware 的惩罚)")
    parser.add_argument('--loss-function', type=str, default='ce', choices=['ce', 'focal'],
                        help="训练损失函数：ce=CrossEntropyLoss, focal=FocalLoss")
    parser.add_argument('--focal-gamma', type=float, default=2.0, help="Focal Loss 的 gamma 参数")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    # 运行时交互：若未提供 --dataset 且 data_root 为默认值，则提示选择预定义数据集
    if args.dataset is None and os.path.abspath(args.data_root) == os.path.abspath(default_data_root):
        print("请选择训练数据集：\n 1) ustc (默认 USTC-TFC2016-master)\n 2) ics (ICS-Pcaps-master)\n 3) mixed (data/mixed_train_pcap)\n 4) combined (data/combined_train_pcap)")
        choice = input("输入编号并回车 (默认 1): ").strip()
        mapping = {'1': 'ustc', '2': 'ics', '3': 'mixed', '4': 'combined'}
        args.dataset = mapping.get(choice, 'ustc')
        print(f"已选择数据集: {args.dataset}")

    if args.dataset is not None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_cache,
        cache_compress=args.cache_compress
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 在提供的数据集中找不到任何被转换成功的有效数据。")
        return
    
    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"\n数据集加载成功！类别: {class_names}, 总类别数量: {num_classes}")

    # 根据训练集标签分布自动计算类别权重，缓解类别不平衡问题。
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
    # 步骤 2: 定义模型字典以便循环遍历
    # ==========================
    models_to_train = {
        "CNN_BiLSTM": {
            "model": CNN_BiLSTM(num_classes=num_classes, hidden_dim=64), # 降低网络宽度，结合代码内部的 BatchNorm 和 Dropout 提高泛化能力
            "save_path": os.path.join(default_checkpoint_root, "final_model.pth")
        },
        "Classic_CNN": {
            "model": ClassicCNN(num_classes=num_classes),
            "save_path": os.path.join(default_checkpoint_root, "classic_cnn.pth")
        },
        "Lightweight_CNN_BiLSTM": {
            "model": Lightweight_CNN_BiLSTM(num_classes=num_classes, hidden_dim=32), # 回归32，强化轻量化的特点，兼顾速度与低过拟合
            "save_path": os.path.join(default_checkpoint_root, "lightweight.pth")
        },
        "Pure_BiLSTM": {
            "model": PureBiLSTM(num_classes=num_classes, hidden_dim=64, num_layers=2), # 降低隐藏层维度和层数，防止过拟合
            "save_path": os.path.join(default_checkpoint_root, "pure_bilstm.pth")
        },
        "MLP": {
            "model": MLP(input_size=28*28, hidden_sizes=[512, 256, 128], num_classes=num_classes),
            "save_path": os.path.join(default_checkpoint_root, "mlp.pth")
        },
        "Transformer": {
            "model": TrafficTransformer(input_dim=28, d_model=64, nhead=4, num_layers=2, num_classes=num_classes, dropout=0.3), # 缩小模型深度和头数，加大 Dropout，抑制死记硬背
            "save_path": os.path.join(default_checkpoint_root, "transformer.pth")
        }
    }

    # 确保检查点保存文件夹存在
    os.makedirs(default_checkpoint_root, exist_ok=True)

    if args.model != 'all':
        # 仅保留用户指定的模型
        models_to_train = {args.model: models_to_train[args.model]}

    # ==========================
    # 步骤 3: 依次分开训练三个模型
    # ==========================
    for model_name, info in models_to_train.items():
        print(f"\n=============================================")
        print(f"  正在训练模型: {model_name}")
        print(f"=============================================")
        
        model = info["model"]
        save_path = info["save_path"]
        
        criterion = build_criterion(args.loss_function, class_weights, device, focal_gamma=args.focal_gamma)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 构建给 wandb 的 config (如果未开启会自动忽略)
        config = {
            'model': model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'class_names': class_names,
            'num_classes': num_classes,
            'loss_function': args.loss_function,
            'focal_gamma': args.focal_gamma,
        }

        # 调用封装好的训练函数，开始训练验证流程
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
        # 步骤 4: 分别保存各自的权重文件 .pth
        # ==========================
        print(f"\n-> [{model_name}] 训练完成，正在保存模型权重至 {save_path} ...")
        # 为兼容 web 端 `main.py` 的读取格式，我们打包一层带有结构信息的字典
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'num_classes': num_classes,
            'class_names': class_names
        }, save_path)
        print(f"-> [{model_name}] 保存成功！\n")
        
    print("\n🎉 所有六个模型均已成功分开训练并保存至 checkpoints 文件夹！")

if __name__ == "__main__":
    main()
