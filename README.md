# 基于深度学习的加密流量快速入侵检测系统

## 项目简介

本系统将原始 PCAP 流量文件转为 28×28 灰度图，利用多种深度学习模型进行二分类（Benign / Malware），
并通过 Web 界面提供文件上传分析和实时抓包检测功能。后端使用 6 个模型的多数投票集成策略做最终判决。

## 核心思路

1. **PCAP → 会话拆分**：按五元组（源/目的 IP、源/目的端口、协议）将混杂的 PCAP 拆成独立会话。
2. **会话 → 灰度图**：每个会话取前 784 字节的有效载荷，填入 28×28 的 `uint8` 矩阵。
3. **训练时 MAC/IP 头部掩码**：将以太网和 IP 头部字节统一置为固定值，降低模型对地址的过拟合。
4. **多模型集成**：训练 CNN-BiLSTM、Classic CNN、Lightweight CNN-BiLSTM、Pure BiLSTM、MLP、Transformer 共 6 个模型，
   推理时按多数投票 + 各自阈值综合判断。

## 目录结构

```
├── data/                     # 数据集
│   ├── USTC-TFC2016-master/  # USTC-TFC2016 原始数据集 (Benign/Malware)
│   ├── combined_train_pcap/  # 混合训练集
│   ├── mixed_train_pcap/     # 混合训练集（另一版本）
│   └── processed/            # 预处理后的特征缓存
├── src/
│   ├── preprocessing/        # 特征提取 (feature_extraction.py) 与数据加载 (data_loader.py)
│   ├── models/               # 6 个模型定义
│   │   ├── cnn_bilstm.py     # CNN-BiLSTM（默认主力模型）
│   │   ├── classic_cnn.py    # 纯 CNN 基线
│   │   ├── lightweight_cnn_bilstm.py  # 轻量 CNN-BiLSTM
│   │   ├── mlp.py            # 三层 MLP
│   │   ├── transformer.py    # Transformer 编码器
│   │   └── test_wandb.py     # wandb 测试脚本
│   ├── training/             # 训练脚本
│   │   ├── train.py          # 通用训练循环、评估函数
│   │   ├── loss_utils.py     # FocalLoss、类别权重计算
│   │   ├── run_training_all.py  # 一键训练全部 6 个模型
│   │   └── train_*.py        # 各模型的独立训练脚本
│   └── utils/                # 工具函数
├── web/
│   ├── backend/main.py       # FastAPI 后端（模型加载、推理、集成投票、实时抓包）
│   ├── templates/            # Jinja2 HTML 模板
│   └── static/               # 前端静态资源
├── checkpoints/              # 训练好的模型权重 (.pth)
├── tools/                    # 辅助脚本（测试集提取、数据划分检查、
│                             #   端到端自检、实时/离线输入一致性验证）
├── run.py                    # 项目入口，启动 Web 服务
└── requirements.txt
```

## 功能特性

- **原始字节图像化**：直接从 PCAP 载荷字节生成 28×28 灰度图，无需手工特征工程。
- **6 模型集成投票**：CNN-BiLSTM、Classic CNN、Lightweight CNN-BiLSTM、Pure BiLSTM、MLP、Transformer
  共同投票，每个模型有独立的恶意判决阈值，比单模型更鲁棒。
- **特征缓存**：首次提取后将结果存为 `.npz` 文件，后续训练直接读取，大幅缩短数据准备时间。
- **PCAP 包级聚合**：支持按比例和绝对数量两种策略，从会话级判决汇总为整包判决。
- **Web 可视化界面**：带登录认证的 Web UI，支持文件上传分析和实时网卡抓包。
- **FocalLoss 支持**：可选用 Focal Loss 缓解类别不平衡问题。

## 环境要求

- Python 3.11（推荐）
- PyTorch（GPU 版需 CUDA）
- Windows 用户需安装 [Npcap](https://npcap.com/) 才能使用 Scapy 抓包

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将 USTC-TFC2016 数据集放入 `data/USTC-TFC2016-master/`，目录结构应为：
```
USTC-TFC2016-master/
├── Benign/
│   └── *.pcap
└── Malware/
    └── *.pcap
```

也可以使用 `--dataset` 参数选择其他预置数据集路径。

### 3. 训练模型

一键训练全部 6 个模型：
```bash
python src/training/run_training_all.py --device auto --epochs 50
```

仅训练指定模型：
```bash
python src/training/run_training_all.py --model CNN_BiLSTM --device cuda
```

使用 Focal Loss 和恶意样本权重加倍：
```bash
python src/training/run_training_all.py --loss-function focal --malware-weight-multiplier 2.0
```

### 4. 启动 Web 服务

```bash
python run.py
```

或者直接使用 uvicorn：
```bash
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。

> 仓库不再附带任何明文账号，首次使用请自备账号，二选一：
>
> 1. 复制 `web/backend/users.example.json` 为 `web/backend/users.json`，把里面的
>    `example_user` / `example_admin` 改成自己的用户名与密码（写明文即可，
>    首次成功登录后会自动升级为 PBKDF2 哈希并原地回写）。该文件已被 `.gitignore` 忽略。
> 2. 或用环境变量覆盖：`DLIDS_ADMIN_USER` / `DLIDS_ADMIN_PASSWORD`。

## 模型说明

| 模型 | 文件 | 特点 |
|------|------|------|
| CNN-BiLSTM | `cnn_bilstm.py` | 主力模型，CNN 提取局部特征 + BiLSTM 捕捉时序上下文 |
| Classic CNN | `classic_cnn.py` | 纯三层 Conv1d 基线，无 RNN 结构 |
| Lightweight CNN-BiLSTM | `lightweight_cnn_bilstm.py` | CNN-BiLSTM 的轻量版，参数量约为标准版的 1/5 |
| Pure BiLSTM | `pure_bilstm.py` | 纯 BiLSTM，28 行作为 28 个时间步 |
| MLP | `mlp.py` | 三层全连接 + BatchNorm + Dropout |
| Transformer | `transformer.py` | Transformer 编码器 + 位置编码 |

Web 端在推理时会用 6 个模型分别预测，然后做多数投票。
每个模型有独立的恶意判决阈值（`MODEL_MALWARE_THRESHOLDS`），
Transformer 的阈值设置较低（0.35），因为它在当前数据集上输出的恶意概率偏低。

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | USTC-TFC2016 | 数据集根目录 |
| `--dataset` | None | 预置数据集：ustc/ics/mixed/combined |
| `--epochs` | 50 | 每个模型的训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--lr` | 0.001 | Adam 学习率 |
| `--loss-function` | ce | 损失函数：ce (交叉熵) 或 focal |
| `--focal-gamma` | 2.0 | Focal Loss 的 gamma |
| `--malware-weight-multiplier` | 1.0 | 恶意类别的额外权重乘子 |
| `--no-cache` | False | 禁用特征缓存 |
| `--rebuild-cache` | False | 强制重建缓存 |
| `--device` | auto | 训练设备：auto/cpu/cuda |

> 模型结构以 `src/training/run_training_all.py` 里的实例化参数与实际产出的
> `checkpoints/*.pth` 为准。Transformer 当前是 `d_model=64, nhead=4, num_layers=2,
> dropout=0.3`（早期文档里写的 128/8/4 从未落地）。改结构必须重新训练，
> 否则 `load_state_dict` 会因形状不匹配失败。

## 许可证

本项目仅供学术研究使用。
