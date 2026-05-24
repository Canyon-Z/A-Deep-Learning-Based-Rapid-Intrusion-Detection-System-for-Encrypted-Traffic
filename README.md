# 基于深度学习的面向加密流量的入侵检测系统（推荐 Python 3.11）

## 项目简介
本项目旨在开发一套针对加密流量的智能入侵检测系统。当前主流程使用 CNN-BiLSTM 对流量进行二分类（Normal/Malicious），并可选接入本地大语言模型生成安全分析建议。Transformer 已接入，但当前更适合作为实验模型，不建议作为默认主力检测器。

## 目录结构
- `data/`: 存放数据集 (raw: 原始pcap, processed: 处理后的特征)
- `src/`: 源代码
  - `preprocessing/`: 流量预处理与特征提取
  - `models/`: 深度学习模型定义
  - `training/`: 训练与评估脚本
  - `utils/`: 工具函数
- `web/`: Web前端与后端接口
  - `backend/`: FastAPI后端
  - `templates/`: HTML模板
  - `static/`: 静态文件

## 功能特性
1. **多维特征提取（6维）**: 包长度、协议、包间隔时间(IAT)、TCP Flags、源端口、目标端口（已归一化）。
2. **CNN-BiLSTM 检测模型**: 作为主判定模型输出分类与置信度。
3. **本地 LLM 建议模块（可选）**: 生成“风险总结 / 处置建议 / 复核建议”。
4. **Web 可视化界面**: 上传 PCAP 后先返回检测结果，再异步更新 AI 建议。

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   注意：Windows 用户可能需要安装 [Npcap](https://npcap.com/) 才能正常使用 Scapy。
   如果你要启用 GPU，建议使用 Python 3.11 + CUDA 版 PyTorch。

2. 训练模型（首次或特征升级后必须执行）：
   ```bash
   C:/Users/19512/AppData/Local/Programs/Python/Python311/python.exe src/training/run_training_all.py --device auto
   ```
   也可以分别运行 `src/training/train_cnn_bilstm.py`、`src/training/train_lightweight_cnn.py`、`src/training/train_transformer.py`，并显式传入 `--device auto|cpu|cuda`。

3. 运行 Web 服务：
   ```bash
   python run.py
   ```
   或者:
   ```bash
   uvicorn web.backend.main:app --reload
   ```

4. 访问浏览器：
   打开 http://127.0.0.1:8000

## 模型文件位置
- 默认检测模型优先加载：`checkpoints/final_model.pth`
- 兼容回退路径：`checkpoints/cnn_bilstm.pth`
- Transformer 权重文件：`checkpoints/transformer.pth`
- 当前后端会优先选择与模型结构完全匹配的权重文件。

## 模型建议
- 默认线上检测优先使用 CNN-BiLSTM 或 Lightweight_CNN_BiLSTM。
- Transformer 当前在本数据集上的测试表现偏弱，建议先重训再考虑作为默认模型。
- 如果训练后出现“偏向单一类别”的结果，优先检查 checkpoint 是否与当前模型结构一致。

## 本地 LLM 建议（可选）
后端会在返回分类结果后异步生成建议，相关配置：

- `MODEL_API_URL`（例：`http://localhost:11451/api/v1/chat`）
- `MODEL_NAME`
- `MODEL_API_KEY`（可选）
- `MODEL_API_CONNECT_TIMEOUT`（默认 `5`）
- `MODEL_API_READ_TIMEOUT`（默认 `45`）

如果不希望启用 LLM 建议，可将 `MODEL_API_URL` 置空。
