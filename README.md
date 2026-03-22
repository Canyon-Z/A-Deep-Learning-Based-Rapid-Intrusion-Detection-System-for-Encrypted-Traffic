# 基于深度学习的面向加密流量的入侵检测系统（Python 3.12）

## 项目简介
本项目旨在开发一套针对加密流量的智能入侵检测系统。当前主流程使用 CNN-BiLSTM 对流量进行二分类（Normal/Malicious），并可选接入本地大语言模型生成安全分析建议。

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

2. 训练模型（首次或特征升级后必须执行）：
   ```bash
   python -m src.training.train --epochs 20 --batch-size 4 --lr 5e-4 --input-dim 6 --save-path checkpoints/cnn_bilstm_best.pth
   ```

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
- 默认检测模型权重路径：`checkpoints/cnn_bilstm_best.pth`
- 可通过环境变量 `MODEL_CHECKPOINT_PATH` 覆盖默认路径。

## 本地 LLM 建议（可选）
后端会在返回分类结果后异步生成建议，相关配置：

- `MODEL_API_URL`（例：`http://localhost:11451/api/v1/chat`）
- `MODEL_NAME`
- `MODEL_API_KEY`（可选）
- `MODEL_API_CONNECT_TIMEOUT`（默认 `5`）
- `MODEL_API_READ_TIMEOUT`（默认 `45`）

如果不希望启用 LLM 建议，可将 `MODEL_API_URL` 置空。
