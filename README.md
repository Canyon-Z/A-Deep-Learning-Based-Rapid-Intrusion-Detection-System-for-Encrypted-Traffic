# 基于深度学习的面向加密流量的入侵检测系统（python3.12 Conda）

## 项目简介
本项目旨在开发一套针对加密流量的智能入侵检测系统。利用深度学习（CNN-BiLSTM）和Transformer模型，结合多维流量指纹提取，实现对加密流量中隐蔽攻击的有效检测。

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
1. **多维特征提取**: 提取数据包长度、时序间隔、字节分布等。
2. **CNN-BiLSTM模型**: 捕捉时空特征与长期依赖。
3. **Transformer模型**: 实现深度语义理解。
4. **Web可视化界面**: 提供流量上传与检测结果展示。

## 快速开始
1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   *注意: Windows用户可能需要安装 [Npcap](https://npcap.com/) 才能正常使用 Scapy。*

2. 运行Web服务:
   ```bash
   python run.py
   ```
   或者:
   ```bash
   uvicorn web.backend.main:app --reload
   ```

3. 访问浏览器:
   打开 http://127.0.0.1:8000
   *注意: 请勿直接双击 index.html 文件打开，必须通过浏览器访问上述地址，否则无法连接后端接口。*
