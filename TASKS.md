# 项目开发任务清单 (Team Tasks)

本文件列出了各成员在当前框架下的具体开发任务。请大家认领并按计划执行。

## 1. 徐麟翔 (数据工程)
**核心目标**: 完成从原始 PCAP 到 PyTorch DataLoader 的完整管道。

- [ ] **完善数据加载器 (`src/preprocessing/data_loader.py`)**
    - 实现 `TrafficDataset` 类中的 `_load_data` 方法，读取 `data/processed/Png` 或 `data/USTC-TFC2016/4_Png` 下的图片。
    - 实现 `__getitem__` 方法，使用 PIL 加载图片并转换为 Tensor。
    - 在 `get_dataloaders` 中定义合适的数据增强 (Transforms)。
- [ ] **特征提取脚本 (`src/preprocessing/feature_extraction.py`)**
    - 确保 `extract_features` 方法能正确处理单个 PCAP 文件，供 Web 端实时调用。
    - 考虑是否需要将 Session2Png 的逻辑集成到 Python 中，以便单文件实时转换。
- [ ] **数据准备**
    - 运行 `data_USTC-TK2016/` 下的脚本，生成训练所需的 PNG 数据集。

## 2. 石凌云 (核心算法)
**核心目标**: 实现模型训练循环并调优模型性能。

- [ ] **模型细节完善 (`src/models/cnn_bilstm.py` & `transformer.py`)**
    - 确认输入 Tensor 的维度 (Batch, Channels, Height, Width) 是否与数据加载器输出一致。
    - 调整 CNN 的 Kernel Size 和 Transformer 的 Head 数量。
- [ ] **编写训练流程 (`src/training/train.py`)**
    - 补全 `train_model` 函数：
        - 定义 Loss 函数 (如 `CrossEntropyLoss`)。
        - 定义优化器 (如 `AdamW`)。
        - 实现 Forward -> Loss -> Backward -> Optimizer Step 流程。
    - 添加模型保存机制 (`torch.save`)，保存效果最好的模型权重。

## 3. 周嘉辉 (系统开发)
**核心目标**: 完成 Web 界面与后端推理引擎的对接。

- [ ] **后端接口实现 (`web/backend/main.py`)**
    - 加载训练好的模型文件 (`.pth`)。
    - 在 `/analyze` 接口中，接收上传的 PCAP，调用 `extractor` 处理，再输入模型进行预测。
- [ ] **前端优化 (`web/templates/index.html`)**
    - 美化上传界面。
    - 使用图表库 (如 ECharts) 展示检测结果 (正常/恶意概率)。
- [ ] **系统测试**
    - 测试大文件上传和高并发请求下的稳定性。

## 4. 组长 (统筹与融合)
**核心目标**: 整合各模块，把控整体进度。

- [ ] **统一入口 (`run.py`)**
    - 确保 `python run.py --mode train` 能跑通训练。
    - 确保 `python run.py --mode web` 能启动演示界面。
- [ ] **模型融合**
    - 在模型训练完成后，尝试编写代码将 CNN_BiLSTM 和 Transformer 的结果进行加权融合。
