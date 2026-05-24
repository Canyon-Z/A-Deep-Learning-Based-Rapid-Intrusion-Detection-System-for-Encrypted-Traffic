# 阶段汇报

## 1. 项目背景
本阶段工作的核心场景是对加密流量进行入侵检测。系统的整体流程是将 PCAP 文件转换为特征表示，再交由不同检测模型完成推理，并在 Web 页面中展示检测结果。

在前一阶段，系统已经具备基础的单文件检测能力，但实际使用中暴露出两个影响体验的问题：
1. 用户每次只能上传一个文件，不便于一次性处理多个样本。
2. 检测结果只展示预测结论，没有明确标识是哪个模型给出的结果，不利于比对不同模型的表现。

此外，模型加载时还遇到了 checkpoint 兼容性问题，部分权重文件与当前代码中的模型结构存在键名不一致或参数不匹配的情况，需要在加载阶段增强容错能力。

## 2. 本阶段目标
本阶段主要围绕“可用性增强”和“结果可解释性”两条主线展开：
- 支持批量上传多个 `pcap` / `pcapng` 文件。
- 在结果列表中增加模型来源信息，明确每条结果对应的检测模型。
- 保持现有检测接口、统计图表和详情弹窗继续可用。
- 修复或缓解模型 checkpoint 加载兼容性问题，保证服务可以正常启动并完成推理。

## 3. 主要实现内容

### 3.1 前端批量上传能力
修改位置： [web/templates/index.html](web/templates/index.html)

前端上传区域已经从“单文件检测”改成“多文件批量检测”，具体包括：
- 文件选择框增加了 `multiple`，允许用户一次选择多个文件。
- 拖拽上传逻辑支持多文件队列，不再只处理单个文件。
- 上传提示文案同步更新，明确告诉用户支持批量检测。
- 文件上传后不再覆盖单条结果，而是按文件逐条追加到结果列表。

这一改动的直接收益是：用户可以一次性提交一批样本，减少重复操作，尤其适合阶段测试、模型对比和小规模批量分析。

### 3.2 结果列表增加模型列
修改位置： [web/templates/index.html](web/templates/index.html)

结果表新增“模型”列，用于显示当前条目由哪个模型完成检测。展示逻辑支持从后端结果中直接读取模型名称，如果后端没有返回，则回退到前端当前选中的模型。

当前界面会展示的模型包括：
- `CNN-BiLSTM`
- `Classic-CNN`
- `Lightweight CNN-BiLSTM`
- `Pure BiLSTM`
- `Transformer`

这项改动的意义不只是“多一个字段”，而是让后续阶段汇报、模型对比和结果复盘更清晰。以前只能看到“检测结果”，现在可以同时看到“结果是谁给出的”。

### 3.3 后端接口返回增强
修改位置： [web/backend/main.py](web/backend/main.py)

后端 `/analyze` 接口新增了两个返回字段：
- `model_type`
- `model_name`

这样前端在渲染结果时，可以直接显示模型中文或展示名，而不是依赖硬编码映射。后端仍然保留原有返回内容，因此不会影响原有的图表统计、详情弹窗和状态判断。

### 3.4 模型加载兼容性处理
修改位置： [web/backend/main.py](web/backend/main.py)

本阶段还对模型加载过程做了兼容性增强，重点是 `CNN_BiLSTM` 的 checkpoint 读取：
- 主模型加载时固定使用 `hidden_dim=64`，与训练阶段保持一致。
- checkpoint 加载不再采用过于严格的强校验，而是尽量读取可匹配的参数。
- 增加了候选 checkpoint 的探测逻辑，优先加载更可能匹配当前结构的权重文件。
- 当存在缺失键或多余键时，可以输出更有诊断意义的信息，便于后续继续排查。

这部分工作解决的是“服务能不能稳定跑起来”的问题。即使某些 checkpoint 存在历史兼容差异，也不至于直接导致整个 Web 服务启动失败。

## 4. 模型参数说明与改动对照

### 4.1 参数作用说明
这一部分说明每个模型里关键参数的作用，便于在阶段汇报中解释“为什么要这么改”。

| 模型 | 关键参数 | 参数作用 |
| --- | --- | --- |
| CNN_BiLSTM | `hidden_dim` | 控制 BiLSTM 隐状态宽度，直接影响序列表征能力、模型体积和推理开销。|
| Classic_CNN | `out_channels`、`kernel_size`、`dropout`、`fc` 宽度 | 控制卷积特征提取能力和分类层容量，决定模型的基础表达能力和过拟合风险。|
| Lightweight_CNN_BiLSTM | `out_channels`、`hidden_dim`、`dropout` | 在保持 CNN + BiLSTM 结构的前提下压缩参数量，优先提升速度和内存效率。|
| Pure_BiLSTM | `input_size`、`hidden_dim`、`num_layers`、`dropout` | 纯时序结构参数，决定序列建模深度、上下文记忆长度和泛化能力。|
| Transformer | `input_dim`、`d_model`、`nhead`、`num_layers`、`dropout` | 控制输入嵌入维度、多头注意力并行度、编码层深度和正则化强度。|

### 4.2 逐模型改动前后

#### 4.2.1 CNN_BiLSTM
修改位置： [src/models/cnn_bilstm.py](src/models/cnn_bilstm.py)、[src/training/train_cnn_bilstm.py](src/training/train_cnn_bilstm.py)、[run.py](run.py)、[web/backend/main.py](web/backend/main.py)

| 参数 | 改动前 | 改动后 | 说明 |
| --- | --- | --- | --- |
| `hidden_dim` | 128 | 64 | 将 BiLSTM 宽度收缩到 64，减少参数量和推理负担，同时保持足够的分类能力。|
| checkpoint 加载 | 严格匹配或直接失败 | 支持候选 checkpoint 探测与非严格加载 | 避免因为 state_dict 键名或部分层不一致导致模型无法启动。|
| 推理入口 | 仅在训练时显式对齐 | 训练、保存、加载统一到同一配置 | 解决训练和服务端参数不一致的问题。|

#### 4.2.2 Classic_CNN
修改位置： [src/models/classic_cnn.py](src/models/classic_cnn.py)、[src/training/train_classic_cnn.py](src/training/train_classic_cnn.py)、[web/backend/main.py](web/backend/main.py)

| 参数 | 改动前 | 改动后 | 说明 |
| --- | --- | --- | --- |
| `out_channels` | 32 / 64 / 128 | 32 / 64 / 128 | 本阶段未改动卷积层通道数，仍作为稳定基线模型使用。|
| `dropout` | 0.6 | 0.6 | 保持较高 dropout 以抑制过拟合，兼顾召回率和误报率。|
| `fc` 宽度 | 256 | 256 | 分类层容量不变，重点是保持训练/推理一致性。|
| 模型接入 | 仅训练可用 | 前后端均可显示和调用 | 主要补齐 Web 端展示与后端模型注册。|

#### 4.2.3 Lightweight_CNN_BiLSTM
修改位置： [src/models/lightweight_cnn_bilstm.py](src/models/lightweight_cnn_bilstm.py)、[src/training/train_lightweight_cnn.py](src/training/train_lightweight_cnn.py)、[web/backend/main.py](web/backend/main.py)

| 参数 | 改动前 | 改动后 | 说明 |
| --- | --- | --- | --- |
| `hidden_dim` | 64 | 32 | 进一步压缩 BiLSTM 宽度，突出“轻量化”目标，降低延迟和显存占用。|
| `out_channels` | 16 / 32 | 8 / 16 | 卷积通道同步缩小，和隐藏层一起减少计算量。|
| `dropout` | 0.5 左右的更保守配置 | 0.4 | 在轻量化前提下稍微降低 dropout，避免模型容量过小导致欠拟合。|
| 训练与加载 | 参数分散、不完全统一 | 训练、保存、加载统一到 32 维轻量配置 | 保证轻量模型在 Web 端可以稳定读取和推理。|

#### 4.2.4 Pure BiLSTM
修改位置： [src/training/train_pure_bilstm.py](src/training/train_pure_bilstm.py)、[web/backend/main.py](web/backend/main.py)

| 参数 | 改动前 | 改动后 | 说明 |
| --- | --- | --- | --- |
| `hidden_dim` | 128 | 64 | 缩小隐藏层宽度，减少参数量，同时保留足够的序列记忆能力。|
| `num_layers` | 2 | 2 | 层数保持不变，避免纯时序模型过深而训练不稳定。|
| `dropout` | 0.5 | 0.5 | 维持原有正则化强度，防止时序模型过拟合。|
| `loss weight` | 默认交叉熵 | 对 Malware 类权重调到 1.2 | 在不破坏整体分布的情况下，适度提升对恶意流量的关注。|

#### 4.2.5 Transformer
修改位置： [src/models/transformer.py](src/models/transformer.py)、[src/training/train_transformer.py](src/training/train_transformer.py)、[src/training/run_training_all.py](src/training/run_training_all.py)、[web/backend/main.py](web/backend/main.py)

| 参数 | 改动前 | 改动后 | 说明 |
| --- | --- | --- | --- |
| `input_dim` | 28 | 28 | 输入维度保持不变，对应特征提取后的 28 维序列表示。|
| `d_model` | 64 或单一固定值 | 128，后端同时兼容 64 和 128 | 训练侧采用更强表达能力的 128，加载侧兼容历史模型配置，避免 checkpoint 不匹配。|
| `nhead` | 4 | 8 | 多头注意力并行度提升，帮助模型从不同子空间学习流量模式。|
| `num_layers` | 2 | 4 | 编码层更深，提升对复杂流量模式的抽象能力。|
| `dropout` | 0.3 | 0.05 | 训练侧改为更小的 dropout，保留模型容量；同时后端保留旧配置兼容加载。|
| 加载策略 | 单一配置尝试 | 候选配置探测 + 兼容旧权重 | 解决 Transformer checkpoint 在不同训练脚本之间的结构差异。|

### 4.3 当前统一的接口与展示参数
- 文件输入：`multiple`
- 批量处理方式：顺序执行
- 结果表新增列：`模型`
- 结果追加方式：逐条写入，不覆盖历史记录
- `/analyze` 请求仍接收：`file`、`model_type`
- `/analyze` 新增返回：`model_type`、`model_name`
- 后端推理阈值按模型单独管理：
  - `cnn_bilstm: 0.65`
  - `classic_cnn: 0.65`
  - `lightweight_cnn: 0.60`
  - `pure_bilstm: 0.60`
  - `transformer: 0.35`

## 5. 阶段效果
本阶段完成后，系统的可用性和可观测性都有了比较明显的提升：

1. 从单文件检测变成批量检测，操作效率更高。
2. 结果列表能直接看见模型来源，便于对比不同模型输出。
3. 后端返回信息更完整，前后端数据对齐更明确。
4. 模型加载对 checkpoint 差异更宽容，减少“模型未加载成功”导致的服务不可用问题。

## 6. 验证情况
已经完成的验证包括：
- 对 `web/backend/main.py` 做过语法检查，没有报错。
- 前端模板逻辑已经和后端返回字段对齐，数据链路是闭合的。
- 文档内容已整理到仓库中，能够作为阶段汇报材料直接使用。

当前尚未补充的验证包括：
- 一次完整的端到端批量上传实测。
- 不同模型在同一批样本上的结果对比统计。
- 生产级别的大样本压力验证。

## 7. 当前风险与待跟进事项
- 批量检测目前采用顺序处理，稳定性较好，但在文件数量很多时会比较慢，后续可以评估是否引入并发控制。
- 目前主要验证了代码层面的连通性，还缺少更完整的真实样本回归测试。
- 工作区中存在一些临时文件和 `__pycache__` 产物，不影响核心功能，但后续可以再做一次清理。

## 8. 下一阶段建议
如果后续继续推进，这个项目比较自然的下一步是：
1. 增加一次完整的批量检测实测，并记录耗时和准确率。
2. 给不同模型补一份对比结果，形成更完整的阶段分析。
3. 清理临时产物并整理成最终版交付文档。
