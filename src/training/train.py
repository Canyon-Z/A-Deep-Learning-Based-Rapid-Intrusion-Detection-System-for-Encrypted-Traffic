import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns

try:
    import wandb
except ImportError:
    wandb = None


def resolve_malware_index(class_names):
    """定位「恶意」类在 ``class_names`` 里的下标。

    缺陷 D-27：这段逻辑原先在本文件有三份独立实现 —— ``_safe_auc`` 与
    ``_compute_metrics`` 用子串匹配（``'malware' in name.lower()``），
    ``compute_extended_metrics`` 用精确匹配（``in ('malware','malicious','attack')``）。
    同一个 ``class_names`` 在两处可能落到不同下标，于是 AUC 和召回率被算在了不同的类上，
    指标之间互相矛盾。统一成一处：先精确匹配规范名，再退回子串匹配。
    """
    names = [str(n).strip().lower() for n in (class_names or [])]
    for canonical in ('malware', 'malicious', 'attack'):
        if canonical in names:
            return names.index(canonical)
    for idx, name in enumerate(names):
        if 'malware' in name or 'malicious' in name:
            return idx
    # 找不到显式命名时保持历史默认：二分类里 1 号类为正类，单类时退化为 0
    return 1 if len(names) > 1 else 0


def _safe_auc(y_true, y_prob, class_names):
    """计算 AUC，自动找到 Malware 类别作为正类"""
    malware_idx = resolve_malware_index(class_names)

    try:
        if len(class_names) == 2 and y_prob.shape[1] > malware_idx:
            return float(roc_auc_score(y_true, y_prob[:, malware_idx]))
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        return None


def _compute_metrics(y_true, y_pred, y_prob, inference_time_sec, class_names):
    """计算分类相关的各项指标，包括准确率、精确率、召回率、F1、FPR、AUC 等"""
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    malware_idx = resolve_malware_index(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    recall_malware = None
    recall_benign = None
    if len(class_names) == 2 and cm.shape == (2, 2):
        neg_idx = 1 - malware_idx
        tp = cm[malware_idx, malware_idx]
        fn = cm[malware_idx, neg_idx]
        fp = cm[neg_idx, malware_idx]
        tn = cm[neg_idx, neg_idx]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        # 检出率只针对恶意类：TP/(TP+FN)，与 1-FNR 严格一致。
        # 不能复用 macro recall，否则与 FNR 口径矛盾。
        recall_malware = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_benign = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        fpr = None
        fnr = None

    auc = _safe_auc(y_true, y_prob, class_names)
    sample_count = max(len(y_true), 1)
    latency_ms = (inference_time_sec / sample_count) * 1000.0
    throughput = len(y_true) / max(inference_time_sec, 1e-8)

    return {
        'loss': None,
        'accuracy': float((y_pred == y_true).mean()) * 100.0 if len(y_true) else 0.0,
        'precision': float(precision) * 100.0,
        'recall': float(recall) * 100.0,
        'f1_score': float(f1) * 100.0,
        'f2_score': float((5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0) * 100.0,
        # detection_rate 与 FNR 同源，都是"恶意类"口径，保证 DR == 100 - FNR；
        # 旧实现直接复用 macro recall，导致同一文件里 DR 与 FNR 互相矛盾。
        'detection_rate': (float(recall_malware) * 100.0) if recall_malware is not None else float(recall) * 100.0,
        'recall_malware': (float(recall_malware) * 100.0) if recall_malware is not None else None,
        'recall_benign': (float(recall_benign) * 100.0) if recall_benign is not None else None,
        'macro_recall': float(recall) * 100.0,
        'fpr': (float(fpr) * 100.0) if fpr is not None else None,
        'fnr': (float(fnr) * 100.0) if fnr is not None else None,
        'auc': auc,
        'latency_ms_per_sample': float(latency_ms),
        'throughput_samples_per_sec': float(throughput)
    }


def _selection_metrics(y_true, y_pred, class_names):
    """模型选择用的轻量指标：只算混淆矩阵派生量，不跑 AUC。

    与 _compute_metrics 的区别是它足够便宜，可以每轮验证都算一次；训练结束后的
    完整指标仍然由 _compute_metrics 给出。
    """
    num_classes = len(class_names) if class_names else 2
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    malware_idx = resolve_malware_index(class_names)

    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total * 100.0) if total > 0 else 0.0
    recall_malware = 0.0
    recall_benign = 0.0
    if num_classes == 2:
        neg_idx = 1 - malware_idx
        tp = float(cm[malware_idx, malware_idx])
        fn = float(cm[malware_idx, neg_idx])
        fp = float(cm[neg_idx, malware_idx])
        tn = float(cm[neg_idx, neg_idx])
        recall_malware = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        recall_benign = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    binary_kwargs = (dict(average='binary', pos_label=malware_idx, zero_division=0)
                     if num_classes == 2 else dict(average='macro', zero_division=0))
    return {
        'accuracy': accuracy,
        'recall_malware': recall_malware,
        'recall_benign': recall_benign,
        'balanced_accuracy': (recall_malware + recall_benign) / 2.0,
        'f1_score': float(f1_score(y_true, y_pred, **binary_kwargs) * 100.0),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0),
    }


def _selection_score(metrics, metric_name):
    """把指标字典折算成模型选择的标量分；未知指标名告警并回退到 accuracy。"""
    if not metrics:
        return -1.0
    name = str(metric_name or 'accuracy').strip().lower()
    key_map = {
        'f1': 'f1_score',
        'f1_malware': 'f1_score',
        'f1_score': 'f1_score',
        'macro_f1': 'macro_f1',
        'f1_macro': 'macro_f1',
        'balanced_accuracy': 'balanced_accuracy',
        'balanced': 'balanced_accuracy',
        'recall_malware': 'recall_malware',
        'detection_rate': 'recall_malware',
        'dr': 'recall_malware',
        'accuracy': 'accuracy',
        'acc': 'accuracy',
    }
    if name not in key_map:
        print(f"Warning: unknown selection_metric {metric_name!r}, falling back to accuracy.")
    key = key_map.get(name, 'accuracy')
    return float(metrics.get(key, 0.0) or 0.0)


def evaluate_model(model, dataloader, criterion, device, class_names):
    """用测试集评估模型，返回各项指标"""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []
    inference_time_sec = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="test Phase"):
            inputs = inputs.to(device, non_blocking=(device.startswith('cuda') if isinstance(device, str) else device.type == 'cuda'))
            labels = labels.to(device, non_blocking=(device.startswith('cuda') if isinstance(device, str) else device.type == 'cuda'))
            infer_start = time.perf_counter()
            outputs = model(inputs)
            inference_time_sec += (time.perf_counter() - infer_start)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    if total_samples == 0:
        return None

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = _compute_metrics(y_true, y_pred, y_prob, inference_time_sec, class_names)
    metrics['loss'] = float(running_loss / total_samples)
    metrics['sample_count'] = int(total_samples)
    return metrics


def train_model(model, train_loader, val_loader, test_loader=None, criterion=None, optimizer=None,
                num_epochs=50, device='cuda', config=None,
                early_stopping_patience=0, early_stopping_min_delta=1e-4,
                use_amp=False, val_every=1, use_scheduler=False, selection_metric='accuracy'):
    """
    标准训练循环：每轮在训练集上前向/反向传播，在验证集上评估，
    保存验证准确率最高的模型权重。

    训练过程中会将 loss、accuracy、混淆矩阵等信息写入 wandb（如已安装）。
    训练结束后把各项指标汇总写入 checkpoints/model_metrics.json。
    """
    if config is None:
        config = {}

    # 初始化 wandb（优先使用离线模式，避免需要 API key）
    if wandb is not None:
        try:
            print("初始化 WandB (如有需要请在终端输入 API Key，或提前运行 wandb offline)...")
            wandb.init(project="intrusion-detection-traffic", config=config, resume="allow", mode="offline")
            print("WandB 初始化完成 (Offline模式).")
        except Exception as e:
            print(f"WandB init failed: {e}. Running without wandb.")
    else:
        print("WandB 未安装，跳过实验日志。")

    if wandb is not None:
        try:
            wandb.watch(model, criterion, log="all", log_freq=10)
        except:
            pass

    model.to(device)
    best_acc = 0.0
    # 模型选择分数。selection_metric='accuracy' 时它就等于 best_acc；换成 f1 /
    # macro_f1 / balanced_accuracy 时判据会变——同源验证集上 accuracy 很快到 99.9%，
    # 所有模型看起来"一样好"，只有 F1 或恶意类召回率这类指标才能把真正可迁移的
    # 模型挑出来（实测跨数据集时 CNN-BiLSTM 与 Pure BiLSTM 的差距就出在这里）。
    best_selection_score = -1.0
    selection_name = str(selection_metric or 'accuracy').strip().lower()
    best_metrics = {}
    best_state_dict = None
    # 最佳 epoch 的原始预测先留在这里，训练结束再算一次完整指标。
    # 旧写法每次验证集提升都要跑一遍 sklearn（AUC/混淆矩阵），50 轮里白算几十次。
    best_predictions = None
    epochs_without_improvement = 0
    stopped_epoch = num_epochs

    class_names = config.get('class_names', [str(i) for i in range(10)])

    # 混合精度：默认关闭。实测（RTX 4070 Laptop + torch 2.6）开启 AMP 后训练会在
    # 第 5-7 轮之间静默退出（无 Traceback、无崩溃转储、exit code 1），三次复现；
    # 关闭后同样配置可稳定跑完。需要时用 --amp 显式打开。
    amp_enabled = bool(use_amp) and str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None

    # 学习率调度：默认关闭。ReduceLROnPlateau 在 val loss 波动时会把 lr 快速压到
    # 1e-6，实测反而让模型在第 2 轮之后就不再更新（val 卡在首轮水平）。
    # 原始 50 轮训练没有调度器也能收敛到 99.9%+，需要时用 use_scheduler=True 打开。
    scheduler = None
    if use_scheduler and optimizer is not None:
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5
            )
        except Exception:
            scheduler = None

    if early_stopping_patience > 0:
        print(f"早停已启用：连续 {early_stopping_patience} 轮验证集无提升即停止（min_delta={early_stopping_min_delta}）")
    if val_every > 1:
        print(f"验证间隔：每 {val_every} 轮验证一次")
    if amp_enabled:
        print("混合精度 (AMP)：已启用")

    print("\n================== 开始训练循环 ==================")
    for epoch in range(num_epochs):
        print(f'\n▶ Epoch [{epoch+1}/{num_epochs}]')
        print('-' * 40)

        # val_every>1 时跳过中间轮的验证，末轮强制验证一次（省掉接近一半的推理开销）
        run_val = (val_every <= 1) or ((epoch + 1) % val_every == 0) or (epoch + 1 == num_epochs)
        for phase in (['train', 'val'] if run_val else ['train']):
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            all_preds = []
            all_labels = []
            all_probs = []
            inference_time_sec = 0.0

            pbar = tqdm(dataloader, desc=f"{phase} Phase", mininterval=1.0)
            for inputs, labels in pbar:
                non_blocking = device.startswith('cuda') if isinstance(device, str) else device.type == 'cuda'
                inputs = inputs.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    infer_start = time.perf_counter()
                    with torch.amp.autocast('cuda', enabled=amp_enabled):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    inference_time_sec += (time.perf_counter() - infer_start)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                # 用 detach 累加，最后只取一次标量：每批 .item() / set_postfix 都会强制 CUDA 同步
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                if phase == 'val':
                    all_preds.extend(preds.view(-1).cpu().numpy())
                    all_labels.extend(labels.view(-1).cpu().numpy())
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.detach().cpu().numpy())

            epoch_loss = float(running_loss) / max(total_samples, 1)
            epoch_acc = running_corrects.double() / total_samples

            print(f'➜ [{phase.upper()}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}% | Samples: {total_samples}')

            if wandb is not None:
                try:
                    wandb.log({
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_accuracy": epoch_acc,
                        "epoch": epoch + 1
                    })
                except:
                    pass

            # 学习率调度按验证集 loss 走（只在做了验证的轮次调用）
            if phase == 'val' and scheduler is not None:
                try:
                    scheduler.step(epoch_loss)
                except Exception:
                    pass

            # 保存验证集上最好的模型。判据由 selection_metric 决定：换成 f1 等指标时
            # 每轮要多算一次轻量指标（几十毫秒），仍然远比逐轮跑 AUC 便宜。
            if phase == 'val':
                if selection_name != 'accuracy':
                    current_score = _selection_score(
                        _selection_metrics(np.array(all_labels), np.array(all_preds), class_names),
                        selection_name,
                    )
                else:
                    current_score = float(epoch_acc) * 100.0

                if current_score > best_selection_score:
                    best_selection_score = current_score
                    best_acc = float(epoch_acc)
                    # 只留原始预测，完整指标等训练结束再算一次，省掉几十轮 sklearn 开销
                    best_predictions = (
                        np.array(all_labels),
                        np.array(all_preds),
                        np.array(all_probs),
                        float(inference_time_sec),
                    )
                    best_metrics = {
                        'loss': float(epoch_loss),
                        'accuracy': float(epoch_acc) * 100.0,
                        'val_loss': float(epoch_loss),
                        'best_accuracy': float(epoch_acc) * 100.0,
                        'selection_metric': selection_name,
                        'selection_score': float(current_score),
                    }
                    # state_dict 先搬到 CPU 再存：比 copy.deepcopy 省显存也更快
                    best_state_dict = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if phase == 'val' and wandb is not None:
                try:
                    wandb.log({
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_labels,
                            preds=all_preds,
                            class_names=class_names
                        )
                    })
                except:
                    pass

        # 早停：连续 N 轮验证集没有提升就收工。实测 6 个模型几乎第 1 轮就到 99.9%+，
        # 没有早停的话后面 40 多轮全是白跑。
        if (early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
                and epoch + 1 < num_epochs):
            stopped_epoch = epoch + 1
            print(f"\n⏹ 早停触发：连续 {epochs_without_improvement} 轮验证集无提升，在第 {stopped_epoch}/{num_epochs} 轮结束训练")
            break

    print('\n================== 训练结束 ==================')
    if selection_name == 'accuracy':
        print(f'✅ Best Validation Accuracy: {best_acc*100:.2f}% (完成 {stopped_epoch}/{num_epochs} 轮)')
    else:
        print(f'✅ Best Validation {selection_name}: {best_selection_score:.4f} '
              f'(对应 accuracy {best_acc*100:.2f}%，完成 {stopped_epoch}/{num_epochs} 轮)')

    # 最佳 epoch 的完整指标（precision/recall/F1/AUC/混淆矩阵等）在这里补算一次
    if best_predictions is not None:
        y_true, y_pred, y_prob, infer_sec = best_predictions
        computed = _compute_metrics(y_true, y_pred, y_prob, infer_sec, class_names)
        # 训练时记录的 val loss/accuracy 优先，其余指标来自补算结果
        best_metrics = {**computed, **best_metrics}

    test_metrics = None
    if test_loader is not None and len(test_loader) > 0:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        test_metrics = evaluate_model(model, test_loader, criterion, device, class_names)
        if test_metrics is not None:
            print(f"✅ Test Accuracy: {test_metrics['accuracy']:.2f}%")

    # 结束 wandb
    if wandb is not None:
        try:
            wandb.finish()
        except:
            pass

    # 保存指标到本地 JSON
    import json

    model_name = config.get('model', 'unknown_model')
    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../checkpoints/model_metrics.json")

    all_metrics = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
        except:
            pass

    def _fmt_pct(mapping, key):
        value = mapping.get(key) if mapping else None
        return f"{float(value):.2f}%" if value is not None else "N/A"

    def _fmt_num(mapping, key, digits=4):
        value = mapping.get(key) if mapping else None
        return f"{float(value):.{digits}f}" if value is not None else "N/A"

    # 验证集口径（验证集上最佳 epoch 的指标），与测试集口径分开保存，
    # 避免前端把两种来源的数字混在一起比较。
    val_block = {
        "precision": _fmt_pct(best_metrics, 'precision'),
        "recall": _fmt_pct(best_metrics, 'recall'),
        "f1_score": _fmt_pct(best_metrics, 'f1_score'),
        "f2_score": _fmt_pct(best_metrics, 'f2_score'),
        "detection_rate": _fmt_pct(best_metrics, 'detection_rate'),
        "fpr": _fmt_pct(best_metrics, 'fpr'),
        "fnr": _fmt_pct(best_metrics, 'fnr'),
        "auc": _fmt_num(best_metrics, 'auc'),
        "latency_ms_per_sample": _fmt_num(best_metrics, 'latency_ms_per_sample'),
        "throughput_samples_per_sec": _fmt_num(best_metrics, 'throughput_samples_per_sec', 2),
        "recall_malware": _fmt_pct(best_metrics, 'recall_malware'),
        "recall_benign": _fmt_pct(best_metrics, 'recall_benign'),
        "macro_recall": _fmt_pct(best_metrics, 'macro_recall'),
        "val_precision": _fmt_pct(best_metrics, 'precision'),
        "val_recall": _fmt_pct(best_metrics, 'recall'),
        "val_f1_score": _fmt_pct(best_metrics, 'f1_score'),
        "val_f2_score": _fmt_pct(best_metrics, 'f2_score'),
        "val_detection_rate": _fmt_pct(best_metrics, 'detection_rate'),
        "val_fpr": _fmt_pct(best_metrics, 'fpr'),
        "val_fnr": _fmt_pct(best_metrics, 'fnr'),
        "val_auc": _fmt_num(best_metrics, 'auc'),
        "val_latency_ms_per_sample": _fmt_num(best_metrics, 'latency_ms_per_sample'),
        "val_throughput_samples_per_sec": _fmt_num(best_metrics, 'throughput_samples_per_sec', 2),
    }

    test_block = {}
    if test_metrics:
        test_block = {
            "test_precision": _fmt_pct(test_metrics, 'precision'),
            "test_recall": _fmt_pct(test_metrics, 'recall'),
            "test_f1_score": _fmt_pct(test_metrics, 'f1_score'),
            "test_f2_score": _fmt_pct(test_metrics, 'f2_score'),
            "test_detection_rate": _fmt_pct(test_metrics, 'detection_rate'),
            "test_fpr": _fmt_pct(test_metrics, 'fpr'),
            "test_fnr": _fmt_pct(test_metrics, 'fnr'),
            "test_auc": _fmt_num(test_metrics, 'auc'),
            "test_latency_ms_per_sample": _fmt_num(test_metrics, 'latency_ms_per_sample'),
            "test_throughput_samples_per_sec": _fmt_num(test_metrics, 'throughput_samples_per_sec', 2),
            "test_recall_malware": _fmt_pct(test_metrics, 'recall_malware'),
            "test_recall_benign": _fmt_pct(test_metrics, 'recall_benign'),
        }

    # 缺陷 D-26：原先这里写的 key 叫 "best_accuracy" / "final_loss"，但取值是
    # **测试集**准确率与测试集损失（有 test_metrics 时），没有 test_metrics 时又退回
    # 验证集最优值。同一个 key 在两种情况下含义不同，读 metrics.json 的人无从分辨。
    # 现在显式拆开：best_val_* 只放验证集，test_* 只放测试集，
    # 并为老键保留带口径后缀的别名（值不改，只把语义写进 key 名）。
    _has_test = test_metrics is not None
    all_metrics[model_name] = {
        "best_val_accuracy": f"{best_metrics.get('best_accuracy', float(best_acc) * 100.0):.2f}%",
        "best_val_loss": f"{best_metrics.get('val_loss', float(epoch_loss)):.4f}",
        "test_accuracy": f"{test_metrics['accuracy']:.2f}%" if _has_test else None,
        "test_loss": f"{test_metrics['loss']:.4f}" if _has_test else None,
        # 兼容旧读取方（/model-status、前端 model_metrics 面板仍按 best_accuracy 取值），
        # 但把「这个数字究竟来自哪一集」写清楚，避免再被当成验证集最优准确率。
        "best_accuracy": f"{test_metrics['accuracy']:.2f}%" if _has_test else f"{best_metrics.get('best_accuracy', float(best_acc) * 100.0):.2f}%",
        "final_loss": f"{test_metrics['loss']:.4f}" if _has_test else f"{best_metrics.get('val_loss', float(epoch_loss)):.4f}",
        "best_accuracy_source": 'test' if _has_test else 'val_best',
        "final_loss_source": 'test' if _has_test else 'val_best',
        "epochs_trained": stopped_epoch,
        "num_epochs_requested": num_epochs,
        "early_stopped": bool(stopped_epoch < num_epochs),
        "classes": list(class_names),
        **val_block,
        **test_block,
    }

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_file}")

    return model, best_state_dict, test_metrics or best_metrics
