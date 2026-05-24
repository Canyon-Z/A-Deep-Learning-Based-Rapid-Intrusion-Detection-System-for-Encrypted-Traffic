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

# from src.models.cnn_bilstm import CNN_BiLSTM 

def _safe_auc(y_true, y_prob, class_names):
    malware_idx = 1
    for idx, cname in enumerate(class_names):
        if isinstance(cname, str) and ('malware' in cname.lower() or 'malicious' in cname.lower()):
            malware_idx = idx
            break

    try:
        if len(class_names) == 2 and y_prob.shape[1] > malware_idx:
            return float(roc_auc_score(y_true, y_prob[:, malware_idx]))
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        return None


def _compute_metrics(y_true, y_pred, y_prob, inference_time_sec, class_names):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    malware_idx = 1
    for idx, cname in enumerate(class_names):
        if isinstance(cname, str) and ('malware' in cname.lower() or 'malicious' in cname.lower()):
            malware_idx = idx
            break

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if len(class_names) == 2 and cm.shape == (2, 2):
        neg_idx = 1 - malware_idx
        tp = cm[malware_idx, malware_idx]
        fn = cm[malware_idx, neg_idx]
        fp = cm[neg_idx, malware_idx]
        tn = cm[neg_idx, neg_idx]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
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
        'detection_rate': float(recall) * 100.0,
        'fpr': (float(fpr) * 100.0) if fpr is not None else None,
        'fnr': (float(fnr) * 100.0) if fnr is not None else None,
        'auc': auc,
        'latency_ms_per_sample': float(latency_ms),
        'throughput_samples_per_sec': float(throughput)
    }


def evaluate_model(model, dataloader, criterion, device, class_names):
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


def train_model(model, train_loader, val_loader, test_loader=None, criterion=None, optimizer=None, num_epochs=50, device='cuda', config=None):
    """
    Experiment Logging: wandb recording
    - Model Structure/Hyperparameters (passed in config)
    - Data Preprocessing info (passed in config)
    - Training curves (loss, accuracy)
    - Confusion Matrix
    """
    
    if config is None:
        config = {}
        
    # Initialize wandb
    # Ensure you are logged in using `wandb login` in terminal before running
    # If wandb is already initialized (e.g. via wandb agent), this will just update config
    if wandb is not None:
        try:
            print("初始化 WandB (如有需要请在终端输入 API Key，或提前运行 wandb offline)...")
            wandb.init(project="intrusion-detection-traffic", config=config, resume="allow", mode="offline")
            print("WandB 初始化完成 (Offline模式).")
        except Exception as e:
            print(f"WandB init failed: {e}. Running without wandb.")
    else:
        print("WandB 未安装，跳过实验日志。")
    
    # Track gradients and model topology
    if wandb is not None:
        try:
            wandb.watch(model, criterion, log="all", log_freq=10)
        except:
            pass
    
    model.to(device)
    best_acc = 0.0
    best_metrics = {}
    best_state_dict = None
    
    class_names = config.get('class_names', [str(i) for i in range(10)])

    print("\n================== 开始训练循环 ==================")
    for epoch in range(num_epochs):
        print(f'\n▶ Epoch [{epoch+1}/{num_epochs}]')
        print('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Lists for Confusion Matrix
            all_preds = []
            all_labels = []
            all_probs = []
            inference_time_sec = 0.0

            # Iterate over data
            # Use tqdm for progress bar
            pbar = tqdm(dataloader, desc=f"{phase} Phase")
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=(device.startswith('cuda') if isinstance(device, str) else device.type == 'cuda'))
                labels = labels.to(device, non_blocking=(device.startswith('cuda') if isinstance(device, str) else device.type == 'cuda'))

                optimizer.zero_grad()

                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    infer_start = time.perf_counter()
                    outputs = model(inputs)
                    inference_time_sec += (time.perf_counter() - infer_start)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                if phase == 'val':
                    all_preds.extend(preds.view(-1).cpu().numpy())
                    all_labels.extend(labels.view(-1).cpu().numpy())
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # Enhanced output format for better visibility in terminal
            print(f'➜ [{phase.upper()}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}% | Samples: {total_samples}')

            # Log metrics to wandb
            # Training curves (loss, accuracy)
            if wandb is not None:
                try:
                    wandb.log({
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_accuracy": epoch_acc,
                        "epoch": epoch + 1
                    })
                except:
                    pass

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                y_true = np.array(all_labels)
                y_pred = np.array(all_preds)
                y_prob = np.array(all_probs)
                best_metrics = _compute_metrics(y_true, y_pred, y_prob, inference_time_sec, class_names)
                best_metrics['loss'] = float(epoch_loss)
                best_metrics['accuracy'] = float(epoch_acc) * 100.0
                best_metrics['val_loss'] = float(epoch_loss)
                best_metrics['best_accuracy'] = float(epoch_acc) * 100.0
                best_state_dict = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), 'best_model.pth')
            
            # Log Confusion Matrix at the end of validation epoch (or just final epoch)
            if phase == 'val' and wandb is not None:
                 # Compute Confusion Matrix Plot
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

    print('\n================== 训练结束 ==================')
    print(f'✅ Best Validation Accuracy: {best_acc*100:.2f}%')

    test_metrics = None
    if test_loader is not None and len(test_loader) > 0:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        test_metrics = evaluate_model(model, test_loader, criterion, device, class_names)
        if test_metrics is not None:
            print(f"✅ Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Finish wandb run
    if wandb is not None:
        try:
            wandb.finish()
        except:
            pass
    
    # ---------------- 新增：保存指标到本地 JSON ----------------
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

    all_metrics[model_name] = {
        "best_val_accuracy": f"{best_metrics.get('best_accuracy', float(best_acc) * 100.0):.2f}%",
        "best_val_loss": f"{best_metrics.get('val_loss', float(epoch_loss)):.4f}",
        "test_accuracy": f"{test_metrics['accuracy']:.2f}%" if test_metrics else None,
        "test_loss": f"{test_metrics['loss']:.4f}" if test_metrics else None,
        "best_accuracy": f"{test_metrics['accuracy']:.2f}%" if test_metrics else f"{best_metrics.get('best_accuracy', float(best_acc) * 100.0):.2f}%",
        "final_loss": f"{test_metrics['loss']:.4f}" if test_metrics else f"{best_metrics.get('val_loss', float(epoch_loss)):.4f}",
        "precision": f"{test_metrics['precision']:.2f}" if test_metrics and test_metrics.get('precision') is not None else (f"{best_metrics['precision']:.2f}" if best_metrics.get('precision') is not None else None),
        "recall": f"{test_metrics['recall']:.2f}" if test_metrics and test_metrics.get('recall') is not None else (f"{best_metrics['recall']:.2f}" if best_metrics.get('recall') is not None else None),
        "f1_score": f"{test_metrics['f1_score']:.2f}" if test_metrics and test_metrics.get('f1_score') is not None else (f"{best_metrics['f1_score']:.2f}" if best_metrics.get('f1_score') is not None else None),
        "f2_score": f"{test_metrics['f2_score']:.2f}" if test_metrics and test_metrics.get('f2_score') is not None else (f"{best_metrics['f2_score']:.2f}" if best_metrics.get('f2_score') is not None else None),
        "detection_rate": f"{test_metrics['detection_rate']:.2f}" if test_metrics and test_metrics.get('detection_rate') is not None else (f"{best_metrics['detection_rate']:.2f}" if best_metrics.get('detection_rate') is not None else None),
        "fpr": f"{test_metrics['fpr']:.2f}" if test_metrics and test_metrics.get('fpr') is not None else (f"{best_metrics['fpr']:.2f}" if best_metrics.get('fpr') is not None else None),
        "fnr": f"{test_metrics['fnr']:.2f}" if test_metrics and test_metrics.get('fnr') is not None else (f"{best_metrics['fnr']:.2f}" if best_metrics.get('fnr') is not None else None),
        "auc": f"{test_metrics['auc']:.4f}" if test_metrics and test_metrics.get('auc') is not None else (f"{best_metrics['auc']:.4f}" if best_metrics.get('auc') is not None else None),
        "latency_ms_per_sample": f"{test_metrics['latency_ms_per_sample']:.4f}" if test_metrics and test_metrics.get('latency_ms_per_sample') is not None else (f"{best_metrics['latency_ms_per_sample']:.4f}" if best_metrics.get('latency_ms_per_sample') is not None else None),
        "throughput_samples_per_sec": f"{test_metrics['throughput_samples_per_sec']:.2f}" if test_metrics and test_metrics.get('throughput_samples_per_sec') is not None else (f"{best_metrics['throughput_samples_per_sec']:.2f}" if best_metrics.get('throughput_samples_per_sec') is not None else None),
        "epochs_trained": num_epochs,
        "classes": class_names
    }

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    print(f"[{model_name}] 评估指标已保存至: {metrics_file}")
    # ---------------- 新增结束 ----------------
    
    return model
