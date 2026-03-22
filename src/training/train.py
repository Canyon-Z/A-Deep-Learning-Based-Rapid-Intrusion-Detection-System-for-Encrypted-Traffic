import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# from src.models.cnn_bilstm import CNN_BiLSTM 

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', config=None):
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
    try:
        print("初始化 WandB (如有需要请在终端输入 API Key，或提前运行 wandb offline)...")
        wandb.init(project="intrusion-detection-traffic", config=config, resume="allow", mode="offline") 
        print("WandB 初始化完成 (Offline模式).")
    except Exception as e:
        print(f"WandB init failed: {e}. Running without wandb.")
    
    # Track gradients and model topology
    try:
        wandb.watch(model, criterion, log="all", log_freq=10)
    except:
        pass
    
    model.to(device)
    best_acc = 0.0
    
    class_names = config.get('class_names', [str(i) for i in range(10)])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

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

            # Iterate over data
            # Use tqdm for progress bar
            pbar = tqdm(dataloader, desc=f"{phase} Phase")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to wandb
            # Training curves (loss, accuracy)
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
                # torch.save(model.state_dict(), 'best_model.pth')
            
            # Log Confusion Matrix at the end of validation epoch (or just final epoch)
            if phase == 'val':
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

    print(f'Best val Acc: {best_acc:4f}')
    
    # Finish wandb run
    try:
        wandb.finish()
    except:
        pass
    
    return model
