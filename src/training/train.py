import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm

# TODO (石凌云): 导入你可以在 models 模块中定义的模型
# from src.models.cnn_bilstm import CNN_BiLSTM
# from src.models.transformer import TrafficTransformer

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    TODO (石凌云): 
    实现完整的训练循环，包括：
    1. 前向传播
    2. 计算 Loss
    3. 反向传播
    4. 验证集评估
    5. 保存最佳模型
    """
    model.to(device)
    best_acc = 0.0

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

            # Iterate over data
            # for inputs, labels in tqdm(dataloader):
            #     inputs = inputs.to(device)
            #     labels = labels.to(device)

            #     optimizer.zero_grad()

            #     with torch.set_grad_enabled(phase == 'train'):
            #         outputs = model(inputs)
            #         _, preds = torch.max(outputs, 1)
            #         loss = criterion(outputs, labels)

            #         if phase == 'train':
            #             loss.backward()
            #             optimizer.step()

            #     running_loss += loss.item() * inputs.size(0)
            #     running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch statistics
            # ...
            
        # Save best model
        # if phase == 'val' and epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     torch.save(model.state_dict(), 'best_model.pth')

    return model

if __name__ == "__main__":
    # 示例调用
    # 1. Load Config (Hyperparameters)
    # 2. Prepare DataLoaders (call src.preprocessing.data_loader)
    # 3. Initialize Model, Criterion, Optimizer
    # 4. Start Training
    pass
