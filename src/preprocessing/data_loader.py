import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class TrafficDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None):
        """
        TODO (徐麟翔): 
        1. 初始化数据集路径
        2. 读取数据索引 (例如读取 csv 文件或遍历目录)
        3. 区分 'train', 'val', 'test' 模式
        """
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.samples = [] # 存储 (image_path, label)
        
        # 示例：遍历 benign 和 malware 文件夹
        # feature_dir = os.path.join(data_root, mode, 'Png')
        # self._load_data(feature_dir)
        pass

    def _load_data(self, dir_path):
        # TODO: 实现读取逻辑
        pass

    def __len__(self):
        # return len(self.samples)
        return 0 # Placeholder

    def __getitem__(self, idx):
        """
        TODO (徐麟翔):
        1. 获取 image path 和 label
        2. 使用 PIL 读取图片
        3. 应用 transform (数据增强)
        4. 返回 tensor 和 label
        """
        # img_path, label = self.samples[idx]
        # img = Image.open(img_path).convert('L') # 转灰度
        # if self.transform:
        #     img = self.transform(img)
        # return img, torch.tensor(label, dtype=torch.long)
        return torch.randn(1, 28, 28), torch.tensor(0) # Placeholder

def get_dataloaders(data_root, batch_size=32):
    """
    TODO (徐麟翔): 
    定义数据增强 (Transforms) 并返回 train/val/test 的 DataLoader
    """
    # transform_train = transforms.Compose([
    #     transforms.Resize((28, 28)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
    
    # train_dataset = TrafficDataset(data_root, mode='train', transform=transform_train)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # return train_loader, val_loader, test_loader
    return None, None, None
