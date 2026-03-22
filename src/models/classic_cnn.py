import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicCNN(nn.Module):
    """
    A classic CNN structure adapted for 28x28 grayscale images (traffic sessions).
    Structure similar to LeNet/MNIST models:
    Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool -> Dropout -> Flatten -> FC -> FC
    """
    def __init__(self, num_classes=2):
        super(ClassicCNN, self).__init__()
        # Input: 1 channel, 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # 28 - 3 + 1 = 26 -> (32, 26, 26)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # 26 - 3 + 1 = 24 -> (64, 24, 24)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 24 / 2 = 12 -> (64, 12, 12)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1) # Flatten all dims except batch
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
