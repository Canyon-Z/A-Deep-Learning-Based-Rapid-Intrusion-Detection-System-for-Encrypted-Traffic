import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicCNN(nn.Module):
    """
    A classic CNN structure adapted for network traffic payloads.
    Referencing CNN_BiLSTM, the 28x28 input is flattened back to a 1D sequence of length 784.
    This model uses pure 1D Convolutions without LSTM, acting as a strong baseline.
    """
    def __init__(self, num_classes=2):
        super(ClassicCNN, self).__init__()
        
        # CNN layers for feature extraction from 1D byte sequence
        # Input: (Batch, 1, 784)
        self.cnn = nn.Sequential(
            # Block 1: (Batch, 32, 392)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1), # Changed ReLU to LeakyReLU for better generalization
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2: (Batch, 64, 196)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3: (Batch, 128, 98)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Fully Connected Layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), # Increased dropout to balance recall and FPR
            nn.Linear(128 * 98, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6), # Increased dropout to balance recall and FPR
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Reference cnn_bilstm.py: x shape from DataLoader is (Batch, 1, 28, 28)
        # We flatten it to (Batch, 1, 784) for 1D CNN processing
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1) 
        
        # Pass through CNN Feature Extractor
        x = self.cnn(x)
        
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1) 
        
        # Classification
        out = self.classifier(x)
        
        return out
