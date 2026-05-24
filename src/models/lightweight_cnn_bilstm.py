import torch
import torch.nn as nn

class Lightweight_CNN_BiLSTM(nn.Module):
    """
    Lightweight CNN + BiLSTM Model for Intrusion Detection.
    Reduced parameters compared to the standard CNN_BiLSTM to speed up inference and lower memory usage.
    Input: (Batch, 1, 28, 28) - Automatically flattens to (Batch, 1, 784)
    """
    def __init__(self, num_classes, hidden_dim=32):
        super(Lightweight_CNN_BiLSTM, self).__init__()
        # CNN layers for spatial/local features extraction
        # Input: (Batch, 1, 784)
        self.cnn = nn.Sequential(
            # Conv1: (Batch, 8, 784) - Reduced channels
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            # MaxPool1: (Batch, 8, 392)
            nn.MaxPool1d(kernel_size=2),
            
            # Conv2: (Batch, 16, 392) - Reduced channels
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            # MaxPool2: (Batch, 16, 196)
            nn.MaxPool1d(kernel_size=2)
        )
        
        # BiLSTM layers for temporal dependency
        # Input features for LSTM = CNN Output Channels = 16
        self.lstm_input_size = 16 
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # Dropout layer (increased to balance recall and reduce FPR)
        self.dropout = nn.Dropout(0.4)
        
        # Fully connected layer
        # BiLSTM output = hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape from DataLoader: (Batch, 1, 28, 28)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1) 
        
        # CNN Phase -> Output: (Batch, 16, 196)
        x = self.cnn(x) 
        
        # Prepare for LSTM: (Batch, SeqLen=196, Features=16)
        x = x.permute(0, 2, 1)
        
        # LSTM Phase -> out: (Batch, SeqLen, HiddenDim*2)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step for classification
        out = out[:, -1, :]
        
        # Apply Dropout and Classification
        out = self.dropout(out)
        out = self.fc(out)
        return out
