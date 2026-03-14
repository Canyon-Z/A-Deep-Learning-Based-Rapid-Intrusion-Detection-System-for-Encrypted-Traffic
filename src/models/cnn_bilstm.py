import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    """
    TODO (石凌云): 可以在此调整 CNN 核大小 (kernel_size) 和 LSTM 层数 (num_layers) 以优化性能。
    注意输入维度 (input_dim) 必须与 src/preprocessing/feature_extraction.py 的输出一致。
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_lstm_layers=1, dropout=0.1):
        super(CNN_BiLSTM, self).__init__()
        # Input: (batch, seq_len, input_dim)
        # Conv1d expects (batch, channels, seq_len), channels = input_dim.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM input size equals CNN output channels.
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1) 
        
        x = self.cnn(x)
        
        # Prepare for LSTM: (batch_size, seq_len_after_pooling, 64)
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
