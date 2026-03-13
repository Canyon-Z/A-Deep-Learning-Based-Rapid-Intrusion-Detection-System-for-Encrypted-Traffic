import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    """
    TODO (石凌云): 可以在此调整 CNN 核大小 (kernel_size) 和 LSTM 层数 (num_layers) 以优化性能。
    注意输入维度 (input_dim) 必须与 src/preprocessing/feature_extraction.py 的输出一致。
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNN_BiLSTM, self).__init__()
        # CNN layers for spatial features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # BiLSTM layers for temporal dependency
        # Input features for LSTM depends on CNN output size
        # Assuming input length allows, adjust input_size accordingly
        self.lstm_input_size = 64 # This needs to be calculated based on input sequence length after pooling
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features) -> needs modification for Conv1d
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1) 
        
        x = self.cnn(x)
        
        # Prepare for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
