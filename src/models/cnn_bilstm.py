import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    """
    CNN + BiLSTM Model for Intrusion Detection.
    Input: (Batch, 1, 28, 28) - Automatically flattens to (Batch, 1, 784)
    """
    def __init__(self, num_classes, hidden_dim=128):
        super(CNN_BiLSTM, self).__init__()
        # CNN layers for spatial/local features extraction
        # Input: (Batch, 1, 784)
        self.cnn = nn.Sequential(
            # Conv1: (Batch, 16, 784)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            # MaxPool1: (Batch, 16, 392)
            nn.MaxPool1d(kernel_size=2),
            
            # Conv2: (Batch, 32, 392)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            # MaxPool2: (Batch, 32, 196)
            nn.MaxPool1d(kernel_size=2)
        )
        
        # BiLSTM layers for temporal dependency
        # Input features for LSTM = CNN Output Channels = 32
        self.lstm_input_size = 32 
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # Dropout layer to prevent overfitting and reduce false positive rate
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer
        # BiLSTM output = hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape from DataLoader: (Batch, 1, 28, 28)
        # Flatten to (Batch, 1, 784) for 1D CNN processing
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1) 
        
        # CNN Phase
        # Input to Conv1d: (Batch, Channels=1, Length=784)
        x = self.cnn(x) 
        # Output from CNN: (Batch, 32, 196)
        
        # Prepare for LSTM: (Batch, SeqLen=196, Features=32)
        x = x.permute(0, 2, 1)
        
        # LSTM Phase
        # out: (Batch, SeqLen, HiddenDim*2)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step for classification
        out = out[:, -1, :]
        
        # Apply Dropout
        out = self.dropout(out)
        
        # Classification
        out = self.fc(out)
        return out
