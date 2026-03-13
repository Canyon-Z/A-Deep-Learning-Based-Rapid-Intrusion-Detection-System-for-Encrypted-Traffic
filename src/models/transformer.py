import torch
import torch.nn as nn

class TrafficTransformer(nn.Module):
    """
    TODO (石凌云): 调整 d_model 和 nhead。注意 d_model 必须能被 nhead 整除。
    尝试不同的 dropout 值以防止过拟合。
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(TrafficTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) # Max seq len 1000
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer expects (seq_len, batch_size, d_model) by default usually, but with batch_first=False
        # PyTorch default is batch_first=False. Let's start with (seq_len, batch, feature)
        x = x.permute(1, 0, 2)
        
        output = self.transformer_encoder(x)
        
        # Take the mean or the first token (CLS equivalent)
        # Using mean pooling here for simplicity
        output = output.mean(dim=0)
        
        output = self.classifier(output)
        return output
