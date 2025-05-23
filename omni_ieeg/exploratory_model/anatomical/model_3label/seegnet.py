import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, mid_channels=30):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y + x


class MultiscaleConvBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.branch1 = nn.Sequential(  # ~0.25s (low freq)
            nn.Conv1d(in_channels, 64, kernel_size=250, stride=20),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=7, stride=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(750)
        )

        self.branch2 = nn.Sequential(  # ~0.08s (mid freq)
            nn.Conv1d(in_channels, 64, kernel_size=80, stride=10),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=7, stride=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(750)
        )

        self.branch3 = nn.Sequential(  # ~0.01s (high freq)
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=8, stride=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(750)
        )

        self.dropout = nn.Dropout(0.5)
        self.se_block = ResidualSEBlock(128 * 3, mid_channels=30)

    def forward(self, x):
        b1, b2, b3 = self.branch1(x), self.branch2(x), self.branch3(x)
        min_len = min(b1.shape[-1], b2.shape[-1], b3.shape[-1])
        b1, b2, b3 = b1[..., :min_len], b2[..., :min_len], b3[..., :min_len]
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.dropout(out)
        return self.se_block(out)  # shape: (B, 384, T)


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.attention_w = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_u = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)  # (B, T, 2*H)

        # Attention mechanism
        u = torch.tanh(self.attention_w(lstm_out))         # (B, T, 2H)
        attn_scores = self.attention_u(u).squeeze(-1)      # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, T)
        attended = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (B, 2H)
        return attended


class SEEGNetBinary(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_extractor = MultiscaleConvBlock(in_channels=in_channels)
        self.temporal_model = BiLSTMWithAttention(input_dim=384, hidden_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Binary classification, return logits
        )

    def forward(self, x):
        x = x[:, None, :]
        x = self.feature_extractor(x)   # (B, 384, T')
        x = self.temporal_model(x)      # (B, 256)
        logits = self.classifier(x)     # (B, 1)
        return logits     # Return (B,1) logits
class SEEGNetPreProcessing():
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data

if __name__ == "__main__":
    # Create dummy input: batch of 2 samples, 1 channel, 60 seconds at 1000 Hz
    dummy_input = torch.randn(2, 1, 60000)

    # Instantiate the model
    model = SEEGNetBinary(in_channels=1)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    print("Logits shape:", logits.shape)
    print("Logits:", logits)
