import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal pe for N × D tensors."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len]

class PatchTransformer(nn.Module):
    """
    PatchTST‑like encoder for univariate signals.
    * sequence is split into non‑overlapping patches,
      each linearly projected to d_model dims
    """
    def __init__(self,
                 seq_len: int = 2000,
                 patch_len: int = 50,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 dim_feedforward: int = 128,
                 dropout: float = 0.2,
                 n_classes: int = 1):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.num_patches = seq_len // patch_len

        # 1 ➜ d_model
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Accommodate CLS token in positional encoding length
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Classifier head adapted from CNN structure
        dropout_val = dropout # Use dropout value from init args
        input_clf_size = d_model
        clf_hidden1 = 64 # Intermediate dimension 1
        clf_hidden2 = 32 # Intermediate dimension 2
        
        # Replace original self.fc sequential block
        self.dropout1 = nn.Dropout(dropout_val) # Adding dropout consistent with LSTM changes
        self.fc1 = nn.Linear(input_clf_size, clf_hidden1)
        self.bn1 = nn.BatchNorm1d(clf_hidden1) # Use BatchNorm1d instead of LayerNorm
        self.relu1 = nn.LeakyReLU()
        
        self.dropout2 = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(clf_hidden1, clf_hidden2)
        self.bn2 = nn.BatchNorm1d(clf_hidden2)
        self.relu2 = nn.LeakyReLU()

        self.fc_out = nn.Linear(clf_hidden2, n_classes)
        # self.final_ac = nn.Sigmoid()
        # weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        x: (batch, seq_len) – raw signal
        """
        B, L = x.shape
        # (batch, num_patches, patch_len)
        patches = x.reshape(B, self.num_patches, -1)
        # project
        patches = self.patch_proj(patches)         # (B, N, d_model)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 1 + N, d_model)

        x = self.pos_encoder(x)
        x = self.transformer(x)

        # take CLS embedding
        cls_emb = x[:, 0]
        
        # Apply new classifier head
        x = self.dropout1(cls_emb)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        logits = self.fc_out(x)
        # logits = self.final_ac(logits)
        return logits



class DummyPreProcessing():
    def __init__(self, event_length, frequency, selected_window_size_ms, random_shift_ms):
        self.event_length = event_length
        self.frequency = frequency
        self.selected_window_size_ms = selected_window_size_ms
        self.random_shift_ms = random_shift_ms
    def enable_random_shift(self):
        self.random_shift = self.random_shift_ms != 0
    def disable_random_shift(self):
        self.random_shift = False
    def __call__(self, data):
        time_crop_index = [self.event_length//2 - self.selected_window_size_ms * self.frequency//1000, self.event_length//2 + self.selected_window_size_ms * self.frequency//1000]
        time_crop_index = np.array(time_crop_index)
        if self.random_shift:
            shift_index = self.random_shift * self.frequency//1000
            shift = np.random.randint(-shift_index, shift_index)
            time_crop_index += shift
        data = data[:,time_crop_index[0]:time_crop_index[1]]
        return data
