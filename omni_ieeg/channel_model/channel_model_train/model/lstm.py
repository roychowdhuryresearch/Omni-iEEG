import torch
import torch.nn as nn
# Data utils and dataloader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn

class AttentiveBiLSTM(nn.Module):
    """
    Bidirectional LSTM ➜ additive attention ➜ dense head.
    """
    def __init__(self,
                 seq_len: int = 2000,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 n_classes: int = 1):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=5),  # (B, 1, 60000) → (B, 32, 12000)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=5), # (B, 64, 12000) → (B, 64, 2400)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4), # (B, 128, 2400) → (B, 128, ~599)
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=128,                
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention parameters (additive / Bahdanau)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

        # Replace original self.classifier sequential block
        dropout_val = dropout # Use dropout value from init args
        input_clf_size = 2 * hidden_size
        clf_hidden1 = 64 # Intermediate dimension 1 (reduced from 128 in attn)
        clf_hidden2 = 32 # Intermediate dimension 2

        self.dropout1 = nn.Dropout(dropout_val)
        self.fc1 = nn.Linear(input_clf_size, clf_hidden1)
        self.bn1 = nn.BatchNorm1d(clf_hidden1)
        self.relu1 = nn.LeakyReLU() # Use LeakyReLU like CNN
        
        self.dropout2 = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(clf_hidden1, clf_hidden2)
        self.bn2 = nn.BatchNorm1d(clf_hidden2)
        self.relu2 = nn.LeakyReLU()

        self.fc_out = nn.Linear(clf_hidden2, n_classes)
        # self.final_ac = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len)  – raw signal
        """
        # (batch, seq_len, 1)
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)

        # LSTM output: (batch, seq_len, 2*hidden)
        lstm_out, _ = self.lstm(x)

        # Attention weights: (batch, seq_len, 1) → (batch, seq_len)
        attn_weights = self.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Context vector: weighted sum over time
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        # Apply new classifier head
        x = self.dropout1(context)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        logits = self.fc_out(x)
        # logits = self.final_ac(logits)
        # Return raw logits
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