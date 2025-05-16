import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification

class PreCompressPatchTST(nn.Module):
    def __init__(self, seq_len=5 * 60 * 200, target_length=1024, num_class=5):
        super().__init__()
        self.target_length = target_length
        self.original_length = seq_len
        # self.compress = nn.Conv1d(
        #     in_channels=1,
        #     out_channels=1,        # keep channel dimension
        #     kernel_size=25,
        #     stride=25              # 60000 → 1200
        # )

        config = PatchTSTConfig(
            patch_len=12,
            stride=12,
            context_length=6000,        # new sequence length
            num_input_channels=1,         # your signal has 1 channel
            num_targets=num_class,
            hidden_size=128,        # you can increase this for a deeper model
            num_attention_heads=4,
            num_hidden_layers=3,
            use_cls_token=True,
        )
        self.model = PatchTSTForClassification(config)

    def forward(self, x):
        # x shape: (B, num_input_channels, input_size)
        x = x[:, None, :]
        # x = self.compress(x).permute(0, 2, 1)  # → (B, target_length, input_size)
        x = x.permute(0, 2, 1)  # → (B, target_length, input_size)
        x = self.model(x)
        return x.prediction_logits
    

class NeuralPatchTSTPreProcessing():
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data