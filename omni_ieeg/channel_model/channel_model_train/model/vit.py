
 
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
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import numpy as np
import pandas as pd
import math
from omni_ieeg.event_model.train.model.cnn import compute_spectrum_batch, normalize_img


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,           # = 1
                out_channels=dim,               # = patch embedding dim
                kernel_size=(patch_height, patch_width),
                stride=(patch_height, patch_width)
            ),
            Rearrange('b d h w -> b (h w) d'),  # flatten patch grid into sequence
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
class NeuralViT(torch.nn.Module):
    def __init__(self, in_channels, outputs, freeze=False, channel_selection=True):
        super(NeuralViT, self).__init__()
        self.in_channels = in_channels
        self.outputs = outputs
        self.channel_selection = channel_selection
        
        # Load pre-trained ViT model
        # self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        patch_height = 32
        patch_width = 500
        self.vit = ViT(
            image_size=(224, 60000),
            patch_size=(patch_height, patch_width),
            num_classes=1,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            pool='cls',
            channels=1,
            dim_head=64,
        )
        
        # # Modify the patch embedding to accept different input channels
        # if in_channels != 3:
        #     # Create new patch embedding with correct number of input channels
        #     new_conv = nn.Conv2d(
        #         in_channels, 
        #         self.vit.conv_proj.out_channels,
        #         kernel_size=self.vit.conv_proj.kernel_size,
        #         stride=self.vit.conv_proj.stride,
        #         padding=self.vit.conv_proj.padding
        #     )
            
        #     # Transfer weights for the available channels
        #     with torch.no_grad():
        #         if in_channels < 3:
        #             # For fewer channels, just use the first few channels of original weights
        #             new_conv.weight.data = self.vit.conv_proj.weight.data[:, :in_channels]
        #         else:
        #             # For more channels, copy existing and initialize new ones
        #             new_conv.weight.data[:, :3] = self.vit.conv_proj.weight.data
        #             # Initialize remaining channels
        #             nn.init.normal_(new_conv.weight.data[:, 3:], std=0.01)
                
        #         # Copy bias if it exists
        #         if self.vit.conv_proj.bias is not None:
        #             new_conv.bias.data = self.vit.conv_proj.bias.data
            
        #     # Replace the original projection
        #     self.vit.conv_proj = new_conv
        
        # # Replace the final classification head
        # self.vit.heads = nn.Identity()  # Remove the classification head
        
        # Create feature layers mirroring NeuralCNN architecture
        self.fc = nn.Linear(768, 32)  # ViT outputs 768-dim features
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()
        
        self.fc_out = nn.Linear(16, self.outputs)
        # self.final_ac = nn.Sigmoid()
        
        # Freeze ViT if requested
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        """
        Forward pass takes input x and returns the model output.
        """
        # ViT forward pass
        batch = self.vit(x)  # This gives us the CLS token as features
        
        # Apply the same processing chain as in NeuralCNN
        # batch = self.bn(self.relu(self.fc(batch)))
        # batch = self.bn1(self.relu1(self.fc1(batch)))
        # batch = self.final_ac(self.fc_out(batch))
        
        # batch = self.final_ac(batch)
        return batch



class NeuralViTPreProcessing():
    def __init__(self, image_size, frequency, freq_range_hz, event_length, selected_window_size_ms, selected_freq_range_hz,
                    random_shift_ms):

        # original data parameter
        self.image_size = image_size
        self.freq_range = freq_range_hz # in HZ
        self.frequency = frequency # in HZ
        self.event_length = event_length # in ms

        # cropped data parameter
        self.crop_time = selected_window_size_ms # in ms
        self.crop_freq = selected_freq_range_hz # in HZ
        self.random_shift_time = random_shift_ms # in ms

        self.initialize()

    def initialize(self):
        self.freq_range_low = self.freq_range[0]   # in HZ
        self.freq_range_high = self.freq_range[1]  # in HZ
        self.time_range = [0, self.event_length] # in ms
        self.crop_range_index = self.crop_time / self.event_length * self.image_size # in index
        self.crop_freq_low = self.crop_freq[0] # in HZ
        self.crop_freq_high = self.crop_freq[1] # in HZ
        self.crop = self.freq_range_low == self.crop_freq_low and self.freq_range_high == self.crop_freq_high and self.crop_time*2 == self.event_length
        self.calculate_crop_index()
        self.random_shift_index = int(self.random_shift_time*(self.image_size/self.event_length)) # in index
        self.random_shift = self.random_shift_time != 0
    
    def check_bound(self, x, text):
        if x < 0 or x > self.image_size:
            raise AssertionError(f"Index out of bound on {text}")
        return True

    def calculate_crop_index(self):
        # calculate the index of the crop, high_freq is low index
        self.crop_freq_index_low = self.image_size - self.image_size / (self.freq_range_high - self.freq_range_low) * (self.crop_freq_low - self.freq_range_low)
        self.crop_freq_index_high = self.image_size - self.image_size / (self.freq_range_high - self.freq_range_low) * (self.crop_freq_high - self.freq_range_low)  
        self.crop_freq_index = np.array([self.crop_freq_index_high, self.crop_freq_index_low]).astype(int) # in index
        self.crop_time_index = np.array([-self.crop_range_index, self.crop_range_index]).astype(int) # in index
        self.crop_time_index_r = self.image_size//2 + self.crop_time_index # in index
        #print("crop freq: ", self.crop_freq, "crop time: ", self.crop_time, "crop freq index: ", self.crop_freq_index, "crop time index: ", self.crop_time_index_r, self.crop_time_index)
        self.check_bound(self.crop_freq_index_low, "selected_freq_range_hz_low")
        self.check_bound(self.crop_freq_index_high, "selected_freq_range_hz_high")
        self.check_bound(self.crop_time_index_r[0], "crop_time")
        self.check_bound(self.crop_time_index_r[1], "crop_time")
        self.crop_index_w = np.abs(self.crop_time_index_r[0]- self.crop_time_index_r[1])
        self.crop_index_h = np.abs(self.crop_freq_index[0]- self.crop_freq_index[1])
    
    def enable_random_shift(self):
        self.random_shift = self.random_shift_time != 0
    
    def disable_random_shift(self):
        self.random_shift = False


    def _cropping(self, data):
        time_crop_index = [self.event_length//2 - self.crop_time * self.frequency//1000, self.event_length//2 + self.crop_time * self.frequency//1000]
        time_crop_index = np.array(time_crop_index)
        if self.random_shift:
            shift_index = self.random_shift * self.frequency//1000
            shift = np.random.randint(-shift_index, shift_index)
            time_crop_index += shift
        # self.crop_freq_index[0] within [0, self.image_size]
        # self.crop_freq_index[1] within [0, self.image_size]
        # time_crop_index[0] within [0, self.image_size]
        # time_crop_index[1] within [0, self.image_size]
        self.crop_freq_index[0] = min(max(0, self.crop_freq_index[0]), self.image_size)
        self.crop_freq_index[1] = min(max(0, self.crop_freq_index[1]), self.image_size)
        # time_crop_index[0] = min(max(0, time_crop_index[0]), self.image_size)
        # time_crop_index[1] = min(max(0, time_crop_index[1]), self.image_size)
        data = data[:,self.crop_freq_index[0]:self.crop_freq_index[1] , time_crop_index[0]:time_crop_index[1]]
        # shape is 128, 224, 570

        # Reshape data from (128, 224, 570) to (128, 224, 224)
        # Add a channel dimension for interpolation
        data = data.unsqueeze(1)  # Shape: (128, 1, 224, 570)
        # data = torch.nn.functional.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert to float32 to match the model's weight type
        data = data.to(torch.float32)
        
        return data

    def __call__(self, data):
        # compute the spectrum of the data
        data = compute_spectrum_batch(data, self.frequency, self.image_size, self.freq_range[0], self.freq_range[1])
        data = self._cropping(data)
        data = normalize_img(data)
        return data
