from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from omni_ieeg.event_model.train.model.cnn import compute_spectrum_batch, normalize_img

class ASTClassifier(nn.Module):
    def __init__(self, num_class=5):
        super().__init__()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-16-16-0.442")
        self.feature_extractor.sampling_rate = 1000   
        # self.feature_extractor.max_length = 60 * 100  # from paper 100t 
        self.feature_extractor.mean = 0.0                   
        self.feature_extractor.std = 0.5                    
        self.feature_extractor.return_attention_mask = False
        print(self.feature_extractor)

        # patch_height = 32
        # patch_width = 600
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=1,           # = 1
        #         out_channels=128,               # = patch embedding dim
        #         kernel_size=(14, 1000),
        #         stride=(14, 936)
        #     ),
        #     Rearrange('b d h w -> b (h w) d'),  # flatten patch grid into sequence
        #     nn.LayerNorm(128)
        # )
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-16-16-0.442", 
            num_labels=1, 
            attn_implementation="sdpa",
            ignore_mismatched_sizes=True,
        )


    def forward(self, inputs):
        inputs = inputs.detach().cpu().numpy()
        inputs = self.feature_extractor(inputs, sampling_rate=1000, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # inputs = self.feature_extractor(inputs)
        # outputs = self.model(input_values=inputs)
        logits = outputs.logits
        return logits

# class NeuralASTPreProcessing():
#     def __init__(self):
#         pass
    
#     def __call__(self, data):
#         return data
    

class NeuralASTPreProcessing():
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
        # data = compute_spectrum_batch(data, self.frequency, self.image_size, self.freq_range[0], self.freq_range[1])
        # data = self._cropping(data)
        # data = normalize_img(data)
        return data