import torch
import torch.nn as nn
# Data utils and dataloader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import pandas as pd
import math

class NeuralCNN(torch.nn.Module):
    def __init__(self, in_channels, outputs, freeze = False, channel_selection = True):
        super(NeuralCNN, self).__init__()
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.in_channels = in_channels
        self.outputs = outputs
        self.channel_selection = channel_selection
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.conv1= nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.cnn.fc = nn.Sequential(nn.Linear(512, 32))
        for param in self.cnn.fc.parameters():
            param.requires_grad = not freeze
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.LeakyReLU()
        self.fc = nn.Linear(32,32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()
        
        self.fc_out = nn.Linear(16, self.outputs)
        # Remove sigmoid activation for multi-class classification
        # For multi-class with CrossEntropyLoss, we need raw logits without activation
        # CrossEntropyLoss internally applies softmax to the logits
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        batch = self.cnn(x)
        batch = self.bn(self.relu(self.fc(batch)))
        batch = self.bn1(self.relu1(self.fc1(batch)))
        # Return raw logits without sigmoid activation
        batch = self.fc_out(batch)
        return batch



class NeuralCNNPreProcessing():
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
        # self.check_bound(self.crop_time_index_r[0], "crop_time")
        # self.check_bound(self.crop_time_index_r[1], "crop_time")
        self.crop_index_w = np.abs(self.crop_time_index_r[0]- self.crop_time_index_r[1])
        self.crop_index_h = np.abs(self.crop_freq_index[0]- self.crop_freq_index[1])
    
    def enable_random_shift(self):
        self.random_shift = self.random_shift_time != 0
    
    def disable_random_shift(self):
        self.random_shift = False


    def _cropping(self, data):
        time_crop_index = [self.event_length//2 - self.crop_time * self.frequency//1000, self.event_length//2 + self.crop_time * self.frequency//1000]
        # time_crop_index = [self.event_length//2 - self.crop_time, self.event_length//2 + self.crop_time]
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
        # data = data[:,self.crop_freq_index[0]:self.crop_freq_index[1], :]
        # shape is 128, 224, 570

        # Reshape data from (128, 224, 570) to (128, 224, 224)
        # Add a channel dimension for interpolation
        data = data.unsqueeze(1)  # Shape: (128, 1, 224, 570)
        last_shape = data.shape[3]
        # print(f"last_shape: {last_shape}")
        data = torch.nn.functional.interpolate(data, size=(224, int(last_shape * 0.5)), mode='bilinear', align_corners=False)
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

    
def compute_spectrum_batch(org_sigs, ps_SampleRate=2000, ps_FreqSeg=512, ps_MinFreqHz=10, ps_MaxFreqHz=500, device='cuda'):
    # Ensure device compatibility
    org_sigs_device = org_sigs.device if org_sigs.is_cuda else torch.device(device)
    
    # Adjust device for computations
    org_sigs = org_sigs.to(org_sigs_device)
    
    batch_size, sig_len = org_sigs.shape
    ii, jj = int(sig_len // 2), int(sig_len // 2 + sig_len)

    # Create extended signals directly on the correct device
    extend_sigs = create_extended_sig_batch(org_sigs)
    ps_StDevCycles = 3
    s_Len = extend_sigs.shape[1]
    s_HalfLen = math.floor(s_Len / 2) + 1

    # Define frequency and window axes on the correct device
    v_WAxis = (torch.linspace(0, 2 * np.pi, s_Len, device=org_sigs_device)[:-1] * ps_SampleRate).float()
    v_WAxisHalf = v_WAxis[:s_HalfLen].repeat(ps_FreqSeg, 1)
    v_FreqAxis = torch.linspace(ps_MaxFreqHz, ps_MinFreqHz, steps=ps_FreqSeg, device=org_sigs_device).float()

    # Initialize FFT window matrix on the correct device
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=org_sigs_device).float()
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5 * (v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)))**2 * (s_StDevSec**2).view(-1, 1))
    
    # Normalize the FFT windows
    v_WinFFT = v_WinFFT * math.sqrt(s_Len) / torch.norm(v_WinFFT, dim=-1).view(-1, 1)

    # Perform FFT on the extended signals and apply windowed filters
    v_InputSignalFFT = torch.fft.fft(extend_sigs, dim=1)
    
    # Corrected reshaping to align dimensions
    res = torch.fft.ifft(v_InputSignalFFT.unsqueeze(1) * v_WinFFT.unsqueeze(0), dim=2)[:, :, ii:jj]
    res = res / torch.sqrt(s_StDevSec).view(1, -1, 1)

    # Return the magnitude, ensuring the result is moved to CPU if required
    res = res.abs()
    # Ensure the function always returns a tensor
    return res

def create_extended_sig_batch(sigs):
    batch_size, s_len = sigs.shape
    s_halflen = int(np.ceil(s_len / 2)) + 1

    # Compute start and end windows for each signal in the batch
    start_win = sigs[:, :s_halflen] - sigs[:, [0]]
    end_win = sigs[:, s_len - s_halflen - 1:] - sigs[:, [-1]]

    start_win = -start_win.flip(dims=[1]) + sigs[:, [0]]
    end_win = -end_win.flip(dims=[1]) + sigs[:, [-1]]

    # Concatenate to form the final extended signals
    final_sigs = torch.cat((start_win[:, :-1], sigs, end_win[:, 1:]), dim=1)

    # Ensure the final signals have an odd length
    if final_sigs.shape[1] % 2 == 0:
        final_sigs = final_sigs[:, :-1]

    return final_sigs
def normalize_img(a):
    batch_num = a.shape[0]
    c = a.shape[1]
    h = a.shape[2]
    w = a.shape[3]
    a_reshape = a.reshape(batch_num * c, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    normalized = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    normalized = normalized.reshape(batch_num,c, h, w)
    return normalized