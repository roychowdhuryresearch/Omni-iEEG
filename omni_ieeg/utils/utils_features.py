import os
import math
from multiprocessing import Process, Pool, get_context

import numpy as np
import pandas as pd
import torch
import mne
from scipy.interpolate import interp1d
import scipy.linalg as LA
from skimage.transform import resize
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from p_tqdm import p_map
from HFODetector import ste, mni

def generate_feature_from_df(df, data, channels_unique, sampling_rate, time_window_ms):
    '''
    df: pandas dataframe, with columns: start, end, channel_name
    '''
    starts = df["start"].values
    ends = df["end"].values
    starts = starts.astype(int)
    ends = ends.astype(int)
    channel_names = df["name"].values
    hfo_waveforms, real_starts, real_ends, is_boundaries = extract_waveforms(data, starts, ends, channel_names, channels_unique, int(time_window_ms/1000*sampling_rate))
    return starts, ends, channel_names, hfo_waveforms, real_starts, real_ends, is_boundaries

def generate_feature_from_df_resample(df, data, channels_unique, annotation_sample_rate,feature_sampling_rate, time_window_ms):
    '''
    df: pandas dataframe, with columns: start, end, channel_name
    '''
    starts = df["start"].values
    ends = df["end"].values
    original_starts = starts.astype(int)
    original_ends = ends.astype(int)
    resampled_starts = starts * feature_sampling_rate // annotation_sample_rate
    resampled_ends = ends * feature_sampling_rate // annotation_sample_rate
    resampled_starts = resampled_starts.astype(int)
    resampled_ends = resampled_ends.astype(int)
    channel_names = df["name"].values
    hfo_waveforms, real_starts, real_ends, is_boundaries = extract_waveforms(data, resampled_starts, resampled_ends, channel_names, channels_unique, int(time_window_ms/1000*feature_sampling_rate))
    return original_starts, original_ends, resampled_starts, resampled_ends, channel_names, hfo_waveforms, real_starts, real_ends, is_boundaries

def calcuate_boundary(start, end, length, win_len=2000):
    if start < win_len: 
        return 0, win_len, True
    if end > length - win_len: 
        return length - win_len, length, True
    return int(0.5*(start + end) - win_len//2), int(0.5*(start + end) + win_len//2), False

def extract_waveforms(data, starts, ends, channel_names, unique_channel_names, window_len=2000):
    '''
    The extracted waveform will be (n_HFOs, widnow_len long) 
    
    '''

    def extract_data(data, start, end, window_len=2000):
        data = np.squeeze(data)
        real_start, real_end, is_boundary = calcuate_boundary(start, end, len(data), win_len=window_len)
        hfo_waveform = data[real_start:real_end]
        return hfo_waveform, real_start, real_end, is_boundary

    hfo_waveforms = np.zeros((len(starts), int(window_len)))
    real_starts = []
    real_ends = []
    is_boundaries = []
    for i in tqdm(range(len(starts))):
        channel_name = channel_names[i]
        start = starts[i]
        end = ends[i]
        channel_index = np.where(unique_channel_names == channel_name)[0]
        if len(channel_index) == 0:
            print(f"Channel {channel_name} not found in {unique_channel_names}")
            exit()
        hfo_waveform, real_start, real_end, is_boundary = extract_data(data[channel_index], start, end, window_len)
        # print(f"Try to find {channel_name} in {unique_channel_names}")
        hfo_waveforms[i] = hfo_waveform
        real_starts.append(real_start)
        real_ends.append(real_end)
        is_boundaries.append(is_boundary)
    return hfo_waveforms, real_starts, real_ends, is_boundaries

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

def compute_spectrum_batch_ehfo(org_sigs, ps_SampleRate=2000, ps_FreqSeg=512, ps_MinFreqHz=10, ps_MaxFreqHz=500, device='cuda'):
    """
    Compute spectrum for a batch of signals using CUDA with vectorized operations,
    but following the mathematical normalization approach of the original function.
    
    Args:
        org_sigs: Tensor of shape [batch_size, signal_length]
        ps_SampleRate: Sampling rate in Hz
        ps_FreqSeg: Number of frequency segments
        ps_MinFreqHz: Minimum frequency in Hz
        ps_MaxFreqHz: Maximum frequency in Hz
        device: Device to run computations on ('cuda' or 'cpu')
        
    Returns:
        Spectrum tensor of shape [batch_size, ps_FreqSeg, signal_length]
    """
    # Ensure device compatibility
    org_sigs_device = org_sigs.device if hasattr(org_sigs, 'is_cuda') and org_sigs.is_cuda else torch.device(device)
    
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
    
    # Create frequency axis exactly like the old method
    v_FreqAxis = torch.linspace(ps_MinFreqHz, ps_MaxFreqHz, steps=ps_FreqSeg, device=org_sigs_device)
    v_FreqAxis = torch.flip(v_FreqAxis, [0])  # Mimic the [::-1] in numpy

    # Initialize FFT window matrix on the correct device
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=org_sigs_device).float()
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5 * (v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)))**2 * (s_StDevSec**2).view(-1, 1))
    
    # Normalize the FFT windows using the global normalization approach from the old method
    # For each window, normalize using the full window's norm (not row-wise)
    for i in range(ps_FreqSeg):
        v_WinFFT[i] = v_WinFFT[i] * math.sqrt(s_Len) / torch.norm(v_WinFFT[i])

    # Perform FFT on the extended signals and apply windowed filters
    v_InputSignalFFT = torch.fft.fft(extend_sigs, dim=1)
    
    # Apply windowed filters in vectorized form
    res = torch.fft.ifft(v_InputSignalFFT.unsqueeze(1) * v_WinFFT.unsqueeze(0), dim=2)[:, :, ii:jj]
    
    # Divide by sqrt of standard deviation
    res = res / torch.sqrt(s_StDevSec).view(1, -1, 1)

    # Return the magnitude, ensuring the result is moved to CPU if required
    res = res.abs()
    return res.cpu() if not org_sigs.is_cuda else res
import numpy as np
import torch
from torch.nn.functional import interpolate

def normalized_cuda(a, max_=2000-11):
    """CUDA version of the normalized function with batch support.
    
    Args:
        a: Input tensor with shape [batch_size, signal_length]
        max_: Maximum value (default: 1989)
        
    Returns:
        Normalized tensor with shape [batch_size, signal_length]
    """
    # Move input to GPU if not already there
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    
    if a.device.type != 'cuda':
        a = a.cuda()
    
    # Handle both single signal and batch of signals
    if a.dim() == 1:
        a = a.unsqueeze(0)  # Add batch dimension
    
    # Get min values for each signal in the batch
    min_vals = a.min(dim=1, keepdim=True)[0]
    # Get peak-to-peak values for each signal
    ptp_vals = a.max(dim=1, keepdim=True)[0] - min_vals
    
    # Normalize and scale
    c = (max_ * (a - min_vals) / ptp_vals).to(torch.int32)
    c = c + 5
    
    return c

def construct_features_ehfo_cuda(raw_signals, length=1000):
    """CUDA version of construct_features_ehfo with batch support.
    
    Args:
        raw_signals: Input signals with shape [batch_size, signal_length]
        length: Canvas size parameter (default: 1000)
        
    Returns:
        Tuple of (spike_images, hfo_images) with shape [batch_size, 224, 224]
    """
    # Move input to GPU if not already there
    if not isinstance(raw_signals, torch.Tensor):
        raw_signals = torch.tensor(raw_signals, dtype=torch.float32)
    
    if raw_signals.device.type != 'cuda':
        raw_signals = raw_signals.cuda()
    
    # Handle both single signal and batch of signals
    if raw_signals.dim() == 1:
        raw_signals = raw_signals.unsqueeze(0)  # Add batch dimension
    
    batch_size, signal_length = raw_signals.shape
    
    # Normalize the signals
    hfo_spikes = normalized_cuda(raw_signals)
    
    # Create index tensor and repeat it for each item in the batch
    index = torch.arange(signal_length, device=raw_signals.device)
    
    # Initialize canvas tensors for all items in the batch
    canvas = torch.zeros((batch_size, 2*length, 2*length), device=raw_signals.device)
    intensity_image = torch.zeros_like(canvas)
    
    # Process each signal in the batch
    for batch_idx in range(batch_size):
        hfo_spike = hfo_spikes[batch_idx]
        
        # Draw lines on canvas for the current signal
        for ii in range(3):
            # Calculate indices
            row_indices = index
            col_indices_minus = hfo_spike - ii
            col_indices_plus = hfo_spike + ii
            
            # Ensure indices are in bounds
            valid_minus = (col_indices_minus >= 0) & (col_indices_minus < 2*length)
            valid_plus = (col_indices_plus >= 0) & (col_indices_plus < 2*length)
            
            # Set values in canvas
            for i, (r, c_minus, c_plus, v_minus, v_plus) in enumerate(zip(
                row_indices, col_indices_minus, col_indices_plus, valid_minus, valid_plus)):
                if v_minus:
                    canvas[batch_idx, r, c_minus] = 256
                if v_plus:
                    canvas[batch_idx, r, c_plus] = 256
        
        # Fill intensity image for the current signal
        intensity_image[batch_idx, index, :] = raw_signals[batch_idx].unsqueeze(1).expand(-1, 2*length)
    
    # Resize images to 224x224
    spike_images = interpolate(canvas.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    hfo_images = interpolate(intensity_image.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    
    return spike_images, hfo_images

# More efficient implementation using vectorized operations
def construct_features_ehfo_cuda_optimized(raw_signals, length=1000):
    """Optimized CUDA version of construct_features_ehfo with batch support.
    
    This version uses more vectorized operations to avoid explicit loops when possible.
    
    Args:
        raw_signals: Input signals with shape [batch_size, signal_length]
        length: Canvas size parameter (default: 1000)
        
    Returns:
        Tuple of (spike_images, hfo_images) with shape [batch_size, 224, 224]
    """
    # Move input to GPU if not already there
    if not isinstance(raw_signals, torch.Tensor):
        raw_signals = torch.tensor(raw_signals, dtype=torch.float32)
    
    if raw_signals.device.type != 'cuda':
        raw_signals = raw_signals.cuda()
    
    # Handle both single signal and batch of signals
    if raw_signals.dim() == 1:
        raw_signals = raw_signals.unsqueeze(0)  # Add batch dimension
    
    batch_size, signal_length = raw_signals.shape
    
    # Normalize the signals
    hfo_spikes = normalized_cuda(raw_signals)
    
    # Create canvas tensors for all items in the batch
    canvas = torch.zeros((batch_size, 2*length, 2*length), device=raw_signals.device)
    
    # Create a batch of indices
    batch_indices = torch.arange(batch_size, device=raw_signals.device).view(-1, 1).repeat(1, signal_length)
    row_indices = torch.arange(signal_length, device=raw_signals.device).repeat(batch_size, 1)
    
    # Process all offsets in a vectorized way
    for ii in range(3):
        # Calculate column indices for minus and plus offsets
        col_indices_minus = hfo_spikes - ii
        col_indices_plus = hfo_spikes + ii
        
        # Flatten all indices for efficient indexing
        flat_batch_minus = batch_indices.flatten()
        flat_batch_plus = batch_indices.flatten()
        flat_rows = row_indices.flatten()
        flat_cols_minus = col_indices_minus.flatten()
        flat_cols_plus = col_indices_plus.flatten()
        
        # Filter out out-of-bounds indices
        valid_minus = (flat_cols_minus >= 0) & (flat_cols_minus < 2*length)
        valid_plus = (flat_cols_plus >= 0) & (flat_cols_plus < 2*length)
        
        # Set values in canvas using masked scatter
        if valid_minus.any():
            valid_batch_minus = flat_batch_minus[valid_minus]
            valid_rows_minus = flat_rows[valid_minus]
            valid_cols_minus = flat_cols_minus[valid_minus]
            canvas[valid_batch_minus, valid_rows_minus, valid_cols_minus] = 256
            
        if valid_plus.any():
            valid_batch_plus = flat_batch_plus[valid_plus]
            valid_rows_plus = flat_rows[valid_plus]
            valid_cols_plus = flat_cols_plus[valid_plus]
            canvas[valid_batch_plus, valid_rows_plus, valid_cols_plus] = 256
    
    # Create intensity images for all signals in the batch
    intensity_image = torch.zeros_like(canvas)
    
    # Expand raw_signals to fill the intensity image rows
    for b in range(batch_size):
        intensity_image[b, :signal_length, :] = raw_signals[b].unsqueeze(1).expand(-1, 2*length)
    
    # Resize images to 224x224
    spike_images = interpolate(canvas.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    hfo_images = interpolate(intensity_image.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    
    return spike_images, hfo_images

def original_normalized(a, max_=2000-11):
    """Original CPU implementation of normalized function."""
    c = (max_*(a - np.min(a))/np.ptp(a)).astype(int)
    c = c + 5 
    return c 

def original_construct_features_ehfo(raw_signal, length=1000):
    """Original CPU implementation of construct_features_ehfo function."""
    #HFO with spike
    canvas = np.zeros((2*length, 2*length))
    hfo_spike = original_normalized(raw_signal)
    index = np.arange(len(hfo_spike))
    for ii in range(3):
        canvas[index, hfo_spike-ii] = 256
        canvas[index, hfo_spike+ii] = 256 
    from skimage.transform import resize
    spike_image = resize(canvas, (224, 224))

    intensity_image = np.zeros_like(canvas)
    intensity_image[index, :] = raw_signal[:, np.newaxis]
    hfo_image = resize(intensity_image, (224, 224))

    return spike_image, hfo_image
