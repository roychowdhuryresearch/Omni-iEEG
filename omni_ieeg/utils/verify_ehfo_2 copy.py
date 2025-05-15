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
        hfo_waveform, real_start, real_end, is_boundary = extract_data(data[channel_index], start, end, window_len)
        # print(f"Try to find {channel_name} in {unique_channel_names}")
        hfo_waveforms[i] = hfo_waveform
        real_starts.append(real_start)
        real_ends.append(real_end)
        is_boundaries.append(is_boundary)
    return hfo_waveforms, real_starts, real_ends, is_boundaries

def generate_feature_from_df_gpu_batch(hfo_waveforms, feature_param, model_name,device='cuda'):
    win_size = feature_param["image_size"]
    ps_MinFreqHz = feature_param["freq_min_hz"]
    ps_MaxFreqHz = feature_param["freq_max_hz"]
    sampling_rate = feature_param["resample"]
    resize_img = True
    # Convert to torch tensor and move to device.
    hfo_waveforms_tensor = torch.tensor(hfo_waveforms).float().to(device)
    assert hfo_waveforms_tensor.shape[1] == feature_param["raw_waveform_length"] * sampling_rate / 1000
    # only take time_window_ms from middle of the waveform
    # Calculate the start and end indices for the middle portion of the waveform
    total_length = feature_param["raw_waveform_length"]
    time_window_length = int(feature_param["model_additional_parameter"][model_name]["time_window_ms"] / 1000 * sampling_rate)
    middle_idx = total_length // 2
    start_idx = middle_idx - time_window_length // 2
    end_idx = start_idx + time_window_length

    # Extract the middle portion of the waveform
    hfo_waveforms_tensor = hfo_waveforms_tensor[:, start_idx:end_idx]
    assert hfo_waveforms_tensor.shape[1] == time_window_length * sampling_rate / 1000
    # Set parameters from feature_param.

    
    # Process all waveforms at once
    time_freq_tensor, amp_tensor = hfo_feature_batch(hfo_waveforms_tensor, sampling_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img=resize_img, device=device)
    
    # Convert the resulting tensors back to numpy arrays.
    # time_frequency_img = time_freq_tensor.cpu().numpy()
    # amplitude_coding_plot = amp_tensor.cpu().numpy()
    
    if feature_param["model_additional_parameter"][model_name]["n_feature"] == 1:
        return time_freq_tensor[:, None, :, :]
    elif feature_param["model_additional_parameter"][model_name]["n_feature"] == 2:
        return torch.cat((time_freq_tensor[:, None, :, :], amp_tensor[:, None, :, :]), dim=1)
    else:
        raise ValueError(f"Invalid number of features: {feature_param['model_additional_parameter'][model_name]['n_feature']}")

def hfo_feature_batch(hfo_waveforms, sample_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img=True, device='cuda'):
    """
    hfo_waveforms: torch.Tensor of shape (B, L)
    sample_rate: sampling rate (ps_SampleRate) for spectrum computation
    win_size: desired output image size (both for ps_FreqSeg and final interpolation)
    ps_MinFreqHz, ps_MaxFreqHz: frequency bounds for the FFT window computation
    resize_img: if True, the computed images are bilinearly interpolated to (win_size, win_size)
    Returns:
       time_frequency_img: torch.Tensor of shape (B, win_size, win_size)
       amplitude_coding_plot: torch.Tensor of shape (B, win_size, win_size)
    """
    # Compute the spectrum image in batch.
    spec = compute_spectrum_batch(hfo_waveforms, ps_SampleRate=sample_rate, ps_FreqSeg=win_size,
                                  ps_MinFreqHz=ps_MinFreqHz, ps_MaxFreqHz=ps_MaxFreqHz, device=device)
    # spec has shape (B, win_size, L_original) where L_original comes from slicing the FFT result.
    spec_tensor = spec  # already on device

    # Compute amplitude coding image (tile the raw waveform).
    amp_tensor = construct_coding_batch(hfo_waveforms, height=win_size)  # shape: (B, win_size, L)

    if resize_img:
        # Interpolate both images to (win_size, win_size). Assume original width may not equal win_size.
        # Add a channel dimension to use interpolate.
        spec_tensor = torch.nn.functional.interpolate(spec_tensor.unsqueeze(1), size=(win_size, win_size), mode='bilinear').squeeze(1)
        amp_tensor = torch.nn.functional.interpolate(amp_tensor.unsqueeze(1), size=(win_size, win_size), mode='bilinear').squeeze(1)

    return spec_tensor, amp_tensor

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
    return res.cpu().numpy() if not org_sigs.is_cuda else res

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

def construct_coding_batch(sigs, height):
    """
    sigs: torch.Tensor of shape (B, L)
    Returns: torch.Tensor of shape (B, height, L) where each signal is repeated along the new height dimension.
    """
    # unsqueeze to (B, 1, L) and repeat along dimension 1 (height axis)
    return sigs.unsqueeze(1).expand(-1, height, -1)
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
    spike_images = interpolate(canvas.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(1)
    hfo_images = interpolate(intensity_image.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(1)
    
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

# Modified comparison function to focus on one change at a time
def compare_implementations(batch_size=4, signal_length=500, run_original=True, 
                           test_variants=False, use_skimage_resize=False):
    """Compare implementations with options to test different parts.
    
    Args:
        batch_size: Number of signals to process
        signal_length: Length of each signal
        run_original: Whether to run the original implementation
        test_variants: Whether to test variant implementations to isolate issues
        use_skimage_resize: Whether to use skimage.resize instead of torch.interpolate
        
    Returns:
        Dictionary containing timing results and output comparisons
    """
    import time
    from skimage.transform import resize
    
    # Generate sample data
    print(f"Generating {batch_size} random signals of length {signal_length}...")
    np.random.seed(42)  # For reproducibility
    raw_signals = np.random.randn(batch_size, signal_length)
    
    results = {}
    
    # Test original implementation (one signal at a time)
    if run_original:
        print("\nTesting original CPU implementation (processing signals one by one)...")
        start_time = time.time()
        original_results = []
        
        for i in range(batch_size):
            spike_img, hfo_img = original_construct_features_ehfo(raw_signals[i], length=1000)
            original_results.append((spike_img, hfo_img))
            
        original_time = time.time() - start_time
        print(f"Original CPU implementation took: {original_time:.4f} seconds")
        
        # Convert results to numpy arrays for comparison
        original_spike_images = np.stack([res[0] for res in original_results])
        original_hfo_images = np.stack([res[1] for res in original_results])
        
        results['original'] = {
            'time': original_time,
            'spike_images': original_spike_images,
            'hfo_images': original_hfo_images
        }
    else:
        print("\nSkipping original CPU implementation due to performance concerns.")
    
    # Convert to PyTorch tensor and move to GPU
    raw_signals_tensor = torch.tensor(raw_signals, dtype=torch.float32)
    if torch.cuda.is_available():
        raw_signals_tensor = raw_signals_tensor.cuda()
        
    # Test CUDA implementation
    print("\nTesting CUDA implementation...")
    start_time = time.time()
    spike_images_cuda, hfo_images_cuda = construct_features_ehfo_cuda(raw_signals_tensor)
    cuda_time = time.time() - start_time
    print(f"CUDA implementation took: {cuda_time:.4f} seconds")
    
    # Test optimized CUDA implementation
    print("\nTesting optimized CUDA implementation...")
    start_time = time.time()
    spike_images_opt, hfo_images_opt = construct_features_ehfo_cuda_optimized(raw_signals_tensor)
    cuda_opt_time = time.time() - start_time
    print(f"Optimized CUDA implementation took: {cuda_opt_time:.4f} seconds")
    
    # Move results to CPU for comparison
    spike_images_cuda_np = spike_images_cuda.cpu().numpy()
    hfo_images_cuda_np = hfo_images_cuda.cpu().numpy()
    spike_images_opt_np = spike_images_opt.cpu().numpy()
    hfo_images_opt_np = hfo_images_opt.cpu().numpy()
    
    results['cuda'] = {
        'time': cuda_time,
        'spike_images': spike_images_cuda_np,
        'hfo_images': hfo_images_cuda_np
    }
    
    results['cuda_optimized'] = {
        'time': cuda_opt_time,
        'spike_images': spike_images_opt_np,
        'hfo_images': hfo_images_opt_np
    }
    
    # Test variants to isolate issues if requested
    if test_variants and run_original:
        # Test variant 1: Original normalization with CUDA implementation
        def construct_features_ehfo_cuda_variant1(raw_signals, length=1000):
            """CUDA with original normalization."""
            # Same as construct_features_ehfo_cuda but uses original_normalized
            if not isinstance(raw_signals, torch.Tensor):
                raw_signals = torch.tensor(raw_signals, dtype=torch.float32)
            
            if raw_signals.device.type != 'cuda':
                raw_signals = raw_signals.cuda()
            
            if raw_signals.dim() == 1:
                raw_signals = raw_signals.unsqueeze(0)
            
            batch_size, signal_length = raw_signals.shape
            
            # Use original normalized instead of CUDA version
            hfo_spikes_list = []
            for i in range(batch_size):
                signal = raw_signals[i].cpu().numpy()
                hfo_spike = original_normalized(signal)
                hfo_spikes_list.append(torch.tensor(hfo_spike, device=raw_signals.device))
            
            hfo_spikes = torch.stack(hfo_spikes_list)
            
            # Rest of the function is the same as construct_features_ehfo_cuda
            index = torch.arange(signal_length, device=raw_signals.device)
            canvas = torch.zeros((batch_size, 2*length, 2*length), device=raw_signals.device)
            intensity_image = torch.zeros_like(canvas)
            
            for batch_idx in range(batch_size):
                hfo_spike = hfo_spikes[batch_idx]
                
                for ii in range(3):
                    row_indices = index
                    col_indices_minus = hfo_spike - ii
                    col_indices_plus = hfo_spike + ii
                    
                    valid_minus = (col_indices_minus >= 0) & (col_indices_minus < 2*length)
                    valid_plus = (col_indices_plus >= 0) & (col_indices_plus < 2*length)
                    
                    for i, (r, c_minus, c_plus, v_minus, v_plus) in enumerate(zip(
                        row_indices, col_indices_minus, col_indices_plus, valid_minus, valid_plus)):
                        if v_minus:
                            canvas[batch_idx, r, c_minus] = 256
                        if v_plus:
                            canvas[batch_idx, r, c_plus] = 256
                
                intensity_image[batch_idx, index, :] = raw_signals[batch_idx].unsqueeze(1).expand(-1, 2*length)
            
            # Use same resize method as original implementation
            if use_skimage_resize:
                spike_images_list = []
                hfo_images_list = []
                
                canvas_np = canvas.cpu().numpy()
                intensity_image_np = intensity_image.cpu().numpy()
                
                for i in range(batch_size):
                    spike_image = resize(canvas_np[i], (224, 224))
                    hfo_image = resize(intensity_image_np[i], (224, 224))
                    
                    spike_images_list.append(spike_image)
                    hfo_images_list.append(hfo_image)
                
                spike_images = torch.tensor(np.stack(spike_images_list), dtype=torch.float32).to(raw_signals.device)
                hfo_images = torch.tensor(np.stack(hfo_images_list), dtype=torch.float32).to(raw_signals.device)
            else:
                spike_images = interpolate(canvas.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
                hfo_images = interpolate(intensity_image.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            
            return spike_images, hfo_images
        
        # Test variant 2: CUDA normalization with skimage resize
        def construct_features_ehfo_cuda_variant2(raw_signals, length=1000):
            """CUDA with skimage resize."""
            # Same as construct_features_ehfo_cuda but uses skimage resize
            if not isinstance(raw_signals, torch.Tensor):
                raw_signals = torch.tensor(raw_signals, dtype=torch.float32)
            
            if raw_signals.device.type != 'cuda':
                raw_signals = raw_signals.cuda()
            
            if raw_signals.dim() == 1:
                raw_signals = raw_signals.unsqueeze(0)
            
            batch_size, signal_length = raw_signals.shape
            
            # Use CUDA normalized 
            hfo_spikes = normalized_cuda(raw_signals)
            
            # Rest of the function is the same as construct_features_ehfo_cuda
            index = torch.arange(signal_length, device=raw_signals.device)
            canvas = torch.zeros((batch_size, 2*length, 2*length), device=raw_signals.device)
            intensity_image = torch.zeros_like(canvas)
            
            for batch_idx in range(batch_size):
                hfo_spike = hfo_spikes[batch_idx]
                
                for ii in range(3):
                    row_indices = index
                    col_indices_minus = hfo_spike - ii
                    col_indices_plus = hfo_spike + ii
                    
                    valid_minus = (col_indices_minus >= 0) & (col_indices_minus < 2*length)
                    valid_plus = (col_indices_plus >= 0) & (col_indices_plus < 2*length)
                    
                    for i, (r, c_minus, c_plus, v_minus, v_plus) in enumerate(zip(
                        row_indices, col_indices_minus, col_indices_plus, valid_minus, valid_plus)):
                        if v_minus:
                            canvas[batch_idx, r, c_minus] = 256
                        if v_plus:
                            canvas[batch_idx, r, c_plus] = 256
                
                intensity_image[batch_idx, index, :] = raw_signals[batch_idx].unsqueeze(1).expand(-1, 2*length)
            
            # Only difference: using skimage.resize instead of interpolate
            spike_images_list = []
            hfo_images_list = []
            
            canvas_np = canvas.cpu().numpy()
            intensity_image_np = intensity_image.cpu().numpy()
            
            for i in range(batch_size):
                spike_image = resize(canvas_np[i], (224, 224))
                hfo_image = resize(intensity_image_np[i], (224, 224))
                
                spike_images_list.append(spike_image)
                hfo_images_list.append(hfo_image)
            
            spike_images = torch.tensor(np.stack(spike_images_list), dtype=torch.float32).to(raw_signals.device)
            hfo_images = torch.tensor(np.stack(hfo_images_list), dtype=torch.float32).to(raw_signals.device)
            
            return spike_images, hfo_images
        
        # Test variant 3: Change intensity_image fill method 
        def construct_features_ehfo_cuda_variant3(raw_signals, length=1000):
            """CUDA with modified intensity image filling."""
            # Same as construct_features_ehfo_cuda but changes how intensity_image is filled
            if not isinstance(raw_signals, torch.Tensor):
                raw_signals = torch.tensor(raw_signals, dtype=torch.float32)
            
            if raw_signals.device.type != 'cuda':
                raw_signals = raw_signals.cuda()
            
            if raw_signals.dim() == 1:
                raw_signals = raw_signals.unsqueeze(0)
            
            batch_size, signal_length = raw_signals.shape
            
            # Use CUDA normalized 
            hfo_spikes = normalized_cuda(raw_signals)
            
            # Rest of the function is the same as construct_features_ehfo_cuda
            index = torch.arange(signal_length, device=raw_signals.device)
            canvas = torch.zeros((batch_size, 2*length, 2*length), device=raw_signals.device)
            intensity_image = torch.zeros_like(canvas)
            
            for batch_idx in range(batch_size):
                hfo_spike = hfo_spikes[batch_idx]
                
                for ii in range(3):
                    row_indices = index
                    col_indices_minus = hfo_spike - ii
                    col_indices_plus = hfo_spike + ii
                    
                    valid_minus = (col_indices_minus >= 0) & (col_indices_minus < 2*length)
                    valid_plus = (col_indices_plus >= 0) & (col_indices_plus < 2*length)
                    
                    for i, (r, c_minus, c_plus, v_minus, v_plus) in enumerate(zip(
                        row_indices, col_indices_minus, col_indices_plus, valid_minus, valid_plus)):
                        if v_minus:
                            canvas[batch_idx, r, c_minus] = 256
                        if v_plus:
                            canvas[batch_idx, r, c_plus] = 256
                
                # Modified intensity image filling - only fill the first signal_length rows
                # and set each element to the signal value rather than expanding
                for i in range(signal_length):
                    intensity_image[batch_idx, i, :] = raw_signals[batch_idx, i]
            
            # Use interpolate for resizing
            spike_images = interpolate(canvas.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            hfo_images = interpolate(intensity_image.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            
            return spike_images, hfo_images
        
        print("\nTesting variant implementations to isolate issues...")
        
        # Test variant 1
        print("\nTesting Variant 1: Original normalization with CUDA implementation...")
        start_time = time.time()
        spike_images_v1, hfo_images_v1 = construct_features_ehfo_cuda_variant1(raw_signals_tensor, length=1000)
        v1_time = time.time() - start_time
        print(f"Variant 1 took: {v1_time:.4f} seconds")
        
        # Test variant 2
        print("\nTesting Variant 2: CUDA with skimage resize...")
        start_time = time.time()
        spike_images_v2, hfo_images_v2 = construct_features_ehfo_cuda_variant2(raw_signals_tensor, length=1000)
        v2_time = time.time() - start_time
        print(f"Variant 2 took: {v2_time:.4f} seconds")
        
        # Test variant 3
        print("\nTesting Variant 3: Modified intensity image filling...")
        start_time = time.time()
        spike_images_v3, hfo_images_v3 = construct_features_ehfo_cuda_variant3(raw_signals_tensor, length=1000)
        v3_time = time.time() - start_time
        print(f"Variant 3 took: {v3_time:.4f} seconds")
        
        # Move variant results to CPU for comparison
        spike_images_v1_np = spike_images_v1.cpu().numpy()
        hfo_images_v1_np = hfo_images_v1.cpu().numpy()
        spike_images_v2_np = spike_images_v2.cpu().numpy() 
        hfo_images_v2_np = hfo_images_v2.cpu().numpy()
        spike_images_v3_np = spike_images_v3.cpu().numpy()
        hfo_images_v3_np = hfo_images_v3.cpu().numpy()
        
        # Compare original vs variants
        print("\nComparing original vs variants to identify the source of differences:")
        
        # Original vs Variant 1 (normalization)
        spike_diff_orig_v1 = np.abs(original_spike_images - spike_images_v1_np).mean()
        hfo_diff_orig_v1 = np.abs(original_hfo_images - hfo_images_v1_np).mean()
        print(f"Original vs Variant 1 (normalization) - Spike diff: {spike_diff_orig_v1:.6f}, HFO diff: {hfo_diff_orig_v1:.6f}")
        
        spike_pct_v1 = spike_diff_orig_v1 / np.max(original_spike_images) * 100
        hfo_pct_v1 = hfo_diff_orig_v1 / np.max(original_hfo_images) * 100
        print(f"  Percentage difference - Spike: {spike_pct_v1:.4f}%, HFO: {hfo_pct_v1:.4f}%")
        
        # Original vs Variant 2 (resize)
        spike_diff_orig_v2 = np.abs(original_spike_images - spike_images_v2_np).mean()
        hfo_diff_orig_v2 = np.abs(original_hfo_images - hfo_images_v2_np).mean()
        print(f"Original vs Variant 2 (resize) - Spike diff: {spike_diff_orig_v2:.6f}, HFO diff: {hfo_diff_orig_v2:.6f}")
        
        spike_pct_v2 = spike_diff_orig_v2 / np.max(original_spike_images) * 100
        hfo_pct_v2 = hfo_diff_orig_v2 / np.max(original_hfo_images) * 100
        print(f"  Percentage difference - Spike: {spike_pct_v2:.4f}%, HFO: {hfo_pct_v2:.4f}%")
        
        # Original vs Variant 3 (intensity fill)
        spike_diff_orig_v3 = np.abs(original_spike_images - spike_images_v3_np).mean()
        hfo_diff_orig_v3 = np.abs(original_hfo_images - hfo_images_v3_np).mean()
        print(f"Original vs Variant 3 (intensity fill) - Spike diff: {spike_diff_orig_v3:.6f}, HFO diff: {hfo_diff_orig_v3:.6f}")
        
        spike_pct_v3 = spike_diff_orig_v3 / np.max(original_spike_images) * 100
        hfo_pct_v3 = hfo_diff_orig_v3 / np.max(original_hfo_images) * 100
        print(f"  Percentage difference - Spike: {spike_pct_v3:.4f}%, HFO: {hfo_pct_v3:.4f}%")
        
        # Store variant results for return
        results['variant1'] = {
            'time': v1_time,
            'spike_images': spike_images_v1_np,
            'hfo_images': hfo_images_v1_np,
            'description': 'Original normalization with CUDA implementation'
        }
        
        results['variant2'] = {
            'time': v2_time,
            'spike_images': spike_images_v2_np,
            'hfo_images': hfo_images_v2_np,
            'description': 'CUDA with skimage resize'
        }
        
        results['variant3'] = {
            'time': v3_time,
            'spike_images': spike_images_v3_np,
            'hfo_images': hfo_images_v3_np,
            'description': 'Modified intensity image filling'
        }
    
    # Compare original vs CUDA if running original
    if run_original:
        print("\nComparing original vs CUDA implementations:")
        # Compare original vs CUDA
        spike_diff = np.abs(original_spike_images - spike_images_cuda_np).mean()
        hfo_diff = np.abs(original_hfo_images - hfo_images_cuda_np).mean()
        print(f"Average difference original vs CUDA - Spike: {spike_diff:.6f}, HFO: {hfo_diff:.6f}")
        
        # Add percentage comparison
        spike_pct = spike_diff / np.max(original_spike_images) * 100
        hfo_pct = hfo_diff / np.max(original_hfo_images) * 100
        print(f"Percentage difference original vs CUDA - Spike: {spike_pct:.4f}%, HFO: {hfo_pct:.4f}%")
        
        # Compare original vs optimized CUDA
        spike_diff_opt = np.abs(original_spike_images - spike_images_opt_np).mean()
        hfo_diff_opt = np.abs(original_hfo_images - hfo_images_opt_np).mean()
        print(f"Average difference original vs optimized CUDA - Spike: {spike_diff_opt:.6f}, HFO: {hfo_diff_opt:.6f}")
        
        # Add percentage comparison
        spike_pct_opt = spike_diff_opt / np.max(original_spike_images) * 100
        hfo_pct_opt = hfo_diff_opt / np.max(original_hfo_images) * 100
        print(f"Percentage difference original vs optimized CUDA - Spike: {spike_pct_opt:.4f}%, HFO: {hfo_pct_opt:.4f}%")
    
    # Compare CUDA vs optimized CUDA
    spike_diff_cuda = np.abs(spike_images_cuda_np - spike_images_opt_np).mean()
    hfo_diff_cuda = np.abs(hfo_images_cuda_np - hfo_images_opt_np).mean()
    print(f"Average difference CUDA vs optimized CUDA - Spike: {spike_diff_cuda:.6f}, HFO: {hfo_diff_cuda:.6f}")
    
    # Print speedup
    if run_original:
        print("\nSpeedup:")
        print(f"CUDA vs original: {original_time / cuda_time:.2f}x faster")
        print(f"Optimized CUDA vs original: {original_time / cuda_opt_time:.2f}x faster")
    print(f"Optimized CUDA vs CUDA: {cuda_time / cuda_opt_time:.2f}x faster")
    
    return results
if __name__ == "__main__":
    test_variants = True
    run_original = True
    compare_implementations(test_variants=test_variants, run_original=run_original) 