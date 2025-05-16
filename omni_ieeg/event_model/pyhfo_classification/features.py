import torch
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import torchvision.transforms.functional as TF

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
    total_samples = hfo_waveforms.shape[1]
    time_window_length = int(feature_param["model_additional_parameter"][model_name]["time_window_ms"] / 1000 * sampling_rate)
    # print(f"total_samples: {total_samples}, time_window_length: {time_window_length}, sampling_rate: {sampling_rate}")
    middle_idx = total_samples // 2
    start_idx = middle_idx - time_window_length // 2
    end_idx = start_idx + time_window_length

    # Extract the middle portion of the waveform
    hfo_waveforms_tensor = hfo_waveforms_tensor[:, start_idx:end_idx]
    # assert hfo_waveforms_tensor.shape[1] == time_window_length * sampling_rate / 1000, f"hfo_waveforms_tensor.shape[1]: {hfo_waveforms_tensor.shape[1]}, time_window_length: {time_window_length}, sampling_rate: {sampling_rate}"
    # Set parameters from feature_param.

    
    # Process all waveforms at once
    time_freq_tensor, amp_tensor = hfo_feature_batch(hfo_waveforms_tensor, sampling_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img=resize_img, device=device)
    
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
        # Keep spec tensor with bilinear pytorch interpolation
        spec_tensor = torch.nn.functional.interpolate(spec_tensor.unsqueeze(1), size=(win_size, win_size), mode='bilinear', align_corners=False).squeeze(1)

        # --- Use skimage.transform.resize for amp_tensor via looping --- 
        # Note: This will be slower than pure torch operations, especially on GPU
        resized_amps = []
        # Move amp_tensor to CPU for numpy conversion and skimage compatibility
        amp_tensor_cpu = amp_tensor.cpu()
        for i in range(amp_tensor_cpu.shape[0]): # Loop through batch
            amp_single_np = amp_tensor_cpu[i].numpy() # Convert single image to numpy
            # Apply skimage resize (defaults: order=1, anti_aliasing=True for downsampling)
            amp_resized_np = resize(amp_single_np, (win_size, win_size))
            # Convert back to tensor and append
            resized_amps.append(torch.from_numpy(amp_resized_np).float())
            
        # Stack the list of resized tensors back into a batch
        amp_tensor = torch.stack(resized_amps)
        # Move back to the original device of the input waveforms
        amp_tensor = amp_tensor.to(hfo_waveforms.device) 

        # --- End skimage resize modification --- 

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




#================Non parallel version================

def hfo_feature(start, end, channel_name, data, sample_rate, win_size, ps_MinFreqHz, ps_MaxFreqHz, resize_img = True):
    # generate one sec time-freqeucny image
    spectrum_img = compute_spectrum(data, ps_SampleRate=sample_rate, ps_FreqSeg=win_size, ps_MinFreqHz=ps_MinFreqHz, ps_MaxFreqHz=ps_MaxFreqHz)
    amplitude_coding_plot = construct_coding(data, win_size) # height * len(selected_data)
    time_frequncy_img = spectrum_img
    if resize_img:
        time_frequncy_img = resize(time_frequncy_img, (win_size, win_size))
        amplitude_coding_plot = resize(amplitude_coding_plot, (win_size, win_size))
    return channel_name, start, end, time_frequncy_img, amplitude_coding_plot

def create_extended_sig(sig):
    s_len = len(sig)
    s_halflen = int(np.ceil(s_len/2)) + 1
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    start_win = -start_win[::-1] + sig[0]
    end_win = -end_win[::-1] + sig[-1]
    final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
    if len(final_sig)%2 == 0:
        final_sig = final_sig[:-1]
    return final_sig

def compute_spectrum(org_sig, ps_SampleRate = 2000, ps_FreqSeg = 512, ps_MinFreqHz = 10, ps_MaxFreqHz = 500, device = 'cpu'):
    device = torch.device(device)
    ii, jj = int(len(org_sig)//2), int(len(org_sig)//2 + len(org_sig))
    extend_sig = create_extended_sig(org_sig)
    extend_sig = torch.from_numpy(extend_sig).float()
    extend_sig = extend_sig.to(device)
    ps_StDevCycles = 3
    s_Len = len(extend_sig)
    s_HalfLen = math.floor(s_Len/2)+1
    v_WAxis = (torch.linspace(0, 2*np.pi, s_Len)[:-1]* ps_SampleRate).float()
    v_WAxisHalf = v_WAxis[:s_HalfLen].to(device).repeat(ps_FreqSeg, 1)
    v_FreqAxis = torch.linspace(ps_MaxFreqHz, ps_MinFreqHz,steps=ps_FreqSeg, device=device).float()
    v_WinFFT = torch.zeros(ps_FreqSeg, s_Len, device=device).float()
    s_StDevSec = (1 / v_FreqAxis) * ps_StDevCycles
    v_WinFFT[:, :s_HalfLen] = torch.exp(-0.5*torch.pow(v_WAxisHalf - (2 * torch.pi * v_FreqAxis.view(-1, 1)), 2) * (s_StDevSec**2).view(-1, 1))
    v_WinFFT = v_WinFFT * np.sqrt(s_Len)/ torch.norm(v_WinFFT, dim = -1).view(-1, 1)
    v_InputSignalFFT = torch.fft.fft(extend_sig)
    res = torch.fft.ifft(v_InputSignalFFT.view(1, -1)* v_WinFFT)[:, ii:jj]/torch.sqrt(s_StDevSec).view(-1,1)
    res = np.abs(res.numpy())
    return res


def construct_coding(raw_signal, height):
    index = np.arange(height)
    intensity_image = np.zeros((height, len(raw_signal)))
    intensity_image[index, :] = raw_signal
    return intensity_image


# ================ Test Function ================

def test_feature_comparison(result_folder='feature_comparison_results', num_waveforms=3):
    """
    Tests the consistency between hfo_feature_batch and hfo_feature.

    Generates random waveforms, processes them with both functions,
    and saves plots comparing the outputs (spectrum, amplitude) and their differences.
    """
    print(f"Starting feature comparison test. Results will be saved in: {result_folder}")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Created result directory: {result_folder}")

    # --- Parameters ---
    waveform_length_ms = 1000  # Example length in ms
    sampling_rate = 1000  # Hz
    win_size = 224
    ps_MinFreqHz = 10
    ps_MaxFreqHz = 500
    resize_img = True
    device = 'cpu' # Force CPU for easier comparison

    waveform_length = int(waveform_length_ms / 1000 * sampling_rate)

    # --- Generate Random Waveforms ---
    print(f"Generating {num_waveforms} random waveforms of length {waveform_length} samples...")
    np.random.seed(42) # for reproducibility
    waveforms_np = np.random.rand(num_waveforms, waveform_length).astype(np.float32) * 2 - 1 # Scale to [-1, 1]

    # --- Batch Processing ---
    print("Processing waveforms with hfo_feature_batch...")
    waveforms_tensor = torch.from_numpy(waveforms_np).to(device)
    # Note: hfo_feature_batch expects shape (B, L), L=win_size for spec, L=waveform_length for amp initially
    # The interpolation happens inside hfo_feature_batch
    batch_spec, batch_amp = hfo_feature_batch(
        waveforms_tensor,
        sample_rate=sampling_rate,
        win_size=win_size,
        ps_MinFreqHz=ps_MinFreqHz,
        ps_MaxFreqHz=ps_MaxFreqHz,
        resize_img=resize_img,
        device=device
    )
    # Ensure results are on CPU and detached
    batch_spec = batch_spec.detach().cpu()
    batch_amp = batch_amp.detach().cpu()
    print("Batch processing complete.")

    # --- Single Processing ---
    print("Processing waveforms individually with hfo_feature...")
    single_specs = []
    single_amps = []
    for i in tqdm(range(num_waveforms), desc="Single Processing"):
        waveform_single_np = waveforms_np[i, :]
        # hfo_feature returns numpy arrays
        # It expects channel_name, start, end but doesn't use them if resize_img is True. Pass placeholders.
        _, _, _, spec_np, amp_np = hfo_feature(
            start=0, end=0, channel_name='test', # Placeholders
            data=waveform_single_np,
            sample_rate=sampling_rate,
            win_size=win_size,
            ps_MinFreqHz=ps_MinFreqHz,
            ps_MaxFreqHz=ps_MaxFreqHz,
            resize_img=resize_img
        )
        single_specs.append(spec_np)
        single_amps.append(amp_np)
    print("Single processing complete.")

    # --- Comparison and Plotting ---
    print("Comparing results and generating plots...")
    max_abs_diff_spec = 0
    max_abs_diff_amp = 0

    for i in range(num_waveforms):
        # Convert single results (numpy) to tensors for comparison
        spec_single_tensor = torch.from_numpy(single_specs[i]).float()
        amp_single_tensor = torch.from_numpy(single_amps[i]).float()

        spec_batch_tensor = batch_spec[i]
        amp_batch_tensor = batch_amp[i]

        # Calculate differences
        diff_spec = torch.abs(spec_batch_tensor - spec_single_tensor)
        diff_amp = torch.abs(amp_batch_tensor - amp_single_tensor)

        max_abs_diff_spec = max(max_abs_diff_spec, diff_spec.max().item())
        max_abs_diff_amp = max(max_abs_diff_amp, diff_amp.max().item())

        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Waveform {i+1} Comparison (Max Spec Diff: {diff_spec.max():.2e}, Max Amp Diff: {diff_amp.max():.2e})')

        # Spectrum Row
        im_spec_batch = axes[0, 0].imshow(spec_batch_tensor.numpy(), aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Batch Spectrum')
        fig.colorbar(im_spec_batch, ax=axes[0, 0])

        im_spec_single = axes[0, 1].imshow(spec_single_tensor.numpy(), aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Single Spectrum')
        fig.colorbar(im_spec_single, ax=axes[0, 1])

        im_spec_diff = axes[0, 2].imshow(diff_spec.numpy(), aspect='auto', cmap='magma')
        axes[0, 2].set_title('Difference (Abs)')
        fig.colorbar(im_spec_diff, ax=axes[0, 2])

        # Amplitude Row
        im_amp_batch = axes[1, 0].imshow(amp_batch_tensor.numpy(), aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Batch Amplitude')
        fig.colorbar(im_amp_batch, ax=axes[1, 0])

        im_amp_single = axes[1, 1].imshow(amp_single_tensor.numpy(), aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Single Amplitude')
        fig.colorbar(im_amp_single, ax=axes[1, 1])

        im_amp_diff = axes[1, 2].imshow(diff_amp.numpy(), aspect='auto', cmap='magma')
        axes[1, 2].set_title('Difference (Abs)')
        fig.colorbar(im_amp_diff, ax=axes[1, 2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        save_path = os.path.join(result_folder, f'comparison_waveform_{i+1}.png')
        plt.savefig(save_path)
        plt.close(fig)
        # print(f"Saved comparison plot to {save_path}")

    print(f"Plotting complete. Max absolute difference observed in Spectrum: {max_abs_diff_spec:.2e}")
    print(f"Plotting complete. Max absolute difference observed in Amplitude: {max_abs_diff_amp:.2e}")
    print("Test finished.")


if __name__ == "__main__":
    # Example usage:
    test_feature_comparison(result_folder='/mnt/SSD1/nipsdataset/verification/pyhfo_verification')

