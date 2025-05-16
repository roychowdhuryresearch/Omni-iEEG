import torch
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import torchvision.transforms.functional as TF
# Import necessary library for norm
from numpy import linalg as LA

# Import the parallel processing utility
from omni_ieeg.utils.utils_multiprocess import parallel_process

def normalize_img_ehfo(a):
    batch_num = a.shape[0]
    h = a.shape[1]
    w = a.shape[2]
    a_reshape = a.reshape(batch_num, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    c = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    c = c.reshape(batch_num,h, w)
    return c

# Define a helper function for processing a single waveform's features
# This function will be called by parallel_process
def _process_single_waveform(args):
    spectrum_slice, feature_slice, sample_rate = args
    spectrum_img = compute_spectrum_ehfo(spectrum_slice, ps_SampleRate=sample_rate)
    spike_image, intensity_image = construct_features_ehfo(feature_slice)
    return spectrum_img, spike_image, intensity_image

def generate_feature_from_df_gpu_batch(hfo_waveforms, feature_param, device='cuda', n_jobs=16):
    # win_size = feature_param["image_size"]
    # ps_MinFreqHz = feature_param["freq_min_hz"]
    # ps_MaxFreqHz = feature_param["freq_max_hz"]
    sampling_rate = feature_param["resample"]
    # resize_img = True
    # Convert to torch tensor and move to device.
    # hfo_waveforms_tensor = torch.tensor(hfo_waveforms).float().to(device) # Keep as numpy for now
    assert hfo_waveforms.shape[1] == feature_param["raw_waveform_length"] * sampling_rate / 1000
    # only take time_window_ms from middle of the waveform
    # Calculate the start and end indices for the middle portion of the waveform
    total_samples = hfo_waveforms.shape[1]
    middle_idx = total_samples // 2
    assert total_samples >= 2 * sampling_rate, "Waveform length must be at least 2 seconds for eHFO classification"
    start_1s, end_1s = middle_idx - sampling_rate, middle_idx + sampling_rate
    start_05s, end_05s = int(middle_idx - sampling_rate / 2), int(middle_idx + sampling_rate / 2)
    
    data_slice_spectrum = np.squeeze(hfo_waveforms[:, start_1s:end_1s])
    data_slice_features = np.squeeze(hfo_waveforms[:, start_05s:end_05s])

    # Handle case where batch size is 1 after squeeze
    if hfo_waveforms.shape[0] == 1:
        data_slice_spectrum = data_slice_spectrum[np.newaxis, :]
        data_slice_features = data_slice_features[np.newaxis, :]

    # Prepare tasks for parallel processing
    tasks = [
        (data_slice_spectrum[i], data_slice_features[i], sampling_rate) 
        for i in range(hfo_waveforms.shape[0])
    ]

    # Run in parallel
    # Note: parallel_process doesn't use GPU; device='cuda' is not used here.
    # Ensure n_jobs is appropriate for the system.
    results = parallel_process(tasks, _process_single_waveform, n_jobs=n_jobs) # front_num=0 to parallelize all

    # Unzip and stack results
    spectrum_imgs, spike_images, intensity_images = zip(*results)
    
    spectrum_batch = np.stack(spectrum_imgs, axis=0)
    spike_batch = np.stack(spike_images, axis=0)
    intensity_batch = np.stack(intensity_images, axis=0)
    

    return spectrum_batch, spike_batch, intensity_batch

def create_extended_sig_ehfo(wave2000):
    #wave2000 = bb
    s_len = len(wave2000)
    s_halflen = int(np.ceil(s_len/2)) + 1
    sig = wave2000
    start_win = sig[:s_halflen] - sig[0]
    end_win = sig[s_len - s_halflen - 1:] - sig[-1]
    start_win = -start_win[::-1] + sig[0]
    end_win = -end_win[::-1] + sig[-1]
    final_sig = np.concatenate((start_win[:-1],sig, end_win[1:]))
    #print(s_halflen, start_win.shape, end_win.shape, sig.shape, final_sig.shape)
    if len(final_sig)%2 == 0:
        final_sig = final_sig[:-1]
    return final_sig

def compute_spectrum_ehfo(org_sig, ps_SampleRate=1000, ps_FreqSeg=512, ps_MinFreqHz=10, ps_MaxFreqHz=500):
    #这个是旧的！！！！ Changed ps_SampleRate default from 2000 to 1000
    # print(org_sig.shape)
    final_sig = create_extended_sig_ehfo(org_sig)
    wave2000 = final_sig
    s_Len = len(final_sig)
    
    #exts_len = len(final_sig)
    s_HalfLen = math.floor(s_Len/2)+1

    v_WAxis = np.linspace(0, 2*np.pi, s_Len, endpoint=False)
    v_WAxis = v_WAxis* ps_SampleRate
    v_WAxisHalf = v_WAxis[:s_HalfLen]
    v_FreqAxis = np.linspace(ps_MinFreqHz, ps_MaxFreqHz,num=ps_FreqSeg)#ps_MinFreqHz:s_FreqStep:ps_MaxFreqHz
    v_FreqAxis = v_FreqAxis[::-1]
    
    v_InputSignalFFT = np.fft.fft(wave2000)
    ps_StDevCycles = 3
    m_GaborWT = np.zeros((ps_FreqSeg, s_Len),dtype=complex)
    for i, s_FreqCounter in enumerate(v_FreqAxis):
        v_WinFFT = np.zeros(s_Len)
        s_StDevSec = (1 / s_FreqCounter) * ps_StDevCycles
        v_WinFFT[:s_HalfLen] = np.exp(-0.5*np.power( v_WAxisHalf - (2* np.pi* s_FreqCounter) , 2)*
            (s_StDevSec**2))
        v_WinFFT = v_WinFFT* np.sqrt(s_Len)/ LA.norm(v_WinFFT, 2)
        m_GaborWT[i, :] = np.fft.ifft(v_InputSignalFFT* v_WinFFT)/np.sqrt(s_StDevSec)
        #original signal is +- 1s算完之后是+-2s, 取中心1s (e.g. +- 0.5s)
    # Slice indices updated for ps_SampleRate=1000 (center 1s = 1000 samples)
    center_idx = len(final_sig) // 2
    half_win_samples = ps_SampleRate // 2 # Samples for 0.5 seconds
    start_idx = center_idx - half_win_samples
    end_idx = center_idx + half_win_samples
    return resize(np.abs(m_GaborWT[:, start_idx:end_idx]), (224,224))

def normalized(a, max_ = 2000-11):
    c = (max_*(a - np.min(a))/np.ptp(a)).astype(int)
    c = c + 5 
    return c 

def construct_features_ehfo(raw_signal):
    # Removed length parameter
    signal_length = len(raw_signal)
    canvas_width = 2000 # Keep width fixed

    # HFO with spike Image
    spike_canvas = np.zeros((signal_length, canvas_width)) # Dynamic height
    hfo_spike_col_idx = normalized(raw_signal)
    time_idx = np.arange(signal_length) # Use signal_length for time index
    for ii in range(3):
        spike_canvas[time_idx, hfo_spike_col_idx-ii] = 256
        spike_canvas[time_idx, hfo_spike_col_idx+ii] = 256
    spike_image = resize(spike_canvas, (224, 224))

    # HFO Intensity Image (Mimic original behavior: fill row with raw_signal value)
    intensity_canvas = np.zeros((signal_length, canvas_width)) # Dynamic height
    # Assign raw_signal reshaped as column vector to broadcast across columns
    intensity_canvas[time_idx, :] = raw_signal[:, np.newaxis]
    hfo_image = resize(intensity_canvas, (224, 224))

    return spike_image, hfo_image

def test_feature_generation(output_folder):
    """
    Generates synthetic data and tests feature generation for 2000Hz and 1000Hz.

    Args:
        output_folder (str): Path to the folder where output images will be saved.
    """
    print(f"Saving test feature images to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    batch_size = 2
    raw_waveform_length_ms = 2000  # Example raw length in ms
    signal_freq_hz = 60  # Frequency of the synthetic sine wave

    # --- Parameters ---
    common_params = {
        "image_size": 224, # Although unused in current func, keep for consistency
        "freq_min_hz": 10,
        "freq_max_hz": 500,
        "raw_waveform_length": 2000,
    }
    model_name = "test_model" # Dummy model name
    device = 'cpu' # Use CPU for testing as parallel_process is CPU-bound
    n_jobs = 4 # Use fewer jobs for testing locally if needed

    # --- Test Case 1: 2000 Hz ---
    print("\n--- Testing 2000 Hz ---")
    fs_2000 = 2000
    num_samples_2000 = raw_waveform_length_ms * fs_2000 // 1000
    t_2000 = np.linspace(0, raw_waveform_length_ms / 1000, num_samples_2000, endpoint=False)
    signal_2000 = np.sin(2 * np.pi * signal_freq_hz * t_2000) + 0.1 * np.random.randn(num_samples_2000)
    # Add a burst in the middle for visual interest in features
    burst_start = num_samples_2000 // 2 - fs_2000 // 10 # 100ms burst
    burst_end = num_samples_2000 // 2 + fs_2000 // 10
    signal_2000[burst_start:burst_end] *= 3
    waveforms_2000 = np.stack([signal_2000] * batch_size, axis=0) # Create batch

    feature_param_2000 = common_params.copy()
    feature_param_2000["resample"] = fs_2000
    print(f"Input shape (2000Hz): {waveforms_2000.shape}")

    spec_batch_2k, spike_batch_2k, intensity_batch_2k = generate_feature_from_df_gpu_batch(
        waveforms_2000, feature_param_2000, model_name, device=device, n_jobs=n_jobs
    )
    print(f"Output shapes (2000Hz): Spectrum={spec_batch_2k.shape}, Spike={spike_batch_2k.shape}, Intensity={intensity_batch_2k.shape}")

    # Remove individual saving
    # for i in range(batch_size):
    #     plt.imsave(os.path.join(output_folder, f"spectrum_2000hz_{i}.png"), spec_batch_2k[i], cmap='viridis')
    #     plt.imsave(os.path.join(output_folder, f"spike_2000hz_{i}.png"), spike_batch_2k[i], cmap='gray')
    #     plt.imsave(os.path.join(output_folder, f"intensity_2000hz_{i}.png"), intensity_batch_2k[i], cmap='gray')

    # --- Test Case 2: 1000 Hz ---
    print("\n--- Testing 1000 Hz ---")
    fs_1000 = 1000
    num_samples_1000 = raw_waveform_length_ms * fs_1000 // 1000
    t_1000 = np.linspace(0, raw_waveform_length_ms / 1000, num_samples_1000, endpoint=False)
    signal_1000 = np.sin(2 * np.pi * signal_freq_hz * t_1000) + 0.1 * np.random.randn(num_samples_1000)
    # Add a burst in the middle
    burst_start = num_samples_1000 // 2 - fs_1000 // 10
    burst_end = num_samples_1000 // 2 + fs_1000 // 10
    signal_1000[burst_start:burst_end] *= 3
    waveforms_1000 = np.stack([signal_1000] * batch_size, axis=0) # Create batch

    feature_param_1000 = common_params.copy()
    feature_param_1000["raw_waveform_length"] = 2000
    feature_param_1000["resample"] = fs_1000
    print(f"Input shape (1000Hz): {waveforms_1000.shape}")

    spec_batch_1k, spike_batch_1k, intensity_batch_1k = generate_feature_from_df_gpu_batch(
        waveforms_1000, feature_param_1000, model_name, device=device, n_jobs=n_jobs
    )
    print(f"Output shapes (1000Hz): Spectrum={spec_batch_1k.shape}, Spike={spike_batch_1k.shape}, Intensity={intensity_batch_1k.shape}")

    # Remove individual saving
    # for i in range(batch_size):
    #     plt.imsave(os.path.join(output_folder, f"spectrum_1000hz_{i}.png"), spec_batch_1k[i], cmap='viridis')
    #     plt.imsave(os.path.join(output_folder, f"spike_1000hz_{i}.png"), spike_batch_1k[i], cmap='gray')
    #     plt.imsave(os.path.join(output_folder, f"intensity_1000hz_{i}.png"), intensity_batch_1k[i], cmap='gray')

    print("\n--- Generating Comparison Plots ---")
    # Iterate through the batch
    for i in range(batch_size):
        print(f"Generating comparison plots for batch item {i}...")

        # --- Spectrum Comparison ---
        fig_spec, axes_spec = plt.subplots(1, 3, figsize=(15, 5))
        fig_spec.suptitle(f'Spectrum Comparison - Item {i}', fontsize=16)

        im0 = axes_spec[0].imshow(spec_batch_2k[i], cmap='viridis', aspect='auto')
        axes_spec[0].set_title('2000 Hz')
        axes_spec[0].axis('off')
        fig_spec.colorbar(im0, ax=axes_spec[0], fraction=0.046, pad=0.04)

        im1 = axes_spec[1].imshow(spec_batch_1k[i], cmap='viridis', aspect='auto')
        axes_spec[1].set_title('1000 Hz')
        axes_spec[1].axis('off')
        fig_spec.colorbar(im1, ax=axes_spec[1], fraction=0.046, pad=0.04)

        spec_diff = spec_batch_2k[i] - spec_batch_1k[i]
        # Use a diverging colormap for difference, center around 0
        vmax = np.abs(spec_diff).max()
        im2 = axes_spec[2].imshow(spec_diff, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        axes_spec[2].set_title('Difference (2k - 1k)')
        axes_spec[2].axis('off')
        fig_spec.colorbar(im2, ax=axes_spec[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(os.path.join(output_folder, f"comparison_spectrum_{i}.png"))
        plt.close(fig_spec)

        # --- Spike Image Comparison ---
        fig_spike, axes_spike = plt.subplots(1, 3, figsize=(15, 5))
        fig_spike.suptitle(f'Spike Image Comparison - Item {i}', fontsize=16)

        axes_spike[0].imshow(spike_batch_2k[i], cmap='gray', aspect='auto')
        axes_spike[0].set_title('2000 Hz')
        axes_spike[0].axis('off')

        axes_spike[1].imshow(spike_batch_1k[i], cmap='gray', aspect='auto')
        axes_spike[1].set_title('1000 Hz')
        axes_spike[1].axis('off')

        spike_diff = spike_batch_2k[i] - spike_batch_1k[i]
        vmax_spike = np.abs(spike_diff).max()
        im2_spike = axes_spike[2].imshow(spike_diff, cmap='coolwarm', vmin=-vmax_spike, vmax=vmax_spike, aspect='auto')
        axes_spike[2].set_title('Difference (2k - 1k)')
        axes_spike[2].axis('off')
        # Add colorbar only if difference is not zero everywhere
        if vmax_spike > 1e-9: # Use a small tolerance for floating point comparison
            fig_spike.colorbar(im2_spike, ax=axes_spike[2], fraction=0.046, pad=0.04)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, f"comparison_spike_{i}.png"))
        plt.close(fig_spike)

        # --- Intensity Image Comparison ---
        fig_intensity, axes_intensity = plt.subplots(1, 3, figsize=(15, 5))
        fig_intensity.suptitle(f'Intensity Image Comparison - Item {i}', fontsize=16)

        axes_intensity[0].imshow(intensity_batch_2k[i], cmap='gray', aspect='auto')
        axes_intensity[0].set_title('2000 Hz')
        axes_intensity[0].axis('off')

        axes_intensity[1].imshow(intensity_batch_1k[i], cmap='gray', aspect='auto')
        axes_intensity[1].set_title('1000 Hz')
        axes_intensity[1].axis('off')

        intensity_diff = intensity_batch_2k[i] - intensity_batch_1k[i]
        vmax_intensity = np.abs(intensity_diff).max()
        im2_intensity = axes_intensity[2].imshow(intensity_diff, cmap='coolwarm', vmin=-vmax_intensity, vmax=vmax_intensity, aspect='auto')
        axes_intensity[2].set_title('Difference (2k - 1k)')
        axes_intensity[2].axis('off')
        if vmax_intensity > 1e-9: # Use a small tolerance
            fig_intensity.colorbar(im2_intensity, ax=axes_intensity[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_folder, f"comparison_intensity_{i}.png"))
        plt.close(fig_intensity)

    print("\nTest finished.")


if __name__ == "__main__":
    output_dir = "verification/ehfo_verification/"
    test_feature_generation(output_dir)


    

