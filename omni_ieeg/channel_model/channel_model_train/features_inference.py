import pandas as pd
import numpy as np
from omni_ieeg.dataloader.datafilter import DataFilter
from omni_ieeg.utils.utils_edf import concate_edf
from tqdm import tqdm
import random
import mne
from omni_ieeg.utils.utils_multiprocess import get_folder_size_gb, robust_parallel_process
from omni_ieeg.utils.utils_edf import concate_edf
import os
import argparse
def process_row(row_data):
    row, filtered_dataset, save_path, frequency = row_data
    folder_size = get_folder_size_gb(save_path)
    if folder_size > 150:
        print(f"Folder size is {folder_size}GB, stopping extraction")
        return None
    patient_name = row['patient_name']
    edf_name = row['edf_name']
    patient_outcome = row['outcome']
    data, edf_channels = concate_edf([edf_name], resample=frequency)
    channels_df = filtered_dataset.get_channels_for_edf(edf_name)
    missing_channels = []
    data_list = []
    channel_names = []
    labels = []
    start_indices = []
    end_indices = []
    good_channels = channels_df[channels_df['good'] == 1]
    for _, inner_row in good_channels.iterrows():
        channel_name = inner_row['name']
        channel_soz = inner_row['soz']
        channel_resection = inner_row['resection']
        label = -1
        if patient_outcome == 1 and channel_resection == 0:
            label = 1
        elif channel_soz == 1:
            label = 0
        matching_indices = np.where(edf_channels == channel_name)[0]
        if len(matching_indices) == 0:
            missing_channels.append(channel_name)
            continue
        channel_idx = matching_indices[0]
        samples, sample_start_indices, sample_end_indices = extract_uniform_samples(data[channel_idx], frequency, 60)
        if len(samples) > 0:
            data_list.extend(samples)
            channel_names.extend([channel_name] * len(samples))
            labels.extend([label] * len(samples))
            start_indices.extend(sample_start_indices)
            end_indices.extend(sample_end_indices)
    
    save_npz_path = os.path.join(save_path, os.path.basename(edf_name).replace(".edf", ".npz"))
    os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
    np.savez(save_npz_path, data=data_list, name=channel_names, labels=labels, start_indices=start_indices, end_indices=end_indices, patient=patient_name, edf_name=edf_name)
    return {
        'patient_name': patient_name,
        'edf_name': edf_name,
        'samples': len(labels),
        'missing_channels': missing_channels,
    }

def extract_labels(final_split, dataset_path, save_path, frequency=300, n_jobs=8):
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    
    # we first want to get interictal, frequency at least 900, and length at least 300
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]
    
    
    # Prepare data for parallel processing
    rows_data = [(row, filtered_dataset, save_path, frequency) for _, row in testing_split.iterrows()]
    
    # Process rows in parallel using the robust implementation
    print(f"Processing {len(rows_data)} positive rows with {n_jobs} parallel jobs")
    edf_info = robust_parallel_process(rows_data, process_row, n_jobs=n_jobs, desc="Extracting positive labels")
    
    # Filter out None values (if any)
    edf_info = [item for item in edf_info if item is not None]
    
    # Save results
    edf_info_df = pd.DataFrame(edf_info)
    edf_info_df.to_csv(os.path.join(save_path, "edf_info.csv"), index=False)

def extract_uniform_samples(channel_data, sampling_rate, duration_seconds):
    """
    Uniformly sample segments from channel data without overlap, excluding first and last 1 second
    
    Parameters:
    - channel_data: 1D array of channel data
    - sampling_rate: Sampling rate in Hz
    - duration_seconds: Duration of each sample in seconds
    
    Returns:
    - Tuple containing:
      - List of data segments
      - List of start indices (relative to original data)
      - List of end indices (relative to original data)
    """
    # Calculate points to skip at beginning and end (1 second each)
    points_to_skip = sampling_rate
    
    # Skip first and last second if there's enough data
    offset = 0
    if len(channel_data) > 2 * points_to_skip:
        usable_data = channel_data[points_to_skip:-points_to_skip]
        offset = points_to_skip  # Track offset for calculating original indices
    else:
        return [], [], []
    
    sample_length = int(duration_seconds * sampling_rate)
    data_length = len(usable_data)
    
    # Skip if data is shorter than required sample length
    if data_length < sample_length:
        return [], [], []
    
    # Calculate number of complete samples we can fit without overlap
    num_samples = data_length // sample_length
    
    # Extract samples
    samples = []
    start_indices = []
    end_indices = []
    
    for i in range(num_samples):
        start_idx = i * sample_length
        end_idx = start_idx + sample_length
        
        # Get sample
        sample = usable_data[start_idx:end_idx]
        samples.append(sample)
        
        # Calculate indices relative to original data
        original_start = offset + start_idx
        original_end = original_start + sample_length
        start_indices.append(original_start)
        end_indices.append(original_end)
    
    return samples, start_indices, end_indices
            
            
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_split_path", type=str, required=True, help="Path to the splitting file, should be omniieeg/derivatives/datasplit/final_split.csv")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset, should be the path to the dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the features")
    args = parser.parse_args()
    
    
    final_split = pd.read_csv(args.final_split_path)
    final_split = final_split[final_split['dataset'] != "Multicenter"]
    dataset_path = args.dataset_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    extract_labels(final_split, dataset_path, save_path)
    
    
    