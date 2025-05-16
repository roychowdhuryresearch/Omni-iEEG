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






def process_positive_row(row_data):
    row, filtered_dataset, save_path, frequency, dataset_path = row_data
    folder_size = get_folder_size_gb(save_path)
    if folder_size > 150:
        print(f"Folder size is {folder_size}GB, stopping extraction")
        return None
    patient_name = row['patient_name']
    edf_name = row['edf_name']
    edf_name = os.path.join(dataset_path, edf_name)
    data, edf_channels = concate_edf([edf_name], resample=frequency)
    channels_df = filtered_dataset.get_channels_for_edf(edf_name)
    good_channels = channels_df[(channels_df['resection'] == 0) & (channels_df['good'] == 1)].name.tolist()
    missing_channels = []
    data_list = []
    channel_names = []
    labels = []
    start_indices = []
    end_indices = []
    for channel_name in good_channels:
        matching_indices = np.where(edf_channels == channel_name)[0]
        if len(matching_indices) == 0:
            missing_channels.append(channel_name)
            continue
        channel_idx = matching_indices[0]
        samples, sample_start_indices, sample_end_indices = extract_random_samples(data[channel_idx], frequency, 60, 5)
        if len(samples) > 0:
            data_list.extend(samples)
            channel_names.extend([channel_name] * len(samples))
            labels.extend([1] * len(samples))
            start_indices.extend(sample_start_indices)
            end_indices.extend(sample_end_indices)
    save_npz_path = os.path.join(save_path, "positive", os.path.basename(edf_name).replace(".edf", ".npz"))
    os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
    np.savez(save_npz_path, data=data_list, name=channel_names, labels=labels, start_indices=start_indices, end_indices=end_indices, patient=patient_name, edf_name=edf_name)
    return {
        'patient_name': patient_name,
        'edf_name': edf_name,
        'samples': len(labels),
        'missing_channels': missing_channels,
    }

def process_negative_row(row_data):
    row, filtered_dataset, save_path, frequency, dataset_path = row_data
    folder_size = get_folder_size_gb(save_path)
    if folder_size > 150:
        print(f"Folder size is {folder_size}GB, stopping extraction")
        return None
    patient_name = row['patient_name']
    edf_name = row['edf_name']
    edf_name = os.path.join(dataset_path, edf_name)
    data, edf_channels = concate_edf([edf_name], resample=frequency)
    channels_df = filtered_dataset.get_channels_for_edf(edf_name)
    if row['outcome'] == 1:
        bad_channels = channels_df[(channels_df['soz'] == 1) & (channels_df['good'] == 1) & (channels_df['resection'] != 0)].name.tolist()
    else:
        bad_channels = channels_df[(channels_df['soz'] == 1) & (channels_df['good'] == 1)].name.tolist()
    missing_channels = []
    data_list = []
    channel_names = []
    labels = []
    start_indices = []
    end_indices = []
    for channel_name in bad_channels:
        matching_indices = np.where(edf_channels == channel_name)[0]
        if len(matching_indices) == 0:
            missing_channels.append(channel_name)
            continue
        channel_idx = matching_indices[0]
        samples, sample_start_indices, sample_end_indices = extract_random_samples(data[channel_idx], frequency, 60, 5)
        if len(samples) > 0:
            data_list.extend(samples)
            channel_names.extend([channel_name] * len(samples))
            labels.extend([0] * len(samples))
            start_indices.extend(sample_start_indices)
            end_indices.extend(sample_end_indices)
    save_npz_path = os.path.join(save_path, "negative", os.path.basename(edf_name).replace(".edf", ".npz"))
    os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
    np.savez(save_npz_path, data=data_list, name=channel_names, labels=labels, start_indices=start_indices, end_indices=end_indices, patient=patient_name, edf_name=edf_name)
    return {
        'patient_name': patient_name,
        'edf_name': edf_name,
        'samples': len(labels),
        'missing_channels': missing_channels,
    }

def extract_positive_labels(final_split, dataset_path, save_path, frequency=300, n_jobs=8):
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    
    # we first want to get interictal, frequency at least 900, and length at least 300
    training_split = final_split[(final_split['split'] == 'train') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]
    
    # we then want to get patient with outcome 1 and has resection
    positive_split = training_split[(training_split['outcome'] == 1) & (training_split['has_resection'] == True)]
    
    # Prepare data for parallel processing
    rows_data = [(row, filtered_dataset, save_path, frequency, dataset_path) for _, row in positive_split.iterrows()]
    
    # Process rows in parallel using the robust implementation
    print(f"Processing {len(rows_data)} positive rows with {n_jobs} parallel jobs")
    edf_info = robust_parallel_process(rows_data, process_positive_row, n_jobs=n_jobs, desc="Extracting positive labels")
    
    # Filter out None values (if any)
    edf_info = [item for item in edf_info if item is not None]
    
    # Save results
    edf_info_df = pd.DataFrame(edf_info)
    edf_info_df.to_csv(os.path.join(save_path, "positive", "edf_info.csv"), index=False)
    
def extract_negative_labels(final_split, dataset_path, save_path, frequency=300, n_jobs=8):
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    
    # we first want to get interictal, frequency at least 900, and length at least 300
    training_split = final_split[(final_split['split'] == 'train') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]
    
    # we then want to get patient with soz
    positive_split = training_split[training_split['has_soz'] == True]
    
    # Prepare data for parallel processing
    rows_data = [(row, filtered_dataset, save_path, frequency, dataset_path) for _, row in positive_split.iterrows()]
    
    # Process rows in parallel using the robust implementation
    print(f"Processing {len(rows_data)} negative rows with {n_jobs} parallel jobs")
    edf_info = robust_parallel_process(rows_data, process_negative_row, n_jobs=n_jobs, desc="Extracting negative labels")
    
    # Filter out None values (if any)
    edf_info = [item for item in edf_info if item is not None]
    
    # Save results
    edf_info_df = pd.DataFrame(edf_info)
    edf_info_df.to_csv(os.path.join(save_path, "negative", "edf_info.csv"), index=False)

def extract_random_samples(channel_data, sampling_rate, duration_seconds, max_samples=5):
    """
    Randomly sample segments from channel data, excluding first and last 1 second
    
    Parameters:
    - channel_data: 1D array of channel data
    - sampling_rate: Sampling rate in Hz
    - duration_seconds: Duration of each sample in seconds
    - max_samples: Maximum number of samples to extract
    
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
    
    # Simple approach: calculate how many windows could fit with minimal overlap
    # For example, if data is 65 seconds and window is 60 seconds, we might want just 1 sample
    # If data is 120 seconds and window is 60 seconds, we might want 2 samples
    
    # Calculate maximum number of samples based on data length
    max_possible_samples = max(1, data_length // (sample_length // 2))  # Allow 50% overlap
    num_samples = min(max_samples, max_possible_samples)
    
    # Determine valid range for start indices
    valid_range = data_length - sample_length
    
    # Randomly sample start indices without replacement if possible
    if valid_range > num_samples:
        random_indices = sorted(random.sample(range(valid_range), num_samples))
    else:
        # If not enough range, space them evenly (fallback)
        random_indices = [i * valid_range // max(1, num_samples-1) for i in range(num_samples)]
    
    # Extract samples
    samples = []
    start_indices = []
    end_indices = []
    
    for idx in random_indices:
        # Get sample
        sample = usable_data[idx:idx+sample_length]
        samples.append(sample)
        
        # Calculate indices relative to original data
        original_start = offset + idx
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
    extract_positive_labels(final_split, dataset_path, save_path)
    extract_negative_labels(final_split, dataset_path, save_path)
    
    
    