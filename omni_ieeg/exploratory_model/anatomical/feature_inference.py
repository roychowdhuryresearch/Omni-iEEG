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
from argparse import ArgumentParser
def process_row(row_data):
    row, filtered_dataset, save_path, frequency, label_mapping, label_mapping_proposal1, label_mapping_proposal2, dataset_path = row_data
    folder_size = get_folder_size_gb(save_path)
    if folder_size > 150:
        print(f"Folder size is {folder_size}GB, stopping extraction")
        return None
    patient_name = row['patient_name']
    edf_name = row['edf_name']
    edf_name = os.path.join(dataset_path, "edf", edf_name)
    save_npz_path = os.path.join(save_path, os.path.basename(edf_name).replace(".edf", ".npz"))
    if os.path.exists(save_npz_path):
        print(f"File {save_npz_path} already exists, skipping")
        return {
            'patient_name': patient_name,
            'edf_name': edf_name,
            'samples': 0,
            'missing_channels': [],
        }
    patient_outcome = row['outcome']
    data, edf_channels = concate_edf([edf_name], resample=frequency)
    channels_df = filtered_dataset.get_channels_for_edf(edf_name)
    missing_channels = []
    data_list = []
    channel_names = []
    labels = []
    labels_proposal1 = []
    labels_proposal2 = []
    start_indices = []
    end_indices = []
    anatomical_labels = []
    for _, inner_row in channels_df.iterrows():
        channel_name = inner_row['name']
        anatomical_label = inner_row['anatomical']
        label = label_mapping[anatomical_label]
        label_proposal1 = label_mapping_proposal1[anatomical_label]
        label_proposal2 = label_mapping_proposal2[anatomical_label]
        # if label == -1:
        #     continue
        matching_indices = np.where(edf_channels == channel_name)[0]
        if len(matching_indices) == 0:
            missing_channels.append(channel_name)
            continue
        channel_idx = matching_indices[0]
        samples, sample_start_indices, sample_end_indices = extract_uniform_samples(data[channel_idx], frequency, 300)
        if len(samples) > 0:
            data_list.extend(samples)
            channel_names.extend([channel_name] * len(samples))
            labels.extend([label] * len(samples))
            labels_proposal1.extend([label_proposal1] * len(samples))
            labels_proposal2.extend([label_proposal2] * len(samples))
            start_indices.extend(sample_start_indices)
            end_indices.extend(sample_end_indices)
            anatomical_labels.extend([anatomical_label] * len(samples))
    os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
    np.savez(save_npz_path, data=data_list, name=channel_names, labels=labels, labels_proposal1=labels_proposal1, labels_proposal2=labels_proposal2, start_indices=start_indices, end_indices=end_indices, patient=patient_name, edf_name=edf_name, anatomical_labels=anatomical_labels)
    return {
        'patient_name': patient_name,
        'edf_name': edf_name,
        'samples': len(labels),
        'missing_channels': missing_channels,
    }

def extract_labels(final_split, dataset_path, save_path, label_mapping, label_mapping_proposal1, label_mapping_proposal2, frequency=200, n_jobs=16):
    random.seed(42)
    np.random.seed(42)
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    
    # we first want to get interictal, frequency at least 900, and length at least 300
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 302)]
    
    testing_split = testing_split[testing_split['has_anatomical'] == True]
    print(f"Got {len(testing_split['patient_name'].unique())} anatomical patients")
    
    # Prepare data for parallel processing
    rows_data = [(row, filtered_dataset, save_path, frequency, label_mapping, label_mapping_proposal1, label_mapping_proposal2, dataset_path) for _, row in testing_split.iterrows()]
    
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
    channel_data = channel_data.astype(np.float32)
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
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--anatomical_mapping_path", type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    final_split_path = os.path.join(dataset_path, "final_split.csv")
    final_split = pd.read_csv(final_split_path)
    final_split = final_split[final_split['dataset'] != "Multicenter"]
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    anatomical_label_mapping = pd.read_csv(args.anatomical_mapping_path)
    label_2_id = {
        "frontal": 0,
        "temporal": 1,
        "parietal": 2,
        "occipital": 3,
        "limbic": 4,
        "-1": -1
    }
    label_2_id_proposal1 = {
    }
    unique_proposals1 = anatomical_label_mapping['Proposal_1'].unique()
    proposal1_id = 0
    for proposal1 in unique_proposals1:
        if proposal1 == "-1":
            continue
        if proposal1 not in label_2_id_proposal1.keys():
            label_2_id_proposal1[proposal1] = proposal1_id
            proposal1_id += 1
    print(f"label_2_id_proposal1: {label_2_id_proposal1}")
    
    label_2_id_proposal2 = {
    }
    unique_proposals2 = anatomical_label_mapping['Proposal_2'].unique()
    proposal2_id = 0
    for proposal2 in unique_proposals2:
        if proposal2 == "-1":
            continue
        if proposal2 not in label_2_id_proposal2.keys():
            label_2_id_proposal2[proposal2] = proposal2_id
            proposal2_id += 1
    print(f"label_2_id_proposal2: {label_2_id_proposal2}")
        
    label_mapping = {}
    label_mapping_proposal1 = {}
    label_mapping_proposal2 = {}
    for index, row in anatomical_label_mapping.iterrows():
        label_mapping[row['label']] = label_2_id[row['subcategory']]
        if row['label'] == "-1":
            label_mapping_proposal1[row['label']] = -1
            label_mapping_proposal2[row['label']] = -1
        else:
            label_mapping_proposal1[row['label']] = label_2_id_proposal1[row['Proposal_1']]
            label_mapping_proposal2[row['label']] = label_2_id_proposal2[row['Proposal_2']]
    print(f"label_mapping: {label_mapping}")
    print(f"label_mapping_proposal1: {label_mapping_proposal1}")
    print(f"label_mapping_proposal2: {label_mapping_proposal2}")
    extract_labels(final_split, dataset_path, save_path, label_mapping, label_mapping_proposal1, label_mapping_proposal2)
    
    
    