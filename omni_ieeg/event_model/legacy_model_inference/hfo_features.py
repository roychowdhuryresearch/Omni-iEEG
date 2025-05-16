from omni_ieeg.utils.utils_edf import concate_edf
import numpy as np
from omni_ieeg.utils.utils_features import generate_feature_from_df
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import subprocess
# Import the necessary dataloader class
from omni_ieeg.dataloader.datafilter import DataFilter
import argparse


def get_folder_size_gb(folder_path):
    """Calculate the total size of a folder in GB using du command"""
    try:
        # Run du command to get size in bytes
        result = subprocess.run(['du', '-sb', folder_path], capture_output=True, text=True, check=True)
        # Extract the size in bytes (first number in the output)
        size_bytes = int(result.stdout.split()[0])
        # Convert to GB
        return size_bytes / (1024**3)
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"Error getting folder size: {e}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset, should be the path to the dataset")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the features")
    parser.add_argument("--final_split_path", type=str, required=True, help="Path to the final split file, should be omniieeg/derivatives/datasplit/final_split.csv")
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate, recommended 1000")
    parser.add_argument("--time_window", type=int, required=True, help="Time window, recommended 2000")
    parser.add_argument("--max_folder_size_gb", type=int, required=True, help="Maximum folder size in GB, recommended 150")
    args = parser.parse_args()
    
    
    dataset_root = args.dataset_root
    output_folder = args.output_folder
    final_split_path = args.final_split_path
    final_split = pd.read_csv(final_split_path)
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]
    hfo_folder = os.path.join(dataset_root, "derivatives", "hfo") # Use dataset_root for hfo_folder base
    sample_rate = args.sample_rate
    time_window = args.time_window
    max_folder_size_gb = args.max_folder_size_gb  # Maximum folder size in GB

    # Initialize DataFilter
    data_filter = DataFilter(dataset_root)
    # Apply filters if needed, otherwise get all data
    # e.g., filtered_dataset = data_filter.apply_filters(patient_filter=lambda row: row['dataset'] == 'zurich')
    filtered_dataset = data_filter.apply_filters()

    all_participants = filtered_dataset.get_patients()

    print(f"Processing {len(all_participants)} participants using DataFilter...")
    for participant_id in tqdm(all_participants, desc="Processing participants"):
        if participant_id not in testing_split['patient_name'].values:
            continue
        edf_files = filtered_dataset.get_edfs_by_patient(participant_id)

        # Check if participant folder exists in derivatives (needed for HFO csv)
        # This assumes HFO detection has already run and created the structure
        pt_hfo_folder_check = os.path.join(hfo_folder, participant_id)
        if not os.path.exists(pt_hfo_folder_check):
             print(f"Warning: Skipping {participant_id} because derivatives/hfo folder does not exist at {pt_hfo_folder_check}")
             continue

        for edf_file in tqdm(edf_files, desc=f"Processing EDFs for {participant_id}", leave=False):
            if edf_file not in testing_split['edf_name'].values:
                # print(f"Skipping {edf_file} because it is not in the testing split")
                continue
            try:
                print(f"Processing {edf_file}")
                # Construct paths based on the edf_file path from the dataloader
                # Assumes edf_file path format like: /path/to/dataset_root/participant_id/session_id/ieeg/run_name_ieeg.edf
                hfo_csv_path = edf_file.replace(".edf", ".csv").replace(dataset_root, hfo_folder)
                save_feature_path = os.path.basename(edf_file).replace(".edf", ".npz")
                save_feature_path = os.path.join(output_folder, save_feature_path)
                channel_df = filtered_dataset.get_channels_for_edf(edf_file)
                channel_df_good = channel_df[channel_df['good'] == 1]
                channel_good_name = channel_df_good['name'].tolist()

                if not os.path.exists(hfo_csv_path):
                    print(f"Warning: Skipping {edf_file} because corresponding HFO CSV not found at {hfo_csv_path}")
                    continue

                if os.path.exists(save_feature_path):
                    print(f"Skipping {edf_file} because feature file already exists at {save_feature_path}")
                    continue

                data, channels = concate_edf([edf_file], resample=sample_rate)

                channels = channels.tolist()
                # Clean channel names
                for i, ch in enumerate(channels):
                    if ch == 'N/A' or ch == 'nan' or ch == 'NaN' or pd.isna(ch):
                        print(f"Channel {i} ('{ch}') in {edf_file} is replaced with empty string")
                channels = ['' if ch == 'N/A' or ch == 'nan' or ch == 'NaN' or pd.isna(ch) else ch for ch in channels]

                # Check for multiple empty channels
                if channels.count('') > 1:
                    print(f"Warning: Expected no more than 1 channel with empty string in {edf_file}, got {channels.count('')}. Proceeding anyway.")
                    # assert channels.count('') <= 1, f"Expected no more than 1 channel with empty string, got {channels.count('')}"


                channels = np.array(channels) # Convert back for feature generation

                df = pd.read_csv(hfo_csv_path)
                # Clean channel names in DataFrame
                df["name"] = df["name"].fillna("")
                df["name"] = df["name"].replace(['N/A', 'nan', 'NaN'], '') # Replace various NaN representations
                before_cleaning_good = len(df)
                df = df[df['name'].isin(channel_good_name)]
                after_cleaning_good = len(df)
                print(f"Before cleaning good: {before_cleaning_good}, after cleaning good: {after_cleaning_good}")

                start, end, channel_name, hfo_waveforms, real_start, real_end, is_boundary = generate_feature_from_df(df, data, channels, sample_rate, time_window)

                # Basic validation before saving
                if not (len(start) == len(end) == len(channel_name) == len(df)):
                     print(f"Warning: Length mismatch after generate_feature_from_df for {edf_file}. Skipping save.")
                     print(f"Expected length {len(df)}, got start:{len(start)}, end:{len(end)}, name:{len(channel_name)}")
                     continue
                if not (len(real_start) == len(real_end) == len(is_boundary) == len(start)):
                    print(f"Warning: Length mismatch in real_start/end/boundary for {edf_file}. Skipping save.")
                    continue
                if hfo_waveforms.shape[0] != len(start):
                     print(f"Warning: Waveform shape mismatch for {edf_file}. Expected {len(start)} waveforms, got {hfo_waveforms.shape[0]}. Skipping save.")
                     continue


                # Save the hfo waveforms
                os.makedirs(os.path.dirname(save_feature_path), exist_ok=True)
                np.savez(save_feature_path, start=start, end=end, real_start = real_start, real_end = real_end, is_boundary=is_boundary, name=channel_name, hfo_waveforms=hfo_waveforms, detector=df["detector"].values, participant=df['participant'].values, session=df['session'].values, file_name=df['file_name'].values)
                print(f"Saved {save_feature_path}")

                # Check folder size after saving
                folder_size_gb = get_folder_size_gb(output_folder)
                print(f"Current output folder size: {folder_size_gb:.2f} GB")

                if folder_size_gb > max_folder_size_gb:
                    print(f"Output folder size ({folder_size_gb:.2f} GB) exceeds the maximum allowed size ({max_folder_size_gb} GB). Stopping job.")
                    exit(1) # Use exit(1) to indicate an error stop

            except Exception as e:
                print(f"Error processing {edf_file} for participant {participant_id}: {e}")
                # Continue to the next file on error

    print("Finished processing all participants.")
