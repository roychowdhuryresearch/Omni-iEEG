from omni_ieeg.utils.utils_edf import concate_edf
import numpy as np
from omni_ieeg.utils.utils_features import generate_feature_from_df
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import mne
# Import the necessary dataloader class
from omni_ieeg.dataloader.datafilter import DataFilter


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
    annotation_csv_path = "/mnt/SSD1/nipsdataset/dataset/annotation/csv"
    annotation_edf_path = "/mnt/SSD1/nipsdataset/dataset/annotation/edf"
    save_feature_path_all = "/mnt/SSD2/chenda/nipsdataset/annotation_validation_feature_zurich"
    csv_files = os.listdir(annotation_csv_path)
    # sample_rate = 1000
    time_window = 2000
    for csv_file in csv_files:
        if csv_file.startswith("sub-zurich"):
            print(f"Processing {csv_file}")
            hfo_csv_path = os.path.join(annotation_csv_path, csv_file)
            edf_file = os.path.join(annotation_edf_path, csv_file.replace(".csv", ".edf"))
            # get raw freq
            raw_content = mne.io.read_raw_edf(edf_file)
            raw_freq = raw_content.info['sfreq']
            try:
                print(f"Processing {edf_file}")
                print(f"Raw Frequency: {raw_freq}")
                # Construct paths based on the edf_file path from the dataloader
                # Assumes edf_file path format like: /path/to/dataset_root/participant_id/session_id/ieeg/run_name_ieeg.edf
                save_feature_path = edf_file.replace(".edf", ".npz").replace(annotation_edf_path, save_feature_path_all)

                if not os.path.exists(hfo_csv_path):
                    print(f"Warning: Skipping {edf_file} because corresponding HFO CSV not found at {hfo_csv_path}")
                    continue

                # if os.path.exists(save_feature_path):
                #     print(f"Skipping {edf_file} because feature file already exists at {save_feature_path}")
                #     continue

                data, channels = concate_edf([edf_file], resample=None)

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

                start, end, channel_name, hfo_waveforms, real_start, real_end, is_boundary = generate_feature_from_df(df, data, channels, raw_freq, time_window)

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
                np.savez(save_feature_path, start=start, end=end, real_start = real_start, real_end = real_end, is_boundary=is_boundary, name=channel_name, hfo_waveforms=hfo_waveforms, detector=df["detector"].values, file_name=df['edf_file'].values, artifact=df['artifact'].values, spike=df['spike'].values)
                print(f"Saved {save_feature_path}")


            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
                # Continue to the next file on error

