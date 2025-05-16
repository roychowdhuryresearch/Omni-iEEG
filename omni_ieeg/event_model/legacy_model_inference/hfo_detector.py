import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from HFODetector import ste, mni, hil
from omni_ieeg.utils.utils_edf import concate_edf
from omni_ieeg.utils.utils_multiprocess import parallel_process
from omni_ieeg.dataloader.datafilter import DataFilter
import argparse

def detect_ste(edf_path, resample=1000):
    data, channels = concate_edf([edf_path], resample)
    upper_freq = 300 if resample == 1000 else 500
    ste_detector = ste.STEDetector(sample_freq=resample, filter_freq=[80, upper_freq], 
                rms_window=3*1e-3, min_window=6*1e-3, min_gap=10 * 1e-3, 
                epoch_len=600, min_osc=6, rms_thres=5, peak_thres=3,
                n_jobs=32, front_num=1)
    channel_names, start_end = ste_detector.detect_multi_channels(data, channels)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    if len(start_end) == 0:
        return pd.DataFrame({"name":[], "start":[], "end":[], "detector": []})
    start_end = np.concatenate(start_end)
    HFO_ste_df = pd.DataFrame({"name":channel_names,"start":start_end[:,0],"end":start_end[:,1], "detector": "ste"})
    return HFO_ste_df
def detect_mni(edf_path, resample=1000):
    data, channels = concate_edf([edf_path], resample)
    upper_freq = 300 if resample == 1000 else 500
    mni_detector = mni.MNIDetector(resample, filter_freq=[80, upper_freq], 
                epoch_time=10, epo_CHF=60, per_CHF=95/100, 
                min_win=10*1e-3, min_gap=10*1e-3, thrd_perc=99.9999/100, 
                base_seg=125*1e-3, base_shift=0.5, base_thrd=0.67, base_min=5,
                n_jobs=32, front_num=1)
    channel_names, start_end = mni_detector.detect_multi_channels(data, channels)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    if len(start_end) == 0:
        return pd.DataFrame({"name":[], "start":[], "end":[], "detector": []})
    start_end = np.concatenate(start_end)
    HFO_mni_df = pd.DataFrame({"name":channel_names,"start":start_end[:,0],"end":start_end[:,1], "detector": "mni"})
    return HFO_mni_df

def detect_hil(edf_path, resample=1000):
    data, channels = concate_edf([edf_path], resample)
    upper_freq = 300 if resample == 1000 else 500
    hil_detector = hil.HILDetector(sample_freq=resample, filter_freq=[80, upper_freq],
                               sd_thres=5, min_window=10*1e-3,
                               epoch_len=3600, n_jobs=32, front_num=1)
    channel_names, start_end = hil_detector.detect_multi_channels(data, channels)
    channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
    start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
    if len(start_end) == 0:
        return pd.DataFrame({"name":[], "start":[], "end":[], "detector": []})
    start_end = np.concatenate(start_end)
    HFO_hil_df = pd.DataFrame({"name": channel_names,"start":start_end[:,0],"end":start_end[:,1], "detector": "hil"})
    return HFO_hil_df

def detect_hfo_single_edf(edf_path, resample=1000):
    # print("test, processing edf file: ", edf_path)
    hfo_ste_df = detect_ste(edf_path, resample)
    hfo_mni_df = detect_mni(edf_path, resample)
    hfo_hil_df = detect_hil(edf_path, resample)
    hfo_df = pd.concat([hfo_ste_df, hfo_mni_df, hfo_hil_df])
    return hfo_df

def detect_hfo_with_dataloader(dataset_root, resample=1000):
    data_filter = DataFilter(dataset_root)
    filtered_dataset = data_filter.apply_filters()

    all_participants = filtered_dataset.get_patients()
    base_save_path = os.path.join(dataset_root, "derivatives", "hfo")

    print(f"Processing {len(all_participants)} participants using DataFilter...")
    for participant_id in tqdm(all_participants, desc="Processing participants"):
        edf_paths = filtered_dataset.get_edfs_by_patient(participant_id)

        for edf_path in tqdm(edf_paths, desc=f"Processing EDFs for {participant_id}", leave=False):
            try:
                run_name = os.path.basename(edf_path)
                ieeg_folder = os.path.dirname(edf_path)
                session_folder = os.path.dirname(ieeg_folder)
                session_id = os.path.basename(session_folder)

                save_run_path_prefix = os.path.join(base_save_path, participant_id, session_id, 'ieeg')
                os.makedirs(save_run_path_prefix, exist_ok=True)
                save_run_path = os.path.join(save_run_path_prefix, run_name.replace(".edf", ".csv"))

                if os.path.exists(save_run_path):
                    print(f"Skipping {edf_path} because it already exists at {save_run_path}")
                    continue

                hfo_df = detect_hfo_single_edf(edf_path, resample)

                hfo_df["participant"] = participant_id
                hfo_df["session"] = session_id
                hfo_df["file_name"] = run_name
                hfo_df.to_csv(save_run_path, index=False)
            except Exception as e:
                 print(f"Error processing {edf_path} for participant {participant_id}: {e}")

    print("Finished processing all participants.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset, should be the path to the dataset")
    parser.add_argument("--resample_freq", type=int, required=True, help="Resample frequency, recommended 1000")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    resample_freq = args.resample_freq

    detect_hfo_with_dataloader(dataset_path, resample=resample_freq)

    
    
    
    
    



    
    
    


