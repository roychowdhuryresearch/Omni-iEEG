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
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import auc

def calculate_metrics(result_dict):
    for model_name, model_df in result_dict.items():
        print(f"=====For {model_name}")
        window_level_ratio = []
        window_level_true = []
        edf_level_list = []
        edf_level_true = []
        unique_edf_names = model_df['edf_name'].unique()
        for edf_name in unique_edf_names:
            edf_df = model_df[model_df['edf_name'] == edf_name]
            grouped_indices = edf_df.groupby('name')['start_indices'].apply(lambda x: tuple(sorted(x.unique())))
            assert grouped_indices.nunique() == 1, f"all channel does not have all start indices" 
            assert len(edf_df['channel_true'].unique()) == 1, f"Channel true values should be same for {edf_name}"
            label = edf_df['channel_true'].iloc[0]
            # unique_start = edf_df['']
            unique_start = edf_df['start_indices'].unique()
            edf_ratio_list = []
            for start_time in unique_start:
                start_df = edf_df[edf_df['start_indices'] == start_time]
                assert start_df['edf_name'].nunique() == 1
                # aggregate predictions across channels for this start time
                channel_pred_sum = start_df['channel_pred'].sum()
                channel_pred_ratio = channel_pred_sum / len(start_df)
                window_level_ratio.append(channel_pred_ratio)
                window_level_true.append(label)
                edf_ratio_list.append(channel_pred_ratio)
            edf_level_list.append(edf_ratio_list)
            edf_level_true.append(label)
        #------------calculate window level metrics-----------------
        num_1s = window_level_true.count(1)
        num_0s = window_level_true.count(0)
        print(f"Number of 1s: {num_1s}, Number of 0s: {num_0s}")


        fpr, tpr, roc_thresholds = roc_curve(window_level_true, window_level_ratio)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)

        optimal_threshold = roc_thresholds[optimal_idx]
        window_level_pred = (window_level_ratio > optimal_threshold).astype(int)
        
        precision_value = precision_score(window_level_true, window_level_pred, average='macro')
        recall_value = recall_score(window_level_true, window_level_pred, average='macro')
        f1_value = f1_score(window_level_true, window_level_pred, average='macro')
        naive_auc = auc(fpr, tpr)
        balanced_accuracy = balanced_accuracy_score(window_level_true, window_level_pred)
        confusion_matrix_result = confusion_matrix(window_level_true, window_level_pred)
        print(f"For {model_name}, window level")
        print(f"Precision: {precision_value}, Recall: {recall_value}, F1: {f1_value}, AUC: {naive_auc}, optimal threshold: {optimal_threshold}, balanced_accuracy: {balanced_accuracy}")
        print(confusion_matrix_result)
        
        

                
                
                
        

if __name__ == "__main__":
    result_dict = {
        'cnn': "/path/to/your/test_metadata.csv",
    }
    result_df = {}
    for model_name, path in result_dict.items():
        current_df = pd.read_csv(path)
        result_df[model_name] =current_df

    calculate_metrics(result_df)
