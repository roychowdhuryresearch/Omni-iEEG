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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import auc

def calculate_metrics(result_dict):
    for model_name, model_df in result_dict.items():
        print(f"=====For {model_name}")
        prob_class_cols = [col for col in model_df.columns if col.startswith('prob_class_') and col[len('prob_class_'):].isdigit()]

        # Optionally, sort them by the class number
        prob_class_cols = sorted(prob_class_cols, key=lambda x: int(x[len('prob_class_'):]))
        print(prob_class_cols)
        channel_pred = []
        channel_true = []
        unique_edf_names = model_df['edf_name'].unique()
        for edf_name in unique_edf_names:
            
            edf_df = model_df[model_df['edf_name'] == edf_name]
            grouped_indices = edf_df.groupby('name')['start_indices'].apply(lambda x: tuple(sorted(x.unique())))
            assert grouped_indices.nunique() == 1, f"all channel does not have all start indices" 
            # unique name
            unique_name = edf_df['name'].unique()
            for channel_name in unique_name:
                channel_df = edf_df[edf_df['name'] == channel_name]
                assert len(channel_df['channel_true'].unique()) == 1, f"Channel true values should be same for {edf_name}"
                label = channel_df['channel_true'].unique()[0]
                if label == -1:
                    continue
                channel_pred_dict = {}
                max_count = 0
                max_col = None
                for prob_class_col in prob_class_cols:
                    # equal to sum of channel_df[prob_class_col]
                    channel_pred_dict[prob_class_col] = channel_df[prob_class_col].sum()
                    if channel_pred_dict[prob_class_col] > max_count:
                        max_count = channel_pred_dict[prob_class_col]
                        max_col = prob_class_col
                max_col = int(max_col[len('prob_class_'):])
                channel_pred.append(max_col)
                channel_true.append(label)
        
        precison_macro = precision_score(channel_true, channel_pred, average='macro')
        recall_macro = recall_score(channel_true, channel_pred, average='macro')
        f1_macro = f1_score(channel_true, channel_pred, average='macro')
        accuracy = accuracy_score(channel_true, channel_pred)
        balanced_accuracy = balanced_accuracy_score(channel_true, channel_pred)
        confusion_matrix_value = confusion_matrix(channel_true, channel_pred)
        print(f"precison_macro: {precison_macro}, recall_macro: {recall_macro}, f1_macro: {f1_macro}, accuracy: {accuracy}, balanced_accuracy: {balanced_accuracy}")
        print(f"{confusion_matrix_value}")
               
if __name__ == "__main__":
    import pandas as pd
    result_path = {
        "cnn": "path/to/lstm/result.csv",
    }
    result_df = {}
    for model_name, path in result_path.items():
        result_df[model_name] = pd.read_csv(path)

calculate_metrics(result_df)