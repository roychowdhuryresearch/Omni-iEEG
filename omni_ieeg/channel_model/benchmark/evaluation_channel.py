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
from sklearn.metrics import precision_recall_curve, auc
def pr_auc(groundtruth, pred):
    precision, recall, thresholds = precision_recall_curve(groundtruth, pred, pos_label=0)
    pr_auc_result = auc(recall, precision)
    return pr_auc_result

def calculate_all_ratios(testing_split, df_dict_raw, dataset_path):
    results = []
    df_dict = {}
    for model_name, df in df_dict_raw.items():
        df_grouped = (
                        df.groupby(['edf_name', 'name'])
                        .agg(
                            channel_pred_sum=('channel_pred', 'sum'),
                            count=('channel_pred', 'count'),
                            unique_true=('channel_true', 'nunique'),
                            channel_true_val=('channel_true', 'first')
                        )
                        .reset_index()
                    )
        assert all(df_grouped['unique_true'] == 1), "Inconsistent 'channel_true' values within a group for model {model_name}"
        df_grouped['ratio'] = df_grouped['channel_pred_sum'] / df_grouped['count']
        df_dict[model_name] = df_grouped
    for _, row in tqdm(testing_split.iterrows(), total=len(testing_split), desc="Calculating ratios"):
        patient_name = row['patient_name']
        edf_name = row['edf_name']
        edf_name = os.path.join(dataset_path, edf_name)
        patient_outcome = row['outcome']
        channels_df = filtered_dataset.get_channels_for_edf(edf_name)
        # print(f"Channels df: {channels_df}")
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
            current_result = {
                'patient_name': patient_name,
                'edf_name': edf_name,
                'channel_name': channel_name,
                'ground_truth': label,
                "channel_resection": channel_resection,
                "channel_soz": channel_soz,
                "patient_outcome": patient_outcome,
                
            }
            for model_name, model_result in df_dict.items():
                current_df = model_result[(model_result['edf_name'] == edf_name) & (model_result['name'] == channel_name)]
                current_result[f"{model_name}_ratio"] = current_df['ratio'].iloc[0]
                current_result[f"{model_name}_count"] = current_df['channel_pred_sum'].iloc[0]
                current_result[f"{model_name}_num_count"] = current_df['count'].iloc[0]
                # current_result[f"{model_name}_groundtruth_count"] = current_df['count'].iloc[0] * label
                current_result[f"{model_name}_count_reversed"] = current_df['count'].iloc[0] - current_df['channel_pred_sum'].iloc[0]
            results.append(current_result)
    results_df = pd.DataFrame(results)
    return results_df

            
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
def evaluate_model_outputs(df, model_output_columns, ground_truth_column='ground_truth'):
    """
    Evaluate multiple model output columns against ground truth, finding optimal thresholds
    based on ROC curve analysis and calculating performance metrics.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing ground truth and model output columns
    model_output_columns : list
        List of column names containing model outputs (probabilities between 0 and 1)
    ground_truth_column : str, default='ground_truth'
        Column name containing ground truth labels (0 or 1)
    threshold_method : str, default='youden'
        Method for determining optimal threshold from ROC curve.
        Options: 'youden' (maximizes sensitivity+specificity-1) or 'distance' (closest to perfect classifier)
        
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame containing evaluation metrics for each model output column
    """
    # Check input
    if ground_truth_column not in df.columns:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in DataFrame")
    
    for col in model_output_columns:
        if f"{col}_ratio" not in df.columns:
            raise ValueError(f"Model output column '{col}' not found in DataFrame")
    
    # Initialize results dictionary
    results = []
    
    # remove rows where ground truth is not 0 or 1
    df = df[df[ground_truth_column].isin([0, 1])]
    # Get ground truth values
    y_true = df[ground_truth_column].values
    y_true = [1-y for y in y_true]
    # Evaluate each model output column
    for col in model_output_columns:
        # Get model predictions (probabilities)
        y_scores = df[f"{col}_ratio"].values
        y_scores = [1-y_score for y_score in y_scores]
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        
        # Calculate AUC
        naive_auc = auc(fpr, tpr)
        
        class_counts = np.bincount(y_true)
        class_weights = len(y_true) / (len(np.unique(y_true)) * class_counts)
        sample_weights = class_weights[y_true]
        auc_roc_auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weights)

        pr_auc_val = pr_auc(y_true, y_scores)
        # Find optimal threshold based on specified method
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    
        
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Apply optimal threshold to get binary predictions
        y_pred = (y_scores >= optimal_threshold).astype(int)
         # Plot distribution of resection ratios for each ground truth class
        # Calculate metrics
        # macro averaged
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision_value = precision_score(y_true, y_pred, average='macro')
        recall_value = recall_score(y_true, y_pred, average='macro')
        f1_value = f1_score(y_true, y_pred, average='macro')
        specificity = tn / (tn + fp)  # true negative rate
        
        # Store results
        results.append({
            'model': col,
            'optimal_threshold': optimal_threshold,
            'precision': precision_value,
            'recall': recall_value,
            'specificity': specificity,
            'f1': f1_value,
            'roc_auc': auc_roc_auc,
            "auc": naive_auc,
            'pr_auc': pr_auc_val
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df     
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
def resection_ratio_calculation_originalmethod(df, model_output_columns, ground_truth_column='ground_truth'):
    results = []

    model_resection_ratio_dict = {}
    for model_name in model_output_columns:
        # model_resection_ratio_dict[model_name] = []
        # model_resection_ratio_dict[model_name + '_num_count'] = []
        # model_resection_ratio_dict[model_name + '_groundtruth_count'] = []
        model_resection_ratio_dict[model_name + '_count_reversed'] = []
    ground_truth = []
    patient_names = []
    for subject in df['patient_name'].unique():
        subject_df = df[df['patient_name'] == subject]
        assert subject_df['patient_outcome'].nunique() == 1, f"subject {subject} has multiple outcomes"
        subject_outcome = subject_df['patient_outcome'].iloc[0]
        if subject_outcome == -1:
            continue
        patient_names.append(subject)
        ground_truth.append(subject_outcome)
        for model_name in model_output_columns:
            # patient_model_total_count = subject_df[f"{model_name}_count"].sum()
            # patient_model_resected_count = subject_df[subject_df['channel_resection'] == 1][f"{model_name}_count"].sum()
            # if patient_model_total_count == 0:
            #     model_resection_ratio_dict[model_name].append(1)
            # else:
            #     model_resection_ratio_dict[model_name].append(patient_model_resected_count / patient_model_total_count)
            # patient_model_total_num_count = subject_df[f"{model_name}_groundtruth_count"].sum()
            # patient_model_resected_num_count = subject_df[subject_df['channel_resection'] == 1][f"{model_name}_groundtruth_count"].sum()
            # if patient_model_total_num_count == 0:
            #     model_resection_ratio_dict[model_name + '_groundtruth_count'].append(1)
            # else:
            #     model_resection_ratio_dict[model_name + '_groundtruth_count'].append(patient_model_resected_num_count / patient_model_total_num_count)
                
            patient_model_total_count_reversed = subject_df[f"{model_name}_count_reversed"].sum()
            patient_model_resected_count_reversed = subject_df[subject_df['channel_resection'] == 1][f"{model_name}_count_reversed"].sum()
            if patient_model_total_count_reversed == 0:
                model_resection_ratio_dict[model_name + '_count_reversed'].append(1)
            else:
                model_resection_ratio_dict[model_name + '_count_reversed'].append(patient_model_resected_count_reversed / patient_model_total_count_reversed)
        # 
        # subject_df['groundtruth_count'] = subject_df['ground_truth'] * subject_df['']
        # ground_truth_total_count = subject_df['groundtruth_count'].sum()
        # ground_truth_resescted_count = subject_df[subject_df['channel_resection'] == 1]['groundtruth_count'].sum()
        # model_resection_ratio_dict['groundtruth_resection_ratio'].append(ground_truth_resescted_count / ground_truth_total_count)
     # Convert ground truth to numpy array
    ground_truth = np.array(ground_truth)
    save_df = {}
    # save to csv
    save_df['patient_name'] = patient_names
    save_df['ground_truth'] = ground_truth
    for model_name, resection_ratios in model_resection_ratio_dict.items():
        save_df[f'{model_name}_resection_ratio'] = resection_ratios
    # print ground truth distribution
    print(f"Ground truth distribution: {np.bincount(ground_truth)}")
    
    # Calculate AUC for each model using logistic regression
    model_auc_dict = {}
    for model_name, resection_ratios in model_resection_ratio_dict.items():
        # Reshape for scikit-learn (needs 2D array)
        # resection_ratios = [1-x for x in resection_ratios]
        # auc = roc_auc_score(ground_truth, resection_ratios)
        class_counts = np.bincount(ground_truth)
        class_weights = len(ground_truth) / (len(np.unique(ground_truth)) * class_counts)
        
        sample_weights = class_weights[ground_truth]
        # auc = roc_auc_score(ground_truth, resection_ratios, average='samples')
        fpr, tpr, roc_thresholds = roc_curve(ground_truth, resection_ratios)
        
        # Calculate AUC
        naive_auc = auc(fpr, tpr)
        pr_auc_val = pr_auc(ground_truth, resection_ratios)
        model_auc_dict[model_name] = auc
        
        # Store results
        results.append({
            'model': model_name,
            'auc': naive_auc,
            'pr_auc': pr_auc_val
        })
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    dataset_path = "path/to/your/dataset"


    final_split_path = os.path.join(dataset_path, "derivatives/datasplit/final_split.csv")
    result_path = {
        "cnn2": "/path/test_metadata.csv",
        "vit": "/path/test_metadata.csv",
    }
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    final_split = pd.read_csv(final_split_path)
    final_split = final_split[final_split['dataset'] != "Multicenter"]
    # we first want to get interictal, frequency at least 900, and length at least 300
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]
    result_df = {}
    for model_name, path in result_path.items():
        result_df[model_name] = pd.read_csv(path)
    print(f"Calculating ratios for {len(result_df)} models")
    ratio_df = calculate_all_ratios(testing_split, result_df, dataset_path)
    # save ratio df
    print(f"Ground truth distribution count: {ratio_df['ground_truth'].value_counts()}")
    print(f"Calculating metrics for {len(ratio_df)} channels")
    channel_metrics_df = evaluate_model_outputs(ratio_df, result_df.keys(), 'ground_truth')
    print(f"======Printing Channel metrics")
    print(channel_metrics_df)
    # for k, v in channel_metrics_df.items():
    #     print(k, v)

    subject_outcome_metrics_df_original = resection_ratio_calculation_originalmethod(ratio_df, result_df.keys(), 'ground_truth')
    print(f"======Printing subject metrics original method")
    print(subject_outcome_metrics_df_original)

