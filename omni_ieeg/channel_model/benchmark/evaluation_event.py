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
def calculate_event_count(df_dict_raw, testing_split):
    results = []
    df_dict = {}
    for model_name, df in df_dict_raw.items():
        if model_name == "pyhfo":
            filtered_df = df[(df['artifact_pruning'].astype(float) > 0.5) & (df['spike_pruning'].astype(float) > 0.5)]
            grouped = filtered_df.groupby(['file_name', 'name']).size().reset_index(name='count')
            df_dict[model_name] = grouped
            print(f"Model {model_name} has max count {grouped['count'].max()}")
        elif model_name == "ehfo":
            filtered_df = df[(df['artifact_pruning'].astype(float) > 0.5) & (df['ehfo_ehfo'].astype(float) > 0.5)]
            grouped = filtered_df.groupby(['file_name', 'name']).size().reset_index(name='count')
            df_dict[model_name] = grouped
            print(f"Model {model_name} has max count {grouped['count'].max()}")
        else:
            filtered_df = df[df['3label_pred'].astype(int) == 2]
            grouped = filtered_df.groupby(['file_name', 'name']).size().reset_index(name='count')
            df_dict[model_name] = grouped
            print(f"Model {model_name} has max count {grouped['count'].max()}")
    for index, row in tqdm(testing_split.iterrows(), total=len(testing_split), desc="Calculating event count"):
        patient_name = row['patient_name']
        edf_name = row['edf_name']
        patient_outcome = row['outcome']
        length = row['length']
        channels_df = filtered_dataset.get_channels_for_edf(edf_name)
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
            # results_dict = aggregate_all_model_results(df_dict, edf_name, channel_name)
            #             # })
            
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
                current_count_df = model_result[(model_result['file_name'] == os.path.basename(edf_name)) & (model_result['name'] == channel_name)]['count']
                if len(current_count_df) == 0:
                    current_result[f"{model_name}_count"] = 0
                    current_result[f"{model_name}_ratio"] = 0
                else:
                    current_result[f"{model_name}_count"] = current_count_df.iloc[0]
                    current_result[f"{model_name}_ratio"] = current_count_df.iloc[0] / length * 100
                # current_result[f"{model_name}_count"] = model_result
                # current_result[f"{model_name}_ratio"] = model_result / length
            results.append(current_result)
    results_df = pd.DataFrame(results)
    return results_df
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model_outputs(df, model_output_columns, ground_truth_column='ground_truth'):
    """
    Evaluate multiple model output columns against ground truth, finding optimal thresholds
    based on ROC curve analysis and calculating performance metrics.
    """
    # Check input
    if ground_truth_column not in df.columns:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in DataFrame")
    
    for col in model_output_columns:
        if f"{col}_ratio" not in df.columns:
            raise ValueError(f"Model output column '{col}_ratio' not found in DataFrame")
    
    # Initialize results dictionary
    results = []
    
    # Remove rows where ground truth is not 0 or 1
    df = df[df[ground_truth_column].isin([0, 1])]
    
    # Get ground truth values
    y_true = df[ground_truth_column].values
    
    # Evaluate each model output column
    for col in model_output_columns:
        # Get model predictions (scores)
        y_scores = df[f"{col}_ratio"].values
        
        # Since higher scores are associated with class 0, we need to reverse the scores
        # for ROC curve calculation (which assumes higher scores mean higher probability of class 1)
        y_scores_reversed = -y_scores  # Negate the scores to reverse the relationship
        
        # Calculate ROC curve with reversed scores
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores_reversed)
        pr_auc_val = pr_auc(y_true, y_scores_reversed)
        
        # Calculate AUC
        navie_auc = auc(fpr, tpr)
        
        # Find optimal threshold based on Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold_reversed = roc_thresholds[optimal_idx]
        
        # Convert back to the original score scale
        optimal_threshold = -optimal_threshold_reversed
        
        # Apply optimal threshold to get binary predictions
        # Since we reversed the scores, we need to reverse the comparison operator too
        y_pred = (y_scores <= optimal_threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # macro averaged
        precision_value = precision_score(y_true, y_pred, zero_division=0, average='macro')
        recall_value = recall_score(y_true, y_pred, zero_division=0, average='macro')
        f1_value = f1_score(y_true, y_pred, zero_division=0, average='macro')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        results.append({
            'model': col,
            'optimal_threshold': optimal_threshold,
            'precision': precision_value,
            'recall': recall_value,
            'f1_score': f1_value,
            'specificity': specificity,
            'navie_auc': navie_auc,
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
        model_resection_ratio_dict[model_name] = []
    ground_truth = []
    for subject in df['patient_name'].unique():
        subject_df = df[df['patient_name'] == subject]
        assert subject_df['patient_outcome'].nunique() == 1, f"subject {subject} has multiple outcomes"
        subject_outcome = subject_df['patient_outcome'].iloc[0]
        if subject_outcome == -1:
            continue
        ground_truth.append(subject_outcome)
        for model_name in model_output_columns:
            patient_model_total_count = subject_df[f"{model_name}_count"].sum()
            patient_model_resected_count = subject_df[subject_df['channel_resection'] == 1][f"{model_name}_count"].sum()
            if patient_model_total_count == 0:
                model_resection_ratio_dict[model_name].append(1)
            else:
                model_resection_ratio_dict[model_name].append(patient_model_resected_count / patient_model_total_count)
     # Convert ground truth to numpy array
    ground_truth = np.array(ground_truth)
    # print ground truth distribution
    print(f"Ground truth distribution: {np.bincount(ground_truth)}")
    
    # Calculate AUC for each model using logistic regression
    model_auc_dict = {}
    for model_name, resection_ratios in model_resection_ratio_dict.items():
        # Reshape for scikit-learn (needs 2D array)
        # X = np.array(resection_ratios).reshape(-1, 1)
        
        # Fit logistic regression model
        # lr = LogisticRegression(solver='liblinear')
        # lr.fit(X, ground_truth)
        
        # Get probabilities and calculate AUC
        # y_proba = lr.predict_proba(X)[:, 1]  # Probability of positive class
        fpr, tpr, roc_thresholds = roc_curve(ground_truth, resection_ratios)
        naive_auc = auc(fpr, tpr)
        # auc = roc_auc_score(ground_truth, resection_ratios)
        # pr_auc_val = pr_auc(ground_truth, resection_ratios)
        model_auc_dict[model_name] = auc
        # Plot distribution of resection ratios for each ground truth class
        # Store results
        results.append({
            'model': model_name,
            'auc': naive_auc,
            # 'pr_auc': pr_auc_val,
            # 'coefficients': lr.coef_[0][0],
            # 'intercept': lr.intercept_[0]
        })
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    dataset_path = "path/to/dataset"
    final_split_path = "path/to/final_split.csv"
    pyhfo_all = pd.read_csv("path/to/pyhfo_all.csv")
    ehfo_all = pd.read_csv("path/to/ehfo_all.csv")
    cnn = pd.read_csv("path/to/cnn.csv")
    data_filter = DataFilter(dataset_path)

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()

    final_split = pd.read_csv(final_split_path)
    final_split = final_split[final_split['dataset'] != "Multicenter"]
    # we first want to get interictal, frequency at least 900, and length at least 300
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]

    # we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
    filtered_dataset = data_filter.apply_filters()
    final_split = pd.read_csv(final_split_path)
    final_split = final_split[final_split['dataset'] != "Multicenter"]
    # we first want to get interictal, frequency at least 900, and length at least 300
    testing_split = final_split[(final_split['split'] == 'test') & (final_split['frequency'] > 900) & (final_split['interictal'] == True) & (final_split['length'] >= 62)]

    pyhfo_all = pyhfo_all.drop_duplicates(subset=['start', 'end', 'name', 'file_name'])
    ehfo_all = ehfo_all.drop_duplicates(subset=['start', 'end', 'name', 'file_name'])

    # merge two using inner
    pyhfo_ehfo_merged = pd.merge(pyhfo_all, ehfo_all, on=['start', 'end', 'name', 'file_name'], how='inner')
    # it should be perfect merge
    assert len(pyhfo_ehfo_merged) == len(pyhfo_all)


    # merge them based on start, end, name, and file_name
    pyhfo_efho_merged = pd.merge(pyhfo_all, ehfo_all, on=['start', 'end', 'name', 'file_name'], how='inner')
    # it should be perfect merge
    # assert len(pyhfo_efho_merged) == len(pyhfo_all)


    cnn = cnn.drop_duplicates(subset=['start', 'end', 'name', 'file_name'])
    # convert start and end to int
    cnn['start'] = cnn['start'].astype(int)
    cnn['end'] = cnn['end'].astype(int)

    # check if they have same length
    assert len(cnn) == len(pyhfo_efho_merged)
    cnn_merged = pd.merge(cnn, pyhfo_efho_merged, on=['start', 'end', 'name', 'file_name'], how='inner')
    assert len(cnn_merged) == len(pyhfo_efho_merged)


    input_dict = {
        "ehfo": pyhfo_efho_merged,
        "pyhfo": pyhfo_efho_merged,
        "cnn": cnn_merged
    }

    print(f"Calculating resectionratios for {len(input_dict)} models")
    resection_ratio_df = calculate_event_count(input_dict, testing_split)
    print(f"Max in pyhfo_count: {resection_ratio_df['pyhfo_count'].max()}")
    print(f"Max in ehfo_count: {resection_ratio_df['ehfo_count'].max()}")
    print(f"Max in cnn_count: {resection_ratio_df['cnn_count'].max()}")

    print(f"max in pyhfo_ratio: {resection_ratio_df['pyhfo_ratio'].max()}")
    print(f"max in ehfo_ratio: {resection_ratio_df['ehfo_ratio'].max()}")
    print(f"max in cnn_ratio: {resection_ratio_df['cnn_ratio'].max()}")
    # print ground truth distribution count
    print(f"Ground truth distribution count: {resection_ratio_df['ground_truth'].value_counts()}")



    print(f"Calculating metrics for {len(resection_ratio_df)} channels")
    channel_metrics_df = evaluate_model_outputs(resection_ratio_df, input_dict.keys(), 'ground_truth')
    print(f"=======Printing channel metrics")
    print(channel_metrics_df)
    subject_outcome_metrics_df = resection_ratio_calculation_originalmethod(resection_ratio_df, input_dict.keys(), 'ground_truth')
    print(f"=======Printing subject outcome metrics")
    print(subject_outcome_metrics_df)
