import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
def calculate_metrics(channel_pred, channel_true):
    # calculate metrics
    # remove where channel_true is -1
    valid_idx = channel_true != -1
    channel_pred = channel_pred[valid_idx]
    channel_true = channel_true[valid_idx]
    metrics = {
        'precision_macro': precision_score(channel_true, channel_pred, average='macro'),
        'recall_macro': recall_score(channel_true, channel_pred, average='macro'),
        'f1_macro': f1_score(channel_true, channel_pred, average='macro'),
        'accuracy': accuracy_score(channel_true, channel_pred),
        "confusion_matrix": confusion_matrix(channel_true, channel_pred)
    }
    return metrics

def calculate_all_metrics(result_df_dicts):
    # check if all dataframe have same length
    lengths = [len(df) for df in result_df_dicts.values()]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All dataframes must have the same length")
    # calculate metrics
    metrics = {}
    for model_name, df in result_df_dicts.items():
        channel_pred = df['channel_pred']
        channel_true = df['channel_true']
        # calculate metrics
        metrics[model_name] = calculate_metrics(channel_pred, channel_true)
    return metrics
if __name__ == "__main__":
    import pandas as pd
    result_path = {
        "lstm": "path/to/lstm/result.csv",
    }
    result_df = {}
    for model_name, path in result_path.items():
        result_df[model_name] = pd.read_csv(path)

    resulting_metrics = calculate_all_metrics(result_df)
    for model_name, metrics in resulting_metrics.items():
        print(model_name, metrics)
        print(metrics['confusion_matrix'])