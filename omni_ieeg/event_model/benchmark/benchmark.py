import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(result_df):
    """
    Calculate classification metrics for multi-class classification.
    
    Args:
        result_df: DataFrame with 'predicted_class' and 'true_class' columns
        
    Returns:
        dict: Dictionary containing various metrics
    """
    y_true = result_df['true_class']
    y_pred = result_df['predicted_class']
    
    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    # For multi-class, we calculate precision, recall, and F1 for each class
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    
    
    # Calculate metrics for each class
    class_names = ["Artifact", "Real", "Spike"]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Return all metrics
    return {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }
    

if __name__ == "__main__":
    checkpoint_mapping = {
        'cnn': "path/to/model/test_metadata.csv",
    }
    import pandas as pd
    for model_name, result_csv in checkpoint_mapping.items():
        print(f"==================Processing {model_name} model")
        result_df = pd.read_csv(result_csv)

        metrics = calculate_metrics(result_df)

        # Print overall accuracy
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")

        # Print macro-averaged metrics (treats all classes equally)
        print("\nMacro-averaged Metrics:")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall: {metrics['recall_macro']:.4f}")
        print(f"F1 Score: {metrics['f1_macro']:.4f}")
        print(f"Confusion Matrix: {metrics['confusion_matrix']}")   