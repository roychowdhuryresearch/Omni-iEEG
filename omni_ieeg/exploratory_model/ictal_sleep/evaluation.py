import pandas as pd
import numpy as np
np.random.seed(41)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
def run_evaluation(result_dict):
    for model_name, result_path in result_dict.items():
        result_df = pd.read_csv(result_path)

        pred = result_df['channel_pred']
        true = result_df['channel_true']

        # apply sigmoid to pred
        pred = 1 / (1 + np.exp(-pred))

        # find a threshold that maximize the auc

        from sklearn.metrics import roc_curve

        fpr, tpr, roc_thresholds = roc_curve(true, pred)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)


        optimal_threshold = roc_thresholds[optimal_idx]

        new_pred = (pred > optimal_threshold).astype(int)

        # reverse true
        true = 1 - true
        new_pred = 1 - new_pred
        tn, fp, fn, tp = confusion_matrix(true, new_pred).ravel()
        precision_value = precision_score(true, new_pred, average='macro')
        recall_value = recall_score(true, new_pred, average='macro')
        f1_value = f1_score(true, new_pred, average='macro')
        confusion_matrix_result = confusion_matrix(true, new_pred)
        print(f"For {model_name},")
        print(f"Precision: {precision_value}, Recall: {recall_value}, F1: {f1_value}")
        print(f"Confusion Matrix: {confusion_matrix_result}")

    # make a random guess list

    random_guess_list = []
    for i in range(len(true)):
        random_guess_list.append(np.random.randint(0, 2))

    # calculate matrix
    precision_value = precision_score(true, random_guess_list, average='macro')
    recall_value = recall_score(true, random_guess_list, average='macro')
    f1_value = f1_score(true, random_guess_list, average='macro')
    confusion_matrix_result = confusion_matrix(true, random_guess_list)

    print(f"For random guess,")
    print(f"Precision: {precision_value}, Recall: {recall_value}, F1: {f1_value}")
    print(f"Confusion Matrix: {confusion_matrix_result}")





if __name__ == "__main__":
    result_dict = {
        'cnn': "/path/to/your/test_metadata.csv",
    }
    run_evaluation(result_dict)
