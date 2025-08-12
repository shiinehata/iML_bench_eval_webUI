import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score
import json

def validate_and_read_inputs(ground_truth_path, prediction_path):
    if not ground_truth_path.is_file():
        raise ValueError(f"Ground truth file does not exist at '{ground_truth_path}'")
    
    if not prediction_path.is_file():
        raise ValueError(f"Prediction file does not exist at '{prediction_path}'")

    try:
        df_gt = pd.read_csv(ground_truth_path)
        df_pred = pd.read_csv(prediction_path)
    except Exception as e:
        raise ValueError(f"Could not read one of the CSV files. Please check the file format. Error: {e}")

    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    if set(df_gt.columns) != set(df_pred.columns):
        raise ValueError(f"Column names do not match. GT: {list(df_gt.columns)}, Pred: {list(df_pred.columns)}")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    try:
        df_truth = df_truth.sort_values('discourse_id').reset_index(drop=True)
        df_pred = df_pred.sort_values('discourse_id').reset_index(drop=True)

        class_labels = ['Ineffective', 'Adequate', 'Effective']

        y_true = df_truth[class_labels].values
        y_pred_proba = df_pred[class_labels].values

        y_true_labels = df_truth[class_labels].idxmax(axis=1)
        y_pred_labels = df_pred[class_labels].idxmax(axis=1)

        loss = log_loss(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

        print("--- RESULTS ---")
        print(f"Log Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print("---------------------------------")

        return {
            'log_loss': loss,
            'accuracy': accuracy,
            'f1_score_macro': f1
        }
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")