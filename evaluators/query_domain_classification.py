import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
        df_truth = df_truth.sort_values('ID').reset_index(drop=True)
        df_pred = df_pred.sort_values('ID').reset_index(drop=True)

        y_true = df_truth['Domain']
        y_pred = df_pred['Domain']

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print("--- Domain Classification Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print("-----------------------------------")

        return {
            'accuracy': accuracy,
            'f1__score_macro': f1_macro
        }

    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")