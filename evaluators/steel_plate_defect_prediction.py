import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score
import json

def validate_and_read_inputs(ground_truth_path, prediction_path):
    """
    Validates input files and reads them into DataFrames.
    Checks for existence, row/column match, and required columns.
    """
    if not ground_truth_path.is_file():
        raise ValueError(f"Ground truth file does not exist at '{ground_truth_path}'")
    
    if not prediction_path.is_file():
        raise ValueError(f"Prediction file does not exist at '{prediction_path}'")

    try:
        df_gt = pd.read_csv(ground_truth_path)
        df_pred = pd.read_csv(prediction_path)
    except Exception as e:
        raise ValueError(f"Could not read one of the CSV files. Please check the file format. Error: {e}")

    # Check for the specific columns in this classification task
    required_cols = {'id', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'}
    
    if not required_cols.issubset(df_gt.columns):
        raise ValueError(f"Ground truth file is missing required columns. Required: {list(required_cols)}")
        
    if not required_cols.issubset(df_pred.columns):
        raise ValueError(f"Prediction file is missing required columns. Required: {list(required_cols)}")

    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    """
    Evaluates multi-label classification predictions.
    Calculates log-loss, subset accuracy, and macro F1-score.
    """
    try:
        df_truth = df_truth.sort_values('id').reset_index(drop=True)
        df_pred = df_pred.sort_values('id').reset_index(drop=True)

        class_labels = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        
        y_true = df_truth[class_labels]
        # Treat prediction columns as probabilities for log-loss calculation
        y_pred_proba = df_pred[class_labels].astype(float)

        # Calculate metrics
        loss = log_loss(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred_proba.round()) # Use rounded values for accuracy
        f1 = f1_score(y_true, y_pred_proba.round(), average='macro', zero_division=0)
        
        print("--- Steel Plate Defect Classification Results ---")
        print(f"Log-loss: {loss:.4f}")
        print(f"Accuracy (Subset): {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print("-------------------------------------------------")

        return {
            'log_loss': loss,
            'accuracy_subset': accuracy,
            'f1_score_macro': f1
        }
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")