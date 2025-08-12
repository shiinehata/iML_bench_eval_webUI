import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
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
        df_gt = pd.read_csv(ground_truth_path, dtype={'Labels': str}).fillna('')
        df_pred = pd.read_csv(prediction_path, dtype={'Labels': str}).fillna('')
    except Exception as e:
        raise ValueError(f"Could not read one of the CSV files. Please check the file format. Error: {e}")

    required_cols = {'ImageID', 'Labels'}
    if not required_cols.issubset(df_gt.columns) or not required_cols.issubset(df_pred.columns):
        raise ValueError("Files must contain 'ImageID' and 'Labels' columns.")
    
    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    """
    Evaluates multi-label classification predictions.
    It transforms space-separated string labels into a binary matrix before scoring.
    """
    try:
        df_truth = df_truth.sort_values('ImageID').reset_index(drop=True)
        df_pred = df_pred.sort_values('ImageID').reset_index(drop=True)

        # Split the space-separated strings into lists of labels
        y_true_labels = [str(s).split() for s in df_truth['Labels']]
        y_pred_labels = [str(s).split() for s in df_pred['Labels']]

        # Use MultiLabelBinarizer to convert lists of labels into a binary matrix
        mlb = MultiLabelBinarizer()
        
        # Fit on the true labels to learn all possible classes
        mlb.fit(y_true_labels)

        # Transform both true and predicted labels into the binary matrix format
        y_true = mlb.transform(y_true_labels)
        y_pred = mlb.transform(y_pred_labels)

        # Calculate metrics
        # accuracy_score computes the subset accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # F1-score with 'macro' average treats each label equally
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print("------------------------------------------")

        return {
            'accuracy_subset': accuracy,
            'f1_score_macro': f1
        }
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")