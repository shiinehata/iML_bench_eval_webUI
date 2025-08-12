import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json

def validate_and_read_inputs(ground_truth_path, prediction_path):
    """
    Validates input files (existence, format, row/column count)
    and reads them into DataFrames.
    """
    # 1. Check for file existence
    if not ground_truth_path.is_file():
        raise ValueError(f"Ground truth file does not exist at '{ground_truth_path}'")

    if not prediction_path.is_file():
        raise ValueError(f"Prediction file does not exist at '{prediction_path}'")

    # 2. Check if CSV files can be read
    try:
        df_gt = pd.read_csv(ground_truth_path)
        df_pred = pd.read_csv(prediction_path)
    except Exception as e:
        raise ValueError(f"Could not read one of the CSV files. Please check the file format. Error: {e}")

    # 3. Check row count
    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    # 4. Check column names
    if set(df_gt.columns) != set(df_pred.columns):
        raise ValueError(f"Column names do not match. GT: {list(df_gt.columns)}, Pred: {list(df_pred.columns)}")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    """
    Calculates accuracy and f1-score metrics from the DataFrames.

    Args:
        df_truth (pd.DataFrame): DataFrame containing the ground truth data.
        df_pred (pd.DataFrame): DataFrame containing the prediction data.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    try:
        # Sort to ensure rows are aligned
        df_truth = df_truth.sort_values('PetID').reset_index(drop=True)
        df_pred = df_pred.sort_values('PetID').reset_index(drop=True)

        y_true = df_truth['AdoptionSpeed']
        y_pred = df_pred['AdoptionSpeed']

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro
        }

    except KeyError:
        raise ValueError("A required column ('PetID' or 'AdoptionSpeed') was not found in the files.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")