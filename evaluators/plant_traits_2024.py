import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import json

def validate_and_read_inputs(ground_truth_path, prediction_path):
    """
    Validates input files and reads them into DataFrames.
    Checks for existence, row/column match, and required columns for the traits task.
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

    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    if set(df_gt.columns) != set(df_pred.columns):
        raise ValueError(f"Column names do not match. GT: {list(df_gt.columns)}, Pred: {list(df_pred.columns)}")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    """
    Evaluates regression predictions by calculating R2 and RMSE for each trait.
    """
    try:
        df_truth = df_truth.sort_values('id').reset_index(drop=True)
        df_pred = df_pred.sort_values('id').reset_index(drop=True)

        trait_labels = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
        r2_scores = []
        rmse_scores = []

        print("--- Plant Traits Regression Results ---")
        for trait in trait_labels:
            y_true = df_truth[trait]
            y_pred = df_pred[trait]
            
            # Calculate metrics for the current trait
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            
            print(f"Trait: {trait} -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        # Calculate mean scores
        mean_r2 = np.mean(r2_scores)
        mean_rmse = np.mean(rmse_scores)
        
        print("---------------------------------------")
        print(f"Mean R2 Score: {mean_r2:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f}")
        print("---------------------------------------")

        return {
            'mean_r2_score': mean_r2,
            'mean_rmse': mean_rmse,
            'individual_r2_scores': {trait: score for trait, score in zip(trait_labels, r2_scores)},
            'individual_rmse_scores': {trait: score for trait, score in zip(trait_labels, rmse_scores)}
        }
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during evaluation: {e}")