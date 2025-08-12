import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json

def validate_and_read_inputs(ground_truth_path, prediction_path):
    """
    Validates input files and reads them into DataFrames.
    Checks for existence, row/column match, and required columns for the classification task.
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

    # Check for required columns for this task
    required_cols = {'image_id', 'label'}
    if not required_cols.issubset(df_gt.columns):
        raise ValueError(f"Ground truth file is missing required columns. Required: {list(required_cols)}")
        
    if not required_cols.issubset(df_pred.columns):
        raise ValueError(f"Prediction file is missing required columns. Required: {list(required_cols)}")

    if len(df_gt) != len(df_pred):
        raise ValueError(f"Row count mismatch: Ground Truth has {len(df_gt)} rows, Prediction has {len(df_pred)} rows.")

    return df_gt, df_pred

def evaluate_predictions(df_truth, df_pred):
    """
    Evaluates classification predictions by calculating Accuracy and Macro F1-Score.
    """
    try:
        df_truth = df_truth.sort_values('image_id').reset_index(drop=True)
        df_pred = df_pred.sort_values('image_id').reset_index(drop=True)

        y_true = df_truth['label']
        y_pred = df_pred['label']
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print("--- Paddy Disease Classification Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print("------------------------------------------")

        return {
            'accuracy': accuracy,
            'f1_score_macro': f1
        }
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        raise

def main():
    if len(sys.argv) != 4:
        print("Error: Exactly 3 arguments are required.")
        print("Usage: python your_script_name.py <path_ground_truth> <path_prediction> <path_json>")
        sys.exit(1)

    ground_truth_path = Path(sys.argv[1])
    prediction_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    print(f"Ground Truth File: {ground_truth_path}")
    print(f"Prediction File: {prediction_path}\n")
    
    df_gt, df_pred = validate_and_read_inputs(ground_truth_path, prediction_path)

    try:
        metrics = evaluate_predictions(df_gt, df_pred)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        print(f"\nEvaluation results saved to '{output_path}'")

    except Exception as e:
        error_info = {"error": str(e)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=4)
        print(f"An error occurred. Error info saved to '{output_path}'")

    print("\nSuccessfully evaluated!")

if __name__ == "__main__":
    main()