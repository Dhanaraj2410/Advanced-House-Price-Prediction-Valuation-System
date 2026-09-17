"""
Dataset loading and verification module for House Price Prediction.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Loads train and test CSV datasets into pandas DataFrames.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def dataset_summary(train, test):
    """
    Returns comprehensive dataset summary dictionary.
    """
    num_cols = train.select_dtypes(include=["number"]).columns
    cat_cols = train.select_dtypes(include=["object", "category", "string"]).columns

    summary = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "target_col": "SalePrice",
        "target_stats": train["SalePrice"].describe().to_dict() if "SalePrice" in train.columns else None,
        "num_numeric_train": len(num_cols),
        "num_categorical_train": len(cat_cols),
        "top_missing_train": (train.isnull().mean() * 100).sort_values(ascending=False).head(10).to_dict()
    }
    return summary

if __name__ == "__main__":
    print("--- Loading Datasets ---")
    train_df, test_df = load_data()
    summary = dataset_summary(train_df, test_df)
    
    print(f"Train Shape: {summary['train_shape']}")
    print(f"Test Shape:  {summary['test_shape']}")
    print("\n--- Target Variable ('SalePrice') Summary ---")
    for k, v in summary['target_stats'].items():
        print(f"  {k}: {v:,.2f}")
    
    print(f"\nNumerical Columns Count: {summary['num_numeric_train']}")
    print(f"Categorical Columns Count: {summary['num_categorical_train']}")
    
    print("\n--- Top 10 Missing Value Columns (%) ---")
    for col, pct in summary['top_missing_train'].items():
        if pct > 0:
            print(f"  {col}: {pct:.2f}%")
