"""
Model Benchmark & Evaluation Metric Suite.
Runs rigorous model comparison across multiple metrics (MAE, RMSE, R2, MAPE, Training Time)
and generates JSON benchmark report and comparative visualization plots.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import load_data
from ml.preprocessing import remove_outliers, FeatureEngineer, build_preprocessing_pipeline

REPORT_DIR = os.path.join(PROJECT_ROOT, "models")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static", "eda")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100)

def run_benchmark():
    print("==================================================")
    print("        MODEL BENCHMARK & EVALUATION SUITE        ")
    print("==================================================")

    train_df, _ = load_data()
    train_df = remove_outliers(train_df)

    X = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = train_df["SalePrice"]
    y_log = np.log1p(y)

    X_train, X_valid, y_train_log, y_valid_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )

    fe = FeatureEngineer()
    X_train_eng = fe.transform(X_train)
    num_cols = X_train_eng.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_eng.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    stacking_estimators = [
        ("ridge", Ridge(alpha=10.0)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1))
    ]

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1),
        "Stacking Ensemble": StackingRegressor(estimators=stacking_estimators, final_estimator=Ridge(alpha=1.0))
    }

    y_valid_orig = np.expm1(y_valid_log)
    benchmark_data = []

    for name, regressor in models.items():
        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
        pipeline.steps.append(("model", regressor))
        
        start_time = time.time()
        pipeline.fit(X_train, y_train_log)
        fit_duration = round(time.time() - start_time, 3)

        pred_log = pipeline.predict(X_valid)
        pred_orig = np.expm1(pred_log)

        mae = mean_absolute_error(y_valid_orig, pred_orig)
        rmse = np.sqrt(mean_squared_error(y_valid_orig, pred_orig))
        r2 = r2_score(y_valid_orig, pred_orig)
        mape = mean_absolute_percentage_error(y_valid_orig, pred_orig)

        metrics = {
            "model": name,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2_score": round(float(r2), 4),
            "mape_percent": round(float(mape), 2),
            "fit_duration_sec": fit_duration
        }
        benchmark_data.append(metrics)
        print(f"  {name:<20} | RMSE: ${rmse:>9,.2f} | R2: {r2:.4f} | MAPE: {mape:>5.2f}% | Time: {fit_duration:>5.2f}s")

    benchmark_df = pd.DataFrame(benchmark_data).sort_values("rmse")
    
    # Export JSON report
    report_path = os.path.join(REPORT_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved benchmark JSON report to: {report_path}")

    # Generate benchmark visualization plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="rmse", y="model", data=benchmark_df, hue="model", legend=False, palette="viridis")
    plt.title("Model Performance Comparison (Root Mean Squared Error)", fontsize=14, fontweight="bold")
    plt.xlabel("RMSE ($) - Lower is Better", fontsize=12)
    plt.ylabel("Model Architecture", fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(STATIC_DIR, "model_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved benchmark plot to: {plot_path}")

    return benchmark_data

if __name__ == "__main__":
    run_benchmark()
