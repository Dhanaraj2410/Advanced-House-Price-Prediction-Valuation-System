"""
Model Training, Evaluation, Hyperparameter Tuning, and Export Module.
Trains Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static", "eda")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def evaluate_predictions(name, y_true_orig, pred_orig):
    """
    Computes MAE, RMSE, and R2 scores given original scale targets and predictions.
    """
    mae = mean_absolute_error(y_true_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, pred_orig))
    r2 = r2_score(y_true_orig, pred_orig)
    return {
        "Model": name,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }

def run_model_training():
    print("==================================================")
    print("   ADVANCED HOUSE PRICE PREDICTION MODEL TRAINER   ")
    print("==================================================")

    # 1. Load Data & Remove Outliers
    train_df, test_df = load_data()
    train_df = remove_outliers(train_df)

    test_ids = test_df["Id"]
    X = train_df.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = train_df["SalePrice"]

    # Target log-transform for stability & skew handling
    y_log = np.log1p(y)

    # 2. Train / Validation Split (80/20)
    X_train, X_valid, y_train_log, y_valid_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )

    # 3. Fit Preprocessing Pipeline
    fe = FeatureEngineer()
    X_train_eng = fe.transform(X_train)
    
    num_cols = X_train_eng.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_eng.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # 4. Define Regressors
    stacking_estimators = [
        ("ridge", Ridge(alpha=10.0)),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1))
    ]

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1),
        "Stacking Ensemble": StackingRegressor(estimators=stacking_estimators, final_estimator=Ridge(alpha=1.0))
    }

    results = []
    fitted_pipelines = {}

    y_valid_orig = np.expm1(y_valid_log)

    print("\n--- Training & Evaluating Models ---")
    for name, regressor in models.items():
        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
        pipeline.steps.append(("model", regressor))
        
        pipeline.fit(X_train, y_train_log)
        
        pred_log = pipeline.predict(X_valid)
        pred_orig = np.expm1(pred_log)

        eval_res = evaluate_predictions(name, y_valid_orig, pred_orig)
        results.append(eval_res)
        fitted_pipelines[name] = pipeline
        
        print(f"  {name:<20} | MAE: ${eval_res['MAE ($)']:>10,.2f} | RMSE: ${eval_res['RMSE ($)']:>10,.2f} | R2: {eval_res['R2 Score']:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE ($)")
    print("\n--- Model Leaderboard ---")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nTop Performing Base Model: {best_model_name}")

    # 5. Hyperparameter Tuning for Top Ensemble Model (Gradient Boosting / Random Forest)
    print("\n--- Hyperparameter Tuning via GridSearchCV ---")
    base_rf = build_preprocessing_pipeline(num_cols, cat_cols)
    base_rf.steps.append(("model", GradientBoostingRegressor(random_state=42)))

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.03, 0.05],
        "model__max_depth": [3, 4]
    }

    grid_search = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train_log)
    best_pipeline = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Evaluate Tuned Model
    tuned_pred_log = best_pipeline.predict(X_valid)
    tuned_pred_orig = np.expm1(tuned_pred_log)
    tuned_eval = evaluate_predictions("Tuned Gradient Boosting", y_valid_orig, tuned_pred_orig)
    print(f"Tuned Model | MAE: ${tuned_eval['MAE ($)']:,.2f} | RMSE: ${tuned_eval['RMSE ($)']:,.2f} | R2: {tuned_eval['R2 Score']:.4f}")

    # 6. 5-Fold Cross Validation
    print("\n--- 5-Fold Cross Validation ---")
    cv_scores = cross_val_score(best_pipeline, X, y_log, cv=5, scoring="neg_root_mean_squared_error")
    rmse_cv = -cv_scores
    print(f"CV RMSE Scores (Log Scale): {np.round(rmse_cv, 4)}")
    print(f"Mean CV RMSE: {rmse_cv.mean():.4f}")

    # 7. Model Serialization
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved Best Trained Model Pipeline to: {model_path}")

    # 8. Feature Importance Analysis
    try:
        preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor_fitted.get_feature_names_out()
        rf_model = best_pipeline.named_steps["model"]

        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            # Clean feature name prefixes
            fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "").str.replace("cat__", "")
            fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

            plt.figure(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=fi_df, hue="Feature", legend=False, palette="mako")
            plt.title("Top 20 Important Features (Tuned Gradient Boosting)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            fi_img_path = os.path.join(STATIC_DIR, "feature_importance.png")
            plt.savefig(fi_img_path, dpi=300)
            plt.close()
            print(f"Saved Feature Importance plot to: {fi_img_path}")
    except Exception as e:
        print(f"Feature importance rendering skipped: {e}")

    # 9. Generate Kaggle-style Test Submission CSV
    print("\n--- Generating Test Dataset Predictions ---")
    test_X = test_df.drop(columns=["Id"], errors="ignore")
    test_pred_log = best_pipeline.predict(test_X)
    test_pred_orig = np.expm1(test_pred_log)

    submission_df = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": np.round(test_pred_orig, 2)
    })
    sub_path = os.path.join(PROJECT_ROOT, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Saved submission.csv to: {sub_path} (Shape: {submission_df.shape})")

    # 10. Actual vs Predicted Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_valid_orig, tuned_pred_orig, alpha=0.6, color="indigo")
    plt.plot([y_valid_orig.min(), y_valid_orig.max()], [y_valid_orig.min(), y_valid_orig.max()], 'r--', lw=2)
    plt.xlabel("Actual Sale Price ($)", fontsize=12)
    plt.ylabel("Predicted Sale Price ($)", fontsize=12)
    plt.title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    act_vs_pred_path = os.path.join(STATIC_DIR, "actual_vs_predicted.png")
    plt.savefig(act_vs_pred_path, dpi=300)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {act_vs_pred_path}")

    return best_pipeline

if __name__ == "__main__":
    run_model_training()
