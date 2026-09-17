"""
Preprocessing and Feature Engineering pipeline module for House Price Prediction.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for domain-specific feature engineering.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Total Square Footage
        total_bsmt = X_out['TotalBsmtSF'].fillna(0) if 'TotalBsmtSF' in X_out.columns else 0
        flr1 = X_out['1stFlrSF'].fillna(0) if '1stFlrSF' in X_out.columns else 0
        flr2 = X_out['2ndFlrSF'].fillna(0) if '2ndFlrSF' in X_out.columns else 0
        X_out['TotalSF'] = total_bsmt + flr1 + flr2

        # 2. Total Bathrooms
        full_bath = X_out['FullBath'].fillna(0) if 'FullBath' in X_out.columns else 0
        half_bath = X_out['HalfBath'].fillna(0) if 'HalfBath' in X_out.columns else 0
        bsmt_full = X_out['BsmtFullBath'].fillna(0) if 'BsmtFullBath' in X_out.columns else 0
        bsmt_half = X_out['BsmtHalfBath'].fillna(0) if 'BsmtHalfBath' in X_out.columns else 0
        X_out['TotalBath'] = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)

        # 3. House Age & Remodel status
        yr_sold = X_out['YrSold'].fillna(2010) if 'YrSold' in X_out.columns else 2010
        yr_built = X_out['YearBuilt'].fillna(1970) if 'YearBuilt' in X_out.columns else 1970
        yr_remod = X_out['YearRemodAdd'].fillna(yr_built) if 'YearRemodAdd' in X_out.columns else yr_built

        X_out['HouseAge'] = yr_sold - yr_built
        X_out['RemodelAge'] = yr_sold - yr_remod
        X_out['IsRemodeled'] = (yr_remod != yr_built).astype(int)

        # 4. Has Garage, Has Basement, Has Porch
        garage_cars = X_out['GarageCars'].fillna(0) if 'GarageCars' in X_out.columns else 0
        X_out['HasGarage'] = (garage_cars > 0).astype(int)
        X_out['HasBasement'] = (total_bsmt > 0).astype(int)

        return X_out

def remove_outliers(df):
    """
    Removes documented GrLivArea outliers (GrLivArea > 4000 & SalePrice < 300000).
    """
    if 'SalePrice' in df.columns and 'GrLivArea' in df.columns:
        outlier_mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        num_outliers = outlier_mask.sum()
        if num_outliers > 0:
            print(f"Removing {num_outliers} extreme outlier(s) from dataset.")
            df = df[~outlier_mask].reset_index(drop=True)
    return df

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Constructs the ColumnTransformer preprocessing pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", preprocessor)
        ]
    )

    return full_pipeline

if __name__ == "__main__":
    from ml.dataset import load_data
    train, test = load_data()
    train = remove_outliers(train)

    X = train.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = train["SalePrice"]

    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.transform(X)

    num_cols = X_engineered.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_engineered.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
    X_processed = pipeline.fit_transform(X)

    print(f"Original X shape: {X.shape}")
    print(f"Engineered X shape: {X_engineered.shape}")
    print(f"Processed matrix shape: {X_processed.shape}")
    print("Preprocessing pipeline build & transform verified successfully!")
