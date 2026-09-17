"""
Exploratory Data Analysis (EDA) module for House Price Prediction.
Generates statistical summaries and saves EDA visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import load_data

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_eda():
    """
    Performs full EDA on the training dataset and saves visualization plots.
    """
    train, test = load_data()
    print("--- 1. Target Variable Analysis ---")
    saleprice = train['SalePrice']
    print(f"Mean SalePrice: ${saleprice.mean():,.2f}")
    print(f"Median SalePrice: ${saleprice.median():,.2f}")
    print(f"Skewness: {saleprice.skew():.2f} (Right-skewed, log transform recommended)")

    # Plot 1: SalePrice Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(saleprice, kde=True, color='teal')
    plt.title('Sale Price Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, 'saleprice_dist.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # Plot 2: Overall Quality vs Sale Price
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=train, hue='OverallQual', palette='viridis', legend=False)
    plt.title('Overall Quality vs. Sale Price', fontsize=14, fontweight='bold')
    plt.xlabel('Overall Quality (1-10)')
    plt.ylabel('Sale Price ($)')
    plt.tight_layout()
    qual_path = os.path.join(OUTPUT_DIR, 'overall_qual_vs_saleprice.png')
    plt.savefig(qual_path, dpi=300)
    plt.close()
    print(f"Saved: {qual_path}")

    # Plot 3: Living Area vs Sale Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, alpha=0.7, color='coral')
    plt.title('Above Ground Living Area vs. Sale Price', fontsize=14, fontweight='bold')
    plt.xlabel('GrLivArea (sq ft)')
    plt.ylabel('Sale Price ($)')
    plt.tight_layout()
    area_path = os.path.join(OUTPUT_DIR, 'grlivarea_vs_saleprice.png')
    plt.savefig(area_path, dpi=300)
    plt.close()
    print(f"Saved: {area_path}")

    # Plot 4: Top 15 Correlated Numeric Features
    numeric_df = train.select_dtypes(include=['number'])
    corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
    top_corr_features = corr.head(15)
    
    print("\n--- Top 15 Features Correlated with SalePrice ---")
    for feat, val in top_corr_features.items():
        print(f"  {feat:<15}: {val:.4f}")

    # Plot Heatmap of Top 10 correlated features
    top_10_cols = top_corr_features.head(10).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[top_10_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (Top 10 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")

    return top_corr_features

if __name__ == "__main__":
    perform_eda()
