# 🏠 House Price Prediction 

## Project Overview  

A Machine Learning project that predicts house sale prices based on property features such as quality, living area, location, garage, and year built.

## Technologies
* Python
* Pandas & NumPy 
* Matplotlib & Seaborn
* Scikit-learn
* Machine Learning Regression

## ML Models 

* Linear Regression

## Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. House Price Prediction

## Dataset 

**Ames Housing Dataset – Kaggle House Prices**



=======
# 🏠 Advanced House Price Prediction & Valuation System

An end-to-end Machine Learning Regression & Production Django Web Application that predicts residential property sale prices using structural, spatial, and quality features from the Ames Housing Dataset.

---

## 📌 Project Overview
- **Dataset**: 1,460 property records (81 columns) in `train.csv` and 1,459 records (80 columns) in `test.csv`.
- **Target Variable**: `SalePrice` (Log-transformed `np.log1p` during training for variance stabilization).
- **Core Algorithms Evaluated**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
- **Top Model Metrics**: Tuned Gradient Boosting Regressor achieving **$14,108.70 MAE**, **$20,023.42 RMSE**, and **0.9274 R² Score** with **0.1204 Mean CV RMSE**.
- **Backend Infrastructure**: Django REST Framework + MySQL Database (`house_price_db`) with prediction history logging and automated AI feature influence insights.
- **Frontend Dashboard**: Responsive Bootstrap 5 interface with interactive property valuation forms and EDA chart visualizations.

---

## 🏗️ Project Architecture

```
User (Browser / API Client)
    │
    ▼
Django Web Application / DRF REST API
    │
    ├──► MySQL Database (house_price_db: Prediction History)
    │
    └──► Modular ML Inference Pipeline (ml/predict.py)
              │
              └──► Preprocessor Pipeline (ml/preprocessing.py)
                        │
                        └──► Trained Model (models/best_model.pkl)
```

## 📁 Repository Folder Structure

```text
House_Price_Prediction/
│
├── data/
│   ├── train.csv                 # 1,460 rows × 81 cols
│   └── test.csv                  # 1,459 rows × 80 cols
│
├── ml/
│   ├── __init__.py
│   ├── dataset.py                # Data loading & summary statistics
│   ├── eda.py                    # Visualizations & distribution analysis
│   ├── preprocessing.py          # Custom FeatureEngineer & ColumnTransformer
│   ├── train.py                  # Model benchmark, tuning, CV, joblib export
│   └── predict.py                # Production prediction API & insights wrapper
│
├── models/
│   └── best_model.pkl            # Serialized Scikit-Learn pipeline
│
├── config/                       # Django project settings & MySQL routing
├── app/                          # Django application (models, views, DRF serializers)
├── templates/                    # Bootstrap 5 HTML templates
├── static/
│   └── eda/                      # High-resolution charts & feature importance plots
│
├── submission.csv                # Kaggle test dataset predictions
├── manage.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 Machine Learning Model Leaderboard

| Model | MAE ($) | RMSE ($) | R² Score |
| :--- | :---: | :---: | :---: |
| **Tuned Gradient Boosting** | **$14,108.70** | **$20,023.42** | **0.9274** |
| **Lasso Regression** | $14,183.80 | $19,571.46 | 0.9307 |
| **XGBoost** | $14,224.54 | $20,059.30 | 0.9272 |
| **Ridge Regression** | $14,437.50 | $19,995.37 | 0.9276 |
| **Linear Regression** | $15,372.76 | $21,652.12 | 0.9151 |
| **Random Forest** | $16,407.76 | $23,611.41 | 0.8991 |
| **Decision Tree** | $28,617.68 | $43,020.82 | 0.6649 |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Data Analysis & Model Training
```powershell
# 1. Dataset Verification
python -m ml.dataset

# 2. Generate EDA Plots
python -m ml.eda

# 3. Train & Evaluate Models (Saves best_model.pkl & submission.csv)
python -m ml.train
```

### 3. Database Setup & Server Launch
```powershell
# Setup MySQL Database
python scratch/create_db.py

# Apply Migrations
python manage.py makemigrations app
python manage.py migrate

# Launch Django Server
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/predict/` | Submit house features & get predicted sale price + insights |
| `GET` | `/api/predictions/` | Fetch historical predictions log |
| `GET` | `/api/predictions/<id>/` | Fetch single prediction record |
| `DELETE` | `/api/predictions/<id>/` | Delete historical prediction record |

### Request Payload Example (`POST /api/predict/`)
```json
{
  "overall_qual": 8,
  "gr_liv_area": 2100,
  "year_built": 2015,
  "garage_cars": 2,
  "total_bsmt_sf": 1100,
  "full_bath": 2,
  "neighborhood": "NridgHt"
}
```

### Response Example  
```json
{
  "id": 1,
  "overall_qual": 8,
  "gr_liv_area": 2100.0,
  "year_built": 2015,
  "garage_cars": 2,
  "total_bsmt_sf": 1100.0,
  "full_bath": 2,
  "neighborhood": "NridgHt",
  "predicted_price": 284520.15,
  "insights": [
    "High Overall Quality rating (8/10) significantly boosts property valuation.",
    "Spacious living area (2100 sq ft) adds substantial valuation premium.",
    "Modern construction (Built 2015) commands high modern buyer appeal."
  ],
  "created_at": "2026-09-17T22:25:00Z"
}


Author
Dhanaraj Lokhande
BE Information Technology | 2026 Graduate 

