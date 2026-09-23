#  🏠  Advanced House Price Prediction & Valuation System  

An end-to-end Machine Learning Regression & Production Django Web Application that predicts residential property sale prices using structural, spatial, quality, and domain-engineered features from the Ames Housing Dataset. 

---

## 📌 Project Overview 

- **Dataset**: 1,460 property records (81 columns) in `train.csv` and 1,459 records (80 columns) in `test.csv`.  
- **Target Variable**: `SalePrice` (Log-transformed `np.log1p` during training for variance stabilization).
- **Core Algorithms Evaluated**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Stacking Ensemble.
- **Top Model Performance**: Stacking Regressor & Tuned Lasso Regressor achieving **$14,028 MAE**, **$19,896 RMSE**, and **0.9283 R² Score** with **0.1188 Mean CV RMSE**.
- **Valuation Confidence**: Automated confidence bounds ($\pm 6.5\%$ standard error interval), price per sq. ft. metrics, valuation tier classification (`Luxury Estate`, `Premium Residential`, `Mid-Range Suburban`, `Budget Friendly`), and dynamic AI feature influence drivers.
- **Batch CSV & JSON Export**: Upload bulk property CSVs via web interface (`/predict/bulk/`) or REST API (`/api/predict/bulk/`) and export history as `.csv` or `.json`.
- **Health Check & Feature Importances API**: RESTful endpoints for system health diagnostics (`/api/health/`) and relative feature weight breakdowns (`/api/feature-importance/`).
- **Input Anomaly Guardrails**: Out-of-bounds parameter validation flagging extreme square footage, construction year, or pricing anomalies.
- **Developer & CLI Workflows**: Custom Django management commands (`retrain_model`, `seed_history`, `export_model_summary`), request performance middleware, and GitHub Actions CI workflow (`.github/workflows/ci.yml`).


---

## 🏗️ System Architecture

```text
User (Web UI / API Client)
    │
    ▼
Django Web Application & DRF REST API (app/)
    │
    ├──► Database Storage (HousePrediction Model with confidence bounds & tags)
    │
    ├──► Bulk Valuation & CSV Export Handler (pandas / csv engine)
    │
    └──► Modular ML Inference Engine (ml/predict.py)
              │
              ├──► Domain Feature Engineer (TotalSF, TotalBath, Quality_LivArea, QualityAgeRatio)
              │
              └──► Preprocessor Pipeline & Stacking Regressor (models/best_model.pkl)
```

---

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
│   ├── predict.py                # Production prediction API & insights wrapper
│   ├── benchmark.py              # Metrics benchmarking & JSON report generator
│   └── tests.py                  # ML pipeline & transformation unit tests
│
├── models/
│   ├── best_model.pkl            # Serialized Scikit-Learn pipeline
│   └── benchmark_report.json     # Automated JSON model leaderboard report
│
├── config/                       # Django project settings & URL routing
├── app/                          # Django web app (models, views, forms, serializers, management commands)
│   ├── management/commands/
│   │   ├── retrain_model.py     # CLI command to retrain model & generate benchmarks
│   │   └── seed_history.py      # CLI command to seed realistic test valuation records
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── forms.py
│   └── tests.py                  # Web view & REST API integration test suite
│
├── templates/                    # Responsive Bootstrap 5 HTML templates
│   ├── base.html                 # Main layout & navigation
│   ├── home.html                 # Landing page
│   ├── predict.html              # Single property & bulk CSV upload form
│   ├── dashboard.html            # Analytics dashboard & benchmark leaderboard
│   └── history.html              # Filterable history table with pagination & CSV export
│
├── static/
│   └── eda/                      # High-resolution benchmark & feature importance plots
│
├── submission.csv                # Kaggle test dataset predictions
├── manage.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 Machine Learning Model Leaderboard

| Model Architecture | MAE ($) | RMSE ($) | R² Score | MAPE (%) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | **$14,028.62** | **$19,942.50** | **0.9280** | **8.71%** | 8.83s |
| **Lasso Regression** | **$14,323.62** | **$19,896.15** | **0.9283** | **8.71%** | 0.19s |
| **Ridge Regression** | $14,526.46 | $20,237.65 | 0.9259 | 8.94% | 0.07s |
| **XGBoost** | $14,773.23 | $20,963.95 | 0.9204 | 9.04% | 0.39s |
| **Gradient Boosting** | $15,027.94 | $20,964.25 | 0.9204 | 9.27% | 1.73s |
| **Linear Regression** | $15,357.12 | $21,636.24 | 0.9153 | 9.49% | 0.20s |
| **Random Forest** | $16,183.30 | $24,314.31 | 0.8930 | 9.83% | 1.31s |
| **Decision Tree** | $22,004.69 | $30,815.66 | 0.8281 | 13.37% | 0.08s |

---

## 🛠️ CLI Management Commands

### Retrain Model & Generate Benchmarks
```powershell
python manage.py retrain_model --benchmark
```

### Seed Test Prediction History
```powershell
python manage.py seed_history --clear
```

---

## 🔌 REST API Endpoints and OpenAPI Documentation 

OpenAPI Schema Route: `GET /api/schema/`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/predict/` | Single property valuation & feature insights |
| `POST` | `/api/predict/bulk/` | Batch valuation from JSON list or CSV file upload |
| `GET` | `/api/predictions/` | Fetch historical predictions list |
| `GET` | `/api/predictions/<id>/` | Fetch single prediction record |
| `DELETE` | `/api/predictions/<id>/` | Delete prediction record |
| `GET` | `/api/schema/` | Generate OpenAPI 3.0 API schema |

### Single Property Request Example (`POST /api/predict/`)
```json
{
  "overall_qual": 8,
  "gr_liv_area": 2100,
  "year_built": 2015,
  "garage_cars": 3,
  "total_bsmt_sf": 1100,
  "full_bath": 2,
  "neighborhood": "NridgHt"
}
```

### Single Property Response Example
```json
{
  "id": 1,
  "overall_qual": 8,
  "gr_liv_area": 2100.0,
  "year_built": 2015,
  "garage_cars": 3,
  "total_bsmt_sf": 1100.0,
  "full_bath": 2,
  "neighborhood": "NridgHt",
  "predicted_price": 226818.57,
  "price_min": 212075.36,
  "price_max": 241561.78,
  "price_per_sqft": 108.01,
  "valuation_tier": "Mid-Range Suburban",
  "insights": [
    "High Overall Quality rating (8/10) significantly boosts property valuation.",
    "Spacious living area (2,100 sq ft) adds substantial valuation premium.",
    "Modern construction (Built 2015) commands strong buyer appeal."
  ],
  "created_at": "2026-09-20T22:35:00Z"
}
```

---

## 🧪 Automated Testing & Verification 

Run the full Django web & REST API test suite:
```powershell
python manage.py test
```

Run the dedicated ML pipeline unit test suite:
```powershell
python -m unittest ml/tests.py
```

Run model evaluation benchmarking:
```powershell
python -m ml.benchmark
```

---

## 📝Author
- **Author**:-
-  Dhanaraj Lokhande 
