# REST API Documentation & Usage Guide

Welcome to the **Advanced House Price Prediction & Valuation API** reference documentation.

## Base URL
Default local development base URL: `http://127.0.0.1:8000`

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health/` | System status, database connectivity & ML engine health |
| `GET`  | `/api/feature-importance/` | Returns relative feature importance weights |
| `POST` | `/api/predict/` | Single property price estimation |
| `POST` | `/api/predict/bulk/` | Batch valuation via JSON array or CSV upload |
| `GET`  | `/api/predictions/` | Paginated list of historical property predictions |
| `GET`  | `/api/predictions/<id>/` | Single prediction detail record |

---

## 1. System Health Check
**`GET /api/health/`**

### Response Example (`200 OK`)
```json
{
  "status": "healthy",
  "database": "ok",
  "model_engine": "loaded",
  "total_predictions_stored": 15,
  "version": "1.2.0"
}
```

---

## 2. Feature Importance
**`GET /api/feature-importance/`**

### Response Example (`200 OK`)
```json
{
  "count": 10,
  "feature_importances": {
    "OverallQual": 0.385,
    "GrLivArea": 0.245,
    "TotalBsmtSF": 0.112,
    "YearBuilt": 0.084,
    "GarageCars": 0.061
  }
}
```

---

## 3. Predict Single Property Price
**`POST /api/predict/`**

### Request Body (`application/json`)
```json
{
  "overall_qual": 8,
  "gr_liv_area": 2200,
  "year_built": 2018,
  "garage_cars": 2,
  "total_bsmt_sf": 1100,
  "full_bath": 2,
  "neighborhood": "NridgHt",
  "notes": "Custom high-spec home"
}
```

### Response Example (`201 Created`)
```json
{
  "id": 16,
  "overall_qual": 8,
  "gr_liv_area": 2200,
  "year_built": 2018,
  "garage_cars": 2,
  "total_bsmt_sf": 1100,
  "full_bath": 2,
  "neighborhood": "NridgHt",
  "predicted_price": 295400.0,
  "price_min": 276200.0,
  "price_max": 314600.0,
  "price_per_sqft": 134.27,
  "valuation_tier": "Premium Residential",
  "created_at": "2026-09-21T23:00:00Z"
}
```
