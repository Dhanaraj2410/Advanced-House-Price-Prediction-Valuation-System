"""
Prediction interface module for loading the trained pipeline and making single/batch house price predictions.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

_model = None

def get_model():
    """
    Lazy-loads the trained best_model.pkl pipeline.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at '{MODEL_PATH}'. Run 'python -m ml.train' first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_default_house_features():
    """
    Returns a dictionary of default house features matching the training schema.
    """
    return {
        "OverallQual": 6,
        "GrLivArea": 1500,
        "YearBuilt": 2000,
        "YearRemodAdd": 2005,
        "TotalBsmtSF": 900,
        "1stFlrSF": 900,
        "2ndFlrSF": 600,
        "FullBath": 2,
        "HalfBath": 1,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 7,
        "Fireplaces": 1,
        "GarageCars": 2,
        "GarageArea": 500,
        "WoodDeckSF": 100,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2010,
        "LotFrontage": 65.0,
        "LotArea": 8450,
        "OverallCond": 5,
        "MasVnrArea": 0.0,
        "BsmtFinSF1": 400.0,
        "BsmtFinSF2": 0.0,
        "BsmtUnfSF": 500.0,
        "LowQualFinSF": 0,
        "GarageYrBlt": 2000.0,
        "MSZoning": "RL",
        "Street": "Pave",
        "Alley": "Grvl",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinType2": "Unf",
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "KitchenQual": "TA",
        "Functional": "Typ",
        "FireplaceQu": "Gd",
        "GarageType": "Attchd",
        "GarageFinish": "RFn",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "PoolQC": "Gd",
        "Fence": "MnPrv",
        "MiscFeature": "Shed",
        "SaleType": "WD",
        "SaleCondition": "Normal",
        "MSSubClass": 60
    }

def predict_price(input_data):
    """
    Predicts house sale price for input dictionary or DataFrame.
    """
    model = get_model()
    
    if isinstance(input_data, dict):
        defaults = get_default_house_features()
        defaults.update(input_data)
        input_df = pd.DataFrame([defaults])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise ValueError("Input data must be a dictionary or pandas DataFrame.")

    # Predict in log scale and convert back
    log_pred = model.predict(input_df)
    predicted_price = float(np.expm1(log_pred[0]))
    
    # Feature impact insights
    overall_qual = input_df.get('OverallQual', [5])[0]
    gr_liv_area = input_df.get('GrLivArea', [1500])[0]
    year_built = input_df.get('YearBuilt', [2000])[0]
    garage_cars = input_df.get('GarageCars', [2])[0]
    
    insights = []
    if overall_qual >= 8:
        insights.append(f"High Overall Quality rating ({overall_qual}/10) significantly boosts property valuation.")
    elif overall_qual <= 4:
        insights.append(f"Below average Quality rating ({overall_qual}/10) reduces property market value.")
        
    if gr_liv_area > 2000:
        insights.append(f"Spacious living area ({gr_liv_area} sq ft) adds substantial valuation premium.")
        
    if year_built >= 2010:
        insights.append(f"Modern construction (Built {year_built}) commands high modern buyer appeal.")

    return {
        "predicted_price": round(predicted_price, 2),
        "formatted_price": f"${predicted_price:,.2f}",
        "insights": insights
    }

if __name__ == "__main__":
    sample = {
        "OverallQual": 8,
        "GrLivArea": 2100,
        "YearBuilt": 2015,
        "GarageCars": 2,
        "FullBath": 2
    }
    res = predict_price(sample)
    print("--- Test Prediction ---")
    print(f"Predicted Price: {res['formatted_price']}")
    print("Insights:", res['insights'])
