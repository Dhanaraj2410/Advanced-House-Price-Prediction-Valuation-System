"""
Unit test suite for ML pipeline, preprocessing feature transformations, and prediction interface.
Usage: python -m unittest ml/tests.py
"""

import unittest
import numpy as np
import pandas as pd
from ml.preprocessing import FeatureEngineer, remove_outliers
from ml.predict import predict_price, get_default_house_features

class TestMLPreprocessing(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineer()
        self.sample_df = pd.DataFrame([{
            'OverallQual': 8,
            'GrLivArea': 2000.0,
            'YearBuilt': 2015,
            'YearRemodAdd': 2018,
            'TotalBsmtSF': 1000.0,
            '1stFlrSF': 1000.0,
            '2ndFlrSF': 1000.0,
            'FullBath': 2,
            'HalfBath': 1,
            'BsmtFullBath': 1,
            'BsmtHalfBath': 0,
            'GarageCars': 2,
            'WoodDeckSF': 100,
            'OpenPorchSF': 50,
            'EnclosedPorch': 0,
            '3SsnPorch': 0,
            'ScreenPorch': 0,
            'YrSold': 2020
        }])

    def test_feature_engineering_transforms(self):
        df_out = self.fe.transform(self.sample_df)
        
        # Total Square Footage
        self.assertIn('TotalSF', df_out.columns)
        self.assertEqual(df_out['TotalSF'].iloc[0], 3000.0)

        # Total Bathrooms (2 + 0.5*1 + 1 + 0 = 3.5)
        self.assertIn('TotalBath', df_out.columns)
        self.assertEqual(df_out['TotalBath'].iloc[0], 3.5)

        # House Age (2020 - 2015 = 5)
        self.assertIn('HouseAge', df_out.columns)
        self.assertEqual(df_out['HouseAge'].iloc[0], 5)

        # Total Porch SF (100 + 50 = 150)
        self.assertIn('TotalPorchSF', df_out.columns)
        self.assertEqual(df_out['TotalPorchSF'].iloc[0], 150)

        # Quality_LivArea (8 * 2000 = 16000)
        self.assertIn('Quality_LivArea', df_out.columns)
        self.assertEqual(df_out['Quality_LivArea'].iloc[0], 16000.0)

    def test_outlier_removal(self):
        outlier_df = pd.DataFrame([
            {'GrLivArea': 4500, 'SalePrice': 200000}, # Outlier
            {'GrLivArea': 2000, 'SalePrice': 250000}  # Normal
        ])
        cleaned = remove_outliers(outlier_df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned['GrLivArea'].iloc[0], 2000)

class TestMLPredict(unittest.TestCase):
    def test_default_features_schema(self):
        defaults = get_default_house_features()
        self.assertIn('OverallQual', defaults)
        self.assertIn('GrLivArea', defaults)
        self.assertIn('Neighborhood', defaults)

    def test_predict_price_dictionary_input(self):
        sample_input = {
            'OverallQual': 8,
            'GrLivArea': 2100,
            'YearBuilt': 2015,
            'GarageCars': 3,
            'FullBath': 2,
            'Neighborhood': 'NridgHt'
        }
        res = predict_price(sample_input)
        
        self.assertIn('predicted_price', res)
        self.assertIn('formatted_price', res)
        self.assertIn('price_min', res)
        self.assertIn('price_max', res)
        self.assertIn('price_per_sqft', res)
        self.assertIn('valuation_tier', res)
        self.assertIn('insights', res)
        self.assertIn('anomalies', res)
        
        self.assertGreater(res['predicted_price'], 0)
        self.assertGreater(res['price_max'], res['price_min'])
        self.assertIsInstance(res['insights'], list)

    def test_predict_price_anomaly_warnings(self):
        anomalous_input = {
            'OverallQual': 15, # Out of 1-10 range
            'GrLivArea': 8000, # Out of 300-6000 range
            'YearBuilt': 1750, # Out of 1850-2026 range
            'TotalBsmtSF': 6000 # Basement > 5000
        }
        res = predict_price(anomalous_input)
        self.assertIn('anomalies', res)
        self.assertGreaterEqual(len(res['anomalies']), 3)

    def test_get_feature_importances(self):
        from ml.predict import get_feature_importances
        importances = get_feature_importances()
        self.assertIsInstance(importances, dict)
        self.assertIn('OverallQual', importances)
        self.assertIn('GrLivArea', importances)

if __name__ == '__main__':
    unittest.main()

