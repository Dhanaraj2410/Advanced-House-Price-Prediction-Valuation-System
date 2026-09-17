from django.test import TestCase, Client
from django.urls import reverse
from app.models import HousePrediction

class HousePredictionAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.sample_payload = {
            "overall_qual": 7,
            "gr_liv_area": 1800,
            "year_built": 2008,
            "garage_cars": 2,
            "total_bsmt_sf": 1000,
            "full_bath": 2,
            "neighborhood": "CollgCr"
        }

    def test_predict_api(self):
        response = self.client.post(
            reverse('api_predict'),
            data=self.sample_payload,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn('predicted_price', data)
        self.assertTrue(data['predicted_price'] > 50000)

    def test_prediction_list_api(self):
        HousePrediction.objects.create(
            overall_qual=8,
            gr_liv_area=2000,
            year_built=2015,
            garage_cars=2,
            total_bsmt_sf=1100,
            full_bath=2,
            neighborhood="NridgHt",
            predicted_price=280000.0
        )
        response = self.client.get(reverse('api_predictions_list'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
