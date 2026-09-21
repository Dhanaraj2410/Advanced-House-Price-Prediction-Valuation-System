from django.test import TestCase, Client
from django.urls import reverse
from django.template import Context
from app.models import HousePrediction

# Python 3.14 + Django test client Context.__copy__ compatibility patch
def _fixed_context_copy(self):
    duplicate = Context()
    duplicate.dicts = getattr(self, 'dicts', [])[:]
    return duplicate

Context.__copy__ = _fixed_context_copy

class HousePredictionWebTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.sample_prediction = HousePrediction.objects.create(
            overall_qual=8,
            gr_liv_area=2000,
            year_built=2015,
            garage_cars=2,
            total_bsmt_sf=1100,
            full_bath=2,
            neighborhood="NridgHt",
            predicted_price=280000.0,
            price_min=261800.0,
            price_max=298200.0,
            price_per_sqft=140.0,
            valuation_tier="Premium Residential",
            notes="Test luxury house"
        )

    def test_home_view(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HouseValuation AI")

    def test_predict_view_get(self):
        response = self.client.get(reverse('predict'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Overall Quality")

    def test_predict_view_post(self):
        payload = {
            'overall_qual': 7,
            'gr_liv_area': 1750,
            'year_built': 2010,
            'garage_cars': 2,
            'total_bsmt_sf': 950,
            'full_bath': 2,
            'neighborhood': 'Somerst'
        }
        response = self.client.post(reverse('predict'), data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predicted Valuation")

    def test_history_view_filtering(self):
        response = self.client.get(reverse('history'), {'q': 'NridgHt'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['predictions']), 1)

    def test_export_history_csv_view(self):
        response = self.client.get(reverse('export_history_csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('NridgHt', response.content.decode('utf-8'))

    def test_dashboard_view(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Valuation Analytics")

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
        self.assertIn('price_min', data)
        self.assertIn('price_max', data)

    def test_bulk_predict_api(self):
        bulk_payload = [
            self.sample_payload,
            {
                "overall_qual": 9,
                "gr_liv_area": 2500,
                "year_built": 2020,
                "garage_cars": 3,
                "total_bsmt_sf": 1300,
                "full_bath": 3,
                "neighborhood": "NoRidge"
            }
        ]
        response = self.client.post(
            reverse('api_predict_bulk'),
            data=bulk_payload,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data['count'], 2)

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

    def test_health_check_api(self):
        response = self.client.get(reverse('api_health'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['database'], 'ok')
        self.assertEqual(data['model_engine'], 'loaded')
        self.assertIn('version', data)


