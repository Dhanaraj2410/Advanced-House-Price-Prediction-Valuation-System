"""
Django Management Command to seed realistic sample property valuation records.
Usage: python manage.py seed_history [--count 15]
"""

import json
from django.core.management.base import BaseCommand
from app.models import HousePrediction
from ml.predict import predict_price

SAMPLE_PROPERTIES = [
    {"OverallQual": 9, "GrLivArea": 2800, "YearBuilt": 2018, "GarageCars": 3, "TotalBsmtSF": 1400, "FullBath": 3, "Neighborhood": "NoRidge", "notes": "Executive luxury home in Northridge"},
    {"OverallQual": 8, "GrLivArea": 2200, "YearBuilt": 2015, "GarageCars": 2, "TotalBsmtSF": 1100, "FullBath": 2, "Neighborhood": "NridgHt", "notes": "Modern single family residential"},
    {"OverallQual": 7, "GrLivArea": 1850, "YearBuilt": 2008, "GarageCars": 2, "TotalBsmtSF": 950, "FullBath": 2, "Neighborhood": "Somerst", "notes": "Suburban multi-level home"},
    {"OverallQual": 6, "GrLivArea": 1500, "YearBuilt": 2002, "GarageCars": 2, "TotalBsmtSF": 850, "FullBath": 2, "Neighborhood": "CollgCr", "notes": "Standard family home near college"},
    {"OverallQual": 5, "GrLivArea": 1200, "YearBuilt": 1985, "GarageCars": 1, "TotalBsmtSF": 700, "FullBath": 1, "Neighborhood": "NAmes", "notes": "Cozy 3-bedroom bungalow"},
    {"OverallQual": 4, "GrLivArea": 950, "YearBuilt": 1960, "GarageCars": 1, "TotalBsmtSF": 600, "FullBath": 1, "Neighborhood": "Edwards", "notes": "Starter starter home needing cosmetic updates"},
    {"OverallQual": 8, "GrLivArea": 2400, "YearBuilt": 2019, "GarageCars": 3, "TotalBsmtSF": 1250, "FullBath": 3, "Neighborhood": "StoneBr", "notes": "High end custom construction"},
    {"OverallQual": 7, "GrLivArea": 1700, "YearBuilt": 2005, "GarageCars": 2, "TotalBsmtSF": 900, "FullBath": 2, "Neighborhood": "Gilbert", "notes": "Quiet cul-de-sac location"},
    {"OverallQual": 6, "GrLivArea": 1400, "YearBuilt": 1998, "GarageCars": 2, "TotalBsmtSF": 800, "FullBath": 2, "Neighborhood": "Sawyer", "notes": "Renovated kitchen & open floor plan"},
    {"OverallQual": 9, "GrLivArea": 3100, "YearBuilt": 2021, "GarageCars": 3, "TotalBsmtSF": 1600, "FullBath": 3, "Neighborhood": "NoRidge", "notes": "Ultra luxury estate with pool"}
]

class Command(BaseCommand):
    help = "Populates database with sample property valuation records."

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing valuation history records before seeding.',
        )

    def handle(self, *args, **options):
        if options['clear']:
            deleted, _ = HousePrediction.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Cleared {deleted} existing prediction history records."))

        created_count = 0
        for prop in SAMPLE_PROPERTIES:
            res = predict_price(prop)
            HousePrediction.objects.create(
                overall_qual=prop['OverallQual'],
                gr_liv_area=prop['GrLivArea'],
                year_built=prop['YearBuilt'],
                garage_cars=prop['GarageCars'],
                total_bsmt_sf=prop['TotalBsmtSF'],
                full_bath=prop['FullBath'],
                neighborhood=prop['Neighborhood'],
                predicted_price=res['predicted_price'],
                price_min=res.get('price_min'),
                price_max=res.get('price_max'),
                price_per_sqft=res.get('price_per_sqft'),
                valuation_tier=res.get('valuation_tier', 'Mid-Range Suburban'),
                notes=prop.get('notes', ''),
                insights_json=json.dumps(res['insights'])
            )
            created_count += 1

        self.stdout.write(self.style.SUCCESS(f"Successfully seeded {created_count} property valuation records!"))
