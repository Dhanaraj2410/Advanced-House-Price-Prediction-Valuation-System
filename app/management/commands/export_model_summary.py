"""
Django Management Command to export model metrics summary report to static/model_summary.json.
Usage: python manage.py export_model_summary
"""

import os
import json
from django.core.management.base import BaseCommand
from app.models import HousePrediction
from ml.predict import get_feature_importances

class Command(BaseCommand):
    help = "Exports overall model performance metrics, database counts, and feature weights to JSON summary report."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Generating model metrics summary report..."))
        try:
            total_predictions = HousePrediction.objects.count()
            importances = get_feature_importances()

            summary = {
                "system_name": "Advanced House Price Prediction & Valuation System",
                "version": "1.2.0",
                "total_stored_predictions": total_predictions,
                "top_feature_importances": importances,
                "supported_valuation_tiers": [
                    "Luxury Estate",
                    "Premium Residential",
                    "Mid-Range Suburban",
                    "Budget Friendly"
                ]
            }

            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "static")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "model_summary.json")

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)

            self.stdout.write(self.style.SUCCESS(f"Successfully exported summary report to '{out_path}'"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Export failed: {str(e)}"))
