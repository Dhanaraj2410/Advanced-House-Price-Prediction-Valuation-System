"""
Django Management Command to retrain ML models and generate evaluation metrics.
Usage: python manage.py retrain_model [--benchmark]
"""

from django.core.management.base import BaseCommand
from ml.train import run_model_training
from ml.benchmark import run_benchmark

class Command(BaseCommand):
    help = "Triggers model training pipeline, fits regressors, and updates model artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Run full model benchmarking suite after training.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting ML model retraining pipeline..."))
        try:
            best_pipeline = run_model_training()
            self.stdout.write(self.style.SUCCESS("Model pipeline trained and saved successfully!"))
            
            if options['benchmark']:
                self.stdout.write(self.style.NOTICE("Running model benchmark comparison suite..."))
                run_benchmark()
                self.stdout.write(self.style.SUCCESS("Benchmark suite complete! JSON report exported."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Model retraining failed: {str(e)}"))
