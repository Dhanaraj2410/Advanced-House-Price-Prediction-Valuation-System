from django.db import models

class HousePrediction(models.Model):
    """
    Model for storing house feature inputs, ML predicted prices, and insights.
    """
    overall_qual = models.IntegerField(default=6, help_text="Overall Material & Finish Quality (1-10)")
    gr_liv_area = models.FloatField(default=1500.0, help_text="Above grade living area in sq ft")
    year_built = models.IntegerField(default=2000, help_text="Original construction year")
    garage_cars = models.IntegerField(default=2, help_text="Garage car capacity")
    total_bsmt_sf = models.FloatField(default=900.0, help_text="Total basement square feet")
    full_bath = models.IntegerField(default=2, help_text="Full bathrooms above grade")
    neighborhood = models.CharField(max_length=50, default="NAmes", help_text="Physical location within Ames city bounds")
    
    predicted_price = models.FloatField(help_text="Estimated Sale Price ($)")
    insights_json = models.TextField(blank=True, default="[]", help_text="AI feature influence insights")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction #{self.id}: ${self.predicted_price:,.2f} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
