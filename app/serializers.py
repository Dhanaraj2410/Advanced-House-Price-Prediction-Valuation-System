import json
from rest_framework import serializers
from app.models import HousePrediction

class HousePredictionSerializer(serializers.ModelSerializer):
    insights = serializers.SerializerMethodField()

    class Meta:
        model = HousePrediction
        fields = [
            'id',
            'overall_qual',
            'gr_liv_area',
            'year_built',
            'garage_cars',
            'total_bsmt_sf',
            'full_bath',
            'neighborhood',
            'predicted_price',
            'price_min',
            'price_max',
            'price_per_sqft',
            'valuation_tier',
            'notes',
            'insights',
            'created_at'
        ]
        read_only_fields = ['id', 'predicted_price', 'price_min', 'price_max', 'price_per_sqft', 'valuation_tier', 'insights', 'created_at']

    def get_insights(self, obj):
        try:
            return json.loads(obj.insights_json)
        except Exception:
            return []

    def validate_overall_qual(self, value):
        if value < 1 or value > 10:
            raise serializers.ValidationError("overall_qual must be between 1 and 10.")
        return value

    def validate_gr_liv_area(self, value):
        if value < 100 or value > 15000:
            raise serializers.ValidationError("gr_liv_area must be between 100 and 15,000 sq ft.")
        return value

