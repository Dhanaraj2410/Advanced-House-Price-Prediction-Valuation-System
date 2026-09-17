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
            'insights',
            'created_at'
        ]
        read_only_fields = ['id', 'predicted_price', 'insights', 'created_at']

    def get_insights(self, obj):
        try:
            return json.loads(obj.insights_json)
        except Exception:
            return []
