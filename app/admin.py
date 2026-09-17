from django.contrib import admin
from app.models import HousePrediction

@admin.register(HousePrediction)
class HousePredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'overall_qual', 'gr_liv_area', 'year_built', 'predicted_price', 'created_at')
    list_filter = ('overall_qual', 'year_built', 'created_at')
    search_fields = ('neighborhood', 'predicted_price')
