from django.urls import path
from rest_framework.schemas import get_schema_view
from app import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('predict/bulk/', views.bulk_predict_view, name='bulk_predict'),
    path('history/', views.history_view, name='history'),
    path('history/export/', views.export_history_csv_view, name='export_history_csv'),
    path('history/delete/<int:pk>/', views.delete_history_view, name='delete_history'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # REST API & OpenAPI Schema Routes
    path('api/schema/', get_schema_view(
        title="Advanced House Price Prediction & Valuation API",
        description="RESTful API endpoints for ML property price inference, batch valuation, and prediction history.",
        version="1.0.0"
    ), name='api_schema'),
    path('api/predict/', views.PredictAPIView.as_view(), name='api_predict'),
    path('api/predict/bulk/', views.BulkPredictAPIView.as_view(), name='api_predict_bulk'),
    path('api/predictions/', views.PredictionListAPIView.as_view(), name='api_predictions_list'),
    path('api/predictions/<int:pk>/', views.PredictionDetailAPIView.as_view(), name='api_prediction_detail'),
    path('api/health/', views.HealthCheckAPIView.as_view(), name='api_health'),
    path('api/feature-importance/', views.FeatureImportanceAPIView.as_view(), name='api_feature_importance'),
]


