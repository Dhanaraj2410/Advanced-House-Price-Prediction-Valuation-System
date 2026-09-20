from django.urls import path
from app import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('predict/bulk/', views.bulk_predict_view, name='bulk_predict'),
    path('history/', views.history_view, name='history'),
    path('history/export/', views.export_history_csv_view, name='export_history_csv'),
    path('history/delete/<int:pk>/', views.delete_history_view, name='delete_history'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # REST API Routes
    path('api/predict/', views.PredictAPIView.as_view(), name='api_predict'),
    path('api/predict/bulk/', views.BulkPredictAPIView.as_view(), name='api_predict_bulk'),
    path('api/predictions/', views.PredictionListAPIView.as_view(), name='api_predictions_list'),
    path('api/predictions/<int:pk>/', views.PredictionDetailAPIView.as_view(), name='api_prediction_detail'),
]
