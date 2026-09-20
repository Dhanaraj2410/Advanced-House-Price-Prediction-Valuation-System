import json
import io
import csv
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from app.models import HousePrediction
from app.forms import HousePredictionForm
from app.serializers import HousePredictionSerializer
from ml.predict import predict_price

def home_view(request):
    """Render home landing page."""
    recent_predictions = HousePrediction.objects.all()[:5]
    total_predictions = HousePrediction.objects.count()
    return render(request, 'home.html', {
        'recent_predictions': recent_predictions,
        'total_predictions': total_predictions
    })

def predict_view(request):
    """Render and process the house price prediction web form."""
    if request.method == 'POST':
        form = HousePredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            feature_dict = {
                'OverallQual': data['overall_qual'],
                'GrLivArea': data['gr_liv_area'],
                'YearBuilt': data['year_built'],
                'GarageCars': data['garage_cars'],
                'TotalBsmtSF': data['total_bsmt_sf'],
                'FullBath': data['full_bath'],
                'Neighborhood': data['neighborhood']
            }
            
            # Predict price using trained ML model
            result = predict_price(feature_dict)
            
            # Save record to database
            prediction_record = HousePrediction.objects.create(
                overall_qual=data['overall_qual'],
                gr_liv_area=data['gr_liv_area'],
                year_built=data['year_built'],
                garage_cars=data['garage_cars'],
                total_bsmt_sf=data['total_bsmt_sf'],
                full_bath=data['full_bath'],
                neighborhood=data['neighborhood'],
                predicted_price=result['predicted_price'],
                price_min=result.get('price_min'),
                price_max=result.get('price_max'),
                price_per_sqft=result.get('price_per_sqft'),
                valuation_tier=result.get('valuation_tier', 'Mid-Range Suburban'),
                insights_json=json.dumps(result['insights'])
            )
            
            return render(request, 'predict.html', {
                'form': form,
                'result': result,
                'prediction_record': prediction_record
            })
    else:
        form = HousePredictionForm()
        
    return render(request, 'predict.html', {'form': form})

def history_view(request):
    """Render prediction history table."""
    predictions = HousePrediction.objects.all()
    return render(request, 'history.html', {'predictions': predictions})

def dashboard_view(request):
    """Render analytics dashboard with charts."""
    predictions = HousePrediction.objects.all()
    total = predictions.count()
    avg_price = sum(p.predicted_price for p in predictions) / total if total > 0 else 0
    
    return render(request, 'dashboard.html', {
        'total': total,
        'avg_price': round(avg_price, 2),
        'predictions': predictions[:10]
    })

def delete_history_view(request, pk):
    """Delete a single prediction history record."""
    record = get_object_or_404(HousePrediction, pk=pk)
    record.delete()
    messages.success(request, f"Prediction #{pk} deleted successfully.")
    return redirect('history')

# REST API Endpoints
class PredictAPIView(APIView):
    """
    POST /api/predict/
    Accepts JSON body with house features and returns predicted price & insights.
    """
    def post(self, request):
        data = request.data
        try:
            feature_dict = {
                'OverallQual': data.get('overall_qual', 6),
                'GrLivArea': data.get('gr_liv_area', 1500),
                'YearBuilt': data.get('year_built', 2000),
                'GarageCars': data.get('garage_cars', 2),
                'TotalBsmtSF': data.get('total_bsmt_sf', 900),
                'FullBath': data.get('full_bath', 2),
                'Neighborhood': data.get('neighborhood', 'NAmes')
            }
            
            res = predict_price(feature_dict)
            
            record = HousePrediction.objects.create(
                overall_qual=feature_dict['OverallQual'],
                gr_liv_area=feature_dict['GrLivArea'],
                year_built=feature_dict['YearBuilt'],
                garage_cars=feature_dict['GarageCars'],
                total_bsmt_sf=feature_dict['TotalBsmtSF'],
                full_bath=feature_dict['FullBath'],
                neighborhood=feature_dict['Neighborhood'],
                predicted_price=res['predicted_price'],
                price_min=res.get('price_min'),
                price_max=res.get('price_max'),
                price_per_sqft=res.get('price_per_sqft'),
                valuation_tier=res.get('valuation_tier', 'Mid-Range Suburban'),
                notes=data.get('notes', ''),
                insights_json=json.dumps(res['insights'])
            )
            
            serializer = HousePredictionSerializer(record)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class PredictionListAPIView(APIView):
    """
    GET /api/predictions/
    List all recorded predictions.
    """
    def get(self, request):
        predictions = HousePrediction.objects.all()
        serializer = HousePredictionSerializer(predictions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class PredictionDetailAPIView(APIView):
    """
    GET /api/predictions/<id>/
    DELETE /api/predictions/<id>/
    """
    def get(self, request, pk):
        record = get_object_or_404(HousePrediction, pk=pk)
        serializer = HousePredictionSerializer(record)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        record = get_object_or_404(HousePrediction, pk=pk)
        record.delete()
        return Response({'message': f'Prediction #{pk} deleted.'}, status=status.HTTP_204_NO_CONTENT)

def bulk_predict_view(request):
    """
    POST /predict/bulk/
    Accepts CSV upload for batch property price predictions.
    """
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)
            results = []
            records_created = []

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                res = predict_price(row_dict)

                overall_qual = int(row_dict.get('OverallQual', 6))
                gr_liv_area = float(row_dict.get('GrLivArea', 1500))
                year_built = int(row_dict.get('YearBuilt', 2000))
                garage_cars = int(row_dict.get('GarageCars', 2))
                total_bsmt = float(row_dict.get('TotalBsmtSF', 900))
                full_bath = int(row_dict.get('FullBath', 2))
                neighborhood = str(row_dict.get('Neighborhood', 'NAmes'))

                record = HousePrediction.objects.create(
                    overall_qual=overall_qual,
                    gr_liv_area=gr_liv_area,
                    year_built=year_built,
                    garage_cars=garage_cars,
                    total_bsmt_sf=total_bsmt,
                    full_bath=full_bath,
                    neighborhood=neighborhood,
                    predicted_price=res['predicted_price'],
                    price_min=res.get('price_min'),
                    price_max=res.get('price_max'),
                    price_per_sqft=res.get('price_per_sqft'),
                    valuation_tier=res.get('valuation_tier', 'Mid-Range Suburban'),
                    insights_json=json.dumps(res['insights'])
                )
                records_created.append(record)

            messages.success(request, f"Successfully processed {len(records_created)} batch property predictions!")
            return render(request, 'predict.html', {
                'form': HousePredictionForm(),
                'bulk_records': records_created
            })
        except Exception as e:
            messages.error(request, f"Error processing bulk CSV file: {str(e)}")
            return redirect('predict')
            
    return redirect('predict')

class BulkPredictAPIView(APIView):
    """
    POST /api/predict/bulk/
    Accepts list of property feature objects or CSV file for bulk valuation.
    """
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request):
        try:
            items = []
            if 'csv_file' in request.FILES:
                df = pd.read_csv(request.FILES['csv_file'])
                items = df.to_dict(orient='records')
            elif isinstance(request.data, list):
                items = request.data
            elif 'properties' in request.data:
                items = request.data['properties']
            else:
                return Response({'error': 'Provide a JSON list or upload a csv_file.'}, status=status.HTTP_400_BAD_REQUEST)

            created_records = []
            for item in items:
                feature_dict = {
                    'OverallQual': item.get('overall_qual', item.get('OverallQual', 6)),
                    'GrLivArea': item.get('gr_liv_area', item.get('GrLivArea', 1500)),
                    'YearBuilt': item.get('year_built', item.get('YearBuilt', 2000)),
                    'GarageCars': item.get('garage_cars', item.get('GarageCars', 2)),
                    'TotalBsmtSF': item.get('total_bsmt_sf', item.get('TotalBsmtSF', 900)),
                    'FullBath': item.get('full_bath', item.get('FullBath', 2)),
                    'Neighborhood': item.get('neighborhood', item.get('Neighborhood', 'NAmes'))
                }
                res = predict_price(feature_dict)

                record = HousePrediction.objects.create(
                    overall_qual=feature_dict['OverallQual'],
                    gr_liv_area=feature_dict['GrLivArea'],
                    year_built=feature_dict['YearBuilt'],
                    garage_cars=feature_dict['GarageCars'],
                    total_bsmt_sf=feature_dict['TotalBsmtSF'],
                    full_bath=feature_dict['FullBath'],
                    neighborhood=feature_dict['Neighborhood'],
                    predicted_price=res['predicted_price'],
                    price_min=res.get('price_min'),
                    price_max=res.get('price_max'),
                    price_per_sqft=res.get('price_per_sqft'),
                    valuation_tier=res.get('valuation_tier', 'Mid-Range Suburban'),
                    insights_json=json.dumps(res['insights'])
                )
                created_records.append(record)

            serializer = HousePredictionSerializer(created_records, many=True)
            return Response({'count': len(created_records), 'results': serializer.data}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

