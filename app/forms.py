from django import forms
from app.models import HousePrediction

NEIGHBORHOOD_CHOICES = [
    ("NAmes", "North Ames"),
    ("CollgCr", "College Creek"),
    ("OldTown", "Old Town"),
    ("Edwards", "Edwards"),
    ("Somerst", "Somerset"),
    ("NridgHt", "Northridge Heights"),
    ("Gilbert", "Gilbert"),
    ("Sawyer", "Sawyer"),
    ("NWAmes", "Northwest Ames"),
    ("SawyerW", "Sawyer West"),
    ("Mitchel", "Mitchell"),
    ("Crawfor", "Crawford"),
    ("IDOTRR", "Iowa DOT and Rail Road"),
    ("Timber", "Timberland"),
    ("NoRidge", "Northridge"),
    ("StoneBr", "Stone Brook"),
    ("SWISU", "South & West of Iowa State University"),
    ("ClearCr", "Clear Creek"),
    ("MeadowV", "Meadow Village"),
    ("BrDale", "Briardale"),
    ("Blmngtn", "Bloomington Heights"),
    ("Veenker", "Veenker"),
    ("NPkVill", "Northpark Villa"),
    ("Blueste", "Bluestem")
]

class HousePredictionForm(forms.ModelForm):
    class Meta:
        model = HousePrediction
        fields = [
            'overall_qual',
            'gr_liv_area',
            'year_built',
            'garage_cars',
            'total_bsmt_sf',
            'full_bath',
            'neighborhood'
        ]
        widgets = {
            'overall_qual': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 10, 'value': 6}),
            'gr_liv_area': forms.NumberInput(attrs={'class': 'form-control', 'min': 200, 'max': 6000, 'value': 1500}),
            'year_built': forms.NumberInput(attrs={'class': 'form-control', 'min': 1800, 'max': 2026, 'value': 2000}),
            'garage_cars': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 5, 'value': 2}),
            'total_bsmt_sf': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 4000, 'value': 900}),
            'full_bath': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 5, 'value': 2}),
            'neighborhood': forms.Select(choices=NEIGHBORHOOD_CHOICES, attrs={'class': 'form-select'}),
        }
