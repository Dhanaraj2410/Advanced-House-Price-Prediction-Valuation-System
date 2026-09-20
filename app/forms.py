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
            'gr_liv_area': forms.NumberInput(attrs={'class': 'form-control', 'min': 200, 'max': 10000, 'value': 1500}),
            'year_built': forms.NumberInput(attrs={'class': 'form-control', 'min': 1800, 'max': 2030, 'value': 2000}),
            'garage_cars': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 10, 'value': 2}),
            'total_bsmt_sf': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 10000, 'value': 900}),
            'full_bath': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 10, 'value': 2}),
            'neighborhood': forms.Select(choices=NEIGHBORHOOD_CHOICES, attrs={'class': 'form-select'}),
        }

    def clean_overall_qual(self):
        val = self.cleaned_data.get('overall_qual')
        if val is not None and (val < 1 or val > 10):
            raise forms.ValidationError("Overall quality rating must be between 1 and 10.")
        return val

    def clean_gr_liv_area(self):
        val = self.cleaned_data.get('gr_liv_area')
        if val is not None and (val < 100 or val > 15000):
            raise forms.ValidationError("Living area must be between 100 and 15,000 sq ft.")
        return val

    def clean_year_built(self):
        val = self.cleaned_data.get('year_built')
        if val is not None and (val < 1800 or val > 2030):
            raise forms.ValidationError("Construction year must be between 1800 and 2030.")
        return val

