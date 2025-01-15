from django.urls import path
from .views import TrainingView, PredictionView, ModelStatusView, PredictionTemplateView

urlpatterns = [
    path('train/', TrainingView.as_view(), name='train'),
    path('predict/', PredictionView.as_view(), name='predict'),
    path('status/', ModelStatusView.as_view(), name='status'),
    path('', PredictionTemplateView.as_view(), name='predict_form'),
]