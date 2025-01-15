from rest_framework import serializers
from .models import ProcessedImage

class ProcessedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessedImage
        fields = ['id', 'image', 'dataset', 'processed_at']

class PredictionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    num_results = serializers.IntegerField(required=False, default=5)
    datasets = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )