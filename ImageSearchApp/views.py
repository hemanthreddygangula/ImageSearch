from django.shortcuts import render

# Create your views here.
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .serializers import ProcessedImageSerializer, PredictionRequestSerializer
from .training.training import Training  # Use this direct import instead
from .prediction.prediction import Prediction  # Use this direct import instead
import logging
import json

logger = logging.getLogger(__name__)

class TrainingView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        try:
            trainer = Training(base_path=settings.BASE_DIR)
            dataset_status = trainer.check_dataset_exists()
            
            status_dict = {}
            for dataset, info in dataset_status.items():
                features_file = os.path.join(settings.BASE_DIR, 'imageFeatures', f"{dataset}features-resnet.pickle")
                filenames_file = os.path.join(settings.BASE_DIR, 'imageFeatures', f"{dataset}filenames.pickle")
                
                status_dict[dataset] = {
                    'features_exist': os.path.exists(features_file),
                    'filenames_exist': os.path.exists(filenames_file),
                    'status': 'Trained' if os.path.exists(features_file) and os.path.exists(filenames_file) else 'Not Trained',
                    'dataset_exists': info['exists'],
                    'image_count': info['image_count']
                }

            return Response({
                'training_status': status_dict
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request, *args, **kwargs):
        try:
            # Initialize training
            trainer = Training(base_path=settings.BASE_DIR)
            
            # Start training process
            result = trainer.train(request.data)
            
            if result['status'] == 200:
                return Response({
                    'message': result['response']
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': result['response']
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PredictionTemplateView(APIView):
    def get(self, request):
        # Get available datasets
        # datasets = ['BNG', 'ERN', 'RNG', 'NEC']
        datasets = settings.DATASETS
        available_datasets = []
        
        for dataset in datasets:
            features_file = os.path.join(settings.BASE_DIR, 'imageFeatures', f"{dataset}features-resnet.pickle")
            if os.path.exists(features_file):
                available_datasets.append(dataset)
        
        return render(request, 'predict.html', {
            'available_datasets': available_datasets
        })

class PredictionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get and validate the uploaded image
            image = request.FILES.get('image')
            if not image:
                return Response({'error': 'No image provided'}, 
                              status=status.HTTP_400_BAD_REQUEST)

            # Parse datasets from JSON string
            datasets = json.loads(request.POST.get('datasets', '[]'))
            num_results = int(request.POST.get('num_results', 5))

            # Save uploaded image temporarily
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Initialize predictor
            predictor = Prediction(base_path=settings.BASE_DIR)
            
            # Prepare prediction request
            prediction_request = {
                'image_path': temp_path,
                'num_results': num_results,
                'datasets': datasets
            }

            # Get predictions
            result = predictor.predict(prediction_request)

            # # Clean up temporary file
            # if os.path.exists(temp_path):
            #     os.remove(temp_path)

            if result['status'] == 200:
                return Response(result['response'])
            else:
                return Response({'error': result['response']}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return Response({'error': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelStatusView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            base_path = settings.BASE_DIR
            pickle_path = os.path.join(base_path, 'imageFeatures')
            
            # Check if model files exist
            # datasets = ['BNG', 'ERN', 'RNG', 'NEC']
            datasets = settings.DATASETS
            status_dict = {}
            
            for dataset in datasets:
                features_file = os.path.join(pickle_path, f"{dataset}features-resnet.pickle")
                filenames_file = os.path.join(pickle_path, f"{dataset}filenames.pickle")
                
                status_dict[dataset] = {
                    'features_exist': os.path.exists(features_file),
                    'filenames_exist': os.path.exists(filenames_file)
                }

            return Response({
                'model_status': status_dict
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)