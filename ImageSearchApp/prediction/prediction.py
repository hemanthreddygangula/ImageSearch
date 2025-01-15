import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sklearn.neighbors import NearestNeighbors

class Prediction:
    def __init__(self, base_path: str):
        """Initialize the Prediction class with necessary paths and models."""
        self.base_path = Path(base_path)
        self.pickle_path = self.base_path / 'imageFeatures'
        self.log_path = self.base_path / 'log'
        
        # Set up logging
        logging.basicConfig(
            filename=self.log_path / 'prediction.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize model and feature data
        self.model = self._build_model()
        self.features_dict = {}
        self.filenames_dict = {}
        self.neighbors_dict = {}
        
        # Load all feature files
        self._load_feature_files()
        
    def _build_model(self) -> tf.keras.Model:
        """Build and return the ResNet50 model with GlobalMaxPooling."""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        return tf.keras.Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])
    
    def _load_feature_files(self) -> None:
        """Load all feature and filename pickle files."""
        try:
            datasets = ['ern', 'nec', 'pnd', 'rng']
            
            for dataset in datasets:
                features_file = self.pickle_path / f"{dataset}features-resnet.pickle"
                filenames_file = self.pickle_path / f"{dataset}filenames.pickle"
                
                if features_file.exists() and filenames_file.exists():
                    with open(features_file, 'rb') as f:
                        self.features_dict[dataset] = pickle.load(f)
                    with open(filenames_file, 'rb') as f:
                        self.filenames_dict[dataset] = pickle.load(f)
                        
                    # Initialize NearestNeighbors for this dataset
                    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
                    neighbors.fit(self.features_dict[dataset])
                    self.neighbors_dict[dataset] = neighbors
                    
                    logging.info(f"Loaded features for dataset {dataset}")
                else:
                    logging.warning(f"Feature files for dataset {dataset} not found")
                    
        except Exception as e:
            logging.error(f"Error loading feature files: {str(e)}")
            raise

    def extract_features(self, img_path: str) -> np.ndarray:
        """Extract features from a single image."""
        try:
            input_shape = (224, 224)
            img = image.load_img(img_path, target_size=input_shape)
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            features = self.model.predict(preprocessed_img, verbose=0)
            flattened_features = features.flatten()
            return flattened_features / norm(flattened_features)
        except Exception as e:
            logging.error(f"Error extracting features from {img_path}: {str(e)}")
            raise

    def find_similar_images(self, 
                          query_features: np.ndarray, 
                          dataset: str, 
                          num_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar images in the specified dataset."""
        from django.conf import settings
        dataset_url = settings.DATASET_URL
        try:
            data_path = os.path.join(dataset_url, dataset)
            distances, indices = self.neighbors_dict[dataset].kneighbors([query_features])
            similar_images = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.filenames_dict[dataset]):  # Safety check
                    similar_images.append({
                        'filename': self.filenames_dict[dataset][idx].split(os.sep)[-1],
                        'url': os.path.join(data_path, self.filenames_dict[dataset][idx].split(os.sep)[-1]),
                        'distance': float(distance),
                        'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity score
                    })
            
            return similar_images[:num_results]
        except Exception as e:
            logging.error(f"Error finding similar images in dataset {dataset}: {str(e)}")
            raise

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction method to find similar images across all datasets.
        
        Expected request format:
        {
            'image_path': str,
            'num_results': int (optional, default=5),
            'datasets': List[str] (optional, default=all datasets)
        }
        """
        try:
            image_path = request.get('image_path')
            if not image_path:
                raise ValueError("Image path not provided in request")
                
            num_results = request.get('num_results', 5)
            datasets = request.get('datasets', list(self.features_dict.keys()))
            
            # Extract features from query image
            query_features = self.extract_features(image_path)
            
            # Get results from each dataset
            results = {}
            for dataset in datasets:
                if dataset in self.features_dict:
                    similar_images = self.find_similar_images(
                        query_features, 
                        dataset, 
                        num_results
                    )
                    results[dataset] = similar_images
                else:
                    logging.warning(f"Dataset {dataset} not found in loaded features")
            
            return {
                'response': {
                    'query_image': image_path,
                    'results': results
                },
                'status': 200
            }

        except Exception as e:
            error_msg = f"Exception during prediction: {str(e)}"
            logging.error(error_msg)
            return {
                'response': error_msg,
                'status': 500
            }
        
        