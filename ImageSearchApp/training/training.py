import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import pickle
from django.conf import settings
from tqdm import tqdm
from typing import List, Dict, Any
import logging
from pathlib import Path


class Training:
    def __init__(self, base_path: str):
        """Initialize the Training class with configuration paths."""
        self.base_path = Path(base_path)
        self.data_paths = [
            self.base_path / 'dataset/ern',
            self.base_path / 'dataset/nec',
            self.base_path / 'dataset/pnd',
            self.base_path / 'dataset/rng'
        ]
        self.pickle_path = self.base_path / 'imageFeatures'
        self.log_path = self.base_path / 'log'
        
        # Set up logging
        logging.basicConfig(
            filename=self.log_path / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize model
        self.model = self._build_model()
    
    def check_dataset_exists(self):
        """Check if datasets exist and have images"""
        dataset_status = {}
        for data_path in self.data_paths:
            dataset_name = os.path.basename(data_path)
            files = self.get_file_list(data_path)
            image_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            dataset_status[dataset_name] = {
                'exists': os.path.exists(data_path),
                'image_count': image_count
            }
        return dataset_status
        
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

    @staticmethod
    def extract_features(img_path: str, model: tf.keras.Model) -> np.ndarray:
        """Extract and normalize features from an image using the model."""
        try:
            input_shape = (224, 224)
            img = image.load_img(img_path, target_size=input_shape)
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            features = model.predict(preprocessed_img, verbose=0)
            flattened_features = features.flatten()
            return flattened_features / norm(flattened_features)
        except Exception as e:
            logging.error(f"Error extracting features from {img_path}: {str(e)}")
            raise

    def get_file_list(self, data_path: Path) -> List[str]:
        """Get a sorted list of image files from the given directory."""
        file_list = []
        try:
            for file_path in data_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    file_list.append(str(file_path))
            
            logging.info(f"Found {len(file_list)} files in {data_path}")
            return sorted(file_list)
        except Exception as e:
            logging.error(f"Error getting file list from {data_path}: {str(e)}")
            raise

    def save_features(self, features: List[np.ndarray], filenames: List[str], dataset_name: str) -> None:
        """Save extracted features and filenames to pickle files."""
        try:
            self.pickle_path.mkdir(parents=True, exist_ok=True)
            
            features_file = self.pickle_path / f"{dataset_name}features-resnet.pickle"
            filenames_file = self.pickle_path / f"{dataset_name}filenames.pickle"
            
            with open(features_file, 'wb') as f:
                pickle.dump(features, f)
            with open(filenames_file, 'wb') as f:
                pickle.dump(filenames, f)
                
            logging.info(f"Saved features and filenames for {dataset_name}")
        except Exception as e:
            logging.error(f"Error saving features for {dataset_name}: {str(e)}")
            raise

    def train(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process all images and extract features using the ResNet50 model."""
        try:
            logging.info("Starting feature extraction process")
            
            for data_path in self.data_paths:
                dataset_name = data_path.name
                logging.info(f"Processing dataset: {dataset_name}")
                
                filenames = self.get_file_list(data_path)
                if not filenames:
                    logging.warning(f"No files found in {data_path}")
                    continue
                
                feature_list = []
                for filename in tqdm(filenames, desc=f"Processing {dataset_name}"):
                    feature_list.append(self.extract_features(filename, self.model))
                
                self.save_features(feature_list, filenames, dataset_name)
            
            logging.info("Feature extraction completed successfully")
            return {
                'response': 'Model trained successfully',
                'status': 200
            }

        except Exception as e:
            error_msg = f"Exception during training: {str(e)}"
            logging.error(error_msg)
            return {
                'response': error_msg,
                'status': 500
            }