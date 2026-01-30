"""
Module 2: Image Preprocessing

This module performs preprocessing operations on raw images from the TLFS23 dataset
to prepare them for hand landmark extraction using MediaPipe. It handles image loading,
format conversion, normalization, and batch processing.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from tqdm import tqdm
import pandas as pd
from PIL import Image
import pickle


class ImagePreprocessor:
    """
    Main image preprocessing class for TLFS23 dataset.
    Prepares images for MediaPipe hand landmark extraction.
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 ensure_rgb: bool = True,
                 normalize: bool = False,
                 quality_check: bool = True):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Optional (width, height) to resize images. None keeps original size.
            ensure_rgb: Whether to ensure all images are in RGB format (default: True)
            normalize: Whether to normalize pixel values to [0, 1] range (default: False)
            quality_check: Whether to perform quality checks on images (default: True)
        """
        self.target_size = target_size
        self.ensure_rgb = ensure_rgb
        self.normalize = normalize
        self.quality_check = quality_check
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'failed_files': [],
            'processing_times': []
        }
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image as numpy array or None if processing failed
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Quality check
            if self.quality_check:
                if not self._check_image_quality(image):
                    raise ValueError(f"Image failed quality check: {image_path}")
            
            # Convert BGR to RGB (OpenCV reads in BGR format)
            if self.ensure_rgb:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    # Grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize if target size is specified
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values if requested
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            else:
                # Ensure uint8 format
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _check_image_quality(self, image: np.ndarray) -> bool:
        """
        Perform quality checks on the image.
        
        Args:
            image: Image as numpy array
        
        Returns:
            True if image passes quality checks, False otherwise
        """
        # Check if image is empty
        if image is None or image.size == 0:
            return False
        
        # Check if image has valid dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False
        
        # Check if image is too small
        if image.shape[0] < 50 or image.shape[1] < 50:
            return False
        
        # Check if image is completely black or white
        if np.all(image == 0) or np.all(image == 255):
            return False
        
        return True
    
    def preprocess_batch(self, image_paths: List[str], 
                        batch_size: int = 32,
                        show_progress: bool = True) -> Tuple[List[np.ndarray], List[str]]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (preprocessed_images, successful_paths)
        """
        preprocessed_images = []
        successful_paths = []
        
        iterator = tqdm(image_paths, desc="Preprocessing images") if show_progress else image_paths
        
        for img_path in iterator:
            self.stats['total_processed'] += 1
            
            preprocessed = self.preprocess_image(img_path)
            
            if preprocessed is not None:
                preprocessed_images.append(preprocessed)
                successful_paths.append(img_path)
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_files'].append(img_path)
        
        return preprocessed_images, successful_paths
    
    def preprocess_from_dataframe(self, df: pd.DataFrame, 
                                  image_path_column: str = 'image_path',
                                  label_column: str = 'label',
                                  batch_size: int = 32,
                                  max_samples: Optional[int] = None) -> Dict:
        """
        Preprocess images from a pandas DataFrame (from Module 1).
        
        Args:
            df: DataFrame containing image paths and labels
            image_path_column: Name of column containing image paths
            label_column: Name of column containing labels
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
        
        Returns:
            Dictionary containing preprocessed data
        """
        # Limit samples if specified
        if max_samples is not None:
            df = df.head(max_samples)
        
        print(f"Preprocessing {len(df)} images from DataFrame...")
        
        # Get image paths and labels
        image_paths = df[image_path_column].tolist()
        labels = df[label_column].tolist()
        
        # Preprocess images
        preprocessed_images, successful_paths = self.preprocess_batch(
            image_paths, batch_size=batch_size, show_progress=True
        )
        
        # Filter labels to match successful preprocessing
        successful_labels = []
        for path in successful_paths:
            idx = image_paths.index(path)
            successful_labels.append(labels[idx])
        
        # Convert to numpy arrays
        X = np.array(preprocessed_images)
        y = np.array(successful_labels)
        
        print(f"\nPreprocessing complete!")
        print(f"Successful: {len(preprocessed_images)}/{len(image_paths)}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Output shape: X={X.shape}, y={y.shape}")
        
        return {
            'X': X,
            'y': y,
            'successful_paths': successful_paths,
            'stats': self.stats.copy()
        }
    
    def save_preprocessed_data(self, data: Dict, output_path: str):
        """
        Save preprocessed data to disk using pickle.
        
        Args:
            data: Dictionary containing preprocessed data
            output_path: Path to save the pickle file
        """
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Preprocessed data saved to: {output_path}")
    
    def load_preprocessed_data(self, input_path: str) -> Dict:
        """
        Load preprocessed data from disk.
        
        Args:
            input_path: Path to the pickle file
        
        Returns:
            Dictionary containing preprocessed data
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Preprocessed data loaded from: {input_path}")
        return data
    
    def get_statistics(self) -> Dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary containing statistics
        """
        success_rate = (self.stats['successful'] / self.stats['total_processed'] * 100 
                       if self.stats['total_processed'] > 0 else 0)
        
        return {
            'total_processed': self.stats['total_processed'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': success_rate,
            'failed_files_count': len(self.stats['failed_files'])
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'failed_files': [],
            'processing_times': []
        }


class DatasetPreprocessor:
    """
    High-level preprocessor for the entire TLFS23 dataset.
    Integrates with Module 1 for dataset loading and preprocessing.
    """
    
    def __init__(self, dataset_loader, 
                 target_size: Optional[Tuple[int, int]] = None,
                 train_split: float = 0.8,
                 random_state: int = 42):
        """
        Initialize dataset preprocessor.
        
        Args:
            dataset_loader: TLFS23DatasetLoader instance from Module 1
            target_size: Optional target image size
            train_split: Fraction of data for training (default: 0.8)
            random_state: Random seed for reproducibility
        """
        self.dataset_loader = dataset_loader
        self.target_size = target_size
        self.train_split = train_split
        self.random_state = random_state
        
        self.preprocessor = ImagePreprocessor(
            target_size=target_size,
            ensure_rgb=True,
            normalize=False,
            quality_check=True
        )
    
    def preprocess_dataset(self, max_samples_per_class: Optional[int] = None,
                          output_dir: Optional[str] = None) -> Dict:
        """
        Preprocess the entire dataset with train/test split.
        
        Args:
            max_samples_per_class: Maximum samples per class (None for all)
            output_dir: Directory to save preprocessed data
        
        Returns:
            Dictionary containing train and test data
        """
        print("="*70)
        print("DATASET PREPROCESSING")
        print("="*70)
        
        # Create DataFrame from Module 1
        df = self.dataset_loader.create_dataframe()
        
        # Sample if requested
        if max_samples_per_class is not None:
            print(f"\nSampling {max_samples_per_class} images per class...")
            df = df.groupby('label').head(max_samples_per_class).reset_index(drop=True)
        
        print(f"\nTotal images to process: {len(df)}")
        
        # Shuffle and split
        from sklearn.model_selection import train_test_split
        
        # Calculate required minimum samples per class for stratified split
        num_classes = df['label'].nunique()
        test_size = 1 - self.train_split
        min_test_samples = int(np.ceil(num_classes / test_size))
        
        # Check if we have enough samples for stratified split
        if len(df) < min_test_samples:
            print(f"\nWarning: Not enough samples for stratified split with {num_classes} classes")
            print(f"  Need at least {min_test_samples} samples, have {len(df)}")
            print(f"  Using simple random split instead")
            stratify_param = None
        else:
            stratify_param = df['label']
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Train set: {len(train_df)} images")
        print(f"Test set: {len(test_df)} images")
        
        # Preprocess training data
        print("\n" + "="*70)
        print("PREPROCESSING TRAINING DATA")
        print("="*70)
        train_data = self.preprocessor.preprocess_from_dataframe(train_df)
        
        # Reset stats for test data
        self.preprocessor.reset_statistics()
        
        # Preprocess testing data
        print("\n" + "="*70)
        print("PREPROCESSING TESTING DATA")
        print("="*70)
        test_data = self.preprocessor.preprocess_from_dataframe(test_df)
        
        # Combine results
        result = {
            'X_train': train_data['X'],
            'y_train': train_data['y'],
            'X_test': test_data['X'],
            'y_test': test_data['y'],
            'train_paths': train_data['successful_paths'],
            'test_paths': test_data['successful_paths'],
            'train_stats': train_data['stats'],
            'test_stats': test_data['stats'],
            'preprocessing_config': {
                'target_size': self.target_size,
                'ensure_rgb': True,
                'normalize': False,
                'train_split': self.train_split,
                'random_state': self.random_state
            }
        }
        
        # Save if output directory specified
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            self.preprocessor.save_preprocessed_data(
                result, 
                str(output_path / 'preprocessed_data.pkl')
            )
            
            # Save metadata
            import json
            metadata = {
                'train_samples': len(train_data['X']),
                'test_samples': len(test_data['X']),
                'image_shape': list(train_data['X'][0].shape),
                'num_classes': len(np.unique(train_data['y'])),
                'train_stats': train_data['stats'],
                'test_stats': test_data['stats'],
                'config': result['preprocessing_config']
            }
            
            with open(output_path / 'preprocessing_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nPreprocessed data saved to: {output_dir}")
        
        return result


def main():
    """
    Main function to demonstrate image preprocessing.
    """
    # Import Module 1
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from mod1 import TLFS23DatasetLoader
    
    # Set paths
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    output_dir = Path(__file__).parent / "output"
    
    # Load dataset using Module 1
    print("Loading dataset using Module 1...")
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Initialize dataset preprocessor
    preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None,  # Keep original size (640x480)
        train_split=0.8,
        random_state=42
    )
    
    # Preprocess dataset (sample for testing)
    print("\nPreprocessing dataset (10 samples per class for testing)...")
    result = preprocessor.preprocess_dataset(
        max_samples_per_class=10,
        output_dir=str(output_dir)
    )
    
    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Training set shape: X={result['X_train'].shape}, y={result['y_train'].shape}")
    print(f"Testing set shape: X={result['X_test'].shape}, y={result['y_test'].shape}")
    print(f"Image dtype: {result['X_train'].dtype}")
    print(f"Label dtype: {result['y_train'].dtype}")
    print(f"Number of classes: {len(np.unique(result['y_train']))}")
    
    print("\nModule 2 execution completed successfully!")


if __name__ == "__main__":
    main()
