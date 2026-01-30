"""
Module 3: Hand Landmark Extraction using MediaPipe

This module extracts 21 hand landmarks from preprocessed images using MediaPipe Hands.
Each landmark has (x, y, z) coordinates, resulting in 63-dimensional feature vectors.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from tqdm import tqdm
import pickle


class HandLandmarkExtractor:
    """
    Extract hand landmarks from images using MediaPipe Hands.
    
    Extracts 21 anatomical landmarks (63 features: 21 × 3 coordinates).
    """
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        enable_retry: bool = True,
        enhance_image: bool = True
    ):
        """
        Initialize MediaPipe Hands landmark extractor.
        
        Args:
            static_image_mode: If True, treats each image independently
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_retry = enable_retry
        self.enhance_image = enhance_image
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 0.0
        }
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Landmark feature vector (63,) or None if no hand detected
            Features: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
        """
        try:
            # Ensure image is RGB
            if len(image.shape) != 3 or image.shape[2] != 3:
                return None
            
            # Enhance image if enabled
            if self.enhance_image:
                image = self._enhance_image(image)
            
            # Process image with MediaPipe
            results = self.hands.process(image)
            
            # Extract landmarks if hand detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # First hand
                
                # Extract coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Convert to numpy array
                feature_vector = np.array(landmarks, dtype=np.float32)
                
                # Verify shape
                if feature_vector.shape[0] != 63:
                    return None
                
                self.stats['successful'] += 1
                return feature_vector
            else:
                # Retry with enhanced processing if enabled
                if self.enable_retry:
                    return self._retry_extraction(image)
                else:
                    self.stats['failed'] += 1
                    return None
                
        except Exception as e:
            # Retry on error if enabled
            if self.enable_retry:
                try:
                    return self._retry_extraction(image)
                except:
                    pass
            self.stats['failed'] += 1
            return None
        finally:
            self.stats['total_processed'] += 1
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better hand detection.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Slight denoising
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return enhanced
        except:
            return image
    
    def _retry_extraction(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Retry landmark extraction with different confidence levels and enhancements.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            Landmark feature vector or None
        """
        # Try with progressively lower confidence thresholds
        confidence_levels = [0.2, 0.1, 0.05]
        
        for confidence in confidence_levels:
            try:
                # Create temporary extractor with lower confidence
                temp_hands = self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence
                )
                
                # Try with enhanced image
                enhanced_image = self._enhance_image(image)
                results = temp_hands.process(enhanced_image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    feature_vector = np.array(landmarks, dtype=np.float32)
                    if feature_vector.shape[0] == 63:
                        temp_hands.close()
                        self.stats['successful'] += 1
                        return feature_vector
                
                temp_hands.close()
            except:
                continue
        
        # All retries failed
        self.stats['failed'] += 1
        return None
    
    def extract_landmarks_batch(
        self,
        images: List[np.ndarray],
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Extract landmarks from multiple images efficiently.
        
        Args:
            images: List of RGB images
            show_progress: Show progress bar
        
        Returns:
            Tuple of (landmark_features, detection_status)
            - landmark_features: List of 63-dim feature vectors
            - detection_status: List of booleans (True if hand detected)
        """
        landmark_features = []
        detection_status = []
        
        iterator = tqdm(images, desc="Extracting landmarks") if show_progress else images
        
        for image in iterator:
            landmarks = self.extract_landmarks(image)
            
            if landmarks is not None:
                landmark_features.append(landmarks)
                detection_status.append(True)
            else:
                landmark_features.append(None)
                detection_status.append(False)
        
        return landmark_features, detection_status
    
    def extract_from_paths(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[bool], List[str]]:
        """
        Extract landmarks from image file paths.
        
        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar
        
        Returns:
            Tuple of (landmark_features, detection_status, successful_paths)
        """
        landmark_features = []
        detection_status = []
        successful_paths = []
        
        iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
        
        for image_path in iterator:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                landmark_features.append(None)
                detection_status.append(False)
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(image_rgb)
            
            if landmarks is not None:
                landmark_features.append(landmarks)
                detection_status.append(True)
                successful_paths.append(str(image_path))
            else:
                landmark_features.append(None)
                detection_status.append(False)
        
        return landmark_features, detection_status, successful_paths
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        image_col: str = 'image_path',
        label_col: str = 'label',
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract landmarks from DataFrame with image paths and labels.
        
        Args:
            df: DataFrame with image paths and labels
            image_col: Column name for image paths
            label_col: Column name for labels
            max_samples: Maximum samples to process (None for all)
        
        Returns:
            Dictionary with:
                'X': Feature matrix (n_samples, 63)
                'y': Label array (n_samples,)
                'image_paths': Array of successful image paths
        """
        if max_samples is not None:
            df = df.head(max_samples)
        
        print(f"\nExtracting landmarks from {len(df)} images...")
        
        # Extract landmarks
        landmarks, status, paths = self.extract_from_paths(
            df[image_col].tolist(),
            show_progress=True
        )
        
        # Filter successful extractions
        valid_indices = [i for i, s in enumerate(status) if s]
        
        X = np.array([landmarks[i] for i in valid_indices], dtype=np.float32)
        y = df.iloc[valid_indices][label_col].values
        image_paths = np.array([df.iloc[i][image_col] for i in valid_indices])
        
        print(f"\nExtraction complete!")
        print(f"Successful: {len(valid_indices)}/{len(df)}")
        print(f"Failed: {len(df) - len(valid_indices)}")
        print(f"Output shape: X={X.shape}, y={y.shape}")
        
        return {
            'X': X,
            'y': y,
            'image_paths': image_paths
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.stats['total_processed'] > 0:
            self.stats['success_rate'] = (
                self.stats['successful'] / self.stats['total_processed'] * 100
            )
        
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 0.0
        }
    
    def visualize_landmarks(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw hand landmarks on image for visualization.
        
        Args:
            image: RGB image
            landmarks: Optional pre-extracted landmarks (if None, extract from image)
        
        Returns:
            Image with landmarks drawn
        """
        # Create copy to avoid modifying original
        annotated_image = image.copy()
        
        # Process image to get landmarks if not provided
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_image
    
    def close(self):
        """Close MediaPipe Hands instance."""
        if self.hands:
            self.hands.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class DatasetLandmarkExtractor:
    """
    Extract landmarks from entire dataset with integration to Module 2.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        enable_retry: bool = True,
        enhance_image: bool = True
    ):
        """
        Initialize dataset landmark extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.extractor = HandLandmarkExtractor(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_retry=enable_retry,
            enhance_image=enhance_image
        )
    
    def extract_from_preprocessed_data(
        self,
        preprocessed_data: Dict[str, np.ndarray],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract landmarks from preprocessed data (Module 2 output).
        
        Args:
            preprocessed_data: Dictionary with X_train, X_test, y_train, y_test
            output_dir: Directory to save landmark features (None to skip saving)
        
        Returns:
            Dictionary with landmark features for train and test sets
        """
        print("=" * 70)
        print("LANDMARK EXTRACTION FROM PREPROCESSED DATA")
        print("=" * 70)
        
        # Extract from training data
        print("\n" + "=" * 70)
        print("EXTRACTING TRAINING LANDMARKS")
        print("=" * 70)
        
        X_train_images = preprocessed_data['X_train']
        y_train = preprocessed_data['y_train']
        
        print(f"Processing {len(X_train_images)} training images...")
        
        train_landmarks, train_status = self.extractor.extract_landmarks_batch(
            X_train_images,
            show_progress=True
        )
        
        # Filter successful extractions
        train_valid_indices = [i for i, s in enumerate(train_status) if s]
        X_train_landmarks = np.array(
            [train_landmarks[i] for i in train_valid_indices],
            dtype=np.float32
        )
        y_train_filtered = y_train[train_valid_indices]
        
        train_stats = self.extractor.get_statistics()
        print(f"\nTraining extraction:")
        print(f"  Successful: {train_stats['successful']}/{train_stats['total_processed']}")
        print(f"  Success rate: {train_stats['success_rate']:.2f}%")
        
        # Reset stats for test data
        self.extractor.reset_statistics()
        
        # Extract from test data
        print("\n" + "=" * 70)
        print("EXTRACTING TEST LANDMARKS")
        print("=" * 70)
        
        X_test_images = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        
        print(f"Processing {len(X_test_images)} test images...")
        
        test_landmarks, test_status = self.extractor.extract_landmarks_batch(
            X_test_images,
            show_progress=True
        )
        
        # Filter successful extractions
        test_valid_indices = [i for i, s in enumerate(test_status) if s]
        X_test_landmarks = np.array(
            [test_landmarks[i] for i in test_valid_indices],
            dtype=np.float32
        )
        y_test_filtered = y_test[test_valid_indices]
        
        test_stats = self.extractor.get_statistics()
        print(f"\nTest extraction:")
        print(f"  Successful: {test_stats['successful']}/{test_stats['total_processed']}")
        print(f"  Success rate: {test_stats['success_rate']:.2f}%")
        
        # Prepare result
        result = {
            'X_train': X_train_landmarks,
            'y_train': y_train_filtered,
            'X_test': X_test_landmarks,
            'y_test': y_test_filtered,
            'train_stats': train_stats,
            'test_stats': test_stats,
            'label_mapping': preprocessed_data.get('label_mapping', None)
        }
        
        # Save if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save landmark features
            landmark_file = output_dir / 'landmark_features.pkl'
            with open(landmark_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"\nLandmark features saved to: {landmark_file}")
            
            # Save individual sets
            train_file = output_dir / 'train_landmarks.pkl'
            with open(train_file, 'wb') as f:
                pickle.dump({
                    'X_train': X_train_landmarks,
                    'y_train': y_train_filtered
                }, f)
            
            test_file = output_dir / 'test_landmarks.pkl'
            with open(test_file, 'wb') as f:
                pickle.dump({
                    'X_test': X_test_landmarks,
                    'y_test': y_test_filtered
                }, f)
            
            print(f"Train landmarks saved to: {train_file}")
            print(f"Test landmarks saved to: {test_file}")
        
        return result
    
    def load_landmark_features(self, file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load landmark features from pickle file.
        
        Args:
            file_path: Path to pickle file
        
        Returns:
            Dictionary with landmark features
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Landmark features loaded from: {file_path}")
        return data
    
    def close(self):
        """Close the landmark extractor."""
        self.extractor.close()


def main():
    """
    Main function demonstrating Module 3 usage.
    """
    print("\n" + "=" * 70)
    print("MODULE 3: HAND LANDMARK EXTRACTION DEMO")
    print("=" * 70)
    
    # Example: Extract landmarks from a sample image
    print("\nTest 1: Single image landmark extraction")
    
    # Initialize extractor
    extractor = HandLandmarkExtractor()
    
    # Load a sample image (you would use your own image path)
    # This is just a demonstration structure
    print("Extractor initialized successfully!")
    print(f"MediaPipe Hands configured:")
    print(f"  Static image mode: {extractor.static_image_mode}")
    print(f"  Max hands: {extractor.max_num_hands}")
    print(f"  Detection confidence: {extractor.min_detection_confidence}")
    
    # Close extractor
    extractor.close()
    
    print("\n✓ Module 3 core functionality ready!")
    print("\nNext: Run test_module3.py for comprehensive testing")


if __name__ == "__main__":
    main()
