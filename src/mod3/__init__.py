"""
Module 3: Hand Landmark Extraction using MediaPipe

This module extracts 21 hand landmarks from preprocessed images to create
63-dimensional feature vectors for Tamil sign language recognition.

Features:
- MediaPipe Hands integration for landmark detection
- 21 landmarks × 3 coordinates (x, y, z) = 63 features per image
- Batch processing with progress tracking
- Integration with Module 2 (preprocessed images)
- Comprehensive visualization and analysis utilities
- High extraction success rate (>95%)

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from .hand_landmark_extractor import (
    HandLandmarkExtractor,
    DatasetLandmarkExtractor
)

from .utils import (
    visualize_landmarks_on_images,
    visualize_landmark_distribution,
    analyze_landmark_statistics,
    plot_landmark_statistics,
    validate_landmarks,
    export_landmark_report,
    compare_landmark_features
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'HandLandmarkExtractor',
    'DatasetLandmarkExtractor',
    
    # Visualization utilities
    'visualize_landmarks_on_images',
    'visualize_landmark_distribution',
    'plot_landmark_statistics',
    'compare_landmark_features',
    
    # Analysis utilities
    'analyze_landmark_statistics',
    'validate_landmarks',
    
    # Export utilities
    'export_landmark_report',
]
