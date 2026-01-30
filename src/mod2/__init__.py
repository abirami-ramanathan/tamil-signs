"""
Module 2: Image Preprocessing for Tamil Sign Language Recognition

This module handles preprocessing of raw TLFS23 dataset images to prepare them
for MediaPipe hand landmark extraction.

Features:
- RGB format conversion
- Optional image resizing
- Normalization support
- Batch processing with progress tracking
- Train/test splitting with stratification
- Quality validation
- Comprehensive utilities

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from .image_preprocessor import ImagePreprocessor, DatasetPreprocessor
from .utils import (
    visualize_preprocessed_images,
    compare_original_vs_preprocessed,
    analyze_image_statistics,
    plot_preprocessing_statistics,
    visualize_data_distribution,
    check_image_quality_batch,
    export_preprocessing_report,
    validate_preprocessed_data
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'ImagePreprocessor',
    'DatasetPreprocessor',
    
    # Visualization utilities
    'visualize_preprocessed_images',
    'compare_original_vs_preprocessed',
    'plot_preprocessing_statistics',
    'visualize_data_distribution',
    
    # Analysis utilities
    'analyze_image_statistics',
    'check_image_quality_batch',
    'validate_preprocessed_data',
    
    # Export utilities
    'export_preprocessing_report',
]
