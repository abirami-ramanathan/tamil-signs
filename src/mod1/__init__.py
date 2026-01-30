"""
Module 1: TLFS23 Dataset Loading and Label Mapping

This module provides functionality for loading and managing the TLFS23
Tamil Sign Language dataset.

Main Components:
- TamilCharacterMapping: Manages label-to-character mappings
- TLFS23DatasetLoader: Main dataset loader class
- Utility functions for visualization and validation

Usage:
    from mod1 import TLFS23DatasetLoader
    
    loader = TLFS23DatasetLoader('path/to/dataset')
    dataset_info = loader.load_dataset_structure()
    df = loader.create_dataframe()

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from .dataset_loader import TamilCharacterMapping, TLFS23DatasetLoader
from .utils import (
    visualize_sample_images,
    visualize_reference_images,
    plot_class_distribution,
    validate_dataset_integrity,
    get_image_statistics,
    export_label_mappings
)

__all__ = [
    'TamilCharacterMapping',
    'TLFS23DatasetLoader',
    'visualize_sample_images',
    'visualize_reference_images',
    'plot_class_distribution',
    'validate_dataset_integrity',
    'get_image_statistics',
    'export_label_mappings'
]

__version__ = '1.0.0'
