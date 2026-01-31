"""
Module 4: Feature Dataset Construction & Scaling

This module consolidates extracted hand landmark features into a structured dataset
suitable for machine learning, applies feature scaling/normalization, and performs
train-test splitting with stratification.

Components:
- FeatureScaler: Handles feature scaling using StandardScaler or MinMaxScaler
- DatasetConstructor: Constructs final dataset with train-test split
- Utilities: Data validation and statistics functions

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from .feature_scaler import FeatureScaler, DatasetConstructor

__all__ = ['FeatureScaler', 'DatasetConstructor']
