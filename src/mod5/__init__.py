"""
Module 5: Model Training & Selection

This module trains multiple classifiers (Random Forest, SVM, Gradient Boosting, XGBoost)
and compares their performance on Tamil sign language recognition.

Components:
- ModelTrainer: Trains and evaluates multiple models
- ModelComparator: Compares performance across models
- Utilities: Visualization and reporting functions

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from .model_trainer import ModelTrainer, ModelComparator

__all__ = ['ModelTrainer', 'ModelComparator']
