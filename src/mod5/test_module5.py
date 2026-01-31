"""
Module 5: Test Script for Model Training & Selection

This script trains and compares multiple classifiers for Tamil sign language recognition.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod5 import ModelComparator


def train_and_compare_models():
    """Train and compare all models."""
    
    # Paths
    scaled_dataset_path = Path(__file__).parent.parent / "mod4" / "output" / "scaled_dataset.pkl"
    output_dir = Path(__file__).parent / "output"
    
    if not scaled_dataset_path.exists():
        print(f"\n✗ Scaled dataset not found: {scaled_dataset_path}")
        print("  Please run Module 4 tests first!")
        return
    
    # Initialize comparator
    comparator = ModelComparator(random_state=42)
    
    # Load data
    dataset = comparator.load_data(str(scaled_dataset_path))
    
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    # Train all models
    comparator.train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    comparison_df = comparator.compare_models()
    
    # Print detailed reports
    comparator.print_detailed_reports()
    
    # Save results
    comparator.save_results(str(output_dir), dataset)
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nAll models saved to: {output_dir}")
    print("\nModule 5 is ready for integration with Module 6!")
    print("Next: Real-Time Prediction & User Interface")


if __name__ == "__main__":
    train_and_compare_models()
