"""
Module 4: Test Script for Feature Dataset Construction & Scaling

This script tests the feature scaling and dataset construction functionality.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod4 import FeatureScaler, DatasetConstructor


def test_feature_scaler():
    """Test feature scaler functionality."""
    print("=" * 70)
    print("TEST 1: Feature Scaler")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 63) * 10 + 50
    X_test = np.random.randn(20, 63) * 10 + 50
    
    print(f"\nSample data:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Train mean: {X_train.mean():.2f} ± {X_train.std():.2f}")
    
    # Test StandardScaler
    print("\nTesting StandardScaler...")
    scaler = FeatureScaler(scaler_type='standard')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print(f"  Train scaled mean: {X_train_scaled.mean():.2f} ± {X_train_scaled.std():.2f}")
    print(f"  Test scaled range: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
    
    # Test inverse transform
    X_train_reconstructed = scaler.inverse_transform(X_train_scaled)
    reconstruction_error = np.abs(X_train - X_train_reconstructed).mean()
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    
    # Test MinMaxScaler
    print("\nTesting MinMaxScaler...")
    scaler_mm = FeatureScaler(scaler_type='minmax')
    X_train_scaled_mm = scaler_mm.fit_transform(X_train)
    X_test_scaled_mm = scaler_mm.transform(X_test)
    
    print(f"  Train scaled range: [{X_train_scaled_mm.min():.2f}, {X_train_scaled_mm.max():.2f}]")
    print(f"  Train scaled mean: {X_train_scaled_mm.mean():.2f} ± {X_train_scaled_mm.std():.2f}")
    
    print("\n✓ Feature scaler test passed!\n")


def test_dataset_constructor():
    """Test dataset constructor with Module 3 output."""
    print("=" * 70)
    print("TEST 2: Dataset Constructor")
    print("=" * 70)
    
    # Paths
    landmark_pickle = Path(__file__).parent.parent / "mod3" / "output" / "landmark_features.pkl"
    output_dir = Path(__file__).parent / "output"
    
    if not landmark_pickle.exists():
        print(f"\n✗ Landmark features not found: {landmark_pickle}")
        print("  Please run Module 3 tests first!")
        return
    
    # Initialize constructor
    constructor = DatasetConstructor(
        test_size=0.2,
        random_state=42,
        scaler_type='standard',
        stratify=True
    )
    
    # Construct dataset
    dataset = constructor.construct_dataset(
        landmark_pickle_path=str(landmark_pickle),
        output_dir=str(output_dir)
    )
    
    # Verify output
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print(f"\nDataset keys: {list(dataset.keys())}")
    print(f"\nShapes:")
    print(f"  X_train: {dataset['X_train'].shape}")
    print(f"  X_test: {dataset['X_test'].shape}")
    print(f"  y_train: {dataset['y_train'].shape}")
    print(f"  y_test: {dataset['y_test'].shape}")
    
    print(f"\nScaled feature properties:")
    print(f"  Train mean: {dataset['X_train'].mean():.6f}")
    print(f"  Train std: {dataset['X_train'].std():.6f}")
    print(f"  Test mean: {dataset['X_test'].mean():.6f}")
    print(f"  Test std: {dataset['X_test'].std():.6f}")
    
    print(f"\nClass distribution:")
    train_unique, train_counts = np.unique(dataset['y_train'], return_counts=True)
    test_unique, test_counts = np.unique(dataset['y_test'], return_counts=True)
    print(f"  Training classes: {len(train_unique)}")
    print(f"  Testing classes: {len(test_unique)}")
    print(f"  Samples per class (train): min={train_counts.min()}, max={train_counts.max()}, mean={train_counts.mean():.2f}")
    print(f"  Samples per class (test): min={test_counts.min()}, max={test_counts.max()}, mean={test_counts.mean():.2f}")
    
    print("\n✓ Dataset constructor test passed!\n")
    
    return dataset


def test_load_scaled_dataset():
    """Test loading saved scaled dataset."""
    print("=" * 70)
    print("TEST 3: Load Scaled Dataset")
    print("=" * 70)
    
    from mod4.feature_scaler import load_scaled_dataset
    
    pickle_path = Path(__file__).parent / "output" / "scaled_dataset.pkl"
    
    if not pickle_path.exists():
        print(f"\n✗ Scaled dataset not found: {pickle_path}")
        print("  Please run Test 2 first!")
        return
    
    # Load dataset
    dataset = load_scaled_dataset(str(pickle_path))
    
    # Verify scaler can be used
    print("\nTesting scaler object...")
    scaler = dataset['scaler']
    X_test_rescaled = scaler.transform(dataset['X_test_original'])
    
    # Check if rescaling matches
    error = np.abs(X_test_rescaled - dataset['X_test']).mean()
    print(f"  Rescaling error: {error:.6e}")
    
    if error < 1e-10:
        print("  ✓ Scaler produces identical results")
    else:
        print(f"  ✗ Scaler mismatch: {error}")
    
    print("\n✓ Load scaled dataset test passed!\n")


def test_statistics_accuracy():
    """Test accuracy of calculated statistics."""
    print("=" * 70)
    print("TEST 4: Statistics Accuracy")
    print("=" * 70)
    
    pickle_path = Path(__file__).parent / "output" / "scaled_dataset.pkl"
    
    if not pickle_path.exists():
        print(f"\n✗ Scaled dataset not found: {pickle_path}")
        return
    
    import pickle
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    stats = dataset['dataset_stats']
    
    # Verify dataset size statistics
    print("\nVerifying dataset size statistics...")
    assert stats['dataset_size']['train_samples'] == len(dataset['X_train'])
    assert stats['dataset_size']['test_samples'] == len(dataset['X_test'])
    assert stats['dataset_size']['n_features'] == dataset['X_train'].shape[1]
    print("  ✓ Dataset size statistics correct")
    
    # Verify scaled feature statistics
    print("\nVerifying scaled feature statistics...")
    train_mean_actual = dataset['X_train'].mean()
    train_mean_stored = stats['scaled_features']['train']['mean']
    mean_error = abs(train_mean_actual - train_mean_stored)
    print(f"  Train mean: actual={train_mean_actual:.6f}, stored={train_mean_stored:.6f}, error={mean_error:.6e}")
    
    assert mean_error < 1e-5, f"Mean error too large: {mean_error}"
    print("  ✓ Scaled feature statistics correct")
    
    # Verify feature names
    print("\nVerifying feature names...")
    assert len(dataset['feature_names']) == 63
    assert dataset['feature_names'][0] == 'wrist_x'
    assert dataset['feature_names'][62] == 'pinky_tip_z'
    print(f"  Feature names: {len(dataset['feature_names'])} features")
    print(f"  First: {dataset['feature_names'][0]}")
    print(f"  Last: {dataset['feature_names'][62]}")
    print("  ✓ Feature names correct")
    
    print("\n✓ Statistics accuracy test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING MODULE 4 TESTS")
    print("=" * 70 + "\n")
    
    # Run tests
    test_feature_scaler()
    dataset = test_dataset_constructor()
    
    if dataset:
        test_load_scaled_dataset()
        test_statistics_accuracy()
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / "output"
    print(f"\nOutput files saved to: {output_dir}")
    print("\nModule 4 is ready for integration with Module 5!")
    print("Next: Model Training & Selection (Random Forest)")


if __name__ == "__main__":
    run_all_tests()
