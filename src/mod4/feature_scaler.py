"""
Module 4: Feature Dataset Construction & Scaling

This module handles feature scaling, dataset construction, and train-test splitting.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import json


class FeatureScaler:
    """
    Handles feature scaling using StandardScaler or MinMaxScaler.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature scaler.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'")
        
        self.is_fitted = False
        self.feature_stats = {}
    
    def fit(self, X: np.ndarray):
        """
        Fit the scaler on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        print(f"Fitting {self.scaler_type} scaler on {X.shape[0]} samples...")
        self.scaler.fit(X)
        self.is_fitted = True
        
        # Calculate statistics
        self.feature_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0)
        }
        
        print(f"✓ Scaler fitted successfully")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Scaled feature matrix
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X_scaled: Scaled feature matrix
        
        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def get_stats(self) -> Dict:
        """Get feature statistics."""
        return self.feature_stats


class DatasetConstructor:
    """
    Constructs the final dataset with feature scaling and train-test split.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        scaler_type: str = 'standard',
        stratify: bool = True
    ):
        """
        Initialize dataset constructor.
        
        Args:
            test_size: Proportion of dataset for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler ('standard' or 'minmax')
            stratify: Whether to stratify split by labels
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.stratify = stratify
        
        self.scaler = None
        self.dataset_info = {}
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> list:
        """Generate feature names for 63 landmark coordinates."""
        feature_names = []
        landmark_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        
        for i, name in enumerate(landmark_names):
            feature_names.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
        
        return feature_names
    
    def load_landmark_features(self, pickle_path: str) -> Dict:
        """
        Load landmark features from Module 3 output.
        
        Args:
            pickle_path: Path to landmark_features.pkl from Module 3
        
        Returns:
            Dictionary with X_train, X_test, y_train, y_test
        """
        print(f"Loading landmark features from: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"\n✓ Loaded landmark features:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Number of features: {X_train.shape[1]}")
        print(f"  Number of classes: {len(np.unique(np.concatenate([y_train, y_test])))}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean data by removing invalid entries.
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Cleaned X and y
        """
        print("\nCleaning data...")
        
        # Check for NaN or Inf
        invalid_rows = np.any(np.isnan(X) | np.isinf(X), axis=1)
        n_invalid = np.sum(invalid_rows)
        
        if n_invalid > 0:
            print(f"  Found {n_invalid} invalid entries (NaN/Inf)")
            X_clean = X[~invalid_rows]
            y_clean = y[~invalid_rows]
            print(f"  Removed invalid entries: {X.shape[0]} → {X_clean.shape[0]} samples")
            return X_clean, y_clean
        else:
            print(f"  ✓ No invalid entries found")
            return X, y
    
    def calculate_dataset_statistics(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Class distribution
        train_class_counts = Counter(y_train)
        test_class_counts = Counter(y_test)
        
        stats = {
            'dataset_size': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X_train) + len(X_test),
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(np.concatenate([y_train, y_test])))
            },
            'split_ratio': {
                'train_ratio': len(X_train) / (len(X_train) + len(X_test)),
                'test_ratio': len(X_test) / (len(X_train) + len(X_test))
            },
            'class_distribution': {
                'train': dict(train_class_counts),
                'test': dict(test_class_counts),
                'train_classes': len(train_class_counts),
                'test_classes': len(test_class_counts)
            },
            'original_features': {
                'mean': float(np.mean(X_train)),
                'std': float(np.std(X_train)),
                'min': float(np.min(X_train)),
                'max': float(np.max(X_train)),
                'median': float(np.median(X_train))
            },
            'scaled_features': {
                'train': {
                    'mean': float(np.mean(X_train_scaled)),
                    'std': float(np.std(X_train_scaled)),
                    'min': float(np.min(X_train_scaled)),
                    'max': float(np.max(X_train_scaled)),
                    'median': float(np.median(X_train_scaled))
                },
                'test': {
                    'mean': float(np.mean(X_test_scaled)),
                    'std': float(np.std(X_test_scaled)),
                    'min': float(np.min(X_test_scaled)),
                    'max': float(np.max(X_test_scaled)),
                    'median': float(np.median(X_test_scaled))
                }
            },
            'per_feature_stats': {
                'original': {
                    'means': X_train.mean(axis=0).tolist(),
                    'stds': X_train.std(axis=0).tolist(),
                    'mins': X_train.min(axis=0).tolist(),
                    'maxs': X_train.max(axis=0).tolist()
                },
                'scaled': {
                    'means': X_train_scaled.mean(axis=0).tolist(),
                    'stds': X_train_scaled.std(axis=0).tolist(),
                    'mins': X_train_scaled.min(axis=0).tolist(),
                    'maxs': X_train_scaled.max(axis=0).tolist()
                }
            }
        }
        
        return stats
    
    def construct_dataset(
        self,
        landmark_pickle_path: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Construct scaled dataset from Module 3 landmark features.
        
        Args:
            landmark_pickle_path: Path to landmark_features.pkl from Module 3
            output_dir: Directory to save outputs (if None, no files saved)
        
        Returns:
            Dictionary with scaled features and metadata
        """
        print("=" * 70)
        print("MODULE 4: FEATURE DATASET CONSTRUCTION & SCALING")
        print("=" * 70)
        
        # Step 1: Load landmark features
        data = self.load_landmark_features(landmark_pickle_path)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Step 2: Clean data
        X_train_clean, y_train_clean = self.clean_data(X_train, y_train)
        X_test_clean, y_test_clean = self.clean_data(X_test, y_test)
        
        # Step 3: Initialize and fit scaler on training data
        print(f"\nApplying {self.scaler_type} scaling...")
        self.scaler = FeatureScaler(scaler_type=self.scaler_type)
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        # Transform test data using fitted scaler
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        print(f"✓ Scaling complete")
        print(f"  Train scaled shape: {X_train_scaled.shape}")
        print(f"  Test scaled shape: {X_test_scaled.shape}")
        
        # Step 4: Calculate statistics
        print("\nCalculating dataset statistics...")
        stats = self.calculate_dataset_statistics(
            X_train_clean, X_test_clean,
            y_train_clean, y_test_clean,
            X_train_scaled, X_test_scaled
        )
        
        # Step 5: Prepare output dataset
        dataset = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_clean,
            'y_test': y_test_clean,
            'X_train_original': X_train_clean,
            'X_test_original': X_test_clean,
            'scaler': self.scaler.scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names,
            'dataset_stats': stats,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        
        # Step 6: Save outputs if directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save scaled dataset
            dataset_pkl = output_path / 'scaled_dataset.pkl'
            with open(dataset_pkl, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"\n✓ Scaled dataset saved: {dataset_pkl}")
            
            # Save statistics as JSON
            stats_json = output_path / 'dataset_statistics.json'
            with open(stats_json, 'w') as f:
                # Convert to JSON-compatible format
                stats_copy = self._make_json_serializable(stats)
                json.dump(stats_copy, f, indent=2)
            print(f"✓ Statistics saved: {stats_json}")
            
            # Save summary report
            report_path = output_path / 'dataset_report.txt'
            self._save_report(dataset, report_path)
            print(f"✓ Report saved: {report_path}")
        
        # Print summary
        self._print_summary(stats)
        
        return dataset
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj
    
    def _print_summary(self, stats: Dict):
        """Print dataset summary."""
        print("\n" + "=" * 70)
        print("DATASET CONSTRUCTION SUMMARY")
        print("=" * 70)
        
        ds = stats['dataset_size']
        print(f"\nDataset Size:")
        print(f"  Total samples: {ds['total_samples']}")
        print(f"  Training: {ds['train_samples']} ({stats['split_ratio']['train_ratio']*100:.1f}%)")
        print(f"  Testing: {ds['test_samples']} ({stats['split_ratio']['test_ratio']*100:.1f}%)")
        print(f"  Features: {ds['n_features']}")
        print(f"  Classes: {ds['n_classes']}")
        
        print(f"\nOriginal Features:")
        orig = stats['original_features']
        print(f"  Range: [{orig['min']:.4f}, {orig['max']:.4f}]")
        print(f"  Mean: {orig['mean']:.4f} ± {orig['std']:.4f}")
        
        print(f"\nScaled Features (Training):")
        scaled = stats['scaled_features']['train']
        print(f"  Range: [{scaled['min']:.4f}, {scaled['max']:.4f}]")
        print(f"  Mean: {scaled['mean']:.4f} ± {scaled['std']:.4f}")
        
        print(f"\nScaled Features (Testing):")
        scaled_test = stats['scaled_features']['test']
        print(f"  Range: [{scaled_test['min']:.4f}, {scaled_test['max']:.4f}]")
        print(f"  Mean: {scaled_test['mean']:.4f} ± {scaled_test['std']:.4f}")
        
        print("\n" + "=" * 70)
    
    def _save_report(self, dataset: Dict, report_path: Path):
        """Save detailed dataset report."""
        stats = dataset['dataset_stats']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("MODULE 4: FEATURE DATASET CONSTRUCTION & SCALING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            ds = stats['dataset_size']
            f.write(f"Total Samples: {ds['total_samples']}\n")
            f.write(f"Training Samples: {ds['train_samples']}\n")
            f.write(f"Testing Samples: {ds['test_samples']}\n")
            f.write(f"Number of Features: {ds['n_features']}\n")
            f.write(f"Number of Classes: {ds['n_classes']}\n")
            f.write(f"Scaler Type: {dataset['scaler_type']}\n")
            f.write(f"Random State: {dataset['random_state']}\n\n")
            
            f.write("TRAIN-TEST SPLIT\n")
            f.write("-" * 70 + "\n")
            split = stats['split_ratio']
            f.write(f"Training Ratio: {split['train_ratio']*100:.2f}%\n")
            f.write(f"Testing Ratio: {split['test_ratio']*100:.2f}%\n\n")
            
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            cd = stats['class_distribution']
            f.write(f"Training Classes: {cd['train_classes']}\n")
            f.write(f"Testing Classes: {cd['test_classes']}\n\n")
            
            f.write("ORIGINAL FEATURES STATISTICS\n")
            f.write("-" * 70 + "\n")
            orig = stats['original_features']
            f.write(f"Mean: {orig['mean']:.6f}\n")
            f.write(f"Std: {orig['std']:.6f}\n")
            f.write(f"Min: {orig['min']:.6f}\n")
            f.write(f"Max: {orig['max']:.6f}\n")
            f.write(f"Median: {orig['median']:.6f}\n\n")
            
            f.write("SCALED FEATURES STATISTICS (Training Set)\n")
            f.write("-" * 70 + "\n")
            scaled = stats['scaled_features']['train']
            f.write(f"Mean: {scaled['mean']:.6f}\n")
            f.write(f"Std: {scaled['std']:.6f}\n")
            f.write(f"Min: {scaled['min']:.6f}\n")
            f.write(f"Max: {scaled['max']:.6f}\n")
            f.write(f"Median: {scaled['median']:.6f}\n\n")
            
            f.write("SCALED FEATURES STATISTICS (Test Set)\n")
            f.write("-" * 70 + "\n")
            scaled_test = stats['scaled_features']['test']
            f.write(f"Mean: {scaled_test['mean']:.6f}\n")
            f.write(f"Std: {scaled_test['std']:.6f}\n")
            f.write(f"Min: {scaled_test['min']:.6f}\n")
            f.write(f"Max: {scaled_test['max']:.6f}\n")
            f.write(f"Median: {scaled_test['median']:.6f}\n\n")
            
            f.write("FEATURE NAMES (63 total)\n")
            f.write("-" * 70 + "\n")
            for i, name in enumerate(dataset['feature_names']):
                f.write(f"{i:2d}. {name}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Dataset ready for Module 5: Model Training & Selection\n")
            f.write("=" * 70 + "\n")


def load_scaled_dataset(pickle_path: str) -> Dict:
    """
    Utility function to load scaled dataset from pickle file.
    
    Args:
        pickle_path: Path to scaled_dataset.pkl
    
    Returns:
        Dictionary with scaled features and metadata
    """
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded scaled dataset from: {pickle_path}")
    print(f"  Train: {dataset['X_train'].shape}")
    print(f"  Test: {dataset['X_test'].shape}")
    print(f"  Scaler: {dataset['scaler_type']}")
    
    return dataset
