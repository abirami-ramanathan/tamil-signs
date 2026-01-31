"""
Export scaled features to CSV format.

This script loads the scaled features from pickle files and exports them
to CSV files with class names and all 63 features (21 landmarks × 3 coordinates).

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
from mod1 import TLFS23DatasetLoader


def export_scaled_to_csv(pickle_path, output_dir, dataset_path):
    """
    Export scaled features to CSV format.
    
    Args:
        pickle_path: Path to the scaled_dataset.pkl file
        output_dir: Directory to save the CSV files
        dataset_path: Path to the TLFS23 dataset to load label mapping
    """
    print("=" * 70)
    print("EXPORTING SCALED DATASET TO CSV")
    print("=" * 70)
    
    # Load label mapping from dataset
    print("\nLoading label mapping from dataset...")
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Create label mapping: tamil character -> label number
    label_mapping = {}
    for label, class_info in loader.class_paths.items():
        label_mapping[class_info['tamil_char']] = label
    
    # Create reverse label mapping (numeric -> class name)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    print(f"✓ Loaded {len(label_mapping)} class labels")
    
    # Load pickle data
    print(f"\nLoading scaled data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\n✓ Loaded scaled dataset:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Scaler type: {data.get('scaler_type', 'unknown')}")
    
    # Create column names for landmarks
    landmark_columns = []
    for landmark_idx in range(21):  # 21 landmarks
        landmark_columns.append(f'landmark_{landmark_idx}_x')
        landmark_columns.append(f'landmark_{landmark_idx}_y')
        landmark_columns.append(f'landmark_{landmark_idx}_z')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export combined dataset (train + test)
    print("\n" + "=" * 70)
    print("EXPORTING COMBINED DATASET (TRAIN + TEST)")
    print("=" * 70)
    
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    class_names_all = [reverse_mapping[label] for label in y_all]
    
    df_combined = pd.DataFrame(X_all, columns=landmark_columns)
    df_combined.insert(0, 'class_name', class_names_all)
    df_combined.insert(1, 'class_label', y_all)
    
    combined_csv = output_dir / 'scaled_features.csv'
    df_combined.to_csv(combined_csv, index=False)
    
    print(f"\n✓ CSV file saved to: {combined_csv}")
    print(f"  Shape: {df_combined.shape}")
    
    # Display sample
    print("\nSample rows:")
    print(df_combined[['class_name', 'class_label', 'landmark_0_x', 'landmark_0_y', 'landmark_0_z']].head(10))
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = df_combined['class_name'].value_counts()
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Samples per class range: {class_counts.min()} - {class_counts.max()}")
    
    # Export train set
    print("\n" + "=" * 70)
    print("EXPORTING TRAIN SET")
    print("=" * 70)
    
    class_names_train = [reverse_mapping[label] for label in y_train]
    
    df_train = pd.DataFrame(X_train, columns=landmark_columns)
    df_train.insert(0, 'class_name', class_names_train)
    df_train.insert(1, 'class_label', y_train)
    
    train_csv = output_dir / 'scaled_train.csv'
    df_train.to_csv(train_csv, index=False)
    
    print(f"\n✓ CSV file saved to: {train_csv}")
    print(f"  Shape: {df_train.shape}")
    
    # Export test set
    print("\n" + "=" * 70)
    print("EXPORTING TEST SET")
    print("=" * 70)
    
    class_names_test = [reverse_mapping[label] for label in y_test]
    
    df_test = pd.DataFrame(X_test, columns=landmark_columns)
    df_test.insert(0, 'class_name', class_names_test)
    df_test.insert(1, 'class_label', y_test)
    
    test_csv = output_dir / 'scaled_test.csv'
    df_test.to_csv(test_csv, index=False)
    
    print(f"\n✓ CSV file saved to: {test_csv}")
    print(f"  Shape: {df_test.shape}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"\nExported 3 CSV files:")
    print(f"  1. {combined_csv.name} - Combined dataset ({len(df_combined)} samples)")
    print(f"  2. {train_csv.name} - Training set ({len(df_train)} samples)")
    print(f"  3. {test_csv.name} - Test set ({len(df_test)} samples)")
    print(f"\nEach CSV contains:")
    print(f"  - class_name: Tamil character")
    print(f"  - class_label: Numeric label (0-{len(label_mapping)-1})")
    print(f"  - {len(landmark_columns)} feature columns (scaled values)")
    print("\n" + "=" * 70)
    
    return df_combined, df_train, df_test


def main():
    """Main execution function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    mod4_output_dir = project_root / "mod4" / "output"  # Use project root mod4/output
    dataset_path = project_root / "TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Path to scaled dataset
    scaled_pickle = mod4_output_dir / "scaled_dataset.pkl"
    
    if not scaled_pickle.exists():
        print(f"Error: Scaled dataset not found at {scaled_pickle}")
        print("Please run Module 4 tests first to generate the scaled dataset.")
        return
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the TLFS23 dataset is available.")
        return
    
    # Export to CSV
    export_scaled_to_csv(
        pickle_path=scaled_pickle,
        output_dir=mod4_output_dir,
        dataset_path=dataset_path
    )


if __name__ == "__main__":
    main()
