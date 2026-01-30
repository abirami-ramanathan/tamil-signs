"""
Export landmark features to CSV format.

This script loads the landmark features from pickle files and exports them
to a CSV file with class names and all 63 features (21 landmarks × 3 coordinates).

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


def export_landmarks_to_csv(pickle_path, output_csv_path, label_mapping):
    """
    Export landmark features to CSV format.
    
    Args:
        pickle_path: Path to the landmark_features.pkl file
        output_csv_path: Path to save the CSV file
        label_mapping: Dictionary mapping class names to numeric labels
    """
    print(f"Loading landmark data from: {pickle_path}")
    
    # Load pickle data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Combine train and test data
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    print(f"\nTotal samples: {len(X_all)}")
    print(f"Features per sample: {X_all.shape[1]}")
    print(f"Number of classes: {len(label_mapping)}")
    
    # Create column names for landmarks
    landmark_columns = []
    for landmark_idx in range(21):  # 21 landmarks
        landmark_columns.append(f'landmark_{landmark_idx}_x')
        landmark_columns.append(f'landmark_{landmark_idx}_y')
        landmark_columns.append(f'landmark_{landmark_idx}_z')
    
    # Create reverse label mapping (numeric -> class name)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Convert numeric labels to class names
    class_names = [reverse_mapping[label] for label in y_all]
    
    # Create DataFrame
    df = pd.DataFrame(X_all, columns=landmark_columns)
    df.insert(0, 'class_name', class_names)
    df.insert(1, 'class_label', y_all)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ CSV file saved to: {output_csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    
    # Display sample
    print("\nSample rows:")
    print(df[['class_name', 'class_label', 'landmark_0_x', 'landmark_0_y', 'landmark_0_z', 
              'landmark_1_x', 'landmark_1_y', 'landmark_1_z']].head(10))
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = df['class_name'].value_counts().sort_index()
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Samples per class range: {class_counts.min()} - {class_counts.max()}")
    print(f"\nTop 10 classes by sample count:")
    print(class_counts.head(10))
    
    return df


def export_separate_train_test_csv(output_dir, dataset_path):
    """
    Export separate CSV files for train and test sets.
    
    Args:
        output_dir: Directory containing pickle files and where CSVs will be saved
        dataset_path: Path to the TLFS23 dataset to load label mapping
    """
    output_dir = Path(output_dir)
    
    # Load label mapping from dataset
    print("Loading label mapping from dataset...")
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Create label mapping: tamil character -> label number
    label_mapping = {}
    for label, class_info in loader.class_paths.items():
        label_mapping[class_info['tamil_char']] = label
    
    print(f"✓ Loaded {len(label_mapping)} class labels\n")
    
    # Export combined dataset
    print("=" * 70)
    print("EXPORTING COMBINED DATASET (TRAIN + TEST)")
    print("=" * 70)
    
    combined_csv = output_dir / 'landmark_features.csv'
    df_combined = export_landmarks_to_csv(
        output_dir / 'landmark_features.pkl',
        combined_csv,
        label_mapping
    )
    
    # Export train set only
    print("\n" + "=" * 70)
    print("EXPORTING TRAIN SET")
    print("=" * 70)
    
    with open(output_dir / 'landmark_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Create landmark columns
    landmark_columns = []
    for landmark_idx in range(21):
        landmark_columns.append(f'landmark_{landmark_idx}_x')
        landmark_columns.append(f'landmark_{landmark_idx}_y')
        landmark_columns.append(f'landmark_{landmark_idx}_z')
    
    # Create train DataFrame
    df_train = pd.DataFrame(X_train, columns=landmark_columns)
    df_train.insert(0, 'class_name', [reverse_mapping[label] for label in y_train])
    df_train.insert(1, 'class_label', y_train)
    
    train_csv = output_dir / 'train_landmarks.csv'
    df_train.to_csv(train_csv, index=False)
    print(f"\n✓ Train CSV saved: {train_csv}")
    print(f"  Shape: {df_train.shape}")
    
    # Export test set only
    print("\n" + "=" * 70)
    print("EXPORTING TEST SET")
    print("=" * 70)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Create test DataFrame
    df_test = pd.DataFrame(X_test, columns=landmark_columns)
    df_test.insert(0, 'class_name', [reverse_mapping[label] for label in y_test])
    df_test.insert(1, 'class_label', y_test)
    
    test_csv = output_dir / 'test_landmarks.csv'
    df_test.to_csv(test_csv, index=False)
    print(f"\n✓ Test CSV saved: {test_csv}")
    print(f"  Shape: {df_test.shape}")
    
    print("\n" + "=" * 70)
    print("CSV EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. {combined_csv.name} - Combined train + test")
    print(f"  2. {train_csv.name} - Training set only")
    print(f"  3. {test_csv.name} - Test set only")


if __name__ == "__main__":
    # Set paths
    output_dir = Path(__file__).parent / "output"
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Export all CSV files
    export_separate_train_test_csv(output_dir, dataset_path)
