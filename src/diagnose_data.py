"""
Diagnose data distribution and check for label/data issues
"""

import pickle
import numpy as np
from collections import Counter
from pathlib import Path

# Load the scaled dataset
scaled_file = Path(r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\mod4\output\scaled_dataset.pkl")

with open(scaled_file, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print("=" * 70)
print("DATASET DIAGNOSIS")
print("=" * 70)

print(f"\nDataset shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

print(f"\nTotal samples: {len(X_train) + len(X_test)}")
print(f"Total classes: {len(np.unique(np.concatenate([y_train, y_test])))}")

# Check class distribution
train_counts = Counter(y_train)
test_counts = Counter(y_test)

print(f"\nClass distribution in training:")
print(f"  Min samples per class: {min(train_counts.values())}")
print(f"  Max samples per class: {max(train_counts.values())}")
print(f"  Mean samples per class: {np.mean(list(train_counts.values())):.2f}")
print(f"  Median samples per class: {np.median(list(train_counts.values())):.0f}")

print(f"\nClass distribution in test:")
print(f"  Min samples per class: {min(test_counts.values())}")
print(f"  Max samples per class: {max(test_counts.values())}")
print(f"  Mean samples per class: {np.mean(list(test_counts.values())):.2f}")
print(f"  Median samples per class: {np.median(list(test_counts.values())):.0f}")

# Check for classes with very few samples
print(f"\nClasses with < 10 training samples:")
few_samples = {k: v for k, v in train_counts.items() if v < 10}
print(f"  Count: {len(few_samples)}")
if len(few_samples) > 0 and len(few_samples) <= 20:
    for class_id, count in sorted(few_samples.items(), key=lambda x: x[1]):
        print(f"    Class {class_id}: {count} samples")

# Check feature statistics
print(f"\nFeature statistics:")
print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"  X_train mean: {X_train.mean():.4f} ± {X_train.std():.4f}")
print(f"  X_test range: [{X_test.min():.4f}, {X_test.max():.4f}]")
print(f"  X_test mean: {X_test.mean():.4f} ± {X_test.std():.4f}")

# Check for NaN or Inf
print(f"\nData quality:")
print(f"  NaN in X_train: {np.isnan(X_train).any()}")
print(f"  Inf in X_train: {np.isinf(X_train).any()}")
print(f"  NaN in X_test: {np.isnan(X_test).any()}")
print(f"  Inf in X_test: {np.isinf(X_test).any()}")

# Check label encoding
print(f"\nLabel encoding:")
print(f"  y_train unique values: {len(np.unique(y_train))}")
print(f"  y_train range: [{y_train.min()}, {y_train.max()}]")
print(f"  y_train dtype: {y_train.dtype}")
print(f"  y_test unique values: {len(np.unique(y_test))}")
print(f"  y_test range: [{y_test.min()}, {y_test.max()}]")
print(f"  y_test dtype: {y_test.dtype}")

# Check for label overlap
train_classes = set(y_train)
test_classes = set(y_test)
missing_in_test = train_classes - test_classes
missing_in_train = test_classes - train_classes

if missing_in_test:
    print(f"\n⚠ WARNING: {len(missing_in_test)} classes in train but NOT in test!")
    if len(missing_in_test) <= 20:
        print(f"  Classes: {sorted(missing_in_test)}")

if missing_in_train:
    print(f"\n⚠ WARNING: {len(missing_in_train)} classes in test but NOT in train!")
    if len(missing_in_train) <= 20:
        print(f"  Classes: {sorted(missing_in_train)}")

print("\n" + "=" * 70)
