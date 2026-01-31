"""
Test a simple Random Forest to understand what's wrong
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# Load the scaled dataset
scaled_file = Path(r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\mod4\output\scaled_dataset.pkl")

with open(scaled_file, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print("=" * 70)
print("SIMPLE RANDOM FOREST TEST")
print("=" * 70)

# Train a simple Random Forest
print("\nTraining Random Forest with 100 trees...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nResults:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")

# Check prediction distribution
test_pred_counts = Counter(test_pred)
print(f"\nTest predictions distribution:")
print(f"  Unique classes predicted: {len(test_pred_counts)}")
print(f"  Most common predictions:")
for class_id, count in test_pred_counts.most_common(10):
    actual_count = np.sum(y_test == class_id)
    print(f"    Class {class_id}: predicted {count} times (actual: {actual_count})")

# Check if model is predicting mostly one class
if len(test_pred_counts) < 50:
    print(f"\n⚠ WARNING: Model only predicting {len(test_pred_counts)} out of 247 classes!")

# Check feature importances
importances = rf.feature_importances_
print(f"\nTop 10 most important features:")
top_indices = np.argsort(importances)[-10:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {importances[idx]:.4f}")

print("\n" + "=" * 70)
