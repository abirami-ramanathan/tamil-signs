"""
Retrain with CONSERVATIVE hyperparameters to improve confidence while maintaining accuracy
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

print("="*80)
print("RETRAINING WITH CONSERVATIVE HYPERPARAMETERS")
print("="*80)

# Load dataset
dataset_path = Path(__file__).parent / "mod4" / "output" / "scaled_dataset.pkl"
print(f"\n1. Loading dataset...")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']
scaler = dataset['scaler']

print(f"   ✓ Training: {X_train.shape[0]:,} samples")
print(f"   ✓ Testing: {X_test.shape[0]:,} samples")
print(f"   ✓ Features: {X_train.shape[1]}")
print(f"   ✓ Classes: {len(np.unique(y_train))}")

# Get label mapping from old model
old_model_path = Path(__file__).parent / "mod5" / "output" / "random_forest_model.pkl"
if old_model_path.exists():
    with open(old_model_path, 'rb') as f:
        old_model_data = pickle.load(f)
        label_mapping = old_model_data.get('label_mapping', {})
else:
    label_mapping = {i: f"Class_{i}" for i in range(len(np.unique(y_train)))}

print(f"\n2. Training with OPTIMIZED parameters...")
print("   Parameters:")
print("     - n_estimators: 300 (more trees = better)")
print("     - max_depth: None (unlimited depth for complex patterns)")
print("     - min_samples_split: 2 (default, not restrictive)")
print("     - min_samples_leaf: 1 (allow detailed splits)")
print("     - max_features: None (use ALL 63 features)")
print("     - criterion: 'entropy' (information gain)")
print("     - min_impurity_decrease: 0.0001 (slight regularization)")

start_time = time.time()

# Train with BETTER parameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,  # USE ALL FEATURES!
    criterion='entropy',
    min_impurity_decrease=0.0001,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n3. Training...")
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✓ Training completed in {training_time/60:.1f} minutes")

# Evaluate
print(f"\n4. Evaluating...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Overfitting Gap: {(train_accuracy-test_accuracy)*100:.2f}%")

# Confidence analysis
print(f"\n5. Analyzing confidence scores...")
y_test_proba = model.predict_proba(X_test)
max_probas = np.max(y_test_proba, axis=1)

print(f"   Mean: {np.mean(max_probas)*100:.2f}%")
print(f"   Median: {np.median(max_probas)*100:.2f}%")
print(f"   Min: {np.min(max_probas)*100:.2f}%")
print(f"   Max: {np.max(max_probas)*100:.2f}%")

# Distribution
conf_ranges = [
    (0.0, 0.2, "Very Low (0-20%)"),
    (0.2, 0.4, "Low (20-40%)"),
    (0.4, 0.6, "Medium (40-60%)"),
    (0.6, 0.8, "Good (60-80%)"),
    (0.8, 1.0, "Excellent (80-100%)")
]

print(f"\n   Distribution:")
for low, high, label in conf_ranges:
    count = np.sum((max_probas >= low) & (max_probas < high))
    pct = (count / len(max_probas)) * 100
    print(f"      {label}: {count:,} ({pct:.1f}%)")

# Save model
output_path = Path(__file__).parent / "mod5" / "output" / "random_forest_model.pkl"
print(f"\n6. Saving model to: {output_path}")

model_data = {
    'model': model,
    'scaler': scaler,
    'label_mapping': label_mapping,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'mean_confidence': np.mean(max_probas)
}

with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

file_size = output_path.stat().st_size / (1024**3)
print(f"   ✓ Saved! Size: {file_size:.2f} GB")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults:")
print(f"  ✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  ✓ Mean Confidence: {np.mean(max_probas)*100:.2f}%")
print(f"  ✓ Training Time: {training_time/60:.1f} minutes")
print("="*80)
